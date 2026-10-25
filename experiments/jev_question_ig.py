#!/usr/bin/env python3
"""
Does Jev prefer high information-gain questions, in the right proportions?

Stress test of the 20-Questions finding at scale, using the CUQ dataset
(128 secrets, 122,913 questions with precomputed oracle bitmasks).

For a game state (a feasible set of items), each candidate question has an
exact expected information gain, computable from its bitmask. We offer Jev a
slate of K questions and ask which best narrows the remaining possibilities,
reading its probabilities + confidence. We then check:

  1. Top-1: is Jev's chosen question the true max-IG question? (vs 1/K random)
  2. Proportions: do Jev's probabilities track the IG vector across the slate
     (Spearman rank correlation, and cosine to the IG-normalized vector)?
  3. Split preference: does Jev down-weight uninformative questions, i.e. is its
     probability mass negatively related to |split_ratio - 0.5|?

Ground-truth IG uses the CUQ bitmasks (T5-XL oracle). That oracle is noisy
(reported nearest-neighbor Jaccard 0.185), and Jev does not see it, so some
disagreement reflects Jev's item-knowledge differing from T5's answers rather
than a failure to value information gain. A Jev-as-oracle self-consistent
variant (rebuild the slate's bitmasks with Jev) would remove that confound and
is the natural follow-up. Bitmask helpers are adapted from
belief-trajectory/src/bitmask.py.

Usage:
    python experiments/jev_question_ig.py --quick
    python experiments/jev_question_ig.py --n-trials 300 --slate 6 \
        --data-dir ~/workspace/repos/genai/bridge-rl-portfolio-private/flan-oracle/cuq/release \
        -o results/esi/jev_question_ig.json
"""

import argparse
import json
import math
import os
import random
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.jev_provider import JevProvider

DEFAULT_DATA_DIR = Path(
    os.path.expanduser("~/workspace/repos/genai/bridge-rl-portfolio-private/flan-oracle/cuq/release")
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_question_ig.json"
NUM_ITEMS = 128


# ---- bitmask ops (adapted from belief-trajectory/src/bitmask.py) ----------- #
def popcount(s: int) -> int:
    return bin(s).count("1")


def full_state(n: int = NUM_ITEMS) -> int:
    return (1 << n) - 1


def yes_set(state: int, mask: int) -> int:
    return state & mask


def no_set(state: int, mask: int) -> int:
    return state & ~mask


def split_ratio(state: int, mask: int) -> float:
    total = popcount(state)
    return popcount(yes_set(state, mask)) / total if total else 0.0


def state_entropy(state: int) -> float:
    k = popcount(state)
    return math.log2(k) if k > 1 else 0.0


def information_gain(state: int, mask: int) -> float:
    k = popcount(state)
    if k <= 1:
        return 0.0
    k_yes = popcount(yes_set(state, mask))
    k_no = k - k_yes
    if k_yes == 0 or k_no == 0:
        return 0.0
    p_yes, p_no = k_yes / k, k_no / k
    return math.log2(k) - (p_yes * math.log2(k_yes) + p_no * math.log2(k_no))


def update_state(state: int, mask: int, answer: bool) -> int:
    return yes_set(state, mask) if answer else no_set(state, mask)


def answer_for_secret(mask: int, secret_idx: int) -> bool:
    return bool(mask & (1 << secret_idx))


def items_in_state(state: int):
    out, i, s = [], 0, state
    while s:
        if s & 1:
            out.append(i)
        s >>= 1
        i += 1
    return out


# ---- stats ----------------------------------------------------------------- #
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rk[order[k]] = avg
        i = j + 1
    return rk


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def cosine(x, y):
    num = sum(a * b for a, b in zip(x, y))
    den = math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y))
    return num / den if den else float("nan")


# ---- data ------------------------------------------------------------------ #
def load_items(data_dir: Path):
    return [l.strip() for l in (data_dir / "items.txt").read_text().splitlines() if l.strip()]


def load_question_pool(data_dir: Path, pool_size: int, rng: random.Random):
    """Reservoir-sample a pool of (qid, text, mask) from questions.jsonl."""
    pool = []
    with (data_dir / "questions.jsonl").open() as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            rec = (d["question_id"], d["question"], int(d["bitmask"]))
            if len(pool) < pool_size:
                pool.append(rec)
            else:
                j = rng.randint(0, n)
                if j < pool_size:
                    pool[j] = rec
    return pool


# ---- trial ----------------------------------------------------------------- #
def make_state(pool, rng, max_depth):
    """Reachable state: apply truthful answers for a random secret."""
    secret = rng.randrange(NUM_ITEMS)
    state = full_state()
    depth = rng.randint(0, max_depth)
    for (_qid, _t, mask) in rng.sample(pool, depth):
        state = update_state(state, mask, answer_for_secret(mask, secret))
        if popcount(state) <= 2:
            break
    return state


def make_slate(pool, state, k, rng, tries=40):
    """Slate of k questions with non-degenerate, varied IG under state."""
    for _ in range(tries):
        slate = rng.sample(pool, k)
        igs = [information_gain(state, m) for (_q, _t, m) in slate]
        if max(igs) > 0 and len({round(x, 6) for x in igs}) >= 2:
            return slate, igs
    return None, None


def run_trial(provider, items, pool, rng, k, max_depth):
    state = make_state(pool, rng, max_depth)
    if popcount(state) < 4:
        return None
    slate, igs = make_slate(pool, state, k, rng)
    if slate is None:
        return None

    cand = [items[i] for i in items_in_state(state)]
    cand_str = ", ".join(cand)
    q_state = (
        "You are playing 20 Questions and choosing your next yes/no question.\n"
        f"The secret is one of these {len(cand)} remaining items:\n{cand_str}\n\n"
        "Pick the question that would best narrow down which item is the secret."
    )
    criteria = {str(i): text for i, (_qid, text, _m) in enumerate(slate)}
    action = provider.choice_probs({"question": q_state, "options": criteria})
    probs = [action.probs.get(str(i), 0.0) for i in range(k)]

    ig_argmax = max(range(k), key=lambda i: igs[i])
    jev_argmax = max(range(k), key=lambda i: probs[i])
    ig_norm = [x / sum(igs) for x in igs] if sum(igs) else [1.0 / k] * k
    splits = [split_ratio(state, m) for (_q, _t, m) in slate]

    return {
        "state_popcount": popcount(state),
        "state_entropy": state_entropy(state),
        "slate": [
            {"qid": qid, "question": text, "ig": igs[i], "split_ratio": splits[i], "jev_prob": probs[i]}
            for i, (qid, text, _m) in enumerate(slate)
        ],
        "ig": igs,
        "jev_probs": probs,
        "confidence": (action.metadata or {}).get("jev_confidence"),
        "ig_argmax": ig_argmax,
        "jev_argmax": jev_argmax,
        "top1_hit": ig_argmax == jev_argmax,
        "spearman_prob_ig": spearman(probs, igs),
        "cosine_prob_ignorm": cosine(probs, ig_norm),
        "corr_prob_splitdist": pearson(probs, [abs(s - 0.5) for s in splits]),
        "max_ig": max(igs),
        "ig_at_jev_choice": igs[jev_argmax],
    }


def main():
    ap = argparse.ArgumentParser(description="Jev question choice vs information gain (CUQ).")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--slate", type=int, default=6, help="K questions per slate.")
    ap.add_argument("--pool-size", type=int, default=5000, help="Question pool to sample slates/states from.")
    ap.add_argument("--max-depth", type=int, default=4, help="Max clues applied when building a state.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--quick", action="store_true", help="30 trials, pool 1500.")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    if args.quick:
        args.n_trials, args.pool_size = 30, 1500

    rng = random.Random(args.seed)
    items = load_items(args.data_dir)
    assert len(items) == NUM_ITEMS, f"expected {NUM_ITEMS} items, got {len(items)}"
    print(f"Loading question pool ({args.pool_size}) from {args.data_dir} ...")
    pool = load_question_pool(args.data_dir, args.pool_size, rng)

    provider = JevProvider(model_name=args.model)
    print(f"Running {args.n_trials} trials, K={args.slate} ...")

    results = []
    attempts = 0
    while len(results) < args.n_trials and attempts < args.n_trials * 4:
        attempts += 1
        try:
            r = run_trial(provider, items, pool, rng, args.slate, args.max_depth)
        except Exception as e:  # noqa: BLE001
            print(f"  trial error: {e}")
            continue
        if r is None:
            continue
        results.append(r)
        if len(results) % 25 == 0:
            acc = st.mean(1.0 if x["top1_hit"] else 0.0 for x in results)
            print(f"  {len(results)}/{args.n_trials}  running top1={acc:.3f}")

    n = len(results)
    top1 = st.mean(1.0 if r["top1_hit"] else 0.0 for r in results)
    sps = [r["spearman_prob_ig"] for r in results if not math.isnan(r["spearman_prob_ig"])]
    cos = [r["cosine_prob_ignorm"] for r in results if not math.isnan(r["cosine_prob_ignorm"])]
    csd = [r["corr_prob_splitdist"] for r in results if not math.isnan(r["corr_prob_splitdist"])]
    ig_capture = st.mean(r["ig_at_jev_choice"] / r["max_ig"] for r in results if r["max_ig"] > 0)

    # Top-1 by state-entropy bucket.
    buckets = {"low(<3b)": [], "mid(3-5b)": [], "high(>=5b)": []}
    for r in results:
        h = r["state_entropy"]
        key = "low(<3b)" if h < 3 else ("mid(3-5b)" if h < 5 else "high(>=5b)")
        buckets[key].append(1.0 if r["top1_hit"] else 0.0)

    agg = {
        "n_trials": n,
        "slate_k": args.slate,
        "random_baseline_top1": 1.0 / args.slate,
        "top1_accuracy": top1,
        "mean_spearman_prob_ig": st.mean(sps) if sps else None,
        "mean_cosine_prob_ignorm": st.mean(cos) if cos else None,
        "mean_corr_prob_splitdist": st.mean(csd) if csd else None,
        "mean_ig_capture_ratio": ig_capture,
        "mean_confidence": st.mean(r["confidence"] for r in results if r["confidence"] is not None),
        "top1_by_state_entropy": {k: (st.mean(v) if v else None) for k, v in buckets.items()},
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "data_dir": str(args.data_dir),
        "note": "Ground-truth IG uses CUQ T5-XL bitmasks (noisy oracle Jev does not see).",
        "aggregate": agg,
        "trials": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
