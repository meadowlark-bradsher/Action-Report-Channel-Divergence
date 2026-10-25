#!/usr/bin/env python3
"""
ESI-Jev - Structural-sensitivity (option-order) runner for the Jev action channel.

This is the Jev-native analog of the ORDER intervention in esi_runner.py. It
measures Layer-3 structural symmetry: is the model's answer distribution a
property of option *content*, or is it anchored to *position*?

For each question we:
  1. Generate K option-order permutations (interventions/option_order.py),
     each a semantic-preserving relabeling of the same options.
  2. Ask Jev a `choice` question per permutation (action channel only).
  3. Remap each distribution back to the original A/B/C/D labels.
  4. Measure stability across permutations.

Metrics per question (all on remapped, original-label distributions):
  - argmax_flip_rate : fraction of permutations whose argmax != the ensemble
                       (mean-distribution) argmax. The repo reports 52-64% for
                       generative models under position bias alone; a value near
                       0 here would mean Jev is largely order-invariant.
  - n_distinct_argmax: how many different options won across permutations.
  - mean_hellinger   : mean Hellinger distance of each permutation's distribution
                       to the ensemble mean (ESI_action analog).
  - ensemble_entropy : entropy of the K-averaged distribution.
  - mean_entropy     : mean per-permutation entropy.
  - mean_confidence  : mean Jev `confidence` scalar across permutations.

Report/reasoning channels are intentionally absent: Jev cannot produce them
(see jev_provider.py). This runner characterizes the action channel only.

Usage:
    python experiments/esi_jev.py --quick
    python experiments/esi_jev.py --variants 6 --categories ambiguous adversarial \
        -o results/esi/jev_order.jsonl
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.jev_provider import JevProvider
from experiments.providers import OPTIONS
from interventions.option_order import shuffle_options_batch, remap_probs

DEFAULT_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "mc_questions.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_order.jsonl"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def entropy(probs: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def hellinger_distance(p: Dict[str, float], q: Dict[str, float], labels: List[str]) -> float:
    sum_sq = sum((math.sqrt(p.get(k, 0.0)) - math.sqrt(q.get(k, 0.0))) ** 2 for k in labels)
    return math.sqrt(sum_sq) / math.sqrt(2)


def argmax(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)


def ensemble_mean(dists: List[Dict[str, float]], labels: List[str]) -> Dict[str, float]:
    n = len(dists)
    return {k: sum(d.get(k, 0.0) for d in dists) / n for k in labels}


# --------------------------------------------------------------------------- #
# Per-question run
# --------------------------------------------------------------------------- #
def run_question(provider: JevProvider, question: dict, n_variants: int, base_seed: int) -> dict:
    labels = [opt for opt in OPTIONS if opt in question["options"]]
    variants = shuffle_options_batch(question, n_variants=n_variants, base_seed=base_seed)

    remapped_dists: List[Dict[str, float]] = []
    per_variant = []
    for variant_q, mapping in variants:
        action = provider.choice_probs(variant_q)
        remapped = remap_probs(action.probs, mapping)
        remapped_dists.append(remapped)
        per_variant.append({
            "mapping": mapping,
            "probs_shuffled": action.probs,
            "probs_remapped": remapped,
            "argmax_remapped": argmax(remapped),
            "entropy": action.entropy,
            "confidence": (action.metadata or {}).get("jev_confidence"),
        })

    ens = ensemble_mean(remapped_dists, labels)
    ens_argmax = argmax(ens)
    flips = sum(1 for d in remapped_dists if argmax(d) != ens_argmax)
    hellingers = [hellinger_distance(d, ens, labels) for d in remapped_dists]
    confidences = [v["confidence"] for v in per_variant if v["confidence"] is not None]

    return {
        "question": question["question"],
        "category": question.get("category"),
        "correct_answer": question.get("correct_answer"),
        "n_variants": n_variants,
        "ensemble_probs": ens,
        "ensemble_argmax": ens_argmax,
        "ensemble_entropy": entropy(ens),
        "argmax_flip_rate": flips / len(remapped_dists),
        "n_distinct_argmax": len({argmax(d) for d in remapped_dists}),
        "mean_hellinger": sum(hellingers) / len(hellingers),
        "mean_entropy": sum(v["entropy"] for v in per_variant) / len(per_variant),
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "variants": per_variant,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_questions(path: Path, categories: List[str], limit: int) -> List[dict]:
    data = json.loads(path.read_text())
    questions = data["questions"] if isinstance(data, dict) else data
    if categories:
        questions = [q for q in questions if q.get("category") in categories]
    if limit:
        # Take a per-category slice so a quick run stays balanced.
        by_cat: Dict[str, List[dict]] = {}
        for q in questions:
            by_cat.setdefault(q.get("category", "_"), []).append(q)
        sliced = []
        for cat_qs in by_cat.values():
            sliced.extend(cat_qs[:limit])
        questions = sliced
    return questions


def summarize(results: List[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    by_cat: Dict[str, List[dict]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "_"), []).append(r)

    cat_summ = {}
    for cat, rs in by_cat.items():
        frs = [r["argmax_flip_rate"] for r in rs]
        cat_summ[cat] = {
            "n": len(rs),
            "mean_argmax_flip_rate": sum(frs) / len(frs),
            "mean_hellinger": sum(r["mean_hellinger"] for r in rs) / len(rs),
            "mean_ensemble_entropy": sum(r["ensemble_entropy"] for r in rs) / len(rs),
        }

    return {
        "n_questions": len(results),
        "mean_argmax_flip_rate": mean("argmax_flip_rate"),
        "mean_hellinger": mean("mean_hellinger"),
        "mean_ensemble_entropy": mean("ensemble_entropy"),
        "mean_confidence": mean("mean_confidence"),
        "by_category": cat_summ,
    }


def main():
    ap = argparse.ArgumentParser(description="ESI-style option-order sensitivity for the Jev action channel.")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_FILE)
    ap.add_argument("--variants", type=int, default=4, help="Permutations per question (K).")
    ap.add_argument("--categories", nargs="*", default=None, help="Filter to these categories.")
    ap.add_argument("--limit", type=int, default=0, help="Max questions per category (0 = all).")
    ap.add_argument("--quick", action="store_true", help="Shortcut for --limit 3 --variants 4.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if args.quick:
        args.limit = args.limit or 3

    provider = JevProvider(model_name=args.model)
    questions = load_questions(args.questions, args.categories, args.limit)
    print(f"Running ESI-Jev (order) on {len(questions)} questions, K={args.variants} ...")

    results = []
    for i, q in enumerate(questions):
        try:
            res = run_question(provider, q, args.variants, base_seed=args.seed + i * 100)
            results.append(res)
            print(f"  [{i+1}/{len(questions)}] {q.get('category'):>11} "
                  f"flip={res['argmax_flip_rate']:.2f} "
                  f"hell={res['mean_hellinger']:.3f} "
                  f"H_ens={res['ensemble_entropy']:.2f} "
                  f"conf={res['mean_confidence']}")
        except Exception as e:  # noqa: BLE001 - keep the batch going, report at end
            print(f"  [{i+1}/{len(questions)}] ERROR: {e}")

    summary = summarize(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "variants": args.variants,
        "questions_file": str(args.questions),
        "channel": "action (Jev; report/reasoning channels unavailable by design)",
        "summary": summary,
    }, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nPer-question JSONL: {args.output}")
    print(f"Summary:            {summary_path}")


if __name__ == "__main__":
    main()
