#!/usr/bin/env python3
"""
20 Questions: does Jev's confidence/entropy reflect the game-state entropy?

Earlier we found Jev's `confidence` is a near-deterministic readout of its own
action-distribution entropy. The sharper question is whether that entropy is
*honest about the game*: as clues narrow the candidate set, does Jev's entropy
fall in step with the true information-theoretic uncertainty?

Setup. A fixed universe of 16 animals with a small attribute table. A game
reveals clues one at a time; after each clue the ORACLE set is the animals
still consistent with all clues, and the oracle entropy is log2(|consistent|)
(uniform posterior). At each step we ask Jev one `choice` question over all 16
animals, given the clues so far, and read its probabilities + confidence.

We then compare, across the whole trajectory:
  - H_jev   vs  H_oracle     (does Jev's entropy track the true entropy?)
  - conf    vs  H_oracle     (does confidence rise as the game narrows?)
  - mass_on_consistent       (honesty: does Jev zero out eliminated animals?)

Jev calls: ~1 per clue-step across the games (cheap). Needs TYPESAFE_API_KEY.

Usage:
    python experiments/jev_twenty_questions.py
    python experiments/jev_twenty_questions.py -o results/esi/jev_twenty_questions.json
"""

import argparse
import json
import math
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.jev_provider import JevProvider

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_twenty_questions.json"

# --------------------------------------------------------------------------- #
# Universe: 16 animals x attributes. Oracle entropy is computed from this table.
# attrs: klass, fly (can fly), aquatic (lives in/near water), domestic, size, diet
# --------------------------------------------------------------------------- #
ANIMALS = {
    "dog":      dict(klass="mammal",  fly=False, aquatic=False, domestic=True,  size="medium", diet="carnivore"),
    "cat":      dict(klass="mammal",  fly=False, aquatic=False, domestic=True,  size="small",  diet="carnivore"),
    "elephant": dict(klass="mammal",  fly=False, aquatic=False, domestic=False, size="large",  diet="herbivore"),
    "lion":     dict(klass="mammal",  fly=False, aquatic=False, domestic=False, size="large",  diet="carnivore"),
    "dolphin":  dict(klass="mammal",  fly=False, aquatic=True,  domestic=False, size="large",  diet="carnivore"),
    "whale":    dict(klass="mammal",  fly=False, aquatic=True,  domestic=False, size="large",  diet="carnivore"),
    "bat":      dict(klass="mammal",  fly=True,  aquatic=False, domestic=False, size="small",  diet="omnivore"),
    "eagle":    dict(klass="bird",    fly=True,  aquatic=False, domestic=False, size="medium", diet="carnivore"),
    "penguin":  dict(klass="bird",    fly=False, aquatic=True,  domestic=False, size="medium", diet="carnivore"),
    "ostrich":  dict(klass="bird",    fly=False, aquatic=False, domestic=False, size="large",  diet="omnivore"),
    "sparrow":  dict(klass="bird",    fly=True,  aquatic=False, domestic=False, size="small",  diet="omnivore"),
    "salmon":   dict(klass="fish",    fly=False, aquatic=True,  domestic=False, size="medium", diet="carnivore"),
    "shark":    dict(klass="fish",    fly=False, aquatic=True,  domestic=False, size="large",  diet="carnivore"),
    "snake":    dict(klass="reptile", fly=False, aquatic=False, domestic=False, size="medium", diet="carnivore"),
    "crocodile":dict(klass="reptile", fly=False, aquatic=True,  domestic=False, size="large",  diet="carnivore"),
    "bee":      dict(klass="insect",  fly=True,  aquatic=False, domestic=False, size="small",  diet="herbivore"),
}

# Each clue: (natural-language text, predicate on an attrs dict).
GAMES = [
    {
        "secret": "penguin",
        "clues": [
            ("It is a bird.",            lambda a: a["klass"] == "bird"),
            ("It cannot fly.",           lambda a: a["fly"] is False),
            ("It lives in the water.",   lambda a: a["aquatic"] is True),
        ],
    },
    {
        "secret": "dog",
        "clues": [
            ("It is a mammal.",          lambda a: a["klass"] == "mammal"),
            ("It is a domestic pet.",    lambda a: a["domestic"] is True),
            ("It is medium-sized.",      lambda a: a["size"] == "medium"),
        ],
    },
    {
        "secret": "shark",
        "clues": [
            ("It is a fish.",            lambda a: a["klass"] == "fish"),
            ("It is large.",             lambda a: a["size"] == "large"),
        ],
    },
]


def entropy(probs):
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def consistent_set(clues_applied):
    """Animals satisfying every applied clue predicate."""
    out = []
    for name, attrs in ANIMALS.items():
        if all(pred(attrs) for _text, pred in clues_applied):
            out.append(name)
    return out


def build_question(clues_applied):
    if clues_applied:
        facts = "\n".join(f"- {text}" for text, _ in clues_applied)
        state = ("You are playing 20 Questions. The secret is one of the listed animals.\n"
                 "Known facts about the secret animal:\n" + facts)
    else:
        state = ("You are playing 20 Questions. The secret is one of the listed animals.\n"
                 "No facts are known yet.")
    criteria = {name: f"the animal commonly known as a {name}" for name in ANIMALS}
    return state, criteria


def main():
    ap = argparse.ArgumentParser(description="20 Questions: Jev confidence/entropy vs oracle entropy.")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    provider = JevProvider(model_name=args.model)
    hmax = math.log2(len(ANIMALS))

    all_points = []   # for aggregate correlations
    games_out = []

    for game in GAMES:
        secret = game["secret"]
        print(f"\n=== Game: secret = {secret} ===")
        print(f"{'step':>4} {'clue':<28} {'nCons':>5} {'H_oracle':>9} {'H_jev':>7} {'conf':>6} {'mass✓':>6} {'pick':>10}")
        steps = []
        # step 0 = no clues, then one row per clue revealed
        for k in range(len(game["clues"]) + 1):
            applied = game["clues"][:k]
            cons = consistent_set(applied)
            h_oracle = math.log2(len(cons)) if cons else 0.0

            state, criteria = build_question(applied)
            action = provider.choice_probs({"question": state, "options": criteria})
            probs = action.probs
            conf = (action.metadata or {}).get("jev_confidence")
            pick = (action.metadata or {}).get("jev_choice") or max(probs, key=probs.get)
            mass_consistent = sum(probs.get(n, 0.0) for n in cons)

            clue_text = applied[-1][0] if applied else "(no clues)"
            row = {
                "step": k,
                "clue": clue_text,
                "n_consistent": len(cons),
                "consistent": cons,
                "H_oracle": h_oracle,
                "H_jev": action.entropy,
                "confidence": conf,
                "mass_on_consistent": mass_consistent,
                "pick": pick,
                "pick_in_consistent": pick in cons,
                "pick_is_secret": (pick == secret),
            }
            steps.append(row)
            all_points.append(row)
            print(f"{k:>4} {clue_text[:28]:<28} {len(cons):>5} {h_oracle:>9.3f} "
                  f"{action.entropy:>7.3f} {str(conf):>6} {mass_consistent:>6.2f} {pick:>10}")
        games_out.append({"secret": secret, "steps": steps})

    # Aggregate correlations across every (step) point.
    H_oracle = [p["H_oracle"] for p in all_points]
    H_jev = [p["H_jev"] for p in all_points]
    conf = [p["confidence"] for p in all_points if p["confidence"] is not None]
    conf_oracle = [p["H_oracle"] for p in all_points if p["confidence"] is not None]

    agg = {
        "n_points": len(all_points),
        "pearson_Hjev_Horacle": pearson(H_jev, H_oracle),
        "pearson_conf_Horacle": pearson(conf, conf_oracle),
        "pearson_Hjev_conf": pearson([p["H_jev"] for p in all_points if p["confidence"] is not None], conf),
        "mean_mass_on_consistent": st.mean(p["mass_on_consistent"] for p in all_points),
        "mean_mass_on_consistent_final_steps": st.mean(
            p["mass_on_consistent"] for p in all_points if p["n_consistent"] == 1),
        "final_step_pick_is_secret_rate": st.mean(
            1.0 if p["pick_is_secret"] else 0.0 for p in all_points if p["n_consistent"] == 1),
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "universe_size": len(ANIMALS),
        "H_max_bits": hmax,
        "aggregate": agg,
        "games": games_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
