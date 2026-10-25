#!/usr/bin/env python3
"""
Jev confidence calibration on ground-truth questions.

Part two of the "net new experiment" that Jev makes possible: Jev exposes a
scalar `confidence` from a non-generative mechanism. The data-only pass (see
docs) shows `confidence` is a near-deterministic function of the action
distribution's entropy. This pass asks the orthogonal, practical question:

    When Jev is confident, is it right?

It runs one unpermuted `choice` request per question that has a ground-truth
`correct_answer`, then reports accuracy, mean confidence, and the gap between
them (a simple calibration error). Only the `easy` category carries
correct_answer in the committed set, so that is the default filter.

Usage:
    python experiments/jev_calibration.py
    python experiments/jev_calibration.py --categories easy -o results/esi/jev_calibration.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.jev_provider import JevProvider

DEFAULT_QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "mc_questions.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_calibration.json"


def main():
    ap = argparse.ArgumentParser(description="Jev confidence calibration on ground-truth questions.")
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_FILE)
    ap.add_argument("--categories", nargs="*", default=["easy"],
                    help="Only these categories (default: easy, the set with ground truth).")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    data = json.loads(args.questions.read_text())
    questions = data["questions"] if isinstance(data, dict) else data
    questions = [q for q in questions
                 if (not args.categories or q.get("category") in args.categories)
                 and q.get("correct_answer") is not None]

    provider = JevProvider(model_name=args.model)
    print(f"Calibration on {len(questions)} ground-truth questions ...")

    records = []
    for i, q in enumerate(questions):
        action = provider.choice_probs(q)
        choice = (action.metadata or {}).get("jev_choice") or max(action.probs, key=action.probs.get)
        correct = choice == q["correct_answer"]
        conf = (action.metadata or {}).get("jev_confidence")
        records.append({
            "question": q["question"],
            "category": q.get("category"),
            "correct_answer": q["correct_answer"],
            "choice": choice,
            "is_correct": correct,
            "confidence": conf,
            "entropy": action.entropy,
            "probs": action.probs,
        })
        mark = "OK " if correct else "XX "
        print(f"  [{i+1}/{len(questions)}] {mark} choice={choice} truth={q['correct_answer']} "
              f"conf={conf} H={action.entropy:.3f}")

    n = len(records)
    acc = sum(r["is_correct"] for r in records) / n if n else 0.0
    confs = [r["confidence"] for r in records if r["confidence"] is not None]
    mean_conf = sum(confs) / len(confs) if confs else None

    # Confidence-binned accuracy (reliability).
    bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.99), (0.99, 1.01)]
    reliability = []
    for lo, hi in bins:
        b = [r for r in records if r["confidence"] is not None and lo <= r["confidence"] < hi]
        if b:
            reliability.append({
                "bin": f"[{lo:.2f},{hi:.2f})",
                "n": len(b),
                "accuracy": sum(r["is_correct"] for r in b) / len(b),
                "mean_confidence": sum(r["confidence"] for r in b) / len(b),
            })

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "categories": args.categories,
        "n": n,
        "accuracy": acc,
        "mean_confidence": mean_conf,
        "calibration_gap": (mean_conf - acc) if mean_conf is not None else None,
        "reliability_bins": reliability,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    print("\n=== CALIBRATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
