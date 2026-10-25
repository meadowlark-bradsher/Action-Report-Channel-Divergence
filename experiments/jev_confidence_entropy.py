#!/usr/bin/env python3
"""
Jev confidence-vs-entropy analysis (data-only, no API calls).

Part one of the "net new experiment" Jev makes possible. Jev returns a scalar
`confidence` alongside the action distribution. This script tests whether that
confidence carries information independent of the action distribution's entropy,
by reading the per-variant (entropy, confidence) pairs already recorded in an
ESI-Jev run (experiments/esi_jev.py).

Hypothesis: confidence is a near-deterministic readout of normalized entropy,
i.e. confidence ~= 1 - H / log2(n_options). If so, Jev's confidence is not an
independent channel -- the opposite of the generative report channel this repo
studies, which diverges from the action distribution.

Usage:
    python experiments/jev_confidence_entropy.py
    python experiments/jev_confidence_entropy.py --input results/esi/jev_order_full.jsonl \
        -o results/esi/jev_confidence_entropy.json
"""

import argparse
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_order_full.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "esi" / "jev_confidence_entropy.json"


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


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


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def collect(path: Path):
    pts = []  # (entropy, confidence, category, n_options)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_opts = len(r["ensemble_probs"])
        for v in r.get("variants", []):
            e, c = v.get("entropy"), v.get("confidence")
            if e is not None and c is not None:
                pts.append((float(e), float(c), r.get("category"), n_opts))
    return pts


def analyze(pts):
    E = [p[0] for p in pts]
    C = [p[1] for p in pts]
    pred, resid = [], []
    for e, c, _cat, no in pts:
        hmax = math.log2(no) if no > 1 else 1.0
        p = 1 - e / hmax
        pred.append(p)
        resid.append(c - p)

    by_cat = {}
    for (e, c, cat, _no), p in zip(pts, pred):
        by_cat.setdefault(cat, {"H": [], "C": [], "resid": []})
        by_cat[cat]["H"].append(e)
        by_cat[cat]["C"].append(c)
        by_cat[cat]["resid"].append(c - p)

    return {
        "n": len(pts),
        "entropy_mean": st.mean(E),
        "confidence_mean": st.mean(C),
        "pearson_conf_entropy": pearson(C, E),
        "spearman_conf_entropy": spearman(C, E),
        "pearson_conf_vs_1minusHnorm": pearson(C, pred),
        "mean_abs_residual": st.mean(abs(x) for x in resid),
        "mean_residual": st.mean(resid),
        "max_abs_residual": max(abs(x) for x in resid),
        "by_category": {
            cat: {
                "n": len(d["C"]),
                "pearson_conf_entropy": pearson(d["C"], d["H"]),
                "mean_abs_residual": st.mean(abs(x) for x in d["resid"]),
            }
            for cat, d in by_cat.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Jev confidence-vs-entropy analysis (data-only).")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="ESI-Jev per-item JSONL.")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    pts = collect(args.input)
    if not pts:
        raise SystemExit(f"No (entropy, confidence) pairs found in {args.input}")
    result = analyze(pts)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["input"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
