# Jev Action Channel

Using [Jev](https://docs.typesafe.ai/introduction) (typesafe.ai) as a native **action channel**, and an ESI-style option-order run showing its structural sensitivity is low and concentrated in hard items—unlike generative multiple-choice.

## What Jev Is

Jev is a "System One" model: it takes a `state` and one or more typed `questions`, and returns typed answers with `probabilities` and `confidence`. There is **no free-text generation** and **no exposed reasoning trace**. Each question is evaluated in isolation, in parallel, against the same state.

This makes Jev almost the inverse of what this project studies. The repository measures divergence between three channels—what a model **does** (action), **says** (report), and **thinks it's doing** (reasoning). Jev deletes two of the three by design, and provides the first in its cleanest form.

## Feasibility Mapping

| Channel / finding | Jev? | Why |
|-------------------|------|-----|
| **Action distribution** (`get_action_probs`) | ✅ Native | A Jev `choice` question over the options returns a probability distribution directly. No per-provider logprob plumbing, no sampling smoothing. This is the cleanest form of the action channel in the repo (cf. the logprobs-vs-sampling warning behind **F5**). |
| **Structural sensitivity / ESI-ORDER** | ✅ Strong fit | Permute option order, run K times, remap and ensemble. Documented below. |
| **Confidence vs entropy** | 🟡 New construct | Jev returns a scalar `confidence` from a *non-generative* mechanism. Carried in `ActionResult.metadata['jev_confidence']`—a candidate new comparison, **not** a report substitute. |
| **Uniform-report attractor** (**F1 / F2**) | ❌ | Jev has no generative reporting policy, so there is nothing to collapse toward 25/25/25/25 and no anti-uniform constraint to apply. |
| **Action-vs-report JS divergence** | ❌ | Requires two distributions in the same option space; Jev provides no prose report distribution. |
| **Probe-order anchoring** (**F4**) | ❌ | Jev evaluates each question in isolation with no context bleed. The Report→Act anchoring effect F4 measures cannot occur. |
| **CoT variance / introspection** (**F6 / F7**) | ❌ | No reasoning trace to perturb or measure. |

In short: Jev can host the behavioral half of the project (action distributions, structural symmetries) but **cannot** host the report or reasoning channels, where roughly half the findings live. Several of those findings (F1, F2, F4) are specifically pathologies of the text-generation machinery Jev removes.

## What This Experiment Tests

Whether the model's answer distribution is a property of option **content** or is anchored to **position** (A/B/C/D labels)—the Layer-3 structural-symmetry question—measured on the action channel only.

### What It Falsifies

- **Claim**: "Option-order (position) bias is intrinsic to multiple-choice elicitation."
- **Result**: Not for a typed action channel. Argmax-flip rate is **7.5%** overall vs the **52–64%** cited for generative MC, and it is **0%** on easy questions.

### What It Does NOT Claim

- That Jev is order-*invariant*—adversarial items still flip 22.5% of the time.
- That this says anything about the report or reasoning channels (Jev has neither).
- That Jev's `confidence` is comparable to a generated self-report (different mechanism).
- That results transfer to non-multiple-choice formats.

## Method

Each repo question `{question, options}` maps onto one Jev `choice` request:

- `state` ← the question stem
- `criteria` ← `{label: option_text}`, so option **content** lives where Jev expects option descriptions
- `probabilities` come back keyed by the labels; `confidence` is the peakedness scalar

Because option content rides in `criteria`, permuting the criteria entries is the faithful analog of the generative ORDER intervention (shuffling which letter holds which answer). The runner reuses `interventions/option_order.py` for the permutation and remap.

For each question: generate K order permutations → call Jev per permutation → remap each distribution to original labels → measure stability.

## Usage

```bash
# Quick slice (3 per category, K=4)
python experiments/esi_jev.py --quick

# Full committed question set, 8 orderings each
python experiments/esi_jev.py --variants 8 -o results/esi/jev_order_full.jsonl

# Smoke-test the provider alone (one call)
python experiments/jev_provider.py
```

Requires `TYPESAFE_API_KEY` in the environment.

### Options

| Flag | Description |
|------|-------------|
| `--variants` | Permutations per question (K), default 4 |
| `--categories` | Filter to listed categories |
| `--limit` | Max questions per category (0 = all) |
| `--quick` | Shortcut for `--limit 3` |
| `--model` | Jev model, default `jev-latest` |
| `-o FILE` | Output JSONL (a `.summary.json` is written alongside) |

## Metrics

| Metric | Definition |
|--------|------------|
| Argmax Flip Rate | Fraction of permutations whose remapped argmax ≠ the ensemble-mean argmax |
| Mean Hellinger | Mean Hellinger distance of each permutation's remapped distribution to the ensemble mean (ESI_action analog) |
| Ensemble Entropy | Entropy of the K-averaged distribution (bits) |
| Mean Confidence | Mean Jev `confidence` scalar across permutations |

!!! note "Metric nuance"
    `esi_runner.py`'s `ESI_action` is a Hellinger distance between the *original* and *intervened* distributions. The Hellinger here is each variant *to the ensemble mean*, which runs slightly smaller. The order-of-magnitude contrast below holds regardless.

## Key Results

Full committed question set (40 questions), K=8 orderings, `jev-1.14.0`.

| Category | n | Argmax Flip Rate | Mean Hellinger | Ensemble Entropy |
|----------|---|------------------|----------------|------------------|
| easy | 15 | **0.0%** | 0.002 | ~0.00 |
| ambiguous | 15 | 5.0% | 0.040 | 0.77 |
| adversarial | 10 | **22.5%** | 0.117 | 1.31 |
| **overall** | 40 | **7.5%** | 0.045 | 0.62 |

Mean confidence 0.78 overall. Artifacts: `results/esi/jev_order_full.{jsonl,summary.json}` (and the quick run `results/esi/jev_order.{jsonl,summary.json}`).

### Contrast with Generative ESI-ORDER

| | Generative MC (OpenAI/Gemini, [ESI](esi.md)) | Jev action channel |
|---|---|---|
| Argmax flip rate | 52–64% | 7.5% (0% easy → 22.5% adversarial) |
| ESI_action (Hellinger) | ~0.50–0.53 | 0.045 mean (0.117 adversarial) |

## Interpretation

The structural symmetry that argmax-flips generative models in the majority of orderings largely vanishes in Jev's typed channel. More than that, the sensitivity **localizes**: it rises monotonically with difficulty (0% → 5% → 22.5%) and tracks ensemble entropy (easy ~0 bits, adversarial 1.31 bits). Jev isn't uniformly robust—it is robust exactly where the content is unambiguous and perturbable only where the question is genuinely hard, which is the behavior a well-behaved action channel should have. Position bias here is a signal of difficulty, not a pervasive nuisance.

## Confidence vs Entropy (the new experiment)

Jev's `confidence` scalar is the one new handle Jev adds over a generative provider. The repository's central finding is that the *report* channel **diverges** from the action channel. Jev has no report channel, but it does emit confidence from a non-generative mechanism, so the natural question is: does that confidence carry information independent of the action distribution, or is it just a restatement of it?

### Part 1 — Confidence is a readout of entropy (data-only)

Reusing the 320 per-permutation `(entropy, confidence)` pairs from the full run—no new API calls:

```bash
python experiments/jev_confidence_entropy.py   # -> results/esi/jev_confidence_entropy.json
```

| Quantity | Value |
|----------|-------|
| Pearson(confidence, entropy) | **−0.965** |
| Spearman(confidence, entropy) | **−0.978** |
| Pearson(confidence, 1 − H/H_max) | +0.965 |
| Mean \|confidence − (1 − H/H_max)\| | 0.075 |
| Mean (confidence − (1 − H/H_max)) | +0.069 |

By category, the fit to `1 − H/H_max` tightens with easiness:

| Category | n | corr(conf, H) | mean \|residual\| |
|----------|---|---------------|-------------------|
| easy | 120 | **−1.000** | 0.0003 |
| ambiguous | 120 | −0.921 | 0.106 |
| adversarial | 80 | −0.942 | 0.139 |

Confidence is a near-deterministic function of action entropy—essentially normalized entropy—exact on easy items and with a small, consistent *upward* bias on harder ones (confidence sits slightly above what entropy alone predicts). It is **not an independent channel**.

### Part 2 — Calibration on ground truth

```bash
python experiments/jev_calibration.py          # -> results/esi/jev_calibration.json
```

Only the `easy` category carries `correct_answer`, so calibration runs there (15 questions, one unpermuted answer each):

| Metric | Value |
|--------|-------|
| Accuracy | 1.00 (15/15) |
| Mean confidence | 1.00 |
| Calibration gap | 0.00 |

!!! warning "Ceiling case, not a stress test"
    Easy questions saturate at confidence 1.0 and entropy 0, so the reliability curve has a single bin. This confirms Jev is not under-confident on easy items, but it cannot probe calibration across the confidence range. The committed set has no ground truth on hard items; a real calibration curve would need labeled adversarial/ambiguous questions.

### Interpretation

This is the clean mirror image of the repository's core result. For generative models the report channel *diverges* from the action channel (the whole point of the project). For Jev there is effectively **nothing to diverge**: `confidence` is a monotone transform of action entropy. Jev collapses the belief/report distinction by construction—confirmed here empirically, not just predicted from its design.

## Still Out of Scope

The report channel (F1/F2 uniform attractor, action-vs-report JS), probe-order anchoring (F4), and reasoning-trace findings (F6/F7) remain **unavailable by construction** and would need a generative provider. Jev contributes the action channel, its structural-sensitivity profile, and the confidence-vs-entropy result above.

## Related

- [ESI](esi.md) — the generative structural-sensitivity experiment this adapts
- [Findings: Structural Sensitivity](../findings/structural-sensitivity.md)
- Code: `experiments/jev_provider.py`, `experiments/esi_jev.py`, `experiments/jev_confidence_entropy.py`, `experiments/jev_calibration.py`
