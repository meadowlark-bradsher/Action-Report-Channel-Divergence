# Experiments

How to run each experiment and what it tests. Each experiment has a clear epistemic job—a hypothesis it tests and claims it can or cannot support.

## Available Experiments

| Experiment | Purpose | Key Output |
|------------|---------|------------|
| [Belief Probe](belief-probe.md) | Establish decoupling + probe-order effects | JS divergence by probe type |
| [Gravitational Pull](gravitational-pull.md) | Show attractors persist under constraints | Boundary mass trends |
| [Alignment Regularizer](alignment-regularizer.md) | Falsify strategic hedging hypothesis | Effect of incentives |
| [ESI](esi.md) | Distinguish noise vs structural sensitivity | ESI_action / ESI_report ratio |
| [Ensemble Bootstrap](ensemble-bootstrap.md) | Turn diagnostics into measurement fix | JS reduction, optimal K |
| [Jev Action Channel](jev-action-channel.md) | Typed action channel + option-order sensitivity | Argmax flip rate, Hellinger |

## Supported Providers

| Provider | Token Beliefs | Setup |
|----------|---------------|-------|
| OpenAI | True logprobs | `OPENAI_API_KEY` in `.env` |
| Gemini | True logprobs | `GEMINI_API_KEY` in `.env` |
| llama.cpp | True logprobs | `llama-server -m model.gguf` |
| Ollama | Sampling-based | `ollama serve` |

!!! warning "Logprobs vs Sampling"
    True logprobs are essential for confidence-related claims. Sampling-based estimation can invert correlations and mask effects. Use OpenAI, Gemini, or llama.cpp for final results.

## Quick Start

```bash
# Environment setup
pip install -r requirements.txt
cp .env.example .env

# Reproduce the checked-in pipeline
make paper-pipeline

# Generate the unconstrained baseline artifact for the uniform-reporting claim
make unconstrained-baseline

# Core experiments
python experiments/belief_probe_baseline.py llama3.1:latest --experiment2 --anti-uniform
python experiments/gravitational_pull.py -p openai --multi-question
python experiments/alignment_regularizer.py -p openai

# ESI experiments
python experiments/esi_runner.py --providers openai gemini --full
python experiments/esi_ensemble.py --providers openai gemini --bootstrap
```

## Output Format

Outputs are script-specific rather than fully standardized:

- `belief_probe_baseline.py`, `compare_providers.py`, `gravitational_pull.py`, and `alignment_regularizer.py` write JSON when `-o/--output` is provided.
- New `belief_probe_baseline.py` outputs include parse-health summaries plus per-probe raw reports and probability vectors for audit.
- `esi_runner.py` writes per-item JSONL plus a companion summary CSV.
- `esi_ensemble.py` writes per-item JSONL and optional bootstrap summaries inside each record.

Checked-in examples live in `data/` and `results/esi/`.

## Key Metrics

| Metric | Definition |
|--------|------------|
| JS Divergence | Jensen-Shannon divergence between action and report |
| ESI_action | Hellinger distance of action under intervention |
| ESI_report | Hellinger distance of report under intervention |
| Boundary Mass | Fraction of probability within 3% of constraint bounds |
| Argmax Flip Rate | Fraction of orderings where modal answer changes |
