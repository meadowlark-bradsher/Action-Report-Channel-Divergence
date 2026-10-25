"""Jev (typesafe.ai) provider for the ACTION channel only.

Jev is a "System One" structured-output model: it takes a `state` and one or
more typed `questions`, and returns typed answers with `probabilities` and
`confidence`. There is no free-text generation and no exposed reasoning trace.

What this means for Epistemic Channel Divergence:

- ACTION channel  -> fully supported. A Jev `choice` question over the A/B/C/D
  options returns a native probability distribution. This is the cleanest form
  of `get_action_probs` in the repo: true probabilities, no per-provider
  logprob plumbing, no sampling smoothing (cf. F5).

- REPORT channel  -> NOT supported. Jev has no generative reporting policy, so
  the free-text self-report (`get_report`, the 25/25/25/25 attractor behind
  F1/F2) cannot exist here. `get_report` raises NotImplementedError.

- REASONING channel -> NOT supported. No CoT trace to perturb or measure
  (F6/F7). `generate` raises NotImplementedError.

Jev exposes a scalar `confidence` computed from how peaked the distribution is.
It comes from a *different mechanism* than a generated self-report, so it is
carried in ActionResult.metadata as a candidate for a new, non-generative
confidence-vs-entropy comparison -- not as a substitute for the report channel.

Docs: https://docs.typesafe.ai/introduction/quickstart.md
API:  POST https://api.typesafe.ai/v1/systemone   (Bearer TYPESAFE_API_KEY)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Work whether imported as `experiments.jev_provider` or run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from providers import BaseProvider, ActionResult, OPTIONS

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_INSTRUCTION = "Select the single best answer to the question."


class JevProvider(BaseProvider):
    """typesafe.ai Jev provider. Action channel only."""

    provider_name = "jev"
    # Not token logprobs, but native probabilities -- treat as a first-class
    # action distribution (not a sampling estimate).
    supports_logprobs = True

    def __init__(self, model_name: str = DEFAULT_JEV_MODEL, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY not set")
        self.endpoint = kwargs.get("endpoint", JEV_ENDPOINT)
        self.instruction = kwargs.get("instruction", DEFAULT_INSTRUCTION)
        self.timeout = kwargs.get("timeout", 60)

        import requests
        self._requests = requests

    # ------------------------------------------------------------------ #
    # Native, structured action channel (preferred entry point)
    # ------------------------------------------------------------------ #
    def choice_probs(
        self,
        question: dict,
        instruction: Optional[str] = None,
    ) -> ActionResult:
        """Action distribution for one MC question via a Jev `choice` question.

        `question` is a repo question dict: {"question": str, "options": {label: text}}.
        Option *content* is passed as Jev `criteria` (label -> description); the
        model returns a probability per label. This keeps option text where Jev
        expects it, so permuting the criteria order is a faithful ESI/order
        intervention (see interventions/option_order.py).
        """
        options = question["options"]
        labels = [opt for opt in OPTIONS if opt in options]
        # Preserve any non-A/B/C/D labels too, just in case.
        labels += [k for k in options if k not in labels]

        criteria = {label: options[label] for label in labels}

        body = {
            "state": question["question"],
            "model": self.model_name,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": instruction or self.instruction,
                    "criteria": criteria,
                }
            },
        }

        resp = self._requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answers", {}).get("answer", {})
        raw_probs = answer.get("probabilities", {}) or {}

        # Normalize over the labels we asked about (Jev values already sum ~1).
        probs = {label: float(raw_probs.get(label, 0.0)) for label in labels}
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {k: 1.0 / len(labels) for k in labels}

        return ActionResult(
            probs=probs,
            entropy=self._entropy(probs),
            method="jev",
            metadata={
                "jev_confidence": answer.get("confidence"),
                "jev_choice": answer.get("choice"),
                "model": data.get("model", self.model_name),
                "usage": data.get("usage"),
            },
        )

    # ------------------------------------------------------------------ #
    # BaseProvider interface
    # ------------------------------------------------------------------ #
    def get_action_probs(self, prompt: str, temperature: float = 1.0) -> ActionResult:
        """Prompt-string path for drop-in compatibility.

        The whole rendered MC prompt is passed as Jev `state` and the model
        chooses among generic A/B/C/D labels. Prefer `choice_probs(question)`
        for ESI work, which keeps option content in `criteria`.
        """
        body = {
            "state": prompt,
            "model": self.model_name,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": self.instruction,
                    "criteria": {opt: f"Option {opt} as presented in the state" for opt in OPTIONS},
                }
            },
        }
        resp = self._requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answers", {}).get("answer", {})
        raw_probs = answer.get("probabilities", {}) or {}
        probs = {opt: float(raw_probs.get(opt, 0.0)) for opt in OPTIONS}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()} if total > 0 else {k: 0.25 for k in OPTIONS}
        return ActionResult(
            probs=probs,
            entropy=self._entropy(probs),
            method="jev",
            metadata={
                "jev_confidence": answer.get("confidence"),
                "jev_choice": answer.get("choice"),
                "model": data.get("model", self.model_name),
            },
        )

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        raise NotImplementedError(
            "Jev has no text generation. The REASONING channel (CoT, F6/F7) "
            "cannot be measured with Jev."
        )

    def get_report(self, prompt: str, introspective: bool = False, anti_uniform: bool = False):
        raise NotImplementedError(
            "Jev has no generative reporting policy. The REPORT channel "
            "(free-text self-reports, the uniform attractor behind F1/F2, and "
            "action-vs-report JS divergence) cannot be measured with Jev. "
            "Use ActionResult.metadata['jev_confidence'] for the separate, "
            "non-generative confidence signal instead."
        )


# Quick smoke test: python experiments/jev_provider.py
if __name__ == "__main__":
    provider = JevProvider()
    q = {
        "question": "What is the capital of France?",
        "options": {"A": "London", "B": "Paris", "C": "Berlin", "D": "Madrid"},
    }
    result = provider.choice_probs(q)
    print(f"model:      {result.metadata.get('model')}")
    print(f"method:     {result.method}")
    print(f"probs:      {' '.join(f'{k}:{v:.3f}' for k, v in sorted(result.probs.items()))}")
    print(f"entropy:    {result.entropy:.3f} bits")
    print(f"confidence: {result.metadata.get('jev_confidence')}")
    print(f"choice:     {result.metadata.get('jev_choice')}")
