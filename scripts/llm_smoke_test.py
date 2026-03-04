#!/usr/bin/env python
"""Smoke-test the LLM client with a single plausibility-scoring request.

Usage::

    python -m scripts.llm_smoke_test
"""

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.models.llm_client import LLMClient  # noqa: E402

_PROMPT = (
    "Rate the plausibility of this fact on a scale 0 to 1: "
    "'Paris is the capital of France'. "
    'Reply in JSON: {"score": <float>}'
)


def main() -> None:
    client = LLMClient()
    print("Sending smoke-test request …")
    try:
        result = client.chat_completion_json(
            messages=[{"role": "user", "content": _PROMPT}],
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("Raw content :", result["content"])
    print("Parsed JSON :", json.dumps(result["parsed_json"], indent=2))
    print("Usage       :", result["usage"])
    print(f"Latency     : {result['latency_s']:.3f}s")
    print("Stats       :", client.stats())


if __name__ == "__main__":
    main()
