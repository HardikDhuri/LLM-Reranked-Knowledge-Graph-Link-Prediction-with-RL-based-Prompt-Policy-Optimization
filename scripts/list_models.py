#!/usr/bin/env python
"""List models available on the configured AI Gateway.

Usage::

    python -m scripts.list_models
"""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.models.llm_client import LLMClient  # noqa: E402


def main() -> None:
    client = LLMClient()
    models = client.list_models()
    if not models:
        print("No models found (or request failed).")
        return
    print(f"Available models ({len(models)}):")
    for m in models:
        print(f"  {m}")


if __name__ == "__main__":
    main()
