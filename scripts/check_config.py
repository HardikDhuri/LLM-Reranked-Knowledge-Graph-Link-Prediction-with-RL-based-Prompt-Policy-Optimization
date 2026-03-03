#!/usr/bin/env python
"""Check configuration: print all settings (API key masked) or report missing keys."""

import sys

from pydantic import ValidationError

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.config import get_settings, print_config  # noqa: E402


def main() -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("ERROR: Configuration is invalid.\n")
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            print(f"  {loc}: {error['msg']}")
        sys.exit(1)

    print_config(settings)


if __name__ == "__main__":
    main()
