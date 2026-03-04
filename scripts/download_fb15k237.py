#!/usr/bin/env python
"""Download and extract the FB15k-237 benchmark dataset.

Usage::

    python -m scripts.download_fb15k237
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.data import download_fb15k237  # noqa: E402


def main() -> None:
    settings = get_settings()
    dataset_path = download_fb15k237(settings.data_dir)
    print(f"Dataset available at: {dataset_path}")


if __name__ == "__main__":
    main()
