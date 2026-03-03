"""Dataset loaders for FB15k-237 and related benchmarks."""

from src.data.fb15k237 import FB15k237Dataset, download_fb15k237, load_fb15k237

__all__ = ["FB15k237Dataset", "download_fb15k237", "load_fb15k237"]
