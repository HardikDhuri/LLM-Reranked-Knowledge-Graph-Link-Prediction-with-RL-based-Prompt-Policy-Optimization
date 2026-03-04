"""Simple file-backed JSON dictionary cache."""

import json
from pathlib import Path
from typing import Any


class JSONCache:
    """Simple file-backed JSON dictionary cache."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath, encoding="utf-8") as f:
                self._data = json.load(f)

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
