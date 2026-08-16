"""Loads and validates ml/config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

_UNSET = object()

REQUIRED_KEYS = (
    "data.csv_path",
    "detectors.contamination",
    "detectors.random_state",
    "ensemble.threshold",
    "surrogate.min_auc",
    "validation.sane_band",
    "storage.local_dir",
)


class Config:
    """Dotted-path access over the parsed config.yaml tree."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self._validate()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path is not None else CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def get(self, dotted: str, default: Any = _UNSET) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _UNSET:
                    raise KeyError(dotted)
                return default
            node = node[part]
        return node

    def _validate(self) -> None:
        for key in REQUIRED_KEYS:
            try:
                self.get(key)
            except KeyError:
                raise ValueError(f"config missing required key: {key}")
