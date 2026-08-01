"""Shared reproducibility configuration helpers.

The project uses ``config/workflow_v1.0_draft.yaml`` as the human-readable
configuration record. PyYAML is preferred for parsing it. If PyYAML is not
installed, scripts fall back to the embedded defaults below so dry-runs and
basic command construction still work.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "workflow_v1.0_draft.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "china_rural_income_1km_2019_2023",
        "version": "v1.0-draft",
        "study_years": [2019, 2020, 2021, 2022, 2023],
        "price_base_year": 2019,
        "main_mask": "strict",
        "main_product": "unbounded",
    },
    "model": {
        "weak_supervised_downscaling": {
            "benchmark_year": 2020,
            "mask": "strict",
            "folds": 5,
            "pixels_per_county": 128,
            "epochs_main_2020": 180,
            "lr": 0.001,
            "threads": 8,
        },
        "spatiotemporal_validation": {
            "run": "all",
            "year": 2020,
            "mask": "strict",
            "province_folds": 5,
            "province_pixels": 128,
            "year_pixels": 64,
            "epochs": 150,
            "lr": 0.001,
            "threads": 8,
        },
        "ensemble_uncertainty": {
            "members": 5,
            "seeds": [1042, 2039, 3036, 4033, 5030],
            "years": [2019, 2020, 2021, 2022, 2023],
            "pixels_per_county": 128,
            "epochs": 150,
            "lr": 0.001,
            "threads": 8,
            "row_block": 128,
        },
        "final_maps": {
            "years": [2019, 2020, 2021, 2022, 2023],
            "pixels_per_county": 128,
            "epochs": 150,
            "lr": 0.001,
            "threads": 8,
            "row_block": 128,
            "reuse_model": False,
        },
    },
}


LOCAL_DEP_PATHS = [
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]


def add_local_dependency_paths() -> None:
    """Make project-local dependency directories importable when present."""
    for path in reversed(LOCAL_DEP_PATHS):
        if path.exists():
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` with values from ``override``."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load reproducibility config, falling back to embedded defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config["_meta"] = {
        "config_path": str(config_path),
        "yaml_available": False,
        "loaded_from_file": False,
    }
    if not config_path.exists():
        return config

    if config_path.suffix.lower() == ".json":
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        parser_name = "json"
    else:
        add_local_dependency_paths()
        try:
            import yaml  # type: ignore
        except Exception:
            return config
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        parser_name = "yaml"
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    config = deep_update(config, loaded)
    config["_meta"] = {
        "config_path": str(config_path),
        "yaml_available": parser_name == "yaml",
        "loaded_from_file": True,
        "parser": parser_name,
    }
    return config


def deep_get(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a nested value from a dict using a dotted path."""
    cur: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def int_list(value: Any, default: list[int] | None = None) -> list[int]:
    """Normalize YAML/list/string values to ``list[int]``."""
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def comma_ints(value: Any, default: list[int] | None = None) -> str:
    """Normalize YAML/list/string values to a comma-separated integer string."""
    return ",".join(str(x) for x in int_list(value, default))
