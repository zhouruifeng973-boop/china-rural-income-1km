"""Run leave-province validation with a versioned county-income target.

This wrapper replaces only the county-target loader and validation output root.
It does not modify historical targets, models, maps, or validation products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for dependency_path in reversed([
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]):
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

import pandas as pd
import torch

from repro_config import deep_get, load_config
import stage3_weak_supervised_downscaling as weak


DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.3_official44_corrected.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_loader(target_path: Path):
    def load_targets(year: int) -> pd.DataFrame:
        frame = pd.read_csv(target_path)
        frame = frame[frame["year"].astype(int) == year].copy()
        frame["county_code"] = frame["county_code"].astype(int)
        frame = frame.dropna(subset=["rural_income_2019yuan", "province_std"])
        return frame[[
            "year", "county_code", "county", "city", "province_std",
            "rural_income_nominal", "rural_income_2019yuan",
        ]]

    return load_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    target_path = ROOT / deep_get(config, "paths.corrected_county_income")
    output_root = ROOT / deep_get(config, "paths.output_root")
    validation_out = output_root / "stage3_weak_supervision" / "spatiotemporal_validation"
    validation_out.mkdir(parents=True, exist_ok=True)

    settings = deep_get(config, "model.spatiotemporal_validation", {})
    year = int(settings.get("year", 2020))
    mask = str(settings.get("mask", "strict"))
    folds = int(settings.get("province_folds", 5))
    pixels = int(settings.get("province_pixels", 128))
    epochs = int(settings.get("epochs", 150))
    lr = float(settings.get("lr", 0.001))
    threads = int(settings.get("threads", 8))

    prefix = validation_out / f"leave_province_{year}_{mask}"
    expected = [
        Path(f"{prefix}_county_predictions.csv"),
        Path(f"{prefix}_fold_metrics.csv"),
        Path(f"{prefix}_overall_metrics.json"),
    ]
    if not args.force and any(path.exists() for path in expected):
        raise RuntimeError(f"Leave-province output already exists under {validation_out}")
    if not target_path.is_file():
        raise FileNotFoundError(target_path)

    weak.load_targets = target_loader(target_path)
    torch.set_num_threads(max(1, threads))

    import stage3_spatiotemporal_validation as validation

    validation.VALIDATION_OUT = validation_out
    predictions, fold_metrics, overall = validation.run_leave_province(
        year, mask, folds, pixels, epochs, lr
    )
    validation.save_leave_province(predictions, fold_metrics, overall)

    manifest = {
        "version": deep_get(config, "project.version"),
        "scheme": "leave_province",
        "config": str(args.config.resolve()),
        "target": str(target_path.relative_to(ROOT)).replace("\\", "/"),
        "target_sha256": sha256(target_path),
        "settings": {
            "year": year,
            "mask": mask,
            "folds": folds,
            "pixels_per_county": pixels,
            "epochs": epochs,
            "lr": lr,
            "threads": threads,
        },
        "overall_metrics": overall,
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in expected
        ],
        "historical_outputs_overwritten": False,
    }
    manifest_path = validation_out / f"leave_province_{year}_{mask}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "overall_metrics": overall,
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
