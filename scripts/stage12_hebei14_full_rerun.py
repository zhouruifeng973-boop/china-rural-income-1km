"""Versioned full rerun using the 14 corrected Hebei 2021 labels.

The script reuses unchanged raster features, rural weights and official
province calibration targets. It writes every model, validation table and map
to ``outputs/hebei14_corrected_full_rerun_v1`` and never overwrites v1.0 files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
for dependency_path in reversed([
    ROOT / ".codex_pydeps_stage8",
    ROOT / ".codex_pydeps",
    ROOT / "stage6_pydeps",
]):
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd
import rasterio
import torch

from repro_config import deep_get, int_list, load_config
import stage3_weak_supervised_downscaling as weak


DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.1_hebei14_corrected.yaml"
ORIGINAL_TARGET = ROOT / "data" / "processed" / "county_income_2019_2023_deflated.csv"
ORIGINAL_MAP_DIR = ROOT / "outputs" / "stage5_final_maps"
NODATA = -9999.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_loader(target_path: Path):
    def load_targets(year: int) -> pd.DataFrame:
        df = pd.read_csv(target_path)
        df = df[df["year"].astype(int) == year].copy()
        df["county_code"] = df["county_code"].astype(int)
        df = df.dropna(subset=["rural_income_2019yuan", "province_std"])
        return df[[
            "year", "county_code", "county", "city", "province_std",
            "rural_income_nominal", "rural_income_2019yuan",
        ]]
    return load_targets


def compare_maps(
    corrected_map_dir: Path,
    comparison_dir: Path,
    years: list[int],
) -> pd.DataFrame:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in years:
        original_path = ORIGINAL_MAP_DIR / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"
        corrected_path = corrected_map_dir / f"rural_income_{year}_final_calibrated_unbounded_2019yuan_strict.tif"
        difference_path = comparison_dir / f"corrected_minus_original_{year}_2019yuan.tif"
        weight_path = ROOT / "data" / "processed" / "rural_weights" / f"rural_pop_weight_{year}_strict.tif"

        differences = []
        weighted_sum = 0.0
        weight_sum = 0.0
        with rasterio.open(original_path) as original, \
             rasterio.open(corrected_path) as corrected, \
             rasterio.open(weight_path) as weight:
            if (
                original.shape != corrected.shape
                or original.crs != corrected.crs
                or original.transform != corrected.transform
            ):
                raise RuntimeError(f"Grid mismatch for {year}")
            profile = corrected.profile.copy()
            profile.update(
                driver="GTiff", dtype="float32", count=1, nodata=NODATA,
                compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
            )
            with rasterio.open(difference_path, "w", **profile) as destination:
                for _, window in corrected.block_windows(1):
                    old = original.read(1, window=window).astype(np.float32)
                    new = corrected.read(1, window=window).astype(np.float32)
                    weights = weight.read(1, window=window).astype(np.float64)
                    old_nodata = original.nodata if original.nodata is not None else NODATA
                    new_nodata = corrected.nodata if corrected.nodata is not None else NODATA
                    valid = (
                        np.isfinite(old) & np.isfinite(new)
                        & (old != old_nodata) & (new != new_nodata)
                    )
                    block = np.full(old.shape, NODATA, dtype=np.float32)
                    block[valid] = new[valid] - old[valid]
                    destination.write(block, 1, window=window)
                    if valid.any():
                        diff = block[valid].astype(np.float32)
                        differences.append(diff)
                        weighted = valid & np.isfinite(weights) & (weights > 0)
                        weighted_sum += float(np.sum(block[weighted].astype(np.float64) * weights[weighted]))
                        weight_sum += float(np.sum(weights[weighted]))

        values = np.concatenate(differences)
        absolute = np.abs(values)
        rows.append({
            "year": year,
            "common_valid_pixels": int(values.size),
            "mean_difference_yuan": float(np.mean(values)),
            "median_difference_yuan": float(np.median(values)),
            "mean_absolute_difference_yuan": float(np.mean(absolute)),
            "p95_absolute_difference_yuan": float(np.quantile(absolute, 0.95)),
            "p99_absolute_difference_yuan": float(np.quantile(absolute, 0.99)),
            "max_absolute_difference_yuan": float(np.max(absolute)),
            "pixels_abs_difference_gt_100_pct": float(100 * np.mean(absolute > 100)),
            "pixels_abs_difference_gt_500_pct": float(100 * np.mean(absolute > 500)),
            "pixels_abs_difference_gt_1000_pct": float(100 * np.mean(absolute > 1000)),
            "population_weighted_mean_difference_yuan": weighted_sum / weight_sum,
            "difference_map": str(difference_path.relative_to(ROOT)).replace("\\", "/"),
        })
        del values, absolute, differences
        gc.collect()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_version = str(deep_get(config, "project.version", "v1.1-hebei14-corrected"))
    run_role = str(deep_get(config, "project.run_role", "corrected"))
    label_description = str(deep_get(
        config,
        "project.label_description",
        "河北张家口市2021年14条高置信异常标签修正版",
    ))

    target_path = ROOT / deep_get(
        config,
        "paths.corrected_county_income",
        "data/corrections/county_income_2019_2023_deflated_hebei14_corrected_v1.csv",
    )
    output_root = ROOT / deep_get(
        config,
        "paths.output_root",
        "outputs/hebei14_corrected_full_rerun_v1",
    )
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.force:
            raise RuntimeError(f"Versioned output directory is not empty: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_cfg = deep_get(config, "model.spatiotemporal_validation", {})
    final_cfg = deep_get(config, "model.final_maps", {})
    years = int_list(final_cfg.get("years"), [2019, 2020, 2021, 2022, 2023])
    threads = int(final_cfg.get("threads", 8))
    torch.set_num_threads(max(1, threads))

    # Patch only the county-label loader and output roots. Raster features,
    # weights and province calibration targets remain the frozen v1.0 inputs.
    weak.load_targets = target_loader(target_path)
    weak.OUT = output_root / "stage3_weak_supervision"
    weak.OUT.mkdir(parents=True, exist_ok=True)

    import stage3_spatiotemporal_validation as validation
    validation.VALIDATION_OUT = weak.OUT / "spatiotemporal_validation"
    validation.VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Leave-year validation ({run_role})", flush=True)
    leave_year = validation.run_leave_year(
        "strict",
        int(validation_cfg.get("year_pixels", 64)),
        int(validation_cfg.get("epochs", 150)),
        float(validation_cfg.get("lr", 0.001)),
    )
    validation.save_leave_year(*leave_year)

    import stage5_final_maps as final_maps
    final_maps.OUT = output_root / "stage5_final_maps"
    final_maps.MODEL_DIR = final_maps.OUT / "model"
    final_maps.OUT.mkdir(parents=True, exist_ok=True)
    weak.set_seed(weak.RANDOM_STATE)
    print("[2/5] Joint weak-supervision model and bounded annual maps", flush=True)
    model, imputer, scaler, metadata = final_maps.train_and_save(
        int(final_cfg.get("pixels_per_county", 128)),
        int(final_cfg.get("epochs", 150)),
        float(final_cfg.get("lr", 0.001)),
    )
    map_stats = []
    for year in years:
        initial_path, province = final_maps.predict_initial(
            year, model, imputer, scaler, int(final_cfg.get("row_block", 128))
        )
        _, stats = final_maps.calibrate_map(
            year, initial_path, province, int(final_cfg.get("row_block", 128))
        )
        map_stats.append(stats)
        gc.collect()
    (final_maps.OUT / "run_summary.json").write_text(
        json.dumps(map_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/5] Unbounded canonical maps and change products", flush=True)
    import stage5_create_unbounded_maps as unbounded
    unbounded.OUT = final_maps.OUT
    unbounded.main()
    import stage5_change_analysis as bounded_change
    bounded_change.OUT = final_maps.OUT
    bounded_change.main()

    print("[4/5] Pixel-level comparison against v1.0 maps", flush=True)
    comparison_dir = output_root / "map_comparison"
    comparison = compare_maps(final_maps.OUT, comparison_dir, years)
    comparison_csv = comparison_dir / "annual_map_difference_summary.csv"
    comparison.to_csv(comparison_csv, index=False, encoding="utf-8-sig")

    original_validation_path = (
        ROOT / "outputs" / "stage3_weak_supervision" / "spatiotemporal_validation"
        / "leave_year_2019_2023_strict_overall_metrics.json"
    )
    original_validation = json.loads(original_validation_path.read_text(encoding="utf-8"))
    corrected_validation = leave_year[2]
    validation_delta = {
        metric: float(corrected_validation[metric] - original_validation[metric])
        for metric in ("r2", "mae", "rmse", "mape", "bias")
    }

    print("[5/5] Manifest and summary", flush=True)
    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    manifest = {
        "version": run_version,
        "run_role": run_role,
        "label_description": label_description,
        "config": str(args.config.resolve()),
        "corrected_target": str(target_path.relative_to(ROOT)).replace("\\", "/"),
        "corrected_target_sha256": sha256(target_path),
        "original_target_sha256": sha256(ORIGINAL_TARGET),
        "original_outputs_overwritten": False,
        "leave_year_validation_original": original_validation,
        "leave_year_validation_corrected": corrected_validation,
        "leave_year_validation_delta": validation_delta,
        "model_metadata": metadata,
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
            if path.suffix.lower() in {".csv", ".json", ".md", ".pkl", ".pt", ".tif"}
        ],
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    max_median = float(comparison["median_difference_yuan"].abs().max())
    max_p95 = float(comparison["p95_absolute_difference_yuan"].max())
    max_gt500 = float(comparison["pixels_abs_difference_gt_500_pct"].max())
    summary = [
        f"# {label_description}完整流水线对比",
        "",
        "- 原始训练标签、模型、地图和发布包均未覆盖。",
        f"- 本次运行使用版本化标签：`{target_path.relative_to(ROOT).as_posix()}`。",
        "- 重新运行：留年验证、五年联合弱监督模型、年度有界/无界地图和变化图。",
        "- 省级校准约束、空间特征、农村人口权重和农村掩膜保持不变。",
        "",
        "## 留年验证变化（修正版减原版）",
        "",
        f"- ΔR²：{validation_delta['r2']:+.6f}",
        f"- ΔMAE：{validation_delta['mae']:+.2f} 元",
        f"- ΔRMSE：{validation_delta['rmse']:+.2f} 元",
        "",
        "## 年度无界地图差异摘要",
        "",
        f"- 五年中最大年度差异中位数绝对值：{max_median:.2f} 元。",
        f"- 五年中最大的像元绝对差异P95：{max_p95:.2f} 元。",
        f"- 五年中绝对差异超过500元的最大像元比例：{max_gt500:.4f}%。",
        "- 逐年结果见 `map_comparison/annual_map_difference_summary.csv`。",
        "",
        "Stage 8多成员不确定性尚未在本脚本中重跑；应先依据本次地图差异决定是否将修正版升级为正式产品。",
    ]
    (output_root / "FULL_RERUN_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(output_root),
        "validation_delta": validation_delta,
        "max_abs_median_map_difference": max_median,
        "max_p95_absolute_map_difference": max_p95,
        "max_pixels_gt500_pct": max_gt500,
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
