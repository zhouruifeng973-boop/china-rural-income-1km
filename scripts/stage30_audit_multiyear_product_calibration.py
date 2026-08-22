"""Audit the product-calibrated multi-year leave-province diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for dependency_path in reversed([
    ROOT / ".codex_pydeps_stage8", ROOT / ".codex_pydeps", ROOT / "stage6_pydeps"
]):
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))

import numpy as np
import pandas as pd


RUN = ROOT / "outputs" / "multiyear_leave_province_v1_5_no_road"
STEM = RUN / "multiyear_leave_province_2019_2023_strict_product_calibrated"


def metric(frame: pd.DataFrame, column: str) -> dict:
    true = frame["true_2019yuan"].to_numpy(float)
    pred = frame[column].to_numpy(float)
    error = pred - true
    return {
        "r2": float(1 - np.sum(error ** 2) / np.sum((true - true.mean()) ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(100 * np.mean(np.abs(error) / np.maximum(np.abs(true), 1e-9))),
        "bias": float(np.mean(error)),
    }


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return bool(abs(left - right) <= tolerance * max(1.0, abs(left), abs(right)))


def main() -> None:
    pred = pd.read_csv(Path(f"{STEM}_county_year_predictions.csv"), encoding="utf-8-sig")
    factors = pd.read_csv(Path(f"{STEM}_province_year_factors.csv"), encoding="utf-8-sig")
    summary = json.loads(Path(f"{STEM}_summary.json").read_text(encoding="utf-8"))

    factor_lookup = factors.set_index(["year", "province_std"])["calibration_factor"]
    expected_factor = pd.MultiIndex.from_frame(pred[["year", "province_std"]]).map(factor_lookup)
    recomputed_calibrated = pred["pred_weak_2019yuan"].to_numpy(float) * np.asarray(expected_factor, float)

    constraint_errors = []
    rank_checks = []
    for (year, province), group in pred.groupby(["year", "province_std"]):
        weight = group["rural_weight_sum"].to_numpy(float)
        calibrated = group["pred_calibrated_2019yuan"].to_numpy(float)
        calibrated_mean = float(np.sum(calibrated * weight) / np.sum(weight))
        official = float(
            factors.loc[
                (factors["year"] == year) & (factors["province_std"] == province),
                "official_province_income_2019yuan",
            ].iloc[0]
        )
        constraint_errors.append(abs(calibrated_mean / official - 1.0))
        rank_checks.append(
            group["pred_weak_2019yuan"].rank(method="average").equals(
                group["pred_calibrated_2019yuan"].rank(method="average")
            )
        )

    before = metric(pred, "pred_weak_2019yuan")
    after = metric(pred, "pred_calibrated_2019yuan")
    metric_names = ["r2", "mae", "rmse", "mape", "bias"]
    checks = {
        "record_count": int(len(pred)),
        "province_count": int(pred["province_std"].nunique()),
        "province_year_count": int(len(factors)),
        "factor_rows_unique": not factors.duplicated(["year", "province_std"]).any(),
        "all_factors_finite_positive": bool(
            np.isfinite(factors["calibration_factor"]).all()
            and (factors["calibration_factor"] > 0).all()
        ),
        "calibrated_values_recompute_match": bool(
            np.allclose(
                recomputed_calibrated,
                pred["pred_calibrated_2019yuan"].to_numpy(float),
                rtol=1e-6,
                atol=1e-4,
            )
        ),
        "before_metrics_recompute_match": all(
            close(before[name], summary["before_calibration"][name]) for name in metric_names
        ),
        "after_metrics_recompute_match": all(
            close(after[name], summary["after_calibration"][name]) for name in metric_names
        ),
        "max_province_constraint_relative_error": float(max(constraint_errors)),
        "all_within_province_year_ranks_unchanged": bool(all(rank_checks)),
        "county_labels_used_to_compute_factor": False,
        "official_province_constraint_used": True,
    }
    passed = (
        checks["record_count"] == 8275
        and checks["province_count"] == 21
        and checks["province_year_count"] == 97
        and checks["factor_rows_unique"]
        and checks["all_factors_finite_positive"]
        and checks["calibrated_values_recompute_match"]
        and checks["before_metrics_recompute_match"]
        and checks["after_metrics_recompute_match"]
        and checks["max_province_constraint_relative_error"] < 1e-6
        and checks["all_within_province_year_ranks_unchanged"]
    )
    checks["audit_status"] = "PASS" if passed else "FAIL"
    (RUN / "stage30_product_calibration_audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

