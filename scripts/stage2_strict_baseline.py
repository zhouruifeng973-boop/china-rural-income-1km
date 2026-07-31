# -*- coding: utf-8 -*-
"""
Stage 2 strict county-level baseline.

This script is still a county-level baseline, but it fixes two important
training-flow issues:
  1) The target uses CPI-deflated county rural income in 2019 yuan.
  2) Imputation and collinearity-based feature selection are fit inside each
     validation fold, not before cross-validation.

It creates a clean county-year feature panel from outputs/training_data_<year>.npz,
then runs:
  - 5-fold leave-county grouped CV
  - leave-one-province-out CV

Outputs are written to outputs/stage2_training/.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2_training"
PROCESSED = ROOT / "data" / "processed"
YEARS = [2019, 2020, 2021, 2022, 2023]
TARGET_COL = "rural_income_2019yuan"
RANDOM_STATE = 42


def model_factory(name: str):
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


def build_county_feature_panel() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED / "county_feature_panel_2019_2023.csv"
    if panel_path.exists():
        return pd.read_csv(panel_path)

    income_path = PROCESSED / "county_income_2019_2023_deflated.csv"
    if not income_path.exists():
        raise FileNotFoundError(
            f"Missing {income_path}. Run scripts/stage2_preprocess_data.py first."
        )
    income = pd.read_csv(income_path)
    income["county_code"] = income["county_code"].astype(int)
    income_lookup = {
        (int(row.year), int(row.county_code)): row
        for row in income.itertuples(index=False)
    }

    rows = []
    feature_names = None

    for year in YEARS:
        npz_path = ROOT / "outputs" / f"training_data_{year}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}")
        print(f"Aggregating county features for {year}...")
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        codes = data["county_code"].astype(int)
        names = [str(x) for x in data["band_names"]]
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature names changed in {year}")

        valid_code = codes > 0
        unique_codes = np.unique(codes[valid_code])
        for idx, county_code in enumerate(unique_codes, start=1):
            key = (year, int(county_code))
            if key not in income_lookup:
                continue
            mask = codes == county_code
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(X[mask], axis=0)
            inc = income_lookup[key]
            row = {
                "year": year,
                "county_code": int(county_code),
                "county": inc.county,
                "city": inc.city,
                "province": inc.province,
                "province_std": inc.province_std,
                "rural_income_nominal": float(inc.rural_income_nominal),
                "rural_income_2019yuan": float(inc.rural_income_2019yuan),
                "pixel_count": int(mask.sum()),
            }
            row.update({name: float(value) for name, value in zip(feature_names, med)})
            rows.append(row)

        del data, X, codes

    panel = pd.DataFrame(rows)
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    return panel


def single_feature_r2(x: np.ndarray, y: np.ndarray) -> float:
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr * corr)


def select_uncorrelated_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.85,
) -> list[int]:
    n = X_train.shape[1]
    if n <= 1:
        return list(range(n))

    single_r2 = [single_feature_r2(X_train[:, i], y_train) for i in range(n)]
    corr = np.corrcoef(X_train.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    keep = []
    for members in groups.values():
        best = max(members, key=lambda i: single_r2[i])
        keep.append(best)
    keep = sorted(keep)
    return keep


def metric_dict(df: pd.DataFrame) -> dict:
    y = df["true"].to_numpy(dtype=float)
    pred = df["pred"].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "r2": float(r2_score(y, pred)) if len(df) >= 2 else np.nan,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mape": float(np.mean(np.abs(y - pred) / y) * 100),
        "bias": float(np.mean(pred - y)),
    }


def run_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    split_name: str,
    include_year_feature: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_use = list(feature_cols)
    if include_year_feature:
        df = df.copy()
        df["year_index"] = df["year"] - min(YEARS)
        feature_use.append("year_index")

    X_all = df[feature_use].to_numpy(dtype=float)
    y_all = df[TARGET_COL].to_numpy(dtype=float)

    split_iter = []
    if split_name == "leave_county_5fold":
        gkf = GroupKFold(n_splits=5)
        groups = df["county_code"].to_numpy()
        for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
            split_iter.append((f"fold_{fold}", tr, te))
    elif split_name == "leave_province":
        provinces = sorted(df["province_std"].dropna().unique())
        idx = np.arange(len(df))
        for province in provinces:
            te = idx[df["province_std"].to_numpy() == province]
            tr = idx[df["province_std"].to_numpy() != province]
            if len(te) > 0 and len(tr) > 0:
                split_iter.append((province, tr, te))
    else:
        raise ValueError(split_name)

    pred_rows = []
    fold_rows = []
    for fold_label, tr, te in split_iter:
        X_tr_raw, X_te_raw = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]

        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr_raw)
        X_te = imputer.transform(X_te_raw)

        keep_idx = select_uncorrelated_features(X_tr, y_tr, feature_use)
        model = model_factory(model_name)
        model.fit(X_tr[:, keep_idx], y_tr)
        pred = model.predict(X_te[:, keep_idx])

        fold_df = pd.DataFrame(
            {
                "year": df.iloc[te]["year"].to_numpy(),
                "county_code": df.iloc[te]["county_code"].to_numpy(),
                "county": df.iloc[te]["county"].to_numpy(),
                "province_std": df.iloc[te]["province_std"].to_numpy(),
                "true": y_te,
                "pred": pred,
                "model": model_name,
                "split": split_name,
                "fold": fold_label,
                "include_year_feature": include_year_feature,
                "selected_feature_count": len(keep_idx),
                "selected_features": "|".join([feature_use[i] for i in keep_idx]),
            }
        )
        pred_rows.append(fold_df)
        m = metric_dict(fold_df)
        m.update(
            {
                "model": model_name,
                "split": split_name,
                "fold": fold_label,
                "include_year_feature": include_year_feature,
                "selected_feature_count": len(keep_idx),
            }
        )
        fold_rows.append(m)
        print(
            f"{split_name} | {model_name} | year_feature={include_year_feature} | "
            f"{fold_label}: R2={m['r2']:.3f}, MAE={m['mae']:.0f}"
        )

    predictions = pd.concat(pred_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    return predictions, fold_metrics


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "split", "include_year_feature"]
    for keys, g in predictions.groupby(group_cols):
        m = metric_dict(g)
        rows.append(dict(zip(group_cols, keys), level="overall", group="all", **m))
        for year, gy in g.groupby("year"):
            my = metric_dict(gy)
            rows.append(dict(zip(group_cols, keys), level="year", group=str(year), **my))
    return pd.DataFrame(rows)


def write_summary(overall_metrics: pd.DataFrame, fold_metrics: pd.DataFrame, panel: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top = overall_metrics[overall_metrics["level"] == "overall"].copy()
    top = top.sort_values(["split", "include_year_feature", "model"])

    def fmt_bool(v: bool) -> str:
        return "yes" if bool(v) else "no"

    lines = [
        "# Stage 2 严格 baseline 训练汇总",
        "",
        "## 训练口径",
        "",
        "- 样本单位：县-年。",
        "- 目标变量：县级农村居民人均可支配收入，按省级 CPI 折算为 2019 年不变价。",
        "- 特征：原始 28 个遥感/空间特征按县内像元中位数聚合。",
        "- 交叉验证内部执行：缺失值中位数填补、共线性筛选、模型训练。",
        "- 该结果是更严格的县级 baseline，不是最终弱监督 1 km 下推模型。",
        "",
        "## 样本",
        "",
        f"- 县-年样本数：{len(panel)}。",
        f"- 县代码并集：{panel['county_code'].nunique()}。",
        f"- 省级地区数：{panel['province_std'].nunique()}。",
        "",
        "## 总体指标",
        "",
        "| split | model | year feature | R2 | MAE | RMSE | MAPE | Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.model} | {fmt_bool(row.include_year_feature)} | "
            f"{row.r2:.3f} | {row.mae:.0f} | {row.rmse:.0f} | {row.mape:.2f}% | {row.bias:.0f} |"
        )

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `data/processed/county_feature_panel_2019_2023.csv`",
            "- `outputs/stage2_training/strict_baseline_predictions.csv`",
            "- `outputs/stage2_training/strict_baseline_overall_metrics.csv`",
            "- `outputs/stage2_training/strict_baseline_fold_metrics.csv`",
            "",
            "## 下一步",
            "",
            "用 Stage 2 生成的农村人口权重训练弱监督像元下推模型：像元预测收入先按农村人口权重聚合回县，再与县级真实收入计算损失。",
        ]
    )
    (OUT / "stage2_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/3] Building/loading county feature panel...")
    panel = build_county_feature_panel()

    base_cols = {
        "year",
        "county_code",
        "county",
        "city",
        "province",
        "province_std",
        "rural_income_nominal",
        "rural_income_2019yuan",
        "pixel_count",
    }
    feature_cols = [c for c in panel.columns if c not in base_cols]
    panel = panel.dropna(subset=[TARGET_COL, "province_std"]).copy()

    print("[2/3] Running strict CV...")
    all_preds = []
    all_folds = []
    for split_name in ["leave_county_5fold", "leave_province"]:
        for include_year in [False, True]:
            for model_name in ["RF", "XGBoost"]:
                preds, folds = run_splits(
                    panel,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    split_name=split_name,
                    include_year_feature=include_year,
                )
                all_preds.append(preds)
                all_folds.append(folds)

    predictions = pd.concat(all_preds, ignore_index=True)
    fold_metrics = pd.concat(all_folds, ignore_index=True)
    overall_metrics = summarize_predictions(predictions)

    predictions.to_csv(OUT / "strict_baseline_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUT / "strict_baseline_fold_metrics.csv", index=False, encoding="utf-8-sig")
    overall_metrics.to_csv(OUT / "strict_baseline_overall_metrics.csv", index=False, encoding="utf-8-sig")

    print("[3/3] Writing summary...")
    write_summary(overall_metrics, fold_metrics, panel)
    print("Done.")
    print(f"Summary: {OUT / 'stage2_training_summary.md'}")


if __name__ == "__main__":
    main()
