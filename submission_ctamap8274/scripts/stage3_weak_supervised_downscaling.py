# -*- coding: utf-8 -*-
"""
Stage 3 weak-supervised 1 km rural income downscaling.

Core idea:
  pixel features -> pixel income predictions -> rural-population-weighted
  aggregation back to county -> compare with county income labels.

This first version is intentionally validation-first. It runs one year at a time
and writes all outputs to outputs/stage3_weak_supervision/ without touching the
old baseline or map products.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import rasterio
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

from repro_config import deep_get, load_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3_weak_supervision"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "features"
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IncomeMLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        init_log_income: float,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.init_log_income = float(init_log_income)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict log income. The offset makes the initial scale reasonable.
        raw = self.net(x).squeeze(-1)
        log_income = raw + self.init_log_income
        return torch.clamp(log_income, math.log(3000.0), math.log(80000.0))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
        "bias": float(np.mean(y_pred - y_true)),
    }


def load_targets(year: int) -> pd.DataFrame:
    path = PROCESSED / "county_income_2019_2023_deflated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run stage2_preprocess_data.py first.")
    df = pd.read_csv(path)
    df = df[df["year"] == year].copy()
    df["county_code"] = df["county_code"].astype(int)
    df = df.dropna(subset=["rural_income_2019yuan", "province_std"])
    return df[
        [
            "year",
            "county_code",
            "county",
            "city",
            "province_std",
            "rural_income_nominal",
            "rural_income_2019yuan",
        ]
    ]


def reconstruct_labeled_weights(year: int, mask_name: str, data: np.lib.npyio.NpzFile) -> np.ndarray:
    weight_path = PROCESSED / "rural_weights" / f"rural_pop_weight_{year}_{mask_name}.tif"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing {weight_path}. Run stage2_preprocess_data.py first.")

    income_csv = ROOT / "data" / f"county_rural_income_{year}.csv"
    income = pd.read_csv(income_csv)
    income_codes = set(income["county_code"].astype(int).tolist())

    valid_mask = data["valid_mask"].astype(bool)
    labeled_codes = data["county_code"].astype(int)

    with rasterio.open(FEATURES / "county_id.tif") as src:
        county_id = src.read(1).astype(np.int64)
    with rasterio.open(weight_path) as src:
        weight = src.read(1).astype(np.float32)

    valid_codes = county_id[valid_mask]
    labeled_mask = np.isin(valid_codes, list(income_codes))
    reconstructed_codes = valid_codes[labeled_mask].astype(int)
    if len(reconstructed_codes) != len(labeled_codes) or not np.array_equal(
        reconstructed_codes, labeled_codes
    ):
        raise RuntimeError(
            "Could not reconstruct pixel positions for training_data_*.npz. "
            "Please regenerate training_data with the current county income CSV."
        )
    return weight[valid_mask][labeled_mask].astype(np.float32)


def load_pixel_panel(year: int, mask_name: str) -> dict:
    npz_path = ROOT / "outputs" / f"training_data_{year}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    targets = load_targets(year)
    target_by_code = dict(zip(targets["county_code"], targets["rural_income_2019yuan"]))

    print(f"Loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    codes = data["county_code"].astype(int)
    band_names = [str(x) for x in data["band_names"]]
    weights = reconstruct_labeled_weights(year, mask_name, data)
    data.close()

    y_county = np.array([target_by_code.get(int(c), np.nan) for c in codes], dtype=np.float32)
    keep = np.isfinite(y_county) & np.isfinite(weights) & (weights > 0) & (codes > 0)
    X = X[keep]
    codes = codes[keep]
    weights = weights[keep]

    county_codes = np.array(sorted(set(codes.tolist()) & set(target_by_code.keys())), dtype=int)
    target_vec = np.array([target_by_code[int(c)] for c in county_codes], dtype=np.float32)

    code_to_group = {int(code): i for i, code in enumerate(county_codes)}
    groups = np.array([code_to_group[int(c)] for c in codes], dtype=np.int64)
    order = np.argsort(groups, kind="stable")
    group_counts = np.bincount(groups, minlength=len(county_codes))
    group_offsets = np.concatenate([[0], np.cumsum(group_counts)])
    group_indices = [
        order[group_offsets[i] : group_offsets[i + 1]] for i in range(len(county_codes))
    ]

    county_meta = targets.set_index("county_code").loc[county_codes].reset_index()

    print(
        f"Rural weighted pixels: {len(X):,}; counties with positive rural weight: {len(county_codes):,}"
    )
    print(f"Features: {len(band_names)}")
    return {
        "X": X,
        "codes": codes,
        "weights": weights,
        "groups": groups,
        "group_indices": group_indices,
        "county_codes": county_codes,
        "target": target_vec,
        "county_meta": county_meta,
        "band_names": band_names,
    }


def sample_pixels_by_county(
    group_indices: list[np.ndarray],
    weights: np.ndarray,
    train_group_ids: np.ndarray,
    pixels_per_county: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for group_id in train_group_ids:
        idx = group_indices[int(group_id)]
        if len(idx) == 0:
            continue
        if len(idx) <= pixels_per_county:
            selected.append(idx)
            continue
        # Sample rural pixels uniformly, then apply population weights in the
        # county aggregation loss. Probability-weighted sampling plus weighted
        # aggregation would accidentally count the population weight twice.
        chosen = rng.choice(idx, size=pixels_per_county, replace=False)
        selected.append(chosen)
    if not selected:
        raise RuntimeError("No pixels were selected for training.")
    return np.concatenate(selected)


def fit_fold_model(
    X: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    group_indices: list[np.ndarray],
    target: np.ndarray,
    train_group_ids: np.ndarray,
    pixels_per_county: int,
    epochs: int,
    lr: float,
    seed: int,
    output_regularization: float = 0.001,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.05,
) -> tuple[IncomeMLP, SimpleImputer, StandardScaler, dict]:
    # Keep pixel sampling, network initialization, and Dropout reproducible
    # for each validation fold and final training run.
    set_seed(seed)
    rng = np.random.default_rng(seed)
    selected_idx = sample_pixels_by_county(
        group_indices, weights, train_group_ids, pixels_per_county, rng
    )

    train_group_map = {int(g): i for i, g in enumerate(train_group_ids)}
    sampled_groups = np.array([train_group_map[int(g)] for g in groups[selected_idx]], dtype=np.int64)
    sampled_weights = weights[selected_idx].astype(np.float32)
    sampled_targets = target[train_group_ids].astype(np.float32)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X[selected_idx])
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)

    x_t = torch.from_numpy(X_scaled)
    w_t = torch.from_numpy(sampled_weights)
    g_t = torch.from_numpy(sampled_groups)
    y_t = torch.from_numpy(sampled_targets)
    log_y_t = torch.log(y_t)

    model = IncomeMLP(
        X.shape[1],
        init_log_income=float(np.log(np.median(sampled_targets))),
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_groups = len(train_group_ids)
    mean_log_y = log_y_t.mean()
    last_loss = np.nan

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        log_pred = model(x_t)
        pred = torch.exp(log_pred)

        num = torch.zeros(n_groups, dtype=torch.float32)
        den = torch.zeros(n_groups, dtype=torch.float32)
        num.index_add_(0, g_t, pred * w_t)
        den.index_add_(0, g_t, w_t)
        county_pred = num / torch.clamp(den, min=1e-6)

        main_loss = torch.mean((torch.log(torch.clamp(county_pred, min=1.0)) - log_y_t) ** 2)
        regularizer = output_regularization * torch.mean((log_pred - mean_log_y) ** 2)
        loss = main_loss + regularizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        last_loss = float(loss.detach().cpu().item())

        if epoch in {1, epochs} or epoch % max(50, epochs // 4) == 0:
            print(f"    epoch {epoch:4d}/{epochs}: loss={last_loss:.6f}")

    stats = {
        "sampled_pixels": int(len(selected_idx)),
        "train_counties": int(len(train_group_ids)),
        "final_loss": last_loss,
        "output_regularization": float(output_regularization),
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "dropout": float(dropout),
    }
    return model, imputer, scaler, stats


@torch.no_grad()
def aggregate_prediction(
    model: IncomeMLP,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    X: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    eval_group_ids: np.ndarray,
    batch_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    local_lookup = np.full(int(groups.max()) + 1, -1, dtype=np.int64)
    local_lookup[eval_group_ids] = np.arange(len(eval_group_ids), dtype=np.int64)
    eval_mask = np.isin(groups, eval_group_ids)
    eval_indices = np.flatnonzero(eval_mask)

    sums = np.zeros(len(eval_group_ids), dtype=np.float64)
    den = np.zeros(len(eval_group_ids), dtype=np.float64)
    pixel_counts = np.zeros(len(eval_group_ids), dtype=np.int64)

    for start in range(0, len(eval_indices), batch_size):
        idx = eval_indices[start : start + batch_size]
        local_groups = local_lookup[groups[idx]]
        X_chunk = scaler.transform(imputer.transform(X[idx])).astype(np.float32)
        pred = torch.exp(model(torch.from_numpy(X_chunk))).cpu().numpy().astype(np.float64)
        w = weights[idx].astype(np.float64)
        np.add.at(sums, local_groups, pred * w)
        np.add.at(den, local_groups, w)
        np.add.at(pixel_counts, local_groups, 1)

    pred_county = sums / np.maximum(den, 1e-9)
    return pred_county.astype(np.float32), den.astype(np.float32), pixel_counts


def run_county_cv(
    panel: dict,
    year: int,
    mask_name: str,
    folds: int,
    pixels_per_county: int,
    epochs: int,
    lr: float,
    output_regularization: float = 0.001,
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = panel["X"]
    groups = panel["groups"]
    group_indices = panel["group_indices"]
    weights = panel["weights"]
    county_codes = panel["county_codes"]
    target = panel["target"]
    meta = panel["county_meta"].copy()

    gkf = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(county_codes))

    pred_rows = []
    fold_rows = []
    for fold, (train_group_ids, test_group_ids) in enumerate(
        gkf.split(dummy, target, groups=county_codes), start=1
    ):
        print(f"\nFold {fold}/{folds}: train counties={len(train_group_ids)}, test={len(test_group_ids)}")
        model, imputer, scaler, train_stats = fit_fold_model(
            X,
            weights,
            groups,
            group_indices,
            target,
            train_group_ids=train_group_ids,
            pixels_per_county=pixels_per_county,
            epochs=epochs,
            lr=lr,
            seed=RANDOM_STATE + fold,
            output_regularization=output_regularization,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
        )
        pred, weight_sum, pixel_count = aggregate_prediction(
            model, imputer, scaler, X, groups, weights, test_group_ids
        )
        true = target[test_group_ids]
        metrics = metric_dict(true, pred)
        metrics.update(
            {
                "year": year,
                "mask": mask_name,
                "fold": fold,
                "pixels_per_county": pixels_per_county,
                "epochs": epochs,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "dropout": dropout,
                **train_stats,
            }
        )
        fold_rows.append(metrics)
        print(
            f"  Fold {fold} validation: R2={metrics['r2']:.3f}, "
            f"MAE={metrics['mae']:.0f}, RMSE={metrics['rmse']:.0f}"
        )

        fold_meta = meta.iloc[test_group_ids].copy()
        fold_meta["fold"] = fold
        fold_meta["true_2019yuan"] = true
        fold_meta["pred_weak_2019yuan"] = pred
        fold_meta["rural_weight_sum"] = weight_sum
        fold_meta["rural_pixel_count"] = pixel_count
        pred_rows.append(fold_meta)

    predictions = pd.concat(pred_rows, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    return predictions, fold_metrics


def write_summary(
    year: int,
    mask_name: str,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    overall = metric_dict(
        predictions["true_2019yuan"].to_numpy(dtype=float),
        predictions["pred_weak_2019yuan"].to_numpy(dtype=float),
    )
    overall.update(
        {
            "year": year,
            "mask": mask_name,
            "folds": args.folds,
            "pixels_per_county": args.pixels_per_county,
            "epochs": args.epochs,
        }
    )

    overall_path = OUT / f"weak_supervised_{year}_{mask_name}_overall_metrics.json"
    overall_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 3 弱监督像元下推验证：{year} {mask_name}",
        "",
        "## 方法",
        "",
        "- 输入：1 km 遥感/空间特征。",
        "- 模型：MLP 预测像元 log income。",
        "- 监督：将像元预测值按农村人口权重聚合回县级，再与县级农村收入标签比较。",
        "- 当前版本：单年县级交叉验证，用来判断弱监督流程是否稳定。",
        "",
        "## 参数",
        "",
        f"- folds: {args.folds}",
        f"- pixels_per_county: {args.pixels_per_county}",
        f"- epochs: {args.epochs}",
        f"- learning_rate: {args.lr}",
        "",
        "## 总体验证结果",
        "",
        f"- R2: {overall['r2']:.4f}",
        f"- MAE: {overall['mae']:.0f} 元",
        f"- RMSE: {overall['rmse']:.0f} 元",
        f"- MAPE: {overall['mape']:.2f}%",
        f"- Bias: {overall['bias']:.0f} 元",
        "",
        "## 输出",
        "",
        f"- `weak_supervised_{year}_{mask_name}_county_predictions.csv`",
        f"- `weak_supervised_{year}_{mask_name}_fold_metrics.csv`",
        f"- `weak_supervised_{year}_{mask_name}_overall_metrics.json`",
        "",
        "## 解释提醒",
        "",
        "这是弱监督像元下推的第一版验证结果。它比普通县级 baseline 更接近论文目标，"
        "但像元值仍然是县级约束下的估计，并不是真实像元收入观测。",
    ]
    (OUT / f"weak_supervised_{year}_{mask_name}_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(ROOT / "config" / "workflow_v1.0_draft.yaml"))
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    defaults = deep_get(cfg, "model.weak_supervised_downscaling", {})
    main_mask = deep_get(cfg, "project.main_mask", "strict")
    default_threads = max(1, min(8, (os.cpu_count() or 4) // 2))

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--year", type=int, default=int(defaults.get("benchmark_year", 2020)))
    parser.add_argument("--mask", choices=["strict", "broad"], default=str(defaults.get("mask", main_mask)))
    parser.add_argument("--folds", type=int, default=int(defaults.get("folds", 5)))
    parser.add_argument("--pixels-per-county", type=int, default=int(defaults.get("pixels_per_county", 128)))
    parser.add_argument("--epochs", type=int, default=int(defaults.get("epochs_main_2020", 180)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 0.001)))
    parser.add_argument("--hidden1", type=int, default=int(defaults.get("hidden1", 64)))
    parser.add_argument("--hidden2", type=int, default=int(defaults.get("hidden2", 32)))
    parser.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.05)))
    parser.add_argument(
        "--output-regularization",
        type=float,
        default=float(defaults.get("output_regularization", 0.001)),
        help="Coefficient for shrinkage of pixel log-income towards the training mean.",
    )
    parser.add_argument("--threads", type=int, default=int(defaults.get("threads", default_threads)))
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    set_seed(RANDOM_STATE)
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 3 weak-supervised rural income downscaling")
    print("=" * 70)
    print(f"year={args.year}, mask={args.mask}, folds={args.folds}")
    print(
        f"pixels_per_county={args.pixels_per_county}, epochs={args.epochs}, "
        f"hidden={args.hidden1}->{args.hidden2}, dropout={args.dropout}, "
        f"threads={args.threads}"
    )

    panel = load_pixel_panel(args.year, args.mask)
    predictions, fold_metrics = run_county_cv(
        panel,
        year=args.year,
        mask_name=args.mask,
        folds=args.folds,
        pixels_per_county=args.pixels_per_county,
        epochs=args.epochs,
        lr=args.lr,
        output_regularization=args.output_regularization,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    )

    pred_path = OUT / f"weak_supervised_{args.year}_{args.mask}_county_predictions.csv"
    fold_path = OUT / f"weak_supervised_{args.year}_{args.mask}_fold_metrics.csv"
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(fold_path, index=False, encoding="utf-8-sig")
    write_summary(args.year, args.mask, predictions, fold_metrics, args)

    overall = metric_dict(
        predictions["true_2019yuan"].to_numpy(dtype=float),
        predictions["pred_weak_2019yuan"].to_numpy(dtype=float),
    )
    print("\nOverall validation:")
    print(
        f"R2={overall['r2']:.4f}, MAE={overall['mae']:.0f}, "
        f"RMSE={overall['rmse']:.0f}, MAPE={overall['mape']:.2f}%, Bias={overall['bias']:.0f}"
    )
    print(f"\nSaved: {pred_path}")
    print(f"Saved: {fold_path}")


if __name__ == "__main__":
    main()
