"""Run or print the v1.0-draft reproducibility pipeline.

Default behavior is dry-run: commands are printed but not executed.
Use --execute to run them.

Example:
    python scripts/run_reproducible_pipeline.py
    python scripts/run_reproducible_pipeline.py --execute --start stage7 --stop stage9
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from repro_config import comma_ints, deep_get, int_list, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.0_draft.yaml"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    description: str


STEP_NAMES = [
    "stage0_input_check",
    "stage2_preprocess",
    "stage2_baseline",
    "stage3_downscaling_2020",
    "stage3_spatiotemporal_validation",
    "stage4_province_calibration",
    "stage5_final_maps",
    "stage5_unbounded_maps",
    "stage5_change_analysis",
    "stage5_raster_statistics",
    "stage6_quality_layers",
    "stage6_package",
    "stage7_uncertainty_proxy",
    "stage7_validation_diagnostics",
    "stage8_ensemble_uncertainty",
    "stage9_audit",
]


def build_pipeline(config_path: str | Path | None = None) -> list[Step]:
    cfg = load_config(config_path)
    main_mask = str(deep_get(cfg, "project.main_mask", "strict"))
    study_years = int_list(deep_get(cfg, "project.study_years"), [2019, 2020, 2021, 2022, 2023])

    weak = deep_get(cfg, "model.weak_supervised_downscaling", {})
    stval = deep_get(cfg, "model.spatiotemporal_validation", {})
    ens = deep_get(cfg, "model.ensemble_uncertainty", {})
    final_maps = deep_get(cfg, "model.final_maps", {})
    config_arg = str(config_path or DEFAULT_CONFIG)

    benchmark_year = int(weak.get("benchmark_year", 2020))
    weak_mask = str(weak.get("mask", main_mask))
    weak_folds = int(weak.get("folds", 5))
    weak_pixels = int(weak.get("pixels_per_county", 128))
    weak_epochs = int(weak.get("epochs_main_2020", 180))
    weak_lr = float(weak.get("lr", 0.001))
    weak_threads = int(weak.get("threads", 8))

    st_run = str(stval.get("run", "all"))
    st_year = int(stval.get("year", benchmark_year))
    st_mask = str(stval.get("mask", main_mask))
    st_province_folds = int(stval.get("province_folds", 5))
    st_province_pixels = int(stval.get("province_pixels", 128))
    st_year_pixels = int(stval.get("year_pixels", 64))
    st_epochs = int(stval.get("epochs", stval.get("epochs_spatiotemporal_validation", 150)))
    st_lr = float(stval.get("lr", 0.001))
    st_threads = int(stval.get("threads", 8))

    ens_members = int(ens.get("members", 5))
    ens_seeds = comma_ints(ens.get("seeds"), [1042, 2039, 3036, 4033, 5030])
    ens_years = [str(y) for y in int_list(ens.get("years"), study_years)]
    ens_pixels = int(ens.get("pixels_per_county", 128))
    ens_epochs = int(ens.get("epochs", 150))
    ens_lr = float(ens.get("lr", 0.001))
    ens_threads = int(ens.get("threads", 8))
    ens_row_block = int(ens.get("row_block", 128))
    final_years = [str(y) for y in int_list(final_maps.get("years"), study_years)]
    final_pixels = int(final_maps.get("pixels_per_county", 128))
    final_epochs = int(final_maps.get("epochs", 150))
    final_lr = float(final_maps.get("lr", 0.001))
    final_threads = int(final_maps.get("threads", 8))
    final_row_block = int(final_maps.get("row_block", 128))
    final_reuse_model = bool(final_maps.get("reuse_model", False))
    final_command = [
        "scripts/stage5_final_maps.py",
        "--config",
        config_arg,
        "--years",
        *final_years,
        "--pixels-per-county",
        str(final_pixels),
        "--epochs",
        str(final_epochs),
        "--lr",
        str(final_lr),
        "--threads",
        str(final_threads),
        "--row-block",
        str(final_row_block),
    ]
    if final_reuse_model:
        final_command.append("--reuse-model")

    return [
        Step(
        "stage0_input_check",
        ["scripts/check_reproducibility_inputs.py", "--config", config_arg],
        "Check expected inputs, intermediate files, final package files, and redistribution boundaries.",
    ),
        Step(
        "stage2_preprocess",
        ["scripts/stage2_preprocess_data.py", "--config", config_arg],
        "Preprocess province income, CPI, WorldPop, GHS-SMOD, and rural weights.",
    ),
    Step(
        "stage2_baseline",
        ["scripts/stage2_strict_baseline.py", "--config", config_arg],
        "Run strict county-level baseline validation.",
    ),
        Step(
        "stage3_downscaling_2020",
        [
            "scripts/stage3_weak_supervised_downscaling.py",
            "--config",
            config_arg,
            "--year",
            str(benchmark_year),
            "--mask",
            weak_mask,
            "--folds",
            str(weak_folds),
            "--pixels-per-county",
            str(weak_pixels),
            "--epochs",
            str(weak_epochs),
            "--lr",
            str(weak_lr),
            "--threads",
            str(weak_threads),
        ],
        "Run weak-supervised pixel downscaling for the 2020 strict benchmark.",
    ),
        Step(
        "stage3_spatiotemporal_validation",
        [
            "scripts/stage3_spatiotemporal_validation.py",
            "--config",
            config_arg,
            "--run",
            st_run,
            "--year",
            str(st_year),
            "--mask",
            st_mask,
            "--province-folds",
            str(st_province_folds),
            "--province-pixels",
            str(st_province_pixels),
            "--year-pixels",
            str(st_year_pixels),
            "--epochs",
            str(st_epochs),
            "--lr",
            str(st_lr),
            "--threads",
            str(st_threads),
        ],
        "Run leave-province and leave-year validation.",
    ),
    Step(
        "stage4_province_calibration",
        ["scripts/stage4_province_calibration.py", "--config", config_arg],
        "Apply province-year calibration.",
    ),
    Step(
        "stage5_final_maps",
        final_command,
        "Generate final annual income maps.",
    ),
    Step(
        "stage5_unbounded_maps",
        ["scripts/stage5_create_unbounded_maps.py", "--config", config_arg],
        "Generate unbounded main-version maps.",
    ),
    Step(
        "stage5_change_analysis",
        ["scripts/stage5_change_analysis.py", "--config", config_arg],
        "Generate 2019-2023 change maps and summaries.",
    ),
    Step(
        "stage5_raster_statistics",
        ["scripts/stage5_add_raster_statistics.py", "--config", config_arg],
        "Add raster summary statistics.",
    ),
    Step(
        "stage6_quality_layers",
        ["scripts/stage6_create_quality_layers.py", "--config", config_arg],
        "Generate quality layers.",
    ),
    Step(
        "stage6_package",
        ["scripts/stage6_prepare_scientific_data_package.py", "--config", config_arg],
        "Prepare the Scientific Data-style package.",
    ),
    Step(
        "stage7_uncertainty_proxy",
        ["scripts/stage7_uncertainty_proxy_layers.py", "--config", config_arg],
        "Generate rule-based uncertainty proxy layers.",
    ),
    Step(
        "stage7_validation_diagnostics",
        ["scripts/stage7_validation_diagnostics.py", "--config", config_arg],
        "Generate additional validation diagnostics tables.",
    ),
        Step(
        "stage8_ensemble_uncertainty",
        [
            "scripts/stage8_ensemble_uncertainty.py",
            "--config",
            config_arg,
            "--members",
            str(ens_members),
            "--seeds",
            ens_seeds,
            "--years",
            *ens_years,
            "--pixels-per-county",
            str(ens_pixels),
            "--epochs",
            str(ens_epochs),
            "--lr",
            str(ens_lr),
            "--threads",
            str(ens_threads),
            "--row-block",
            str(ens_row_block),
        ],
        "Generate multi-seed ensemble uncertainty layers.",
    ),
    Step(
        "stage9_audit",
        ["scripts/stage9_data_product_audit.py", "--config", config_arg],
        "Audit final raster alignment, NoData, naming, manifests, and documentation.",
    ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands. Without this flag, only print the pipeline.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to workflow configuration YAML.",
    )
    parser.add_argument(
        "--start",
        default=STEP_NAMES[0],
        choices=STEP_NAMES,
        help="First step to include.",
    )
    parser.add_argument(
        "--stop",
        default=STEP_NAMES[-1],
        choices=STEP_NAMES,
        help="Last step to include.",
    )
    return parser.parse_args()


def selected_steps(pipeline: list[Step], start: str, stop: str) -> list[Step]:
    names = [step.name for step in pipeline]
    i = names.index(start)
    j = names.index(stop)
    if i > j:
        raise ValueError(f"--start must come before --stop, got {start!r} > {stop!r}")
    return pipeline[i : j + 1]


def format_command(step: Step) -> list[str]:
    return [sys.executable, *step.command]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = build_pipeline(args.config)
    steps = selected_steps(pipeline, args.start, args.stop)

    print("=" * 78)
    print("v1.0-draft reproducibility pipeline")
    print("=" * 78)
    print(f"Project root: {ROOT}")
    print(f"Config: {args.config}")
    if not deep_get(cfg, "_meta.loaded_from_file", False):
        print("Config status: using embedded defaults; install PyYAML to read the YAML file.")
    else:
        print("Config status: loaded from YAML file.")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Steps: {steps[0].name} -> {steps[-1].name}")
    print()

    for idx, step in enumerate(steps, start=1):
        cmd = format_command(step)
        print(f"[{idx}/{len(steps)}] {step.name}")
        print(f"    {step.description}")
        print("    " + " ".join(cmd))
        if args.execute:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        print()


if __name__ == "__main__":
    main()
