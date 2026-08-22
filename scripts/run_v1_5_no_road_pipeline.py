"""Safe orchestration entry for the canonical v1.5 no-road workflow.

The command prints the planned commands by default. Pass ``--execute`` only
after the restricted local inputs described in ``data/README_expected_inputs.md``
have been supplied and checked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "workflow_v1.5_no_road_core.json"
STEPS = [
    (
        "core_no_road",
        "Train the no-road core model, run leave-year validation, and regenerate annual/change maps.",
        ["scripts/stage12_hebei14_full_rerun.py"],
    ),
    (
        "leave_province_2020",
        "Run the retained single-year leave-province diagnostic without road density.",
        ["scripts/stage19_v3_leave_province_validation.py"],
    ),
    (
        "ensemble",
        "Regenerate the five-member no-road ensemble uncertainty products.",
        ["scripts/stage8_ensemble_uncertainty.py"],
    ),
    (
        "multiyear_leave_province",
        "Run 2019–2023 GroupKFold with every province held out across all years.",
        ["scripts/stage27_multiyear_leave_province_validation.py"],
    ),
    (
        "audit_multiyear",
        "Audit fold assignment, feature exclusion and metric recomputation.",
        ["scripts/stage28_audit_multiyear_leave_province_validation.py"],
    ),
    (
        "product_calibration",
        "Apply the operational official province-year constraints to held-out predictions.",
        ["scripts/stage29_multiyear_leave_province_product_calibration.py"],
    ),
    (
        "audit_product_calibration",
        "Audit product-calibration factors, constraints, metrics and rank preservation.",
        ["scripts/stage30_audit_multiyear_product_calibration.py"],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print("=" * 78)
    print(f"v1.5-no-road pipeline — {mode}")
    print(f"Config: {config}")
    print("=" * 78)
    for index, (name, description, command) in enumerate(STEPS, start=1):
        full = [sys.executable, *command]
        if name in {"core_no_road", "leave_province_2020", "ensemble", "multiyear_leave_province"}:
            full.extend(["--config", str(config)])
        print(f"[{index}/{len(STEPS)}] {name}: {description}")
        print("    " + " ".join(full))
        if args.execute:
            subprocess.run(full, cwd=ROOT, check=True)
    if not args.execute:
        print("\nNo files were changed. Re-run with --execute after checking local inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
