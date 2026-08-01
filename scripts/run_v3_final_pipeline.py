"""Safe orchestration entry for the v1.3 official44-corrected workflow.

The default mode is DRY-RUN. Use --execute only for a fresh versioned output
location. No step deletes or overwrites historical v1.0 products by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "workflow_v1.3_official44_corrected.json"


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    description: str


STEPS = [
    Step(
        "full_rerun",
        ["scripts/stage12_hebei14_full_rerun.py", "--config", str(CONFIG)],
        "Leave-year validation, joint five-year training, annual maps and change maps.",
    ),
    Step(
        "leave_province",
        ["scripts/stage19_v3_leave_province_validation.py", "--config", str(CONFIG)],
        "Five-fold leave-province validation with the versioned v3 target.",
    ),
    Step(
        "ensemble_uncertainty",
        ["scripts/stage8_ensemble_uncertainty.py", "--config", str(CONFIG)],
        "Five-member ensemble uncertainty for 2019–2023.",
    ),
    Step(
        "stage_public_safe_products",
        ["scripts/stage20_stage_v3_release_candidate.py"],
        "Copy public-safe annual, change and aggregate validation products.",
    ),
    Step(
        "quality_layers",
        [
            "scripts/stage6_create_quality_layers.py",
            "--stage5-root", "outputs/official44_corrected_full_rerun_v3/stage5_final_maps",
            "--package-root", "outputs/official44_corrected_stage8_v3/package",
            "--target-path", "data/corrections/county_income_2019_2023_deflated_official44_corrected_v3.csv",
        ],
        "Generate quality/support layers from v3 maps and labels.",
    ),
    Step(
        "quality_risk_layers",
        [
            "scripts/stage7_uncertainty_proxy_layers.py",
            "--package-root", "outputs/official44_corrected_stage8_v3/package",
        ],
        "Generate v3 quality-risk / uncertainty-proxy layers.",
    ),
    Step(
        "refresh_manifest",
        ["scripts/stage20_stage_v3_release_candidate.py"],
        "Refresh the release-candidate manifest after derived layers are generated.",
    ),
    Step(
        "final_data_audit",
        ["scripts/stage21_audit_v3_release_candidate.py"],
        "Run the 19-item raster, hash, documentation and publication-boundary audit.",
    ),
]


def main() -> None:
    names = [step.name for step in STEPS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start", choices=names, default=names[0])
    parser.add_argument("--stop", choices=names, default=names[-1])
    args = parser.parse_args()
    start = names.index(args.start)
    stop = names.index(args.stop)
    if start > stop:
        raise ValueError("--start must not come after --stop")
    selected = STEPS[start:stop + 1]

    print("=" * 78)
    print("v1.3-official44-corrected pipeline")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Config: {CONFIG}")
    print(f"Steps: {selected[0].name} -> {selected[-1].name}")
    print("=" * 78)
    for index, step in enumerate(selected, 1):
        command = [sys.executable, *step.command]
        print(f"[{index}/{len(selected)}] {step.name}: {step.description}")
        print("    " + subprocess.list2cmdline(command))
        if args.execute:
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
