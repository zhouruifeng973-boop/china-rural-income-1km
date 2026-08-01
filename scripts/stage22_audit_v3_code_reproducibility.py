"""Audit the v3 reproducibility entry, configuration and key scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage22_v3_code_reproducibility_audit"
CONFIG = ROOT / "config" / "workflow_v1.3_official44_corrected.json"
TARGET = ROOT / "data" / "corrections" / "county_income_2019_2023_deflated_official44_corrected_v3.csv"
AUDIT = ROOT / "outputs" / "stage21_v3_release_candidate_audit" / "audit.json"
PIPELINE = ROOT / "scripts" / "run_v3_final_pipeline.py"
SCRIPTS = [
    "stage12_hebei14_full_rerun.py",
    "stage19_v3_leave_province_validation.py",
    "stage8_ensemble_uncertainty.py",
    "stage20_stage_v3_release_candidate.py",
    "stage6_create_quality_layers.py",
    "stage7_uncertainty_proxy_layers.py",
    "stage21_audit_v3_release_candidate.py",
    "run_v3_final_pipeline.py",
]
REQUIRED_REQUIREMENTS = {
    "numpy", "pandas", "rasterio", "scikit-learn", "torch", "pyyaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    check("config_is_json_and_readable", True, str(CONFIG.relative_to(ROOT)))
    configured_target = ROOT / config["paths"]["corrected_county_income"]
    check("configured_target_exists", configured_target.is_file(), str(configured_target))
    check("configured_target_is_v3", configured_target.resolve() == TARGET.resolve(), str(configured_target))
    expected_hash = "f0fc49f981f581d477ead02d2e72a1d99658cec9845d6dfcea1ecce6bb0d3d44"
    check("target_hash_frozen", TARGET.is_file() and sha256(TARGET) == expected_hash, sha256(TARGET) if TARGET.is_file() else "missing")

    compile_errors = []
    for name in SCRIPTS:
        try:
            py_compile.compile(str(ROOT / "scripts" / name), doraise=True)
        except Exception as exc:
            compile_errors.append(f"{name}: {exc}")
    check("key_scripts_compile", not compile_errors, repr(compile_errors))

    dry_run = subprocess.run(
        [sys.executable, str(PIPELINE)], cwd=ROOT, capture_output=True, text=True
    )
    check("pipeline_dry_run_succeeds", dry_run.returncode == 0, dry_run.stderr.strip())
    expected_steps = [
        "full_rerun", "leave_province", "ensemble_uncertainty",
        "stage_public_safe_products", "quality_layers", "quality_risk_layers",
        "refresh_manifest", "final_data_audit",
    ]
    check("pipeline_lists_all_steps", all(step in dry_run.stdout for step in expected_steps), dry_run.stdout)

    requirement_lines = {
        line.strip().lower() for line in
        (ROOT / "requirements_reproducibility.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_requirements = sorted(REQUIRED_REQUIREMENTS - requirement_lines)
    check("core_dependencies_declared", not missing_requirements, repr(missing_requirements))

    data_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    check("v3_data_audit_passed", data_audit.get("ready_for_publication") is True, str(AUDIT.relative_to(ROOT)))

    absolute_path_hits = []
    for name in SCRIPTS:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        if "C:\\Users\\" in text:
            absolute_path_hits.append(name)
    check("no_user_specific_absolute_paths", not absolute_path_hits, repr(absolute_path_hits))

    result = {
        "version": config["project"]["version"],
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "ready": all(item["passed"] for item in checks),
        "checks": checks,
    }
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item for item in checks if not item["passed"]]
    lines = [
        "# v3 代码复现性终审",
        "",
        f"- 通过：{result['checks_passed']}/{result['checks_total']}。",
        f"- 可复现入口就绪：{'是' if result['ready'] else '否'}。",
        "",
        "## 未通过项",
        "",
    ]
    lines.extend(f"- {item['check']}：{item['detail']}" for item in failed)
    if not failed:
        lines.append("- 无。")
    lines.extend([
        "",
        "默认入口仅打印命令；只有显式传入 `--execute` 才会运行。历史输出默认不会被删除或覆盖。",
    ])
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
