"""Regenerate CODE_FILE_MANIFEST.csv for the public repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CODE_FILE_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path != OUTPUT
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix()):
            writer.writerow(
                [path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)]
            )
    print(f"manifest={OUTPUT}")
    print(f"files={len(files)}")


if __name__ == "__main__":
    main()
