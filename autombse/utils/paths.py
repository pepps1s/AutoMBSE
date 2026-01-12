from __future__ import annotations

from pathlib import Path


def find_repo_root(start_dir: Path) -> Path:
    start_dir = start_dir.resolve()
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "AutoMBSE" / "pyproject.toml").is_file():
            return candidate
    return start_dir
