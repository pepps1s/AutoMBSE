from __future__ import annotations

import sys
from pathlib import Path


def add_legacy_import_path(*, repo_root: Path, legacy_root: str) -> None:
    legacy_path = (repo_root / legacy_root).resolve()
    if legacy_path.is_dir() and legacy_path.as_posix() not in sys.path:
        sys.path.insert(0, legacy_path.as_posix())

