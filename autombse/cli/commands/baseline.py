from __future__ import annotations

import json
import platform
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineEntry:
    path: str
    size_bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_default_baseline_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = [
        repo_root / "AutoMBSE/out/views/res.json",
        repo_root / "AutoMBSE/out/sysml_tree.json",
        repo_root / "AutoMBSE/out/rule_states.json",
    ]
    paths.extend(sorted((repo_root / "AutoMBSE/cache").glob("*_similarity*.json")))
    return paths


def baseline_snapshot(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    _ = config
    default_output = repo_root / "AutoMBSE/artifacts/baseline_manifest.json"
    output_path = Path(args.output).expanduser() if getattr(args, "output", None) else default_output

    baseline_paths = _collect_default_baseline_paths(repo_root)

    entries: list[BaselineEntry] = []
    missing: list[str] = []
    for path in baseline_paths:
        if not path.exists():
            missing.append(path.as_posix())
            continue
        entries.append(
            BaselineEntry(
                path=path.relative_to(repo_root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": repo_root.as_posix(),
        "entries": [asdict(e) for e in entries],
        "missing": missing,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if missing:
        print("baseline snapshot created with missing files:")
        for m in missing:
            print(f"- {m}")
        return 1

    print(f"baseline snapshot written: {output_path}")
    return 0
