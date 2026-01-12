from __future__ import annotations

from typing import Any


def stub_cmd(*, args: Any, config: dict[str, Any], repo_root: Any) -> int:
    _ = args
    _ = config
    _ = repo_root
    raise SystemExit("not implemented (stage 1): command stub")

