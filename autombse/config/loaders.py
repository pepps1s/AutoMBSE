from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..utils.dicts import deep_merge


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"unsupported config extension: {path}")


def _default_params_path(repo_root: Path) -> Path:
    return repo_root / "AutoMBSE" / "resource" / "autombse.params.v1.yaml"


def _resolve_repo_relative(repo_root: Path, value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path)


def _params_path_candidates(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for raw in (os.environ.get("AUTOMBSE_PARAMS_FILE"), os.environ.get("AUTOMBSE_RESOURCE_PARAMS")):
        if raw:
            candidates.append(_resolve_repo_relative(repo_root, raw))
    candidates.append(_default_params_path(repo_root))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


@lru_cache(maxsize=16)
def _load_defaults_from_params(params_path: str) -> dict[str, Any]:
    path = Path(params_path)
    if not path.is_file():
        return {}
    loaded = _load_yaml(path)
    defaults = loaded.get("defaults")
    if isinstance(defaults, dict):
        return defaults

    # Backwards compatibility:
    # - previous format used `contracts.templates["autombse.*.v1.yaml"]`.
    contracts = loaded.get("contracts") or {}
    if isinstance(contracts, dict):
        templates = contracts.get("templates") or {}
        if isinstance(templates, dict):
            base: dict[str, Any] = {}
            for filename in (
                "autombse.llm.v1.yaml",
                "autombse.qdrant.v1.yaml",
                "autombse.pipeline.v1.yaml",
                "autombse.eval.v1.yaml",
            ):
                chunk = templates.get(filename)
                if isinstance(chunk, dict):
                    base = deep_merge(base, chunk)
            if base:
                return base

    # Last resort: treat the mapping (minus known metadata keys) as config.
    stripped = dict(loaded)
    for key in ("version", "template", "cli", "contracts"):
        stripped.pop(key, None)
    return stripped


def _default_config_for_command(repo_root: Path, command_path: tuple[str, ...]) -> dict[str, Any]:
    _ = command_path
    for params_path in _params_path_candidates(repo_root):
        defaults = _load_defaults_from_params(str(params_path))
        if defaults:
            return dict(defaults)
    return {}


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    llm = dict(config.get("llm") or {})

    if not llm.get("api_key"):
        llm["api_key"] = os.environ.get("OPENAI_API_KEY") or os.environ.get("AUTOMBSE_API_KEY")

    if os.environ.get("AUTOMBSE_BASE_URL"):
        llm["base_url"] = os.environ["AUTOMBSE_BASE_URL"]

    if llm:
        config = dict(config)
        config["llm"] = llm
    return config


def load_config_for_command(
    *,
    repo_root: Path,
    command_path: tuple[str, ...],
    config_path: Optional[str],
    legacy_layout: bool,
) -> dict[str, Any]:
    config: dict[str, Any] = _default_config_for_command(repo_root, command_path)

    if config_path:
        config = deep_merge(config, load_config_file(Path(config_path).expanduser()))

    if legacy_layout:
        paths = dict(config.get("paths") or {})
        if not paths.get("legacy_root") and (repo_root / "generationPipeline").is_dir():
            paths["legacy_root"] = "generationPipeline"
        config = dict(config)
        config["paths"] = paths

    config = _apply_env_overrides(config)
    from .templating import apply_resource_templating

    return apply_resource_templating(config=config, repo_root=repo_root)
