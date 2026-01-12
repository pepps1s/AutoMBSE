from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..utils.dicts import deep_merge

_PROMPT_REF_PREFIX = "@prompt:"
_TEMPLATE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")


def _resource_dir(repo_root: Path) -> Path:
    return repo_root / "AutoMBSE" / "resource"


def _default_params_path(repo_root: Path) -> Path:
    return _resource_dir(repo_root) / "autombse.params.v1.yaml"


def _default_prompts_path(repo_root: Path) -> Path:
    return _resource_dir(repo_root) / "autombse.prompts.v1.yaml"


def _resolve_repo_relative(repo_root: Path, value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path)


def _load_mapping_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if isinstance(loaded, dict) else {}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    raise ValueError(f"unsupported mapping extension: {path}")


def _flatten_prompt_templates(node: Any, *, prefix: str, out: dict[str, str]) -> None:
    if prefix and isinstance(node, str):
        out[prefix] = node
        return

    if isinstance(node, dict):
        template = node.get("template")
        if prefix and isinstance(template, str):
            out[prefix] = template
            return

        for key, value in node.items():
            if not isinstance(key, str):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_prompt_templates(value, prefix=child_prefix, out=out)


def _load_prompt_library(repo_root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    resource_cfg = dict(config.get("resource") or {})
    prompts_raw = (
        resource_cfg.get("prompts_file")
        or os.environ.get("AUTOMBSE_PROMPTS_FILE")
        or os.environ.get("AUTOMBSE_RESOURCE_PROMPTS")
        or _default_prompts_path(repo_root)
    )
    prompts_path = _resolve_repo_relative(repo_root, prompts_raw)
    prompts_cfg = _load_mapping_file(prompts_path)
    prompts_root = prompts_cfg.get("prompts") or {}

    flattened: dict[str, str] = {}
    _flatten_prompt_templates(prompts_root, prefix="", out=flattened)
    return prompts_path, flattened


def _load_template_vars(repo_root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any], bool]:
    resource_cfg = dict(config.get("resource") or {})
    params_raw = (
        resource_cfg.get("params_file")
        or os.environ.get("AUTOMBSE_PARAMS_FILE")
        or os.environ.get("AUTOMBSE_RESOURCE_PARAMS")
        or _default_params_path(repo_root)
    )
    params_path = _resolve_repo_relative(repo_root, params_raw)
    params_cfg = _load_mapping_file(params_path)

    params_template = dict(params_cfg.get("template") or {})
    params_vars = dict(params_template.get("vars") or params_cfg.get("vars") or {})

    cfg_template = dict(config.get("template") or {})
    cfg_vars = dict(cfg_template.get("vars") or {})

    strict = bool(cfg_template.get("strict") or params_template.get("strict") or False)
    merged_vars = deep_merge(params_vars, cfg_vars)
    return params_path, merged_vars, strict


def _lookup_path(context: dict[str, Any], path: str) -> Any:
    cur: Any = context
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        raise KeyError(path)
    return cur


def _render_template_string(text: str, *, context: dict[str, Any], strict: bool) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            value = _lookup_path(context, key)
        except KeyError:
            if strict:
                raise
            return match.group(0)
        return "" if value is None else str(value)

    rendered = text
    for _ in range(3):
        updated = _TEMPLATE_VAR_RE.sub(repl, rendered)
        if updated == rendered:
            break
        rendered = updated
    return rendered


def _resolve_prompt_ref(text: str, *, prompts: dict[str, str], prompts_path: Path) -> str:
    stripped = text.strip()
    if not stripped.startswith(_PROMPT_REF_PREFIX):
        return text
    prompt_id = stripped[len(_PROMPT_REF_PREFIX) :].strip()
    if not prompt_id:
        raise ValueError(f"invalid prompt ref: {text!r}")
    if prompt_id not in prompts:
        known = ", ".join(sorted(prompts.keys())[:20])
        raise KeyError(f"unknown prompt id: {prompt_id} (from {prompts_path}; known: {known}...)")
    return prompts[prompt_id]


def _walk(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, fn) for v in obj]
    return fn(obj)


def apply_resource_templating(*, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """
    Apply two passes to config:
    1) Resolve @prompt:<id> references using the prompt library.
    2) Render {{vars}} placeholders using template vars.
    """

    enabled = bool((config.get("template") or {}).get("enabled", True))
    if not enabled:
        return config

    prompts_path, prompts = _load_prompt_library(repo_root, config)
    params_path, template_vars, strict = _load_template_vars(repo_root, config)

    context: Dict[str, Any] = {"vars": template_vars, **template_vars}

    def resolve_and_render(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        resolved = _resolve_prompt_ref(value, prompts=prompts, prompts_path=prompts_path)
        try:
            return _render_template_string(resolved, context=context, strict=strict)
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"missing template var: {missing} (params: {params_path})") from e

    return _walk(config, resolve_and_render)


__all__ = ["apply_resource_templating"]

