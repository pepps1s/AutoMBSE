from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _schema_dir(repo_root: Path) -> Path:
    preferred = repo_root / "AutoMBSE" / "resource" / "contracts"
    if preferred.is_dir():
        return preferred

    legacy = repo_root / "docs" / "contracts"
    if legacy.is_dir():
        return legacy

    return preferred


def _load_schema(repo_root: Path, filename: str) -> Any:
    schema_path = _schema_dir(repo_root) / filename
    return _load_json(schema_path)


def _validate(instance: Any, schema: Any) -> tuple[list[str], int]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.json_path)
    rendered = [f"{e.json_path or '$'}: {e.message}" for e in errors]
    return rendered, len(rendered)


def _default_input(repo_root: Path, relpath: str) -> Path:
    return repo_root / relpath


def validate_res_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    default_input = paths_cfg.get("views_res_json") or "AutoMBSE/out/views/res.json"
    input_arg = getattr(args, "input", None)
    input_path = Path(input_arg).expanduser() if input_arg else _default_input(repo_root, default_input)
    schema = _load_schema(repo_root, "legacy.res.v1.schema.json")
    instance = _load_json(input_path)
    errors, total = _validate(instance, schema)
    if total:
        print(f"INVALID: {input_path}")
        max_errors_arg = getattr(args, "max_errors", None)
        if max_errors_arg is None:
            max_errors_arg = (config.get("validate") or {}).get("max_errors")
        max_errors = int(max_errors_arg if max_errors_arg is not None else 50)
        for err in errors[:max_errors]:
            print(f"- {err}")
        if max_errors and total > max_errors:
            print(f"... {total - max_errors} more")
        return 1
    print(f"OK: {input_path}")
    return 0


def validate_sysml_tree_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    default_input = paths_cfg.get("sysml_tree_json") or "AutoMBSE/out/sysml_tree.json"
    input_arg = getattr(args, "input", None)
    input_path = Path(input_arg).expanduser() if input_arg else _default_input(repo_root, default_input)
    schema = _load_schema(repo_root, "legacy.sysml_tree.v1.schema.json")
    instance = _load_json(input_path)
    errors, total = _validate(instance, schema)
    if total:
        print(f"INVALID: {input_path}")
        max_errors_arg = getattr(args, "max_errors", None)
        if max_errors_arg is None:
            max_errors_arg = (config.get("validate") or {}).get("max_errors")
        max_errors = int(max_errors_arg if max_errors_arg is not None else 50)
        for err in errors[:max_errors]:
            print(f"- {err}")
        if max_errors and total > max_errors:
            print(f"... {total - max_errors} more")
        return 1
    print(f"OK: {input_path}")
    return 0


def validate_rule_states_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    default_input = paths_cfg.get("rule_states_json") or "AutoMBSE/out/rule_states.json"
    input_arg = getattr(args, "input", None)
    input_path = Path(input_arg).expanduser() if input_arg else _default_input(repo_root, default_input)
    schema = _load_schema(repo_root, "legacy.rule_states.v1.schema.json")
    instance = _load_json(input_path)
    errors, total = _validate(instance, schema)
    if total:
        print(f"INVALID: {input_path}")
        max_errors_arg = getattr(args, "max_errors", None)
        if max_errors_arg is None:
            max_errors_arg = (config.get("validate") or {}).get("max_errors")
        max_errors = int(max_errors_arg if max_errors_arg is not None else 50)
        for err in errors[:max_errors]:
            print(f"- {err}")
        if max_errors and total > max_errors:
            print(f"... {total - max_errors} more")
        return 1
    print(f"OK: {input_path}")
    return 0


def validate_example_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    default_input = paths_cfg.get("examples_json") or "AutoMBSE/resource/examples/example.json"
    input_arg = getattr(args, "input", None)
    input_path = Path(input_arg).expanduser() if input_arg else _default_input(repo_root, default_input)
    schema = _load_schema(repo_root, "legacy.example.v1.schema.json")
    instance = _load_json(input_path)
    errors, total = _validate(instance, schema)
    if total:
        print(f"INVALID: {input_path}")
        max_errors_arg = getattr(args, "max_errors", None)
        if max_errors_arg is None:
            max_errors_arg = (config.get("validate") or {}).get("max_errors")
        max_errors = int(max_errors_arg if max_errors_arg is not None else 50)
        for err in errors[:max_errors]:
            print(f"- {err}")
        if max_errors and total > max_errors:
            print(f"... {total - max_errors} more")
        return 1
    print(f"OK: {input_path}")
    return 0
