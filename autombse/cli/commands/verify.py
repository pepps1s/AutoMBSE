from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from ...sysml.package_tree import package_to_dict, process_sysml_content_to_tree
from ...verification.engine import Rules


def _resolve_repo_relative(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path)


def verify_tree_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    input_arg = getattr(args, "input", None)
    input_path = (
        Path(input_arg).expanduser()
        if input_arg
        else (repo_root / (paths_cfg.get("views_res_json") or "AutoMBSE/out/views/res.json"))
    )
    output_arg = getattr(args, "output", None)
    output_raw = output_arg or paths_cfg.get("sysml_tree_json") or "AutoMBSE/out/sysml_tree.json"
    output_path = _resolve_repo_relative(repo_root, output_raw)

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            json.load(f)
        print(f"exists: {output_path}")
        return 0

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    key_trees: Dict[str, Dict[Any, Any]] = {}

    for item in data:
        project_id = item.get("project_id", 0)

        for key, value in item.items():
            if not isinstance(value, str) or key.endswith("_time") or key == "description":
                continue

            current_key = key
            if current_key not in key_trees:
                key_trees[current_key] = {}

            project_trees, package_dict = process_sysml_content_to_tree(value, project_id, key_trees[current_key], {})
            _ = package_dict
            key_trees[current_key] = project_trees

    serializable_trees: Dict[str, Dict[Any, Any]] = {}
    for key, project_trees in key_trees.items():
        serializable_trees[key] = {}
        for project_id, views in project_trees.items():
            serializable_trees[key][project_id] = {}
            for view_type, packages in views.items():
                serializable_trees[key][project_id][view_type] = [package_to_dict(package) for package in packages]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable_trees, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output_path}")
    return 0


def verify_rules_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    state_file_arg = getattr(args, "state_file", None)
    state_file = state_file_arg or paths_cfg.get("rule_states_json")
    if state_file:
        os.environ["AUTOMBSE_RULE_STATE_FILE"] = _resolve_repo_relative(repo_root, state_file).as_posix()

    sysml_arg = getattr(args, "sysml", None)
    tree_arg = getattr(args, "tree", None)
    if bool(sysml_arg) == bool(tree_arg):
        raise SystemExit("provide exactly one of --sysml/--tree")

    rule_type_arg = getattr(args, "rule_type", None)
    rules_cfg = ((config.get("verify") or {}).get("rules") or {})
    rule_type = rule_type_arg or rules_cfg.get("rule_type") or "all"

    if sysml_arg:
        from ...sysml.package_tree import parse_packages

        sysml_path = Path(sysml_arg).expanduser()
        content = sysml_path.read_text(encoding="utf-8")
        package_dict: dict = {}
        packages = parse_packages(content, package_dict)
        rules = Rules(packages)
    else:
        tree_path = Path(tree_arg).expanduser()
        tree_obj = json.loads(tree_path.read_text(encoding="utf-8"))
        if isinstance(tree_obj, dict) and {"name", "type", "children"}.issubset(tree_obj.keys()):
            tree_obj = [tree_obj]
        rules = Rules(tree_obj)

    errors, warnings = rules.validate_by_type(rule_type)
    print(json.dumps({"errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
    return 0


__all__ = ["verify_tree_cmd", "verify_rules_cmd"]
