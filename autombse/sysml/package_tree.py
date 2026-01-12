from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable=None, *args, **kwargs):
        _ = args
        _ = kwargs
        return iterable if iterable is not None else []


def has_cycle(edges):
    graph = {}
    visited = {}
    for u_name, u_type, v_name, v_type in edges:
        node_u = (u_name, u_type)
        node_v = (v_name, v_type)
        graph.setdefault(node_u, []).append(node_v)
        visited[node_u] = 0
        visited[node_v] = 0

    stack = []
    cycle_path = []
    has_cycle_flag = False

    def dfs(node):
        nonlocal cycle_path
        if visited[node] == 1:
            if not cycle_path:
                idx = stack.index(node)
                cycle_path = stack[idx:] + [node]
            return True
        if visited[node] == 2:
            return False

        visited[node] = 1
        stack.append(node)
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
        stack.pop()
        visited[node] = 2
        return False

    for node in visited:
        if visited[node] == 0:
            if dfs(node):
                has_cycle_flag = True
                break
    return has_cycle_flag, cycle_path


class Package:
    def __init__(self, name, level, type, project_id=None, belongsto=None, redefined=False, legal=True):
        self.name = name
        self.level = level
        self.type = type
        self.project_id = project_id
        self.children = []
        self.redefined = redefined
        self.belongsto = belongsto
        self.legal = legal
        self.view_type = None

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return f"Package('{self.name}', Level: {self.level}, Type: {self.type}, Project: {self.project_id}, View: {self.view_type})"


def name_match(name_type, line):
    return re.match(
        fr"(?:\w*\s*|\s*)?{name_type}\s+(?:def\s+|:>>\s*|redefines\s+)?(\s*(?:[a-zA-Z_][a-zA-Z0-9_\.]*(?:\[[^\]]*\])*)|'[^']*'|\"[^\"]*\")?(?:\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*(?:(?:::|\.)[a-zA-Z_][a-zA-Z0-9_]*)*)\b)?",
        line,
    )


def parse_packages(content, package_dict):
    packages = []
    stack = []
    level = 0
    pending_satisfy = None
    in_block_comment = False

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue
        if "/*" in line:
            if "*/" not in line:
                in_block_comment = True
            continue
        if "//" in line:
            line = line.split("//", 1)[0].strip()
            if not line:
                continue
        open_brace = "{" in line
        close_brace = "}" in line

        if pending_satisfy:
            combined = (pending_satisfy + " " + line).strip()
            satisfy_satisfies_match = re.match(
                r"^satisfy\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s+satisfies\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s*;?\s*$",
                combined,
            )
            satisfy_by_match = re.match(
                r"^satisfy\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s+by\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s*;?\s*$",
                combined,
            )
            if satisfy_satisfies_match:
                design_ref = (satisfy_satisfies_match.group(1) or "").strip()
                req_ref = (satisfy_satisfies_match.group(2) or "").strip()
                node = Package(req_ref, level, "satisfy", belongsto=design_ref)
                if stack:
                    parent = stack[-1]
                    if parent != "None":
                        parent.add_child(node)
                else:
                    packages.append(node)
                pending_satisfy = None
                continue
            if satisfy_by_match:
                req_ref = (satisfy_by_match.group(1) or "").strip()
                impl_ref = (satisfy_by_match.group(2) or "").strip()
                node = Package(req_ref, level, "satisfy", belongsto=impl_ref)
                if stack:
                    parent = stack[-1]
                    if parent != "None":
                        parent.add_child(node)
                else:
                    packages.append(node)
                pending_satisfy = None
                continue
            if ";" in line:
                pending_satisfy = None
            continue

        import_match = name_match("import", line)
        package_match = name_match("package", line)
        requirement_match = name_match("requirement", line)
        attribute_match = name_match("attribute", line)
        part_match = name_match("part", line)
        port_match = name_match("port", line)
        action_match = name_match("action", line)
        block_match = name_match("block", line)
        # Support both `constraint {` and `constraint{` (the latter appears in our prompts/knowledge).
        constraint_match = name_match("constraint", re.sub(r"constraint\s*\{", "constraint {", line))
        connector_match = name_match("connector", line) or name_match("connection", line)
        signal_match = name_match("signal", line)
        redefined = "redefines" in line or ":>>" in line
        referred = re.search(r"\bref\b", line)
        satisfy_satisfies_match = re.match(
            r"^satisfy\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s+satisfies\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s*;?\s*$",
            line,
        )
        satisfy_by_match = re.match(
            r"^satisfy\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s+by\s+([a-zA-Z_][a-zA-Z0-9_:\.]*|'[^']*'|\"[^\"]*\")\s*;?\s*$",
            line,
        )
        legal = True

        if import_match or referred:
            continue
        if "defs" in line:
            legal = False

        def add_node(node: Package) -> Optional[Package]:
            if not stack:
                packages.append(node)
                return node
            parent = stack[-1]
            if parent == "None":
                return None
            parent.add_child(node)
            return node

        def push_node(node: Optional[Package]) -> None:
            if not open_brace:
                return
            if node is None:
                stack.append("None")
            else:
                stack.append(node)

        if package_match:
            raw_name = (package_match.group(1) or "").strip()
            package_name = raw_name or f"unnamed_package_{level}"
            package_dict.setdefault(
                package_name,
                Package(package_name, level, "package", redefined=redefined, legal=legal),
            )
            push_node(add_node(Package(package_name, level, "package", redefined=redefined, legal=legal)))
        elif requirement_match:
            raw_name = (requirement_match.group(1) or "").strip()
            req_name = raw_name or f"unnamed_requirement_{level}"
            req_type = (requirement_match.group(2) or None)
            push_node(add_node(Package(req_name, level, "requirement", belongsto=req_type, redefined=redefined, legal=legal)))
        elif attribute_match:
            raw_name = (attribute_match.group(1) or "").strip()
            attr_name = raw_name or f"unnamed_attribute_{level}"
            attr_type = (attribute_match.group(2) or None)
            push_node(add_node(Package(attr_name, level, "attribute", belongsto=attr_type, redefined=redefined, legal=legal)))
        elif part_match:
            raw_name = re.sub(r"\[[^\]]*\]", "", (part_match.group(1) or "")).strip()
            part_name = raw_name or f"unnamed_part_{level}"
            belongs_to = (part_match.group(2) or None)
            push_node(add_node(Package(part_name, level, "part", belongsto=belongs_to, redefined=redefined, legal=legal)))
        elif port_match:
            raw_name = re.sub(r"\[[^\]]*\]", "", (port_match.group(1) or "")).strip()
            port_name = raw_name or f"unnamed_port_{level}"
            belongs_to = (port_match.group(2) or None)
            push_node(add_node(Package(port_name, level, "port", belongsto=belongs_to, redefined=redefined, legal=legal)))
        elif action_match:
            raw_name = (action_match.group(1) or "").strip()
            action_name = raw_name or f"unnamed_action_{level}"
            belongs_to = (action_match.group(2) or None)
            push_node(add_node(Package(action_name, level, "action", belongsto=belongs_to, redefined=redefined, legal=legal)))
        elif block_match:
            raw_name = (block_match.group(1) or "").strip()
            block_name = raw_name or f"unnamed_block_{level}"
            belongs_to = (block_match.group(2) or None)
            push_node(add_node(Package(block_name, level, "block", belongsto=belongs_to, redefined=redefined, legal=legal)))
        elif constraint_match:
            raw_name = (constraint_match.group(1) or "").strip()
            constraint_name = raw_name or f"unnamed_constraint_{level}"
            belongs_to = (constraint_match.group(2) or None)
            push_node(
                add_node(
                    Package(
                        constraint_name,
                        level,
                        "constraint",
                        belongsto=belongs_to,
                        redefined=redefined,
                        legal=legal,
                    )
                )
            )
        elif satisfy_satisfies_match:
            design_ref = (satisfy_satisfies_match.group(1) or "").strip()
            req_ref = (satisfy_satisfies_match.group(2) or "").strip()
            add_node(Package(req_ref, level, "satisfy", belongsto=design_ref, redefined=redefined, legal=legal))
        elif satisfy_by_match:
            req_ref = (satisfy_by_match.group(1) or "").strip()
            impl_ref = (satisfy_by_match.group(2) or "").strip()
            add_node(Package(req_ref, level, "satisfy", belongsto=impl_ref, redefined=redefined, legal=legal))
        elif line.startswith("satisfy "):
            pending_satisfy = line
        elif connector_match:
            raw_name = (connector_match.group(1) or "").strip()
            conn_name = raw_name or f"unnamed_connector_{level}"
            belongs_to = (connector_match.group(2) or None)
            push_node(add_node(Package(conn_name, level, "connector", belongsto=belongs_to, redefined=redefined, legal=legal)))
        elif signal_match:
            raw_name = (signal_match.group(1) or "").strip()
            sig_name = raw_name or f"unnamed_signal_{level}"
            sig_type = (signal_match.group(2) or None)
            push_node(add_node(Package(sig_name, level, "signal", belongsto=sig_type, redefined=redefined, legal=legal)))

        if open_brace:
            level += 1
        if close_brace:
            if stack:
                stack.pop()
            level = max(0, level - 1)

    return packages


def process_sysml_content_to_tree(content, project_id, project_trees=None, package_dict=None):
    if project_trees is None:
        project_trees = {}
    if package_dict is None:
        package_dict = {}

    view_keywords = {
        "rd": ["requirement", "req"],
        "bdd": ["block", "part", "port", "service", "system"],
        "ibd": ["connection", "connector", "flow", "interface"],
        "ad": ["activity", "action", "flow", "control"],
        "smd": ["state", "transition", "event", "state_machine"],
        "pd": ["constraint", "parametric", "equation"],
    }

    def determine_view_type(content):
        content_lower = content.lower()
        for view_type, keywords in view_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return view_type
        return "bdd"

    def organize_by_view_type(packages):
        view_packages = {"rd": [], "bdd": [], "ibd": [], "ad": [], "smd": [], "pd": []}

        def collect_packages(pkg):
            if pkg.view_type in view_packages:
                view_packages[pkg.view_type].append(pkg)
            for child in pkg.children:
                collect_packages(child)

        for package in packages:
            collect_packages(package)

        return view_packages

    packages = parse_packages(content, package_dict)

    view_type = determine_view_type(content)

    for package in packages:
        package.project_id = project_id
        package.view_type = view_type

        def process_children(pkg):
            for child in pkg.children:
                child.project_id = project_id
                child.view_type = view_type
                process_children(child)

        process_children(package)

    if project_id not in project_trees:
        project_trees[project_id] = organize_by_view_type(packages)
    else:
        new_views = organize_by_view_type(packages)
        for view_type in project_trees[project_id]:
            project_trees[project_id][view_type].extend(new_views[view_type])

    return project_trees, package_dict


def package_to_dict(package):
    return {
        "name": package.name,
        "type": package.type,
        "level": package.level,
        "project_id": package.project_id,
        "view_type": package.view_type,
        "belongsto": package.belongsto,
        "redefined": package.redefined,
        "children": [package_to_dict(child) for child in package.children],
    }


def process_directory(directory_path, sysml_tree_path: str = "AutoMBSE/out/sysml_tree.json"):
    if os.path.exists(sysml_tree_path):
        print("Found existing sysml_tree.json; loading it directly...")
        with open(sysml_tree_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("sysml_tree.json not found; starting JSON processing...")

    key_trees: Dict[str, Dict[Any, Any]] = {}

    with open(directory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Processing objects in the JSON file...")
    for item in tqdm(data, desc="Processing objects"):
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

    print("Saving results to sysml_tree.json...")
    with open(sysml_tree_path, "w", encoding="utf-8") as f:
        json.dump(serializable_trees, f, ensure_ascii=False, indent=2)

    return key_trees


__all__ = [
    "Package",
    "has_cycle",
    "name_match",
    "package_to_dict",
    "parse_packages",
    "process_directory",
    "process_sysml_content_to_tree",
]
