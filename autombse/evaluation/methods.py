from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple


_DEFAULT_EXCLUDED_KEYS = {
    "itemID",
    "code",
    "description",
    "stage",
}

_DEFAULT_EXCLUDED_SUFFIXES = (
    "_time",
    "_bleu",
    "_similarity",
    "_similarity_project",
)

_AUTO_TOKENS = {"*", "all", "auto"}


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_methods_config(value: Any) -> tuple[list[str], bool]:
    """
    Normalize `evaluation.methods` from config.

    Returns (methods, is_auto).

    Rules:
    - null / [] / "*" / "auto" / "all" => auto-discovery (is_auto=True)
    - string => single method (is_auto=False)
    - list => string members as methods (is_auto=False), unless it contains an auto token
    """

    if value is None:
        return [], True

    if isinstance(value, str):
        token = value.strip().lower()
        if token in _AUTO_TOKENS:
            return [], True
        return [value], False

    if isinstance(value, Sequence):
        methods: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()
            if token in _AUTO_TOKENS:
                return [], True
            methods.append(item)
        if not methods:
            return [], True
        return _unique_preserve_order(methods), False

    return [], True


def discover_methods(
    examples: Iterable[dict[str, Any]],
    *,
    exclude_keys: Optional[Set[str]] = None,
    exclude_suffixes: Optional[Tuple[str, ...]] = None,
) -> list[str]:
    """
    Discover baseline method ids from legacy `res.json` objects.

    Heuristic:
    - exclude known meta keys (code/description/itemID/stage)
    - exclude metric/time derived keys by suffix (e.g. *_time)
    - only keep keys whose values are non-empty strings (baseline outputs)
    """

    excluded = set(_DEFAULT_EXCLUDED_KEYS)
    if exclude_keys:
        excluded.update(exclude_keys)

    suffixes = exclude_suffixes if exclude_suffixes is not None else _DEFAULT_EXCLUDED_SUFFIXES

    methods: set[str] = set()
    for example in examples:
        if not isinstance(example, dict):
            continue
        for key, value in example.items():
            if not isinstance(key, str) or key in excluded:
                continue
            if any(key.endswith(suffix) for suffix in suffixes):
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            methods.add(key)

    return sorted(methods)


def filter_methods_present(methods: Sequence[str], examples: Sequence[dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Split methods into (present, missing) based on whether the key exists in any example.
    """

    present: list[str] = []
    missing: list[str] = []
    for method in methods:
        if any(isinstance(example, dict) and method in example for example in examples):
            present.append(method)
        else:
            missing.append(method)
    return present, missing


__all__ = ["discover_methods", "filter_methods_present", "normalize_methods_config"]
