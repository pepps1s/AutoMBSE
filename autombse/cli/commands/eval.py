from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from ...evaluation.methods import discover_methods, filter_methods_present, normalize_methods_config
from ...evaluation.bert_embedding_cache import BertCLOSEmbeddingCache
from ...evaluation.view_metrics import (
    ElementPRF,
    PartElementExtractor,
    avg_elements_per_example,
    avg_semantic_similarity,
    bleu_score,
    element_prf_semantic,
    extract_sysml_fenced,
    part_elements_legacy_semicolon,
    part_elements_strict,
)


_REMOVED_METHOD_IDS = {
    "LLM_Case_Obj_code",
    "LLM_Case_code",
    "LLM_code",
    "LLM_wo_case",
    "LLM_wo_feedback",
    "LLM_wo_knowledge",
    "LLM_wo_memory",
}

_METHOD_ID_RENAMES = {
    "LLM_wo_cache": "MBSE_wo_cache",
    "LLM_wo_rules": "MBSE_wo_rules",
}

_METHOD_DERIVED_SUFFIXES = ("", "_time", "_bleu", "_similarity", "_similarity_project")


def _resolve_repo_relative(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path)


def _set_cache_dir(repo_root: Path, cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return
    os.environ["AUTOMBSE_CACHE_DIR"] = _resolve_repo_relative(repo_root, cache_dir).as_posix()


def _load_examples(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _maybe_rename_method_fields(examples: List[dict], *, old: str, new: str) -> bool:
    changed = False
    for ex in examples:
        if not isinstance(ex, dict):
            continue

        for suffix in _METHOD_DERIVED_SUFFIXES:
            old_key = f"{old}{suffix}"
            if old_key not in ex:
                continue
            new_key = f"{new}{suffix}"
            old_value = ex.get(old_key)
            new_value = ex.get(new_key)

            should_fill_new = new_key not in ex
            if not should_fill_new and isinstance(old_value, str):
                should_fill_new = not isinstance(new_value, str) or not new_value.strip()
            if not should_fill_new and isinstance(old_value, (int, float)):
                should_fill_new = not isinstance(new_value, (int, float))

            if should_fill_new:
                ex[new_key] = old_value

            del ex[old_key]
            changed = True

    return changed


def _normalize_eval_methods(methods: List[str]) -> List[str]:
    renamed_pairs: List[tuple[str, str]] = []
    removed: List[str] = []
    normalized: List[str] = []
    for method in methods:
        if not isinstance(method, str):
            continue
        m = method.strip()
        if not m:
            continue
        new_name = _METHOD_ID_RENAMES.get(m)
        if new_name is not None and new_name != m:
            renamed_pairs.append((m, new_name))
            m = new_name
        if m in _REMOVED_METHOD_IDS:
            removed.append(m)
            continue
        normalized.append(m)

    normalized = _unique_preserve_order(normalized)

    if renamed_pairs:
        renamed_text = ", ".join([f"{old}->{new}" for old, new in renamed_pairs])
        print(f"warning: method id renamed: {renamed_text}")
    if removed:
        print("warning: removed method(s) will be skipped:", ", ".join(_unique_preserve_order(removed)))

    return normalized


def _select_methods(*, args: Any, config: dict[str, Any], examples: List[dict], allow_missing: bool = False) -> List[str]:
    cli_methods = getattr(args, "method", None) or []
    cli_methods = [m for m in cli_methods if isinstance(m, str) and m.strip()]
    if cli_methods:
        selected = cli_methods
    else:
        raw_cfg = (config.get("evaluation") or {}).get("methods")
        cfg_methods, cfg_auto = normalize_methods_config(raw_cfg)
        selected = discover_methods(examples) if cfg_auto else cfg_methods

    selected = _normalize_eval_methods(list(selected))
    present, missing = filter_methods_present(selected, examples)
    if allow_missing:
        if missing:
            print("warning: method(s) not found in input (will attempt generation if supported):", ", ".join(missing))
        return selected

    if missing:
        print("warning: method(s) not found in input and will be skipped:", ", ".join(missing))
    return present


def _parse_element_extractor_overrides(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not k.strip():
                continue
            out[k.strip()] = str(v).strip()
        return out
    if isinstance(raw, list):
        pairs = raw
    elif isinstance(raw, str):
        pairs = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        pairs = []
    out: dict[str, str] = {}
    for item in pairs:
        if not isinstance(item, str) or "=" not in item:
            continue
        method, mode = item.split("=", 1)
        method = method.strip()
        mode = mode.strip()
        if method:
            out[method] = mode
    return out


def _resolve_element_extractor(
    *,
    method: str,
    args: Any,
    config: dict[str, Any],
) -> PartElementExtractor:
    eval_cfg = config.get("evaluation") or {}

    overrides: dict[str, str] = {}
    overrides.update(_parse_element_extractor_overrides(eval_cfg.get("element_extractor_by_method")))
    overrides.update(_parse_element_extractor_overrides(getattr(args, "element_extractor_by_method", None)))

    mode_raw = overrides.get(method)
    if mode_raw is None:
        mode_raw = getattr(args, "element_extractor", None)
    if mode_raw is None:
        mode_raw = eval_cfg.get("element_extractor")

    mode = str(mode_raw or "auto").strip().lower()
    if mode == "auto":
        # Heuristic: keep historical semicolon parsing for DS outputs to reproduce prior numbers.
        mode = "legacy_semicolon" if method.endswith("_ds") else "strict"

    if mode in {"legacy", "legacy_semicolon", "compat", "compat_semicolon"}:
        return part_elements_legacy_semicolon
    return part_elements_strict


def eval_views_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    input_arg = getattr(args, "input", None)
    input_path = (
        Path(input_arg).expanduser()
        if input_arg
        else (repo_root / (paths_cfg.get("views_res_json") or "AutoMBSE/out/views/res.json"))
    )

    cache_dir = getattr(args, "cache_dir", None) or paths_cfg.get("cache_dir")
    _set_cache_dir(repo_root, cache_dir)

    examples = _load_examples(input_path)

    renamed_fields = False
    for old, new in _METHOD_ID_RENAMES.items():
        renamed_fields = _maybe_rename_method_fields(examples, old=old, new=new) or renamed_fields

    threshold = getattr(args, "threshold", None)
    if threshold is None:
        threshold = ((config.get("evaluation") or {}).get("thresholds") or {}).get("view_level_similarity")
    threshold = float(threshold if threshold is not None else 0.9)

    bleu_weight = getattr(args, "bleu_mbse_weight", None)
    if bleu_weight is None:
        bleu_weight = ((config.get("evaluation") or {}).get("thresholds") or {}).get("bleu_mbse_weight")
    bleu_weight = float(0.2 if bleu_weight is None else bleu_weight)

    update = getattr(args, "update", None)
    if update is None:
        update = (config.get("evaluation") or {}).get("update_res_json")
    update = bool(True if update is None else update)

    force = bool(getattr(args, "force", False))
    generate_missing = bool(getattr(args, "generate_missing", False))
    regenerate = bool(getattr(args, "regenerate", False))
    max_generate_raw = getattr(args, "max_generate", None)
    max_generate = int(max_generate_raw) if isinstance(max_generate_raw, (int, float)) else None

    if (generate_missing or regenerate) and not update:
        raise SystemExit("generation requires --update (remove --no-update)")

    methods = _select_methods(args=args, config=config, examples=examples, allow_missing=(generate_missing or regenerate))
    if not methods:
        print("warning: no evaluatable methods found in input")
        return 2

    generated_methods: set[str] = set()

    if generate_missing or regenerate:
        from ...evaluation.resjson_generators import generate_method, resolve_llm_client_config

        llm_cfg = resolve_llm_client_config(config)
        supported = {"MBSE_wo_cache", "MBSE_wo_rules"}
        targets = [m for m in methods if m in supported]
        skipped = [m for m in methods if m not in supported]
        if skipped:
            print("warning: generation not supported for method(s):", ", ".join(skipped))

        generated = 0
        for method in targets:
            time_key = f"{method}_time"
            for idx, ex in enumerate(examples):
                if max_generate is not None and generated >= max_generate:
                    break
                if not isinstance(ex, dict):
                    continue
                desc = ex.get("description")
                if not isinstance(desc, str) or not desc.strip():
                    continue

                existing = ex.get(method)
                if not regenerate and isinstance(existing, str) and existing.strip():
                    continue

                try:
                    text, elapsed = generate_method(method, description=desc, cfg=llm_cfg)
                except Exception as e:
                    print(f"warning: generation failed for {method} item {idx}: {e}")
                    continue

                ex[method] = text
                ex[time_key] = float(elapsed)
                generated += 1
                generated_methods.add(method)
                _write_json_atomic(input_path, examples)

            if max_generate is not None and generated >= max_generate:
                break

    embed_cache = BertCLOSEmbeddingCache()

    def ensure_similarity_field(method: str, *, force_field: bool) -> bool:
        key = f"{method}_similarity"
        changed = False
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            if not force_field and isinstance(ex.get(key), (int, float)):
                continue
            ref = ex.get("code")
            pred_raw = ex.get(method)
            if not isinstance(ref, str) or not isinstance(pred_raw, str):
                continue
            pred = extract_sysml_fenced(pred_raw)
            if not pred.strip():
                ex[key] = 0.0
                changed = True
                continue
            embed_cache.ensure([ref, pred])
            ref_vec = embed_cache.vectors([ref])[0]
            pred_vec = embed_cache.vectors([pred])[0]
            ex[key] = float((ref_vec @ pred_vec).item())
            changed = True
        return changed

    def ensure_bleu_field(method: str, *, force_field: bool) -> bool:
        key = f"{method}_bleu"
        changed = False
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            if not force_field and isinstance(ex.get(key), (int, float)):
                continue
            ref = ex.get("code")
            pred_raw = ex.get(method)
            if not isinstance(ref, str) or not isinstance(pred_raw, str):
                continue
            pred = extract_sysml_fenced(pred_raw)
            ex[key] = float(bleu_score(pred, ref))
            changed = True
        return changed

    wrote = False
    for method in methods:
        force_field = force or (method in generated_methods)
        wrote = ensure_similarity_field(method, force_field=force_field) or wrote
        wrote = ensure_bleu_field(method, force_field=force_field) or wrote

    if update and (wrote or renamed_fields):
        _write_json_atomic(input_path, examples)

    def avg_nonzero(values: List[float]) -> float:
        nz = [v for v in values if v != 0.0]
        if not nz:
            return 0.0
        return float(sum(nz)) / float(len(nz))

    print("Method & Time(s) & #Elements & Recall(%) & Precision(%) & F1-score(%) & Semantic Similarity & BLEU & BLEU$_{mbse}$")
    for method in methods:
        element_extractor = _resolve_element_extractor(method=method, args=args, config=config)
        elements = avg_elements_per_example(examples, method=method, element_extractor=element_extractor)
        times: List[float] = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            v = ex.get(f"{method}_time")
            if isinstance(v, (int, float)):
                times.append(float(v))
        time_avg = avg_nonzero(times)
        prf: ElementPRF = element_prf_semantic(
            examples,
            method=method,
            threshold=threshold,
            embed_cache=embed_cache,
            element_extractor=element_extractor,
        )
        sem_sim = avg_semantic_similarity(examples, method=method, ignore_zero=True)
        bleu_vals: List[float] = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            v = ex.get(f"{method}_bleu")
            if isinstance(v, (int, float)):
                bleu_vals.append(float(v))
        bleu_avg = avg_nonzero(bleu_vals)
        bleu_mbse = (1.0 - bleu_weight) * bleu_avg + bleu_weight * sem_sim

        print(
            f"{method} & {time_avg:.2f} & {elements:.2f} & {prf.recall*100:.2f} & {prf.precision*100:.2f} & {prf.f1*100:.2f} & {sem_sim:.4f} & {bleu_avg:.4f} & {bleu_mbse:.4f} \\\\"
        )

    if embed_cache.stats.added > 0:
        embed_cache.save()

    return 0


def eval_projects_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    paths_cfg = config.get("paths") or {}
    input_arg = getattr(args, "input", None)
    input_path = (
        Path(input_arg).expanduser()
        if input_arg
        else (repo_root / (paths_cfg.get("views_res_json") or "AutoMBSE/out/views/res.json"))
    )

    cache_dir = getattr(args, "cache_dir", None) or paths_cfg.get("cache_dir")
    _set_cache_dir(repo_root, cache_dir)

    examples = _load_examples(input_path)

    for old, new in _METHOD_ID_RENAMES.items():
        _maybe_rename_method_fields(examples, old=old, new=new)

    threshold = getattr(args, "threshold", None)
    if threshold is None:
        threshold = ((config.get("evaluation") or {}).get("thresholds") or {}).get("project_level_similarity")
    threshold = float(threshold if threshold is not None else 0.955)

    use_qdrant = getattr(args, "use_qdrant", None)
    if use_qdrant is None:
        use_qdrant = (((config.get("evaluation") or {}).get("project_metric") or {}).get("use_qdrant"))
    use_qdrant = bool(True if use_qdrant is None else use_qdrant)

    created_flag = getattr(args, "created_flag", None)
    if created_flag is None:
        created_flag = (((config.get("evaluation") or {}).get("project_metric") or {}).get("created_flag"))
    created_flag = bool(created_flag) if created_flag is not None else False

    methods = _select_methods(args=args, config=config, examples=examples)
    if not methods:
        print("warning: no evaluatable methods found in input")
        return 2

    try:
        from ...evaluation.metrics_parts import exampleComparison
    except Exception as e:
        raise SystemExit("project-level metrics require optional deps (scipy, qdrant-client)") from e

    for method in methods:
        if use_qdrant:
            res = exampleComparison.partCompare_sematicSimilarity_project_qdrant(examples, method, threshold, created_flag)
        else:
            res = exampleComparison.partCompare_sematicSimilarity_project(examples, method)
        print(f"{method}:\t\t", res)

    return 0


__all__ = ["eval_views_cmd", "eval_projects_cmd"]
