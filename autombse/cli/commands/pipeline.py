from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

from ...pipeline.runner import Pipeline, ReplayConfig


def _resolve_repo_relative(repo_root: Path, value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path)


def _default_system_prompt() -> str:
    return (
        "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our "
        "Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed code on how to implement SysML v2 for "
        "creating robust MBSE models. "
    )


def _resolve_target_system(*, repo_root: Path, config: dict[str, Any]) -> str:
    cfg_template = config.get("template") or {}
    cfg_vars = (cfg_template.get("vars") or {}) if isinstance(cfg_template, dict) else {}
    if isinstance(cfg_vars, dict):
        val = cfg_vars.get("target_system")
        if isinstance(val, str) and val.strip():
            return val.strip()

    env_val = os.environ.get("AUTOMBSE_TARGET_SYSTEM")
    if env_val:
        return env_val.strip()

    # Fallback to params.yaml template.vars (defaults aren't merged into config).
    try:
        import yaml

        params_raw = os.environ.get("AUTOMBSE_PARAMS_FILE") or os.environ.get("AUTOMBSE_RESOURCE_PARAMS") or (
            repo_root / "AutoMBSE" / "resource" / "autombse.params.v1.yaml"
        )
        params_path = _resolve_repo_relative(repo_root, params_raw)
        if params_path.is_file():
            loaded = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                template = loaded.get("template") or {}
                if isinstance(template, dict):
                    vars_obj = template.get("vars") or {}
                    if isinstance(vars_obj, dict):
                        val = vars_obj.get("target_system")
                        if isinstance(val, str) and val.strip():
                            return val.strip()
    except Exception:
        pass

    return "target_system"


def pipeline_run_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    llm_cfg = config.get("llm") or {}
    api_key_arg = getattr(args, "api_key", None)
    api_key = api_key_arg if api_key_arg is not None else llm_cfg.get("api_key")

    base_url_arg = getattr(args, "base_url", None)
    base_url = base_url_arg if base_url_arg is not None else llm_cfg.get("base_url")

    model_arg = getattr(args, "model", None)
    model = model_arg if model_arg is not None else llm_cfg.get("model")

    system_prompt = llm_cfg.get("system_prompt")
    max_tokens = int(llm_cfg.get("max_tokens") or 1024 * 2)
    temperature = float(llm_cfg.get("temperature") or 0.2)
    system_prompt = system_prompt or _default_system_prompt()

    if not api_key:
        raise SystemExit("missing LLM api key: pass --api-key or set OPENAI_API_KEY/AUTOMBSE_API_KEY")

    pipeline_cfg = config.get("pipeline") or {}
    long_period = bool(getattr(args, "long_period", False))

    paths_cfg = config.get("paths") or {}
    out_dir_arg = getattr(args, "out_dir", None)
    out_dir_raw = out_dir_arg if out_dir_arg is not None else (paths_cfg.get("out_dir") or "AutoMBSE/out")
    out_dir = _resolve_repo_relative(repo_root, out_dir_raw)

    if long_period:
        from ...pipeline.long_period import run_long_period

        lp_cfg = pipeline_cfg.get("long_period") or {}
        rag_cfg = lp_cfg.get("rag") or {}
        safety_margin_tokens = int(lp_cfg.get("safety_margin_tokens") or 2000)
        max_context_tokens_arg = getattr(args, "max_context_tokens", None)
        max_context_tokens = int(max_context_tokens_arg or lp_cfg.get("max_context_tokens") or 180000)

        planner_cfg = lp_cfg.get("planner") or {}
        planner_target_items = int(planner_cfg.get("target_items") or 120)

        builder_cfg = lp_cfg.get("builder") or {}
        builder_max_retries = int(builder_cfg.get("max_retries") or 2)
        validate_rules = bool(builder_cfg.get("validate_rules") if builder_cfg.get("validate_rules") is not None else True)
        rule_type = str(builder_cfg.get("rule_type") or "all")
        cross_view_validation = str(builder_cfg.get("cross_view_validation") or "auto")

        sem_cfg = builder_cfg.get("semantic_entropy") or {}
        sem_samples_arg = getattr(args, "semantic_entropy_samples", None)
        sem_samples = sem_samples_arg if sem_samples_arg is not None else sem_cfg.get("samples")
        sem_temperature_arg = getattr(args, "semantic_entropy_temperature", None)
        sem_temperature = sem_temperature_arg if sem_temperature_arg is not None else sem_cfg.get("temperature")
        sem_threshold_arg = getattr(args, "semantic_entropy_threshold", None)
        sem_threshold = sem_threshold_arg if sem_threshold_arg is not None else sem_cfg.get("similarity_threshold")

        semantic_entropy_samples = int(sem_samples) if sem_samples is not None else 1
        semantic_entropy_temperature = float(sem_temperature) if sem_temperature is not None else 0.7
        semantic_entropy_similarity_threshold = float(sem_threshold) if sem_threshold is not None else 0.85

        run_id = getattr(args, "run_id", None)
        resume_arg = getattr(args, "resume", None)
        resume = bool(resume_arg) if resume_arg is not None else True

        max_diagrams_arg = getattr(args, "max_diagrams", None)
        max_diagrams_cfg = builder_cfg.get("max_diagrams")
        max_diagrams = max_diagrams_arg if max_diagrams_arg is not None else max_diagrams_cfg
        if isinstance(max_diagrams, str) and max_diagrams.strip().lower() == "null":
            max_diagrams = None
        if isinstance(max_diagrams, (int, float)):
            max_diagrams = int(max_diagrams)
        else:
            max_diagrams = None

        input_paths = [Path(p).expanduser() for p in (getattr(args, "input_paths", None) or [])]
        diagram_plan_raw = getattr(args, "diagram_plan", None)
        diagram_plan_path = Path(diagram_plan_raw).expanduser() if diagram_plan_raw else None

        qdrant_cfg = config.get("qdrant") or {}
        qdrant_host = qdrant_cfg.get("host") or "localhost"
        qdrant_port = int(qdrant_cfg.get("port") or 6333)
        qdrant_collections = qdrant_cfg.get("collections") or {}
        examples_collection = qdrant_collections.get("examples") or "examples_vec"
        knowledge_collection = qdrant_collections.get("knowledge") or "knowledge_chunk"

        knowledge_path: Union[Path, None] = None
        knowledge_raw = paths_cfg.get("knowledge_txt")
        if isinstance(knowledge_raw, (str, Path)) and str(knowledge_raw).strip():
            knowledge_path = _resolve_repo_relative(repo_root, knowledge_raw)

        rag_mode = str(rag_cfg.get("mode") or "auto").strip() or "auto"

        replay: Any = None
        record_path = getattr(args, "record", None)
        replay_path = getattr(args, "replay", None)
        if record_path and replay_path:
            raise SystemExit("use only one of --record/--replay")
        if record_path:
            replay = ReplayConfig(mode="record", path=Path(record_path).expanduser())
        if replay_path:
            replay = ReplayConfig(mode="replay", path=Path(replay_path).expanduser())

        target_system = _resolve_target_system(repo_root=repo_root, config=config)

        summary = run_long_period(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            out_dir=out_dir,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            examples_collection=examples_collection,
            knowledge_collection=knowledge_collection,
            replay=replay,
            target_system=target_system,
            input_paths=input_paths,
            knowledge_path=knowledge_path,
            rag_mode=rag_mode,
            diagram_plan_path=diagram_plan_path,
            run_id=run_id,
            resume=resume,
            max_diagrams=max_diagrams,
            max_context_tokens=max_context_tokens,
            safety_margin_tokens=safety_margin_tokens,
            planner_target_items=planner_target_items,
            builder_max_retries=builder_max_retries,
            validate_rules=validate_rules,
            rule_type=rule_type,
            cross_view_validation=cross_view_validation,
            semantic_entropy_samples=semantic_entropy_samples,
            semantic_entropy_temperature=semantic_entropy_temperature,
            semantic_entropy_similarity_threshold=semantic_entropy_similarity_threshold,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    max_context_bytes = int(pipeline_cfg.get("max_context_bytes") or 96 * 1024)

    qdrant_cfg = config.get("qdrant") or {}
    qdrant_host = qdrant_cfg.get("host") or "localhost"
    qdrant_port = int(qdrant_cfg.get("port") or 6333)
    qdrant_collections = qdrant_cfg.get("collections") or {}
    examples_collection = qdrant_collections.get("examples") or "examples_vec"
    knowledge_collection = qdrant_collections.get("knowledge") or "knowledge_chunk"

    replay: Any = None
    record_path = getattr(args, "record", None)
    replay_path = getattr(args, "replay", None)
    if record_path and replay_path:
        raise SystemExit("use only one of --record/--replay")
    if record_path:
        replay = ReplayConfig(mode="record", path=Path(record_path).expanduser())
    if replay_path:
        replay = ReplayConfig(mode="replay", path=Path(replay_path).expanduser())

    pipeline_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "max_context_bytes": max_context_bytes,
        "qdrant_host": qdrant_host,
        "qdrant_port": qdrant_port,
        "replay": replay,
    }
    if base_url is not None:
        pipeline_kwargs["base_url"] = base_url
    if model is not None:
        pipeline_kwargs["model"] = model

    pipeline = Pipeline(**pipeline_kwargs)
    pipeline.shortenHistory = bool(pipeline_cfg.get("shorten_history") or False)

    try:
        stages = (config.get("pipeline") or {}).get("stages")
        if isinstance(stages, list) and stages:
            res = []
            for stage in stages:
                stage_id = stage.get("id")
                task = stage.get("task")
                if not stage_id or not task:
                    continue
                examples_top_k = int(stage.get("examples_top_k") or 1)
                domain_top_k = int(stage.get("domain_top_k") or 2)
                context = pipeline.context_generate(
                    stage_id,
                    task,
                    examples_top_k=examples_top_k,
                    domain_top_k=domain_top_k,
                    examples_collection=examples_collection,
                    knowledge_collection=knowledge_collection,
                )
                res.append(pipeline.response(task=task, context=context))
        else:
            res = pipeline.pipeline()
    except Exception as e:
        raise SystemExit(f"pipeline failed: {e}") from e

    log_history_arg = getattr(args, "log_history", None)
    log_history_raw = (
        log_history_arg if log_history_arg is not None else (paths_cfg.get("log_history_md") or (out_dir / "log-history.md"))
    )
    log_code_arg = getattr(args, "log_code", None)
    log_code_raw = log_code_arg if log_code_arg is not None else (paths_cfg.get("log_code_md") or (out_dir / "log-code.md"))
    log_history = _resolve_repo_relative(repo_root, log_history_raw)
    log_code = _resolve_repo_relative(repo_root, log_code_raw)

    log_history.parent.mkdir(parents=True, exist_ok=True)
    log_code.parent.mkdir(parents=True, exist_ok=True)

    history = getattr(pipeline, "history", [])
    with log_history.open("w+", encoding="utf-8") as f:
        f.write("# History\n\n" + "\n".join(history))
        f.write("# answer--\n\n" + "\n".join(res))
    with log_code.open("w", encoding="utf-8") as f:
        f.write("# answer code\n\n" + "\n".join(getattr(pipeline, "resCode", [])))

    print(f"wrote: {log_history}")
    print(f"wrote: {log_code}")
    return 0
