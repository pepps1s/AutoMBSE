from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .runner import ReplayConfig


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize_id(raw: str) -> str:
    cleaned = _SAFE_ID_RE.sub("_", raw).strip("._-")
    return cleaned or "diagram"


def _read_text_file(path: Path, *, max_bytes: int = 512 * 1024) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _collect_input_text(paths: Sequence[Path]) -> Tuple[str, list[str]]:
    chunks: list[str] = []
    used: list[str] = []
    for raw in paths:
        path = raw.expanduser()
        if not path.exists():
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if not child.is_file():
                    continue
                if child.name.startswith("."):
                    continue
                used.append(str(child))
                chunks.append(f"\n\n# {child}\n\n{_read_text_file(child)}")
        else:
            used.append(str(path))
            chunks.append(f"\n\n# {path}\n\n{_read_text_file(path)}")
    return "\n".join(chunks).strip(), used


def _extract_json_blob(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if 0 <= start_arr < end_arr:
        return text[start_arr : end_arr + 1].strip()
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if 0 <= start_obj < end_obj:
        return text[start_obj : end_obj + 1].strip()
    return text.strip()


def extract_sysml_code(text: str) -> str:
    pattern_sysml = r"```sysml(.*?)```"
    pattern_general = r"```(.*?)```"
    matches_sysml = re.findall(pattern_sysml, text, re.DOTALL | re.IGNORECASE)
    if matches_sysml:
        return matches_sysml[-1].strip()
    matches_general = re.findall(pattern_general, text, re.DOTALL)
    if matches_general:
        return matches_general[-1].strip()
    return text.strip()


def _normalize_rag_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in {"", "auto"}:
        return "auto"
    if mode in {"qdrant", "qdrant_only", "vector", "vec"}:
        return "qdrant_only"
    if mode in {"local", "local_only", "file", "file_only", "degraded", "degrade", "fallback"}:
        return "local_only"
    return mode


def _normalize_rule_type(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"bdd", "cross", "all"}:
        return value
    return "all"


def _normalize_cross_view_validation(raw: str, *, rule_type: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"", "auto"}:
        return "strict" if rule_type == "cross" else "warn"
    if value in {"off", "false", "0", "none", "disable", "disabled"}:
        return "off"
    if value in {"warn", "warning", "log"}:
        return "warn"
    if value in {"strict", "error", "fail"}:
        return "strict"
    return "warn"


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


def _find_repo_root_for_resources(start: Path) -> Optional[Path]:
    for parent in [start] + list(start.parents):
        candidate = parent / "AutoMBSE" / "resource" / "autombse.prompts.v1.yaml"
        if candidate.is_file():
            return parent
    return None


@lru_cache(maxsize=1)
def _load_prompt_library() -> dict[str, str]:
    if yaml is None:
        return {}

    raw_path = os.environ.get("AUTOMBSE_PROMPTS_FILE") or os.environ.get("AUTOMBSE_RESOURCE_PROMPTS")
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            repo_root = _find_repo_root_for_resources(Path.cwd()) or _find_repo_root_for_resources(Path(__file__).resolve())
            path = (repo_root or Path.cwd()) / path
    else:
        repo_root = _find_repo_root_for_resources(Path(__file__).resolve()) or _find_repo_root_for_resources(Path.cwd())
        if repo_root is None:
            return {}
        path = repo_root / "AutoMBSE" / "resource" / "autombse.prompts.v1.yaml"

    if not path.is_file():
        return {}

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompts_root = loaded.get("prompts") if isinstance(loaded, dict) else {}
    flattened: dict[str, str] = {}
    _flatten_prompt_templates(prompts_root, prefix="", out=flattened)
    return flattened


def _prompt_template(prompt_id: str, *, default: str = "") -> str:
    lib = _load_prompt_library()
    template = lib.get(prompt_id)
    if isinstance(template, str) and template.strip():
        return template.strip()
    return default.strip()


_RULE_FIX_HINTS: dict[str, str] = {
    "RD-1": "Provide an explicit design realization for each requirement constraint: represent it via attributes/actions/behaviors, and explicitly link it using `satisfy` where possible.",
    "RD-2": "Provide an implementation for each requirement: create corresponding blocks/parts/behaviors in BDD/IBD/AD and add `satisfy` relationships; do not merely rename things without implementation.",
    "BBD-1": "Concrete attributes that appear in the BDD must be instantiated in the IBD: create corresponding part instances or reflect these attributes in the instance context.",
    "IBD-1": "All connectors must be named: avoid `unnamed_connector_*`, and ensure connected ports/parts are defined in the BDD.",
    "IBD-2": "Signal types used in the IBD must be defined in the BDD with exactly matching names; do not invent signal names ad hoc.",
    "IBD-3": "All ports must be named: avoid `unnamed_port_*`, and ensure the implementation block for each port exists in the BDD.",
    "AD-1": "Actions/parts in the activity diagram must be implemented in the BDD (e.g., as block behaviors/operations) and reuse the same identifiers.",
    "AD-2": "Guards/conditions in the activity diagram should be reflected in requirements/constraints; if conditional branches exist, add the corresponding requirement/constraint and link it.",
    "PD-1": "Parametric diagram constraints must reference attributes that actually exist on design blocks with exactly matching names; ensure constraint expressions are parsable.",
}


def _truncate(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _stringify_validation_detail(detail: Any) -> str:
    if isinstance(detail, dict):
        desc = str(detail.get("description") or "").strip()
        loc = str(detail.get("location") or "").strip()
        if desc and loc:
            return f"{desc} ({loc})"
        if desc:
            return desc
        if loc:
            return loc
        return json.dumps(detail, ensure_ascii=False)
    if isinstance(detail, (list, tuple)):
        return ", ".join([_stringify_validation_detail(d) for d in detail[:4]])
    return str(detail)


def _summarize_validation_errors(errors: Sequence[Any], *, max_rules: int = 8, max_detail_items: int = 3) -> list[str]:
    lines: list[str] = []
    for err in list(errors)[:max_rules]:
        if isinstance(err, dict):
            rule_id = str(err.get("rule_id") or "").strip()
            desc = str(err.get("description") or "").strip()
            details = err.get("errors")
            detail_items: list[str] = []
            if isinstance(details, list):
                for d in details[:max_detail_items]:
                    s = _stringify_validation_detail(d).strip()
                    if s:
                        detail_items.append(s)
            elif details is not None:
                s = _stringify_validation_detail(details).strip()
                if s:
                    detail_items.append(s)

            head = " ".join([p for p in [rule_id, desc] if p]).strip()
            if not head:
                head = _stringify_validation_detail(err).strip()
            if detail_items:
                head += " | examples: " + "; ".join([_truncate(s, max_chars=240) for s in detail_items])
            lines.append(f"- {head}")
        else:
            s = str(err).strip()
            if s:
                lines.append(f"- {_truncate(s, max_chars=360)}")
    return lines


def _requirements_from_errors(errors: Sequence[Any], *, max_items: int = 10) -> list[str]:
    requirements: list[str] = []
    seen: set[str] = set()
    for err in errors:
        rule_id = ""
        desc = ""
        if isinstance(err, dict):
            rule_id = str(err.get("rule_id") or "").strip().upper()
            desc = str(err.get("description") or "").strip()
        key = rule_id or desc or str(err)
        if not key or key in seen:
            continue
        seen.add(key)

        if rule_id == "BBD-1" and isinstance(err, dict):
            detail = err.get("errors")
            owners: list[str] = []
            attrs: list[str] = []
            if isinstance(detail, list):
                for item in detail:
                    if not isinstance(item, dict):
                        continue
                    loc = str(item.get("location") or "").strip()
                    m = re.search(r"package\\s*([A-Za-z0-9_]+)", loc, flags=re.IGNORECASE)
                    if m:
                        owners.append(m.group(1))
                    d = str(item.get("description") or "").strip()
                    m2 = re.search(r"attribute\\s*'([^']+)'", d, flags=re.IGNORECASE)
                    if m2:
                        attrs.append(m2.group(1))
            owners = [o for o in dict.fromkeys(owners) if o]
            attrs = [a for a in dict.fromkeys(attrs) if a]
            if owners:
                container_like: list[str] = []
                likely_types: list[str] = []
                for owner in owners:
                    lowered = owner.strip().lower()
                    if lowered.startswith(("ibd", "bdd", "rd", "ad", "pd")) and "_" in lowered:
                        container_like.append(owner)
                    else:
                        likely_types.append(owner)

                owners_text = ", ".join((likely_types or owners)[:6])
                attrs_text = ", ".join(attrs[:8])
                if likely_types:
                    hint = (
                        f"- [BBD-1] You must create type instances (not `part def`): for each of {owners_text}, create at least one `part <inst> : <Type>;`, "
                        "and use the instance name in connectors/references (e.g., `<inst>.port`); do not connect directly using the type name. "
                        f"{(' Involved attributes: ' + attrs_text + '.') if attrs_text else ''}"
                        "Type names must match exactly the definitions in dependency views (case-sensitive); do not rename arbitrarily or invent new types."
                    )
                    requirements.append(hint)
                if container_like:
                    bad = ", ".join(container_like[:4])
                    requirements.append(
                        f"- [BBD-1] If the owner in the error location looks like an untyped container/local part (e.g., {bad}), "
                        "do not define `attribute` directly under an untyped container; instead: assign an existing type (from dependency BDD) to the container `part`, "
                        "or move the attribute onto an already-instantiated typed part to avoid triggering BBD-1."
                    )
                if len(requirements) >= max_items:
                    break
                continue

        hint = _RULE_FIX_HINTS.get(rule_id) if rule_id else None
        if hint:
            requirements.append(f"- [{rule_id}] {hint}")
        elif rule_id and desc:
            requirements.append(f"- [{rule_id}] Satisfy rule: {desc}")
        elif desc:
            requirements.append(f"- Satisfy validation requirement: {desc}")
        else:
            requirements.append(f"- Fix issue: {_truncate(str(err), max_chars=360)}")
        if len(requirements) >= max_items:
            break
    return requirements


def _retry_previous_sysml_prompt(*, previous_sysml: str) -> str:
    code = (previous_sysml or "").strip()
    if not code:
        return ""
    guidance = _prompt_template(
        "pipeline.long_period.retry_previous_sysml",
        default=(
            "Below is the SysML you generated in the previous attempt (use it as the editing baseline). Make the minimal changes needed to fix errors; do not regenerate from scratch, and do not change names of elements that already appear in dependency views.\n"
            "Your final output must still contain only a single ```sysml code block```.\n"
        ),
    )
    truncated = _truncate(code, max_chars=14000)
    lines = []
    if guidance:
        lines.append(guidance.strip())
    lines.append("```sysml")
    lines.append(truncated)
    lines.append("```")
    return "\n".join(lines).strip()


def _retry_failure_history_prompt(
    *,
    diagram_id: str,
    next_attempt: int,
    history: Sequence[dict[str, Any]],
) -> str:
    if not history or next_attempt <= 1:
        return ""

    guidance = _prompt_template(
        "pipeline.long_period.retry_guidance",
        default=(
            "You are retrying generation for the same diagram (not a new diagram).\n"
            "You must fix, in order, the historical failure reasons listed below and satisfy each corresponding requirement; do not only fix the last failure.\n"
            "Make the minimal changes on top of the 'Previous Attempt SysML': keep correct structure and only fix what caused validation to fail.\n"
            "Cross-view consistency: any element/name/type that already appeared in dependency views must be reused exactly (case-sensitive); do not rename arbitrarily or redefine duplicates.\n"
            "Output requirement: output only a single ```sysml code block```, with no explanatory text.\n"
        ),
    )

    lines: list[str] = []
    retry_no = max(0, int(next_attempt) - 1)
    lines.append(f"=== Retry #{retry_no} for diagram_id={diagram_id} (attempt {next_attempt}) ===")
    if guidance:
        lines.append(guidance.strip())
    lines.append("=== Failure history and required fixes (chronological) ===")
    for entry in history:
        attempt = int(entry.get("attempt") or 0) if isinstance(entry, dict) else 0
        title = f"[Attempt {attempt}]"
        lines.append(title)
        cross_errors = entry.get("cross_errors") or [] if isinstance(entry, dict) else []
        errors = entry.get("errors") or [] if isinstance(entry, dict) else []
        warnings = entry.get("warnings") or [] if isinstance(entry, dict) else []
        scope = entry.get("cross_scope") or [] if isinstance(entry, dict) else []

        lines.append("Cause:")
        if scope:
            lines.append(f"- Cross-view check scope: {', '.join([str(s) for s in scope[:40]])}")
        if cross_errors:
            lines.append("- Cross-view consistency validation failed (enforced cross rules):")
            lines.extend(_summarize_validation_errors(cross_errors))
        if errors:
            lines.append("- Validation failed (all errors):")
            lines.extend(_summarize_validation_errors(errors))
        if warnings:
            lines.append(f"- Warnings: {_truncate(json.dumps(warnings, ensure_ascii=False), max_chars=900)}")

        lines.append("Requirements:")
        req_lines = _requirements_from_errors(cross_errors) if cross_errors else []
        if not req_lines:
            req_lines = _requirements_from_errors(errors)
        if req_lines:
            lines.extend(req_lines)
        lines.append(
            "- Re-validate after fixes: do not introduce new rule failures; add `satisfy` relationships / connectors / port naming, etc. if needed."
        )
        lines.append("")
    return "\n".join(lines).strip()


def _prepare_local_knowledge_for_prompt(text: str) -> str:
    if not text:
        return ""
    # Drop front-matter/notes before the first markdown section heading to avoid polluting generation prompts.
    match = re.search(r"^##\\s+", text, flags=re.MULTILINE)
    cleaned = text[match.start() :] if match else text
    return cleaned.strip()


def _status_counts_for_plan(state: "RunState") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in state.plan:
        status = str(state.status_by_id.get(item.diagram_id) or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


class TokenCounter:
    def __init__(self, *, model: str) -> None:
        self.model = model
        self._encoder = None
        try:  # optional dependency
            import tiktoken  # type: ignore

            try:
                self._encoder = tiktoken.encoding_for_model(model)
            except Exception:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Conservative fallback.
        return max(1, int(len(text) / 4))

    def estimate_messages(self, messages: Sequence[dict[str, Any]]) -> int:
        # Rough approximation for chat messages.
        total = 0
        for msg in messages:
            total += 4
            total += self.estimate_text(str(msg.get("role") or ""))
            total += self.estimate_text(str(msg.get("content") or ""))
        return total + 2


_CONTEXT_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", flags=re.MULTILINE)


def _estimate_component_breakdown(context: str, token_counter: TokenCounter) -> list[dict[str, Any]]:
    text = (context or "").strip()
    if not text:
        return []

    matches = list(_CONTEXT_SECTION_RE.finditer(text))
    if not matches:
        return [{"name": "__context__", "tokens_est": token_counter.estimate_text(text)}]

    items: list[dict[str, Any]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            items.append({"name": "__preamble__", "tokens_est": token_counter.estimate_text(preamble)})

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        name = (match.group(1) or "").strip() or f"section_{idx + 1}"
        items.append({"name": name, "tokens_est": token_counter.estimate_text(chunk)})
    return items


class EventLogger:
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        token: Optional[dict[str, Any]] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> str:
        span = span_id or uuid.uuid4().hex
        obj: dict[str, Any] = {
            "ts": _utc_ts(),
            "event_type": event_type,
            "span_id": span,
            "parent_span_id": parent_span_id,
            "payload": payload,
        }
        if token is not None:
            obj["token"] = token
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return span


class TokenLedger:
    REQUIRED_COLUMNS = [
        "ts",
        "event_type",
        "span_id",
        "budget_total",
        "prompt_est",
        "prompt_actual",
        "completion_actual",
        "total_actual",
        "delta_vs_prev",
        # Prompt breakdown (estimated, per prompt section).
        "prompt_est_system",
        "prompt_est_task",
        "prompt_est_components",
        "prompt_est_parts_total",
        "components_breakdown",
    ]

    def __init__(self, *, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] = []
        self._ensure_header()

    def _read_header(self) -> list[str]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size <= 0:
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return [str(c).strip() for c in header if str(c).strip()]

    def _rewrite_with_fieldnames(self, fieldnames: list[str]) -> None:
        rows: list[dict[str, Any]] = []
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        tmp_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        tmp_path.replace(self.csv_path)

    def _ensure_header(self) -> None:
        existing = self._read_header()
        if not existing:
            self._fieldnames = list(self.REQUIRED_COLUMNS)
            self._rewrite_with_fieldnames(self._fieldnames)
            return

        missing = [c for c in self.REQUIRED_COLUMNS if c not in existing]
        if missing:
            self._fieldnames = existing + missing
            self._rewrite_with_fieldnames(self._fieldnames)
            return

        self._fieldnames = existing

    def _ensure_columns(self, columns: Iterable[str]) -> None:
        new_cols = [c for c in columns if c not in self._fieldnames]
        if not new_cols:
            return
        self._fieldnames = self._fieldnames + new_cols
        self._rewrite_with_fieldnames(self._fieldnames)

    def append(self, row: dict[str, Any]) -> None:
        self._ensure_columns(row.keys())
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writerow({k: row.get(k) for k in self._fieldnames})


class ReplayStore:
    def __init__(self, cfg: Optional[ReplayConfig]) -> None:
        self.cfg = cfg
        self._loaded: Optional[dict[tuple[int, str, str], str]] = None

    def _ensure_loaded(self) -> None:
        if self._loaded is not None:
            return
        self._loaded = {}
        if not self.cfg or self.cfg.mode != "replay":
            return
        if not self.cfg.path.exists():
            raise FileNotFoundError(self.cfg.path)
        with self.cfg.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = (int(obj.get("round") or 0), str(obj.get("task") or ""), str(obj.get("context") or ""))
                self._loaded[key] = str(obj.get("response") or "")

    def lookup(self, *, round_id: int, task: str, context: str) -> Optional[str]:
        if not self.cfg or self.cfg.mode != "replay":
            return None
        self._ensure_loaded()
        assert self._loaded is not None
        return self._loaded.get((round_id, task, context))

    def record(self, *, round_id: int, task: str, context: str, response: str, extra: dict[str, Any]) -> None:
        if not self.cfg or self.cfg.mode != "record":
            return
        self.cfg.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"round": round_id, "task": task, "context": context, "response": response, **extra}
        with self.cfg.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
        replay: Optional[ReplayConfig],
        events: EventLogger,
        ledger: TokenLedger,
        token_counter: TokenCounter,
        budget_total: int,
        safety_margin_tokens: int,
        state: "RunState",
    ) -> None:
        from openai import OpenAI  # local import

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._replay = ReplayStore(replay)
        self.events = events
        self.ledger = ledger
        self.token_counter = token_counter
        self.budget_total = budget_total
        self.safety_margin_tokens = safety_margin_tokens
        self.state = state
        self._prev_total_actual: Optional[int] = None

    def chat(
        self,
        *,
        task: str,
        context: str,
        system_prompt: str,
        parent_span_id: Optional[str],
        tags: dict[str, Any],
        temperature: Optional[float] = None,
    ) -> Tuple[str, dict[str, Any]]:
        self.state.llm_round += 1
        round_id = int(self.state.llm_round)
        span_id = uuid.uuid4().hex

        user_content = task + ("\n\n" + context if context else "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_est = self.token_counter.estimate_messages(messages)
        prompt_est_system = self.token_counter.estimate_text(system_prompt)
        prompt_est_task = self.token_counter.estimate_text(task)
        prompt_est_components = self.token_counter.estimate_text(context)
        prompt_est_parts_total = int(prompt_est_system) + int(prompt_est_task) + int(prompt_est_components)
        components_breakdown = json.dumps(_estimate_component_breakdown(context, self.token_counter), ensure_ascii=False)
        budget = max(0, int(self.budget_total) - int(self.max_tokens) - int(self.safety_margin_tokens))
        if prompt_est > budget:
            raise ValueError(f"prompt token estimate exceeds budget: est={prompt_est} budget={budget}")

        replayed = self._replay.lookup(round_id=round_id, task=task, context=context)
        if replayed is not None:
            response_text = replayed
            usage: dict[str, Any] = {}
            event_type = "llm.replay"
        else:
            effective_temperature = self.temperature if temperature is None else float(temperature)
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=effective_temperature,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
            usage_raw = getattr(completion, "usage", None)
            usage = dict(usage_raw.model_dump()) if usage_raw is not None else {}
            event_type = "llm.call"
            self._replay.record(
                round_id=round_id,
                task=task,
                context=context,
                response=response_text,
                extra={
                    "ts": _utc_ts(),
                    "model": self.model,
                    "usage": usage,
                    "tags": tags,
                    "temperature": effective_temperature,
                },
            )

        prompt_actual = usage.get("prompt_tokens")
        completion_actual = usage.get("completion_tokens")
        total_actual = usage.get("total_tokens")
        delta = None
        if isinstance(total_actual, int) and self._prev_total_actual is not None:
            delta = int(total_actual) - int(self._prev_total_actual)
        if isinstance(total_actual, int):
            self._prev_total_actual = int(total_actual)

        token_payload = {
            "budget_total": self.budget_total,
            "prompt_est": prompt_est,
            "prompt_actual": prompt_actual,
            "completion_actual": completion_actual,
            "total_actual": total_actual,
            "delta_vs_prev": delta,
            "prompt_est_system": prompt_est_system,
            "prompt_est_task": prompt_est_task,
            "prompt_est_components": prompt_est_components,
            "prompt_est_parts_total": prompt_est_parts_total,
            "components_breakdown": components_breakdown,
        }
        self.events.log(
            event_type=event_type,
            span_id=span_id,
            parent_span_id=parent_span_id,
            payload={
                "round": round_id,
                "model": self.model,
                "tags": tags,
                "temperature": (self.temperature if temperature is None else float(temperature)),
                "prompt": {"messages": messages},
                "response": response_text,
            },
            token=token_payload,
        )
        self.ledger.append(
            {
                "ts": _utc_ts(),
                "event_type": event_type,
                "span_id": span_id,
                **token_payload,
            }
        )
        return response_text, token_payload


@dataclass
class DiagramPlanItem:
    diagram_id: str
    type: str
    title: str
    goal: str
    dependencies: list[str] = field(default_factory=list)
    acceptance_checks: list[str] = field(default_factory=list)

    @classmethod
    def from_obj(cls, obj: Any) -> "DiagramPlanItem":
        if not isinstance(obj, dict):
            raise TypeError("diagram plan item must be an object")
        diagram_id = str(obj.get("diagram_id") or obj.get("id") or "").strip()
        if not diagram_id:
            raise ValueError("missing diagram_id")
        diag_type = str(obj.get("type") or "").strip() or "diagram"
        title = str(obj.get("title") or diagram_id).strip()
        goal = str(obj.get("goal") or "").strip()
        deps_raw = obj.get("dependencies") or []
        deps = [str(d).strip() for d in deps_raw if str(d).strip()] if isinstance(deps_raw, list) else []
        checks_raw = obj.get("acceptance_checks") or []
        checks = [str(c).strip() for c in checks_raw if str(c).strip()] if isinstance(checks_raw, list) else []
        return cls(
            diagram_id=diagram_id,
            type=diag_type,
            title=title,
            goal=goal,
            dependencies=deps,
            acceptance_checks=checks,
        )


@dataclass
class RunState:
    version: int = 1
    run_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    llm_round: int = 0
    plan: list[DiagramPlanItem] = field(default_factory=list)
    status_by_id: dict[str, str] = field(default_factory=dict)
    attempts_by_id: dict[str, int] = field(default_factory=dict)
    last_error_by_id: dict[str, str] = field(default_factory=dict)
    replans_by_stage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "llm_round": self.llm_round,
            "plan": [asdict(i) for i in self.plan],
            "status_by_id": dict(self.status_by_id),
            "attempts_by_id": dict(self.attempts_by_id),
            "last_error_by_id": dict(self.last_error_by_id),
            "replans_by_stage": dict(self.replans_by_stage),
        }

    @classmethod
    def from_dict(cls, obj: Any) -> "RunState":
        if not isinstance(obj, dict):
            raise TypeError("state must be an object")
        plan_raw = obj.get("plan") or []
        plan = [DiagramPlanItem.from_obj(i) for i in plan_raw] if isinstance(plan_raw, list) else []
        return cls(
            version=int(obj.get("version") or 1),
            run_id=str(obj.get("run_id") or ""),
            created_at=str(obj.get("created_at") or ""),
            updated_at=str(obj.get("updated_at") or ""),
            llm_round=int(obj.get("llm_round") or 0),
            plan=plan,
            status_by_id=dict(obj.get("status_by_id") or {}),
            attempts_by_id=dict(obj.get("attempts_by_id") or {}),
            last_error_by_id=dict(obj.get("last_error_by_id") or {}),
            replans_by_stage=dict(obj.get("replans_by_stage") or {}),
        )


class StageReplanRequested(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        diagram_id: str,
        cross_errors: list[Any],
        cross_scope: list[str],
        last_errors: list[Any],
        last_warnings: list[Any],
    ) -> None:
        super().__init__(f"cross-view strict validation failed for stage={stage} diagram_id={diagram_id}")
        self.stage = stage
        self.diagram_id = diagram_id
        self.cross_errors = list(cross_errors)
        self.cross_scope = list(cross_scope)
        self.last_errors = list(last_errors)
        self.last_warnings = list(last_warnings)


def _detect_dependency_cycle(items: Sequence[DiagramPlanItem]) -> Optional[list[str]]:
    graph: dict[str, list[str]] = {i.diagram_id: list(i.dependencies) for i in items}
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> Optional[list[str]]:
        if node in visited:
            return None
        if node in visiting:
            if node in stack:
                idx = stack.index(node)
                return stack[idx:] + [node]
            return [node, node]
        visiting.add(node)
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                continue
            found = dfs(dep)
            if found:
                return found
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in graph.keys():
        found = dfs(node)
        if found:
            return found
    return None


def _summarize_packages(packages: Sequence[Any], *, max_names: int = 40) -> dict[str, Any]:
    counts: dict[str, int] = {}
    names: list[str] = []

    def walk(node: Any) -> None:
        t = getattr(node, "type", None)
        n = getattr(node, "name", None)
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
        if isinstance(n, str) and n.strip() and len(names) < max_names:
            names.append(n.strip())
        for child in getattr(node, "children", []) or []:
            walk(child)

    for pkg in packages:
        walk(pkg)
    return {"counts": counts, "sample_names": names}


def _flatten_packages(packages: Sequence[Any]) -> list[Any]:
    flat: list[Any] = []

    def walk(node: Any) -> None:
        flat.append(node)
        for child in getattr(node, "children", []) or []:
            walk(child)

    for pkg in packages:
        walk(pkg)
    return flat


def _build_context_with_budget(
    *,
    task: str,
    components: list[tuple[str, str, bool]],
    system_prompt: str,
    token_counter: TokenCounter,
    max_context_tokens: int,
    reserve_completion_tokens: int,
    safety_margin_tokens: int,
    events: EventLogger,
    parent_span_id: Optional[str],
) -> Tuple[str, dict[str, Any]]:
    """
    Build a context string that keeps estimated prompt tokens within budget.

    components: list of (name, text, drop_first) where drop_first means low priority.
    """
    kept: list[tuple[str, str, bool]] = [(n, t, d) for (n, t, d) in components if t.strip()]

    def assemble(parts: Sequence[tuple[str, str, bool]]) -> str:
        rendered: list[str] = []
        for name, text, _drop in parts:
            rendered.append(f"## {name}\n\n{text.strip()}")
        return "\n\n".join(rendered).strip()

    budget = max(0, int(max_context_tokens) - int(reserve_completion_tokens) - int(safety_margin_tokens))
    initial_context = assemble(kept)
    initial_est = token_counter.estimate_messages(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": task + ("\n\n" + initial_context if initial_context else "")}]
    )
    if initial_est <= budget:
        events.log(
            event_type="context.build",
            parent_span_id=parent_span_id,
            payload={
                "budget_total": max_context_tokens,
                "budget_prompt": budget,
                "prompt_est": initial_est,
                "components": [{"name": n, "tokens_est": token_counter.estimate_text(t)} for (n, t, _d) in kept],
            },
        )
        return initial_context, {"prompt_est": initial_est, "budget_prompt": budget}

    before = {"prompt_est": initial_est, "budget_prompt": budget}
    dropped: list[str] = []
    truncated: dict[str, int] = {}

    # Drop low-priority components first (from the end) while preserving order.
    dropped_set: set[str] = set()
    droppable_names = [name for (name, _text, drop_first) in kept if drop_first]
    for name in reversed(droppable_names):
        dropped.append(name)
        dropped_set.add(name)
        cur = [c for c in kept if c[0] not in dropped_set]
        ctx = assemble(cur)
        est = token_counter.estimate_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": task + ("\n\n" + ctx if ctx else "")}]
        )
        if est <= budget:
            events.log(
                event_type="context.compress",
                parent_span_id=parent_span_id,
                payload={"reason": "drop_low_priority", "before": before, "after": {"prompt_est": est}, "dropped": dropped},
            )
            return ctx, {"prompt_est": est, "budget_prompt": budget, "dropped": dropped}

    # Truncate remaining context as last resort (from the top).
    ctx = assemble([c for c in kept if c[0] not in dropped_set])
    est = token_counter.estimate_messages(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": task + ("\n\n" + ctx if ctx else "")}]
    )
    if est <= budget:
        events.log(
            event_type="context.compress",
            parent_span_id=parent_span_id,
            payload={"reason": "drop_all_low_priority", "before": before, "after": {"prompt_est": est}, "dropped": dropped},
        )
        return ctx, {"prompt_est": est, "budget_prompt": budget, "dropped": dropped}

    # Keep only the tail of the context to fit budget.
    # We approximate truncation by characters using token heuristic.
    allowed_tokens_for_context = max(0, budget - token_counter.estimate_messages([{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]))
    approx_chars = int(allowed_tokens_for_context * 4)
    if approx_chars <= 0:
        final_ctx = ""
        final_est = token_counter.estimate_messages([{"role": "system", "content": system_prompt}, {"role": "user", "content": task}])
    else:
        final_ctx = ctx[-approx_chars:]
        truncated["__tail_chars__"] = approx_chars
        final_est = token_counter.estimate_messages(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": task + ("\n\n" + final_ctx if final_ctx else "")}]
        )
    events.log(
        event_type="context.compress",
        parent_span_id=parent_span_id,
        payload={
            "reason": "truncate_tail",
            "before": before,
            "after": {"prompt_est": final_est, "budget_prompt": budget},
            "dropped": dropped,
            "truncated": truncated,
        },
    )
    return final_ctx, {"prompt_est": final_est, "budget_prompt": budget, "dropped": dropped, "truncated": truncated}


def _load_plan_from_file(path: Path) -> list[DiagramPlanItem]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "items" in obj:
        obj = obj["items"]
    if not isinstance(obj, list):
        raise TypeError("diagram plan must be a list or {items:[...]}")
    items = [DiagramPlanItem.from_obj(i) for i in obj]
    _validate_plan_unique_ids(items)
    return items


def _validate_plan_unique_ids(items: Sequence[DiagramPlanItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.diagram_id in seen:
            raise ValueError(f"duplicate diagram_id: {item.diagram_id}")
        seen.add(item.diagram_id)


def _prune_unknown_dependencies(items: Sequence[DiagramPlanItem]) -> list[dict[str, Any]]:
    known = {i.diagram_id for i in items}
    pruned: list[dict[str, Any]] = []
    for item in items:
        before = list(item.dependencies)
        after = [d for d in before if d in known and d != item.diagram_id]
        if before != after:
            item.dependencies = after
            pruned.append({"diagram_id": item.diagram_id, "before": before, "after": after})
    return pruned


def _planner_prompt(*, target_system: str, input_text: str, target_items: int) -> str:
    return (
        "You are planning a long-period incremental SysML v2 modeling run.\n\n"
        f"Target system: {target_system}\n\n"
        "Inputs (requirements/spec excerpts):\n"
        f"{input_text}\n\n"
        f"Generate a diagram plan with at least {target_items} items.\n"
        "IMPORTANT: diagram_id MUST be unique and stable. Avoid dependency cycles.\n"
        "Return STRICT JSON (no comments) as a JSON array. Each item must include:\n"
        "- diagram_id: string (unique, stable)\n"
        "- type: string (e.g., RD, BDD, IBD, AD, PD, SMD)\n"
        "- title: string\n"
        "- goal: string\n"
        "- dependencies: string[] (diagram_id references)\n"
        "- acceptance_checks: string[]\n"
    )


PHASE_SEQUENCE: tuple[str, ...] = ("RD", "BDD", "IBD", "AD", "PD")
MAX_STAGE_REPLANS_PER_STAGE: int = 2


_CROSS_RULE_MIN_STAGE: dict[str, str] = {
    # RD rules require downstream design views to exist.
    "RD-1": "BDD",
    "RD-2": "BDD",
    # "BBD" rule needs IBD instantiation information.
    "BBD-1": "IBD",
    # IBD rules only become meaningful once an IBD exists.
    "IBD-1": "IBD",
    "IBD-2": "IBD",
    "IBD-3": "IBD",
    # AD rules depend on AD artifacts.
    "AD-1": "AD",
    "AD-2": "AD",
    # PD rules depend on PD artifacts.
    "PD-1": "PD",
}


def _cross_rule_min_stage(rule_id: str) -> str:
    rid = (rule_id or "").strip().upper()
    if not rid:
        return "RD"
    if rid in _CROSS_RULE_MIN_STAGE:
        return _CROSS_RULE_MIN_STAGE[rid]
    if rid.startswith("RD-"):
        return "BDD"
    if rid.startswith("BBD-") or rid.startswith("IBD-"):
        return "IBD"
    if rid.startswith("AD-"):
        return "AD"
    if rid.startswith("PD-"):
        return "PD"
    return "RD"


def _filter_cross_errors_for_stage(cross_errors: Sequence[Any], *, stage: str) -> list[Any]:
    stage_norm = _map_type_to_stage(stage)
    phase_rank = {s: idx for idx, s in enumerate(PHASE_SEQUENCE)}
    stage_idx = phase_rank.get(stage_norm, 0)

    filtered: list[Any] = []
    for err in cross_errors:
        if isinstance(err, dict) and "rule_id" in err:
            min_stage = _cross_rule_min_stage(str(err.get("rule_id") or ""))
            if phase_rank.get(min_stage, 0) > stage_idx:
                continue
        filtered.append(err)
    return filtered


def _is_rd_cross_error(err: Any) -> bool:
    if not isinstance(err, dict):
        return False
    rule_id = str(err.get("rule_id") or "").strip().upper()
    return rule_id.startswith("RD-")


def _phase_target_distribution(total_target_items: int) -> dict[str, int]:
    """
    Distribute a total diagram target count across phases.

    This is used for per-phase incremental planning. Keep it robust even when
    total_target_items is small (e.g., dev/test runs).
    """

    total = max(0, int(total_target_items))
    counts = {stage: 0 for stage in PHASE_SEQUENCE}
    if total <= 0:
        return counts

    # Guarantee at least 1 per phase when the target allows it.
    base = 1 if total >= len(PHASE_SEQUENCE) else 0
    for stage in PHASE_SEQUENCE:
        counts[stage] = base
    remaining = total - base * len(PHASE_SEQUENCE)
    if remaining <= 0:
        return counts

    weights = {"RD": 0.2, "BDD": 0.24, "IBD": 0.34, "AD": 0.09, "PD": 0.13}
    raw = {stage: float(remaining) * float(weights.get(stage, 0.0)) for stage in PHASE_SEQUENCE}

    for stage in PHASE_SEQUENCE:
        counts[stage] += int(raw[stage])

    allocated = sum(counts.values())
    leftover = total - allocated
    if leftover > 0:
        order = sorted(PHASE_SEQUENCE, key=lambda s: raw[s] - int(raw[s]), reverse=True)
        idx = 0
        while leftover > 0:
            counts[order[idx % len(order)]] += 1
            leftover -= 1
            idx += 1

    return counts


def _phase_planner_prompt(*, stage: str, target_system: str, target_items: int) -> str:
    stage_norm = _map_type_to_stage(stage)
    return (
        "You are an incremental planning agent for a long-period SysML v2 modeling run.\n\n"
        f"Target system: {target_system}\n"
        f"Current phase (ONLY): {stage_norm}\n\n"
        f"Generate a diagram plan for ONLY this phase with at least {int(target_items)} items.\n"
        "Use the provided context (input specs, existing view summaries, domain knowledge, and any feedback) to plan.\n\n"
        "Constraints:\n"
        "- Return STRICT JSON as a JSON array (no markdown, no comments).\n"
        "- Each item must include: diagram_id, type, title, goal, dependencies, acceptance_checks.\n"
        "- diagram_id MUST be unique and stable.\n"
        "- type MUST correspond to the current phase.\n"
        "- dependencies may ONLY reference diagram_id values that appear in the provided context or other items in this plan.\n"
        "- Avoid dependency cycles.\n"
    )


def _compress_errors_prompt(*, stage: str, diagram_id: str) -> str:
    stage_norm = _map_type_to_stage(stage)
    return (
        "You are compressing cross-view consistency validation failures for a SysML v2 modeling pipeline.\n\n"
        f"Phase: {stage_norm}\n"
        f"Trigger diagram_id: {diagram_id}\n\n"
        "Summarize the errors into a compact, actionable form for replanning this phase.\n"
        "Output requirements:\n"
        "- Plain text only.\n"
        "- Include: (1) root causes, (2) impacted identifiers, (3) concrete fixes to the plan.\n"
        "- Keep it short (prefer <= 20 bullets / <= 1200 chars).\n"
    )


def _prune_unknown_dependencies_with_known(items: Sequence[DiagramPlanItem], *, known: set[str]) -> list[dict[str, Any]]:
    pruned: list[dict[str, Any]] = []
    for item in items:
        before = list(item.dependencies)
        after = [d for d in before if d in known and d != item.diagram_id]
        if before != after:
            item.dependencies = after
            pruned.append({"diagram_id": item.diagram_id, "before": before, "after": after})
    return pruned


def _extract_markdown_bullets_under_heading(text: str, *, heading_contains: str) -> list[str]:
    needle = heading_contains.strip().lower()
    if not needle:
        return []
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##") and needle in stripped.lower():
            start = idx + 1
            break
    if start is None:
        return []
    bullets: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("##"):
            break
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item:
                bullets.append(item)
    return bullets


def _fallback_diagram_plan(*, target_system: str, input_text: str, target_items: int) -> list[DiagramPlanItem]:
    subsystems = _extract_markdown_bullets_under_heading(input_text, heading_contains="Suggested Subsystems") or []
    if not subsystems:
        subsystems = [
            "System",
            "Subsystem A",
            "Subsystem B",
            "Subsystem C",
            "Subsystem D",
            "Subsystem E",
            "Subsystem F",
            "Subsystem G",
            "Subsystem H",
            "Subsystem I",
        ]

    # Cap to keep IDs compact and stable.
    subsystems = subsystems[:12]
    sub_codes = [f"SUB{i:02d}" for i in range(1, len(subsystems) + 1)]
    sub_by_code = dict(zip(sub_codes, subsystems))

    rd_nf_count = min(20, max(10, int(target_items) // 6))
    rd_f_count = min(20, max(10, int(target_items) // 6))
    ad_count = min(12, max(8, int(target_items) // 12))
    pd_count = min(12, max(8, int(target_items) // 12))
    base = rd_nf_count + rd_f_count + ad_count + pd_count
    remaining = max(0, int(target_items) - base)
    bdd_count = remaining // 2
    ibd_count = remaining - bdd_count

    nf_topics = [
        "Performance (image quality / resolution / sensitivity)",
        "Reliability & availability for scheduled observing",
        "Maintainability (segment handling / calibration / modular replacement)",
        "Safety & compliance constraints (site, operations, laser safety if applicable)",
        "Telemetry/logging completeness for diagnostics",
        "Environmental constraints (wind/thermal/seeing) handling",
        "International collaboration & interface governance (config control)",
        "Security (network, data integrity)",
        "Scalability of observation/data processing",
        "Fault tolerance & safe-state transitions",
    ]
    f_topics = [
        "Target pointing, slewing, tracking, and stabilization",
        "Segment alignment & phasing loop",
        "Wavefront sensing and alignment verification",
        "Instrument selection, configuration, and control interfaces",
        "Observation sequencing and execution (OCS)",
        "Calibration execution and scheduling hooks",
        "Data acquisition with metadata capture",
        "Health monitoring, alarms, and operator interaction",
        "Fault detection, isolation, recovery, and safe-state entry",
        "Maintenance mode support (diagnostics, segment replacement)",
    ]
    ad_scenarios = [
        "Observatory startup",
        "Slew to target",
        "Target acquisition",
        "Alignment & phasing sequence",
        "AO acquisition (optional)",
        "Science exposure loop",
        "Calibration sequence",
        "Fault response and safe-state",
        "Shutdown / park",
        "Maintenance operation",
    ]
    pd_budgets = [
        "Power budget",
        "Mass budget",
        "Thermal budget",
        "Data rate & storage budget",
        "Availability budget",
        "Observation time budget",
        "Pointing/tracking error budget",
        "Wavefront error budget",
        "Vibration budget",
        "Safety risk budget",
    ]

    items: list[DiagramPlanItem] = []

    for idx in range(1, rd_nf_count + 1):
        topic = nf_topics[(idx - 1) % len(nf_topics)]
        items.append(
            DiagramPlanItem(
                diagram_id=f"RD-NF-{idx:03d}",
                type="RD",
                title=f"Non-functional requirement: {topic}",
                goal=f"Define a non-functional requirement for {target_system}: {topic}.",
                dependencies=[],
                acceptance_checks=["Defines at least 1 requirement element", "Uses consistent naming", "Parsable SysML v2 package"],
            )
        )

    for idx in range(1, rd_f_count + 1):
        topic = f_topics[(idx - 1) % len(f_topics)]
        items.append(
            DiagramPlanItem(
                diagram_id=f"RD-F-{idx:03d}",
                type="RD",
                title=f"Functional requirement: {topic}",
                goal=f"Define a functional requirement for {target_system}: {topic}.",
                dependencies=[],
                acceptance_checks=["Defines at least 1 requirement element", "Includes trace hooks to design", "Parsable SysML v2 package"],
            )
        )

    bdd_counters: dict[str, int] = {code: 0 for code in sub_codes}
    for i in range(bdd_count):
        code = sub_codes[i % len(sub_codes)]
        bdd_counters[code] += 1
        seq = bdd_counters[code]
        sub_name = sub_by_code.get(code, code)
        items.append(
            DiagramPlanItem(
                diagram_id=f"BDD-{code}-{seq:03d}",
                type="BDD",
                title=f"BDD {code}: {sub_name}",
                goal=f"Define the structural decomposition (parts/attributes/interfaces) of {target_system} for subsystem: {sub_name}.",
                dependencies=["RD-F-001", "RD-NF-001"],
                acceptance_checks=["Contains package + part definitions", "Includes key interfaces/ports or connections", "Parsable SysML v2 package"],
            )
        )

    available_codes = [c for c in sub_codes if bdd_counters.get(c, 0) > 0] or list(sub_codes)
    ibd_counters: dict[str, int] = {code: 0 for code in available_codes}
    for i in range(ibd_count):
        code = available_codes[i % len(available_codes)]
        ibd_counters[code] += 1
        seq = ibd_counters[code]
        sub_name = sub_by_code.get(code, code)
        dep = f"BDD-{code}-001" if bdd_counters.get(code, 0) > 0 else "RD-F-001"
        items.append(
            DiagramPlanItem(
                diagram_id=f"IBD-{code}-{seq:03d}",
                type="IBD",
                title=f"IBD {code}: {sub_name} internal connectivity",
                goal=f"Model internal connections/interfaces for {target_system} subsystem: {sub_name}.",
                dependencies=[dep],
                acceptance_checks=["Contains package + internal connections", "Uses consistent part names", "Parsable SysML v2 package"],
            )
        )

    for idx in range(1, ad_count + 1):
        scenario = ad_scenarios[(idx - 1) % len(ad_scenarios)]
        items.append(
            DiagramPlanItem(
                diagram_id=f"AD-OPS-{idx:03d}",
                type="AD",
                title=f"Operational activity: {scenario}",
                goal=f"Model the operational flow for {target_system}: {scenario}.",
                dependencies=["RD-F-001"],
                acceptance_checks=["Contains package + activity/behavior elements", "Has clear start/end or triggers", "Parsable SysML v2 package"],
            )
        )

    for idx in range(1, pd_count + 1):
        budget = pd_budgets[(idx - 1) % len(pd_budgets)]
        items.append(
            DiagramPlanItem(
                diagram_id=f"PD-BUD-{idx:03d}",
                type="PD",
                title=f"Parametric model: {budget}",
                goal=f"Create a parametric view for {target_system} to capture and compute: {budget}.",
                dependencies=["RD-NF-001"],
                acceptance_checks=["Contains package + constraint/parametric elements", "Defines parameters/attributes", "Parsable SysML v2 package"],
            )
        )

    _validate_plan_unique_ids(items)
    return items


def _diagram_task_prompt(item: DiagramPlanItem) -> str:
    checks = "\n".join([f"- {c}" for c in item.acceptance_checks]) if item.acceptance_checks else "- (none)"
    deps = ", ".join(item.dependencies) if item.dependencies else "(none)"
    return (
        "Generate SysML v2 code for the following diagram/view. Return ONLY a single ```sysml fenced block```.\n\n"
        f"diagram_id: {item.diagram_id}\n"
        f"type: {item.type}\n"
        f"title: {item.title}\n"
        f"goal: {item.goal}\n"
        f"dependencies: {deps}\n\n"
        "Acceptance checks:\n"
        f"{checks}\n"
    )


def _map_type_to_stage(diagram_type: str) -> str:
    t = (diagram_type or "").strip().upper()
    if not t:
        return "BDD"
    if t == "REQ" or t == "REQUIREMENT" or t.startswith("RD"):
        return "RD"
    if t.startswith("BDD"):
        return "BDD"
    if t.startswith("IBD"):
        return "IBD"
    if t.startswith("AD"):
        return "AD"
    if t.startswith("PD"):
        return "PD"
    return "BDD"


def _try_build_rag_context(
    *,
    stage: str,
    task: str,
    rag_mode: str,
    local_domain_knowledge: str,
    qdrant_host: str,
    qdrant_port: int,
    examples_collection: str,
    knowledge_collection: str,
    examples_top_k: int,
    domain_top_k: int,
    events: EventLogger,
    parent_span_id: Optional[str],
) -> str:
    try:
        from ..sysml.knowledge import SysMLKnowledge

        sysml_knowledge = SysMLKnowledge.get(stage) or ""
        events.log(
            event_type="knowledge.sysml.load",
            parent_span_id=parent_span_id,
            payload={"stage": stage, "bytes": len(sysml_knowledge)},
        )
    except Exception as e:
        sysml_knowledge = ""
        events.log(
            event_type="knowledge.sysml.load",
            parent_span_id=parent_span_id,
            payload={"stage": stage, "error": str(e)},
        )

    mode = _normalize_rag_mode(rag_mode)
    examples_text = ""
    domain_text = ""

    if mode != "local_only":
        # Optional Qdrant RAG retrieval.
        try:
            from ..rag.retriever import searchVec
            from ..rag.domain_knowledge import domainKnowledge_with_params

            retriever = searchVec(host=qdrant_host, port=qdrant_port)

            examples = retriever.find_top_k_similar(task, examples_top_k, examples_collection, False)
            events.log(
                event_type="rag.examples.search",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "k": examples_top_k, "collection": examples_collection, "hits": len(examples)},
            )
            examples_text = "".join([str(item.get("code") or "") for item in examples])

            domain_text = domainKnowledge_with_params(
                task,
                k=domain_top_k,
                collection=knowledge_collection,
                retriever=retriever,
            )
            events.log(
                event_type="rag.domain.search",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "k": domain_top_k, "collection": knowledge_collection},
            )
        except Exception as e:
            events.log(
                event_type="rag.degrade",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "error": str(e)},
            )
            events.log(
                event_type="rag.examples.search",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "error": str(e), "k": examples_top_k, "collection": examples_collection},
            )
            events.log(
                event_type="rag.domain.search",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "error": str(e), "k": domain_top_k, "collection": knowledge_collection},
            )
            examples_text = ""
            domain_text = ""

    if mode != "qdrant_only":
        if local_domain_knowledge.strip():
            domain_text = local_domain_knowledge.strip()
            events.log(
                event_type="knowledge.domain.local",
                parent_span_id=parent_span_id,
                payload={"mode": mode, "bytes": len(domain_text)},
            )

    context = ""
    if sysml_knowledge.strip():
        context += "-- SysML Related Knowledge --\n\n" + sysml_knowledge.strip() + "\n\n"
    if examples_text.strip():
        context += "-- Similar example in SysML-v2 --\n\n" + examples_text.strip() + "\n\n"
    if domain_text.strip():
        context += "-- Domain Related Knowledge --\n\n" + domain_text.strip() + "\n"
    return context.strip()


def run_long_period(
    *,
    api_key: str,
    base_url: Optional[str],
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    out_dir: Path,
    qdrant_host: str,
    qdrant_port: int,
    examples_collection: str,
    knowledge_collection: str,
    replay: Optional[ReplayConfig],
    target_system: str,
    input_paths: Sequence[Path],
    knowledge_path: Optional[Path],
    rag_mode: str,
    diagram_plan_path: Optional[Path],
    run_id: Optional[str],
    resume: bool,
    max_diagrams: Optional[int],
    max_context_tokens: int,
    safety_margin_tokens: int,
    planner_target_items: int,
    builder_max_retries: int,
    validate_rules: bool,
    rule_type: str,
    cross_view_validation: str,
    semantic_entropy_samples: int = 1,
    semantic_entropy_temperature: float = 0.7,
    semantic_entropy_similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    model_name = (model or "deepseek-coder").strip()
    out_dir = out_dir.expanduser()
    long_root = out_dir / "long_period"
    long_root.mkdir(parents=True, exist_ok=True)

    latest_ptr = long_root / "latest.json"
    latest_run_id: Optional[str] = None
    if latest_ptr.is_file():
        try:
            latest_obj = json.loads(latest_ptr.read_text(encoding="utf-8"))
            if isinstance(latest_obj, dict) and isinstance(latest_obj.get("run_id"), str):
                latest_run_id = latest_obj["run_id"]
        except Exception:
            latest_run_id = None

    def new_run_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]

    chosen_run_id = run_id or (latest_run_id if resume else None) or new_run_id()
    run_dir = long_root / chosen_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagrams").mkdir(parents=True, exist_ok=True)

    # Keep rule state isolated from legacy runs.
    os.environ["AUTOMBSE_RULE_STATE_FILE"] = str(run_dir / "rule_states.json")

    latest_ptr.write_text(json.dumps({"run_id": chosen_run_id}, ensure_ascii=False) + "\n", encoding="utf-8")

    state_path = run_dir / "state.json"
    index_path = run_dir / "index.json"
    events_path = run_dir / "events.jsonl"
    token_csv_path = run_dir / "token_trace.csv"
    run_meta_path = run_dir / "run.json"
    plan_out_path = run_dir / "diagram_plan.json"

    events = EventLogger(path=events_path)
    ledger = TokenLedger(csv_path=token_csv_path)
    token_counter = TokenCounter(model=model_name)

    if state_path.is_file():
        state = RunState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
        if not state.run_id:
            state.run_id = chosen_run_id
    else:
        state = RunState(run_id=chosen_run_id, created_at=_utc_ts(), updated_at=_utc_ts())

    index: dict[str, Any] = {}
    if index_path.is_file():
        try:
            obj = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                index = obj
        except Exception:
            index = {}

    if state_path.is_file():
        events.log(
            event_type="state.load",
            payload={
                "path": str(state_path),
                "run_id": chosen_run_id,
                "llm_round": state.llm_round,
                "counts": _status_counts_for_plan(state),
            },
        )
        in_progress_ids = [k for k, v in (state.status_by_id or {}).items() if v == "in_progress"]
        if in_progress_ids:
            for diagram_id in in_progress_ids:
                state.status_by_id[diagram_id] = "pending"
            state.updated_at = _utc_ts()
            state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            events.log(
                event_type="state.recover",
                payload={"reset": in_progress_ids, "from": "in_progress", "to": "pending"},
            )

    if index_path.is_file():
        events.log(
            event_type="artifact.read",
            payload={"path": str(index_path), "kind": "index", "items": len(index)},
        )

    events.log(
        event_type="state.save",
        payload={"run_id": chosen_run_id, "resume": bool(resume and state_path.is_file())},
    )
    run_meta_path.write_text(
        json.dumps(
            {
                "run_id": chosen_run_id,
                "created_at": state.created_at or _utc_ts(),
                "model": model_name,
                "base_url": base_url,
                "max_context_tokens": max_context_tokens,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "out_dir": str(out_dir),
                "rag_mode": str(rag_mode or "auto"),
                "knowledge_path": str(knowledge_path) if knowledge_path else None,
                "validate_rules": bool(validate_rules),
                "rule_type_requested": str(rule_type or ""),
                "cross_view_validation": str(cross_view_validation or ""),
                "semantic_entropy": {
                    "samples": int(semantic_entropy_samples),
                    "temperature": float(semantic_entropy_temperature),
                    "similarity_threshold": float(semantic_entropy_similarity_threshold),
                },
                "qdrant": {
                    "host": qdrant_host,
                    "port": qdrant_port,
                    "collections": {
                        "examples": examples_collection,
                        "knowledge": knowledge_collection,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    input_text, used_inputs = _collect_input_text(list(input_paths))

    llm = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        replay=replay,
        events=events,
        ledger=ledger,
        token_counter=token_counter,
        budget_total=max_context_tokens,
        safety_margin_tokens=safety_margin_tokens,
        state=state,
    )

    rag_mode_norm = _normalize_rag_mode(rag_mode)
    local_domain_knowledge = ""
    if knowledge_path is not None and knowledge_path.is_file():
        try:
            raw = _read_text_file(knowledge_path)
            local_domain_knowledge = _prepare_local_knowledge_for_prompt(raw)
            events.log(
                event_type="knowledge.domain.load",
                payload={
                    "path": str(knowledge_path),
                    "bytes_raw": len(raw),
                    "bytes_prompt": len(local_domain_knowledge),
                },
            )
        except Exception as e:
            events.log(
                event_type="knowledge.domain.load",
                payload={"path": str(knowledge_path), "error": str(e)},
            )
            local_domain_knowledge = ""

    events.log(
        event_type="rag.mode",
        payload={
            "rag_mode": rag_mode,
            "normalized": rag_mode_norm,
            "knowledge_path": str(knowledge_path) if knowledge_path else None,
        },
    )

    requested_rule_type = _normalize_rule_type(rule_type)
    cross_view_mode = _normalize_cross_view_validation(cross_view_validation, rule_type=requested_rule_type)
    events.log(
        event_type="rules.mode",
        payload={
            "validate_rules": bool(validate_rules),
            "rule_type_requested": str(rule_type or ""),
            "rule_type": requested_rule_type,
            "cross_view_validation": str(cross_view_validation or ""),
            "cross_view_mode": cross_view_mode,
        },
    )

    # Plan loading (optional). When no external plan is provided, the pipeline uses
    # per-phase incremental planning (RD -> BDD -> IBD -> AD -> PD) during execution.
    if not state.plan:
        if diagram_plan_path and diagram_plan_path.is_file():
            items = _load_plan_from_file(diagram_plan_path)
            pruned = _prune_unknown_dependencies(items)
            events.log(
                event_type="artifact.read",
                payload={"path": str(diagram_plan_path), "kind": "diagram_plan.load", "items": len(items), "pruned": len(pruned)},
            )
            if pruned:
                events.log(
                    event_type="consistency.deps_prune",
                    payload={"scope": "diagram_plan", "count": len(pruned), "sample": pruned[:20]},
                )

            cycle = _detect_dependency_cycle(items)
            events.log(
                event_type="consistency.cycle_check",
                payload={"scope": "diagram_plan", "has_cycle": bool(cycle), "cycle": cycle},
            )
            if cycle:
                raise ValueError(f"diagram plan has dependency cycle: {' -> '.join(cycle)}")

            state.plan = items
        else:
            events.log(
                event_type="planner.defer",
                payload={
                    "reason": "incremental_phase_planning",
                    "target_items": planner_target_items,
                    "inputs": used_inputs,
                },
            )

    # Ensure state maps cover current plan (resume-safe).
    for item in state.plan:
        state.status_by_id.setdefault(item.diagram_id, "pending")
        state.attempts_by_id.setdefault(item.diagram_id, 0)
    state.updated_at = _utc_ts()
    state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plan_by_id: dict[str, DiagramPlanItem] = {}

    def refresh_plan_by_id() -> None:
        plan_by_id.clear()
        plan_by_id.update({item.diagram_id: item for item in state.plan})

    refresh_plan_by_id()

    def dependency_closure(diagram_id: str) -> list[str]:
        item = plan_by_id.get(diagram_id)
        if not item:
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        stack = list(reversed(item.dependencies))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            ordered.append(cur)
            dep_item = plan_by_id.get(cur)
            if dep_item:
                stack.extend(list(reversed(dep_item.dependencies)))
        return ordered

    # Build order: repeatedly pick next buildable item (unlocks deps within this run).
    def deps_satisfied(item: DiagramPlanItem) -> bool:
        for dep in item.dependencies:
            if state.status_by_id.get(dep) != "done":
                return False
        return True

    # --- Incremental phase planning & execution ---
    #
    # TODO(long_period.phase_planning):
    # - Consider updating later-phase plans when an earlier phase replans (today we only replan the active phase).
    # - Replace package-count summaries with semantic summaries (dedicated LLM call) for higher-quality planning.
    # - Add optional phase-level cross-view validation gates (in addition to per-diagram checks).

    phase_targets = _phase_target_distribution(planner_target_items)

    def _phase_items(stage: str) -> list[DiagramPlanItem]:
        stage_norm = _map_type_to_stage(stage)
        return [i for i in state.plan if _map_type_to_stage(i.type) == stage_norm]

    def _phase_active_items(stage: str) -> list[DiagramPlanItem]:
        return [i for i in _phase_items(stage) if state.status_by_id.get(i.diagram_id) != "expired"]

    def _phase_blocked_ids(stage: str) -> list[str]:
        blocked_ids: list[str] = []
        for candidate in _phase_active_items(stage):
            if state.status_by_id.get(candidate.diagram_id) not in {"pending", "failed"}:
                continue
            if not deps_satisfied(candidate):
                blocked_ids.append(candidate.diagram_id)
        return blocked_ids

    def _write_plan_snapshot(*, parent_span_id: Optional[str], source: str) -> None:
        plan_out_path.write_text(
            json.dumps([asdict(i) for i in state.plan], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        events.log(
            event_type="artifact.write",
            parent_span_id=parent_span_id,
            payload={"path": str(plan_out_path), "kind": "diagram_plan.write", "items": len(state.plan), "source": source},
        )

    phase_rank = {stage: idx for idx, stage in enumerate(PHASE_SEQUENCE)}

    def _format_done_view_summaries(*, stages: Sequence[str], max_items: int = 40) -> str:
        allowed = {str(s).strip().upper() for s in stages if str(s).strip()}
        if not allowed:
            return ""
        lines: list[str] = []
        for item in state.plan:
            if _map_type_to_stage(item.type) not in allowed:
                continue
            if state.status_by_id.get(item.diagram_id) != "done":
                continue
            meta = index.get(item.diagram_id) if isinstance(index.get(item.diagram_id), dict) else {}
            summary = meta.get("summary")
            snippet = json.dumps(summary, ensure_ascii=False) if summary is not None else ""
            lines.append(f"- {item.diagram_id} ({_map_type_to_stage(item.type)}): {item.title} | {snippet}")
            if len(lines) >= max_items:
                break
        return "\n".join(lines).strip()

    def _format_phase_plan_snapshot(stage: str, *, max_items: int = 60) -> str:
        lines: list[str] = []
        for item in _phase_items(stage):
            status = state.status_by_id.get(item.diagram_id) or "pending"
            deps = ", ".join(item.dependencies) if item.dependencies else "(none)"
            goal = (item.goal or "").strip()
            goal = (goal[:200] + "…") if len(goal) > 200 else goal
            lines.append(f"- {item.diagram_id} [{status}] {item.title} | deps: {deps} | goal: {goal}")
            if len(lines) >= max_items:
                break
        return "\n".join(lines).strip()

    def _apply_phase_plan(
        *,
        stage: str,
        items: list[DiagramPlanItem],
        reset_statuses: bool,
        parent_span_id: Optional[str],
        source: str,
    ) -> None:
        stage_norm = _map_type_to_stage(stage)
        existing_other_ids = {i.diagram_id for i in state.plan if _map_type_to_stage(i.type) != stage_norm}
        existing_stage_items = [i for i in state.plan if _map_type_to_stage(i.type) == stage_norm]
        existing_stage_by_id = {i.diagram_id: i for i in existing_stage_items}
        new_ids = {i.diagram_id for i in items}

        conflicts = sorted([d for d in new_ids if d in existing_other_ids])
        if conflicts:
            raise ValueError(f"phase plan diagram_id conflicts with other phases: {conflicts[:10]}")

        removed = sorted([d for d in existing_stage_by_id.keys() if d not in new_ids])
        for diagram_id in removed:
            state.status_by_id[diagram_id] = "expired"
            state.last_error_by_id.pop(diagram_id, None)

        for item in items:
            existing = existing_stage_by_id.get(item.diagram_id)
            if existing is None:
                state.plan.append(item)
            else:
                existing.type = item.type
                existing.title = item.title
                existing.goal = item.goal
                existing.dependencies = list(item.dependencies)
                existing.acceptance_checks = list(item.acceptance_checks)

            state.status_by_id.setdefault(item.diagram_id, "pending")
            state.attempts_by_id.setdefault(item.diagram_id, 0)

            if reset_statuses:
                state.status_by_id[item.diagram_id] = "pending"
                state.attempts_by_id[item.diagram_id] = 0
                state.last_error_by_id.pop(item.diagram_id, None)

        refresh_plan_by_id()
        state.updated_at = _utc_ts()
        state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        events.log(
            event_type="planner.phase.apply",
            parent_span_id=parent_span_id,
            payload={
                "stage": stage_norm,
                "items": len(items),
                "removed": len(removed),
                "reset_statuses": bool(reset_statuses),
                "source": source,
            },
        )
        _write_plan_snapshot(parent_span_id=parent_span_id, source=source)

    def _ensure_phase_plan(*, stage: str, feedback: str, force: bool, parent_span_id: Optional[str]) -> None:
        stage_norm = _map_type_to_stage(stage)
        target_items = int(phase_targets.get(stage_norm) or 0)
        if target_items <= 0:
            events.log(
                event_type="planner.phase.skip",
                parent_span_id=parent_span_id,
                payload={"stage": stage_norm, "reason": "target_items_zero"},
            )
            return

        existing = _phase_items(stage_norm)
        active = _phase_active_items(stage_norm)
        needs_plan = force or (not active)
        if not needs_plan:
            return

        reset_statuses = force or (bool(existing) and not active)

        span = events.log(
            event_type="context.build",
            parent_span_id=parent_span_id,
            payload={"kind": "phase_plan.prompt", "stage": stage_norm, "target_items": target_items},
        )
        planner_task = _phase_planner_prompt(stage=stage_norm, target_system=target_system, target_items=target_items)

        prev_stages = [s for s in PHASE_SEQUENCE if phase_rank.get(s, 0) < phase_rank.get(stage_norm, 0)]
        done_views = _format_done_view_summaries(stages=prev_stages)
        existing_plan = _format_phase_plan_snapshot(stage_norm)

        rag_context = _try_build_rag_context(
            stage=stage_norm,
            task=planner_task,
            rag_mode=rag_mode_norm,
            local_domain_knowledge=local_domain_knowledge,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            examples_collection=examples_collection,
            knowledge_collection=knowledge_collection,
            examples_top_k=1,
            domain_top_k=2,
            events=events,
            parent_span_id=span,
        )

        components: list[tuple[str, str, bool]] = [
            ("Target System", target_system, False),
            ("Input Specs", input_text, stage_norm != "RD"),
            ("Existing Views (done)", done_views, False),
            ("Existing Phase Plan", existing_plan, True),
            ("Feedback", feedback.strip(), False),
            ("RAG/Knowledge", rag_context, True),
        ]
        planner_context, ctx_meta = _build_context_with_budget(
            task=planner_task,
            components=components,
            system_prompt=system_prompt,
            token_counter=token_counter,
            max_context_tokens=max_context_tokens,
            reserve_completion_tokens=max_tokens,
            safety_margin_tokens=safety_margin_tokens,
            events=events,
            parent_span_id=span,
        )

        attempts = 3
        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            attempt_span = events.log(
                event_type="planner.phase.attempt.start",
                parent_span_id=span,
                payload={"stage": stage_norm, "attempt": attempt, "max_attempts": attempts},
            )
            response_text, _token_payload = llm.chat(
                task=planner_task,
                context=planner_context,
                system_prompt=system_prompt,
                parent_span_id=attempt_span,
                tags={"kind": "phase_plan", "stage": stage_norm, "attempt": attempt, **ctx_meta},
            )
            blob = _extract_json_blob(response_text)
            try:
                plan_obj = json.loads(blob)
                if isinstance(plan_obj, dict) and "items" in plan_obj:
                    plan_obj = plan_obj["items"]
                if not isinstance(plan_obj, list):
                    raise TypeError("phase plan must be a list")

                candidate = [DiagramPlanItem.from_obj(i) for i in plan_obj]
                _validate_plan_unique_ids(candidate)

                for item in candidate:
                    if _map_type_to_stage(item.type) != stage_norm:
                        item.type = stage_norm

                known = {i.diagram_id for i in state.plan if _map_type_to_stage(i.type) != stage_norm}
                known.update([i.diagram_id for i in candidate])
                _prune_unknown_dependencies_with_known(candidate, known=known)

                cycle = _detect_dependency_cycle(candidate)
                if cycle:
                    raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
                if len(candidate) < target_items:
                    raise ValueError(f"too few items: got {len(candidate)} < {target_items}")

                _apply_phase_plan(
                    stage=stage_norm,
                    items=candidate,
                    reset_statuses=reset_statuses,
                    parent_span_id=attempt_span,
                    source="llm",
                )
                events.log(
                    event_type="planner.phase.attempt.accept",
                    parent_span_id=attempt_span,
                    payload={"stage": stage_norm, "attempt": attempt, "items": len(candidate)},
                )
                return
            except Exception as e:
                last_error = str(e)
                events.log(
                    event_type="planner.phase.attempt.reject",
                    parent_span_id=attempt_span,
                    payload={"stage": stage_norm, "attempt": attempt, "error": last_error},
                )

        events.log(
            event_type="planner.phase.fallback",
            parent_span_id=span,
            payload={"stage": stage_norm, "reason": "llm_invalid_plan", "error": last_error, "target_items": target_items},
        )
        fallback = [i for i in _fallback_diagram_plan(target_system=target_system, input_text=input_text, target_items=planner_target_items) if _map_type_to_stage(i.type) == stage_norm]
        _apply_phase_plan(stage=stage_norm, items=fallback, reset_statuses=reset_statuses, parent_span_id=span, source="fallback")

    def _expire_phase_views(*, stage: str, reason: str, parent_span_id: Optional[str]) -> list[str]:
        stage_norm = _map_type_to_stage(stage)
        expired_dir = run_dir / "diagrams" / "expired"
        expired_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        expired: list[str] = []

        for item in _phase_items(stage_norm):
            status = state.status_by_id.get(item.diagram_id)
            if status not in {"done", "failed", "in_progress"}:
                continue

            meta = index.get(item.diagram_id) if isinstance(index.get(item.diagram_id), dict) else None
            moved_path: Optional[Path] = None
            if isinstance(meta, dict):
                path_raw = meta.get("path")
                if path_raw:
                    path = Path(str(path_raw)).expanduser()
                    if path.is_file():
                        moved_path = expired_dir / f"{_sanitize_id(item.diagram_id)}__{stamp}.sysml"
                        try:
                            path.replace(moved_path)
                        except Exception:
                            moved_path = path

                history = meta.get("history")
                if not isinstance(history, list):
                    history = []
                prior = {k: meta.get(k) for k in meta.keys() if k != "history"}
                prior.update({"status": meta.get("status") or status, "expired_at": _utc_ts(), "reason": reason})
                if moved_path is not None:
                    prior["path"] = str(moved_path)
                history.append(prior)

                meta["history"] = history
                meta["status"] = "expired"
                meta["reason"] = reason
                meta["expired_at"] = _utc_ts()
                if moved_path is not None:
                    meta["path"] = str(moved_path)
                meta["updated_at"] = _utc_ts()
                index[item.diagram_id] = meta

            state.status_by_id[item.diagram_id] = "expired"
            state.attempts_by_id[item.diagram_id] = 0
            state.last_error_by_id.pop(item.diagram_id, None)
            expired.append(item.diagram_id)

        if expired:
            index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            state.updated_at = _utc_ts()
            state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            events.log(
                event_type="diagram.expire",
                parent_span_id=parent_span_id,
                payload={"stage": stage_norm, "reason": reason, "count": len(expired), "sample": expired[:20]},
            )
        return expired

    def _compress_errors_for_replan(*, req: StageReplanRequested, parent_span_id: Optional[str]) -> str:
        task = _compress_errors_prompt(stage=req.stage, diagram_id=req.diagram_id)
        components = [
            ("Cross Scope", "\n".join(req.cross_scope), False),
            ("Cross Errors (JSON)", json.dumps(req.cross_errors, ensure_ascii=False, indent=2), False),
            ("All Errors (last attempt)", json.dumps(req.last_errors, ensure_ascii=False, indent=2), True),
            ("Warnings (last attempt)", json.dumps(req.last_warnings, ensure_ascii=False, indent=2), True),
        ]
        ctx, ctx_meta = _build_context_with_budget(
            task=task,
            components=components,
            system_prompt=system_prompt,
            token_counter=token_counter,
            max_context_tokens=max_context_tokens,
            reserve_completion_tokens=max_tokens,
            safety_margin_tokens=safety_margin_tokens,
            events=events,
            parent_span_id=parent_span_id,
        )
        try:
            text, _token_payload = llm.chat(
                task=task,
                context=ctx,
                system_prompt=system_prompt,
                parent_span_id=parent_span_id,
                tags={"kind": "errors.compress", "stage": _map_type_to_stage(req.stage), **ctx_meta},
            )
            return (text or "").strip()
        except Exception as e:
            events.log(
                event_type="planner.replan.compress.fail",
                parent_span_id=parent_span_id,
                payload={"stage": _map_type_to_stage(req.stage), "diagram_id": req.diagram_id, "error": str(e)},
            )
            return json.dumps(req.cross_errors, ensure_ascii=False)[:2000]

    def _handle_stage_replan(*, req: StageReplanRequested, parent_span_id: Optional[str]) -> None:
        stage_norm = _map_type_to_stage(req.stage)
        state.replans_by_stage[stage_norm] = int(state.replans_by_stage.get(stage_norm) or 0) + 1
        state.updated_at = _utc_ts()
        state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        events.log(
            event_type="planner.replan.start",
            parent_span_id=parent_span_id,
            payload={
                "stage": stage_norm,
                "diagram_id": req.diagram_id,
                "replans": int(state.replans_by_stage.get(stage_norm) or 0),
                "max_replans": MAX_STAGE_REPLANS_PER_STAGE,
            },
        )

        feedback = _compress_errors_for_replan(req=req, parent_span_id=parent_span_id)
        _expire_phase_views(stage=stage_norm, reason="cross_view_strict", parent_span_id=parent_span_id)
        _ensure_phase_plan(stage=stage_norm, feedback=feedback, force=True, parent_span_id=parent_span_id)

        # Allow the phase items to be processed again within this run.
        for item in _phase_items(stage_norm):
            processed_in_run.discard(item.diagram_id)

        events.log(
            event_type="planner.replan.end",
            parent_span_id=parent_span_id,
            payload={"stage": stage_norm, "diagram_id": req.diagram_id, "items": len(_phase_items(stage_norm))},
        )

    built: list[str] = []
    failed: list[str] = []
    processed_in_run: set[str] = set()
    processed_total = 0
    stage_cursor = 0

    max_to_process = max(0, int(max_diagrams)) if max_diagrams is not None else None
    events.log(
        event_type="queue.init",
        payload={
            "max_diagrams": max_to_process,
            "counts": _status_counts_for_plan(state),
            "phase_targets": dict(phase_targets),
        },
    )

    while True:
        if max_to_process is not None and processed_total >= max_to_process:
            events.log(
                event_type="run.stop",
                payload={"reason": "max_diagrams", "max_diagrams": max_to_process, "processed": processed_total},
            )
            break

        if stage_cursor >= len(PHASE_SEQUENCE):
            break

        active_stage = PHASE_SEQUENCE[stage_cursor]
        stage_target = int(phase_targets.get(active_stage) or 0)

        if not _phase_active_items(active_stage):
            if stage_target <= 0:
                stage_cursor += 1
                continue
            _ensure_phase_plan(stage=active_stage, feedback="", force=False, parent_span_id=None)
            continue

        next_item: Optional[DiagramPlanItem] = None
        # Prefer fresh "pending" items; retry "failed" items only if no pending is ready.
        for candidate in state.plan:
            if _map_type_to_stage(candidate.type) != active_stage:
                continue
            if candidate.diagram_id in processed_in_run:
                continue
            if state.status_by_id.get(candidate.diagram_id) != "pending":
                continue
            if deps_satisfied(candidate):
                next_item = candidate
                break
        if next_item is None:
            for candidate in state.plan:
                if _map_type_to_stage(candidate.type) != active_stage:
                    continue
                if candidate.diagram_id in processed_in_run:
                    continue
                if state.status_by_id.get(candidate.diagram_id) != "failed":
                    continue
                if deps_satisfied(candidate):
                    next_item = candidate
                    break

        if next_item is None:
            blocked_ids = _phase_blocked_ids(active_stage)
            if blocked_ids:
                events.log(
                    event_type="stage.blocked",
                    payload={"stage": active_stage, "blocked": len(blocked_ids), "sample": blocked_ids[:20]},
                )
                break
            stage_cursor += 1
            continue

        item = next_item
        processed_total += 1
        processed_in_run.add(item.diagram_id)
        diag_span = events.log(
            event_type="context.build",
            payload={
                "kind": "diagram.start",
                "diagram_id": item.diagram_id,
                "type": item.type,
                "title": item.title,
                "processed": processed_total,
                "counts": _status_counts_for_plan(state),
            },
        )
        prev_status = state.status_by_id.get(item.diagram_id)
        state.status_by_id[item.diagram_id] = "in_progress"
        state.updated_at = _utc_ts()
        state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        events.log(
            event_type="diagram.status",
            parent_span_id=diag_span,
            payload={"diagram_id": item.diagram_id, "from": prev_status, "to": "in_progress"},
        )

        # Dependency summaries as compact memory.
        dep_summaries: list[str] = []
        for dep in item.dependencies:
            meta = index.get(dep) or {}
            summary = meta.get("summary")
            if summary:
                dep_summaries.append(f"- {dep}: {json.dumps(summary, ensure_ascii=False)}")
        dep_summary_text = "\n".join(dep_summaries).strip()
        events.log(
            event_type="diagram.deps.read",
            parent_span_id=diag_span,
            payload={
                "diagram_id": item.diagram_id,
                "deps": list(item.dependencies),
                "summaries": len(dep_summaries),
            },
        )

        stage = _map_type_to_stage(item.type)
        rag_context = _try_build_rag_context(
            stage=stage,
            task=item.goal or item.title,
            rag_mode=rag_mode_norm,
            local_domain_knowledge=local_domain_knowledge,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            examples_collection=examples_collection,
            knowledge_collection=knowledge_collection,
            examples_top_k=1,
            domain_top_k=2,
            events=events,
            parent_span_id=diag_span,
        )

        attempt = 0
        attempt_history: list[dict[str, Any]] = []
        last_errors: list[Any] = []
        last_warnings: list[Any] = []
        last_response_text = ""
        sysml_code = ""
        last_cross_errors: list[Any] = []
        last_cross_scope_ids: list[str] = []
        item_replan_request: Optional[StageReplanRequested] = None

        while attempt <= builder_max_retries:
            attempt += 1
            previous_sysml_code = sysml_code
            events.log(
                event_type="diagram.attempt.start",
                parent_span_id=diag_span,
                payload={
                    "diagram_id": item.diagram_id,
                    "attempt": attempt,
                    "max_attempts": int(builder_max_retries) + 1,
                },
            )
            state.attempts_by_id[item.diagram_id] = int(state.attempts_by_id.get(item.diagram_id) or 0) + 1

            base_task = _diagram_task_prompt(item)
            if attempt > 1:
                history_prompt = _retry_failure_history_prompt(
                    diagram_id=item.diagram_id,
                    next_attempt=attempt,
                    history=attempt_history,
                )
                if history_prompt:
                    base_task += "\n\n" + history_prompt + "\n"

            components: list[tuple[str, str, bool]] = [
                ("Target System", target_system, False),
                ("Input Specs", input_text, True),
                ("Dependency Summaries", dep_summary_text, False),
                ("Previous Attempt SysML", _retry_previous_sysml_prompt(previous_sysml=previous_sysml_code), False),
                ("RAG/Knowledge", rag_context, True),
            ]
            context, ctx_meta = _build_context_with_budget(
                task=base_task,
                components=components,
                system_prompt=system_prompt,
                token_counter=token_counter,
                max_context_tokens=max_context_tokens,
                reserve_completion_tokens=max_tokens,
                safety_margin_tokens=safety_margin_tokens,
                events=events,
                parent_span_id=diag_span,
            )

            try:
                last_response_text, _token_payload = llm.chat(
                    task=base_task,
                    context=context,
                    system_prompt=system_prompt,
                    parent_span_id=diag_span,
                    tags={
                        "kind": "diagram.generate",
                        "diagram_id": item.diagram_id,
                        "type": item.type,
                        "attempt": attempt,
                        **ctx_meta,
                    },
                )
            except Exception as e:
                state.last_error_by_id[item.diagram_id] = str(e)
                last_errors = [{"error": str(e)}]
                last_warnings = []
                last_cross_errors = []
                last_cross_scope_ids = []
                attempt_history.append(
                    {
                        "attempt": attempt,
                        "errors": list(last_errors),
                        "warnings": list(last_warnings),
                        "cross_errors": list(last_cross_errors),
                        "cross_scope": list(last_cross_scope_ids),
                        "sysml_code": "",
                    }
                )
                events.log(
                    event_type="llm.call",
                    parent_span_id=diag_span,
                    payload={
                        "diagram_id": item.diagram_id,
                        "attempt": attempt,
                        "error": str(e),
                        "prompt": {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": base_task + ("\n\n" + context if context else "")},
                            ]
                        },
                        "response": None,
                    },
                )
                events.log(
                    event_type="diagram.attempt.end",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "attempt": attempt, "success": False, "error": str(e)},
                )
                continue

            sysml_code = extract_sysml_code(last_response_text)
            events.log(
                event_type="sysml.extract_code",
                parent_span_id=diag_span,
                payload={"diagram_id": item.diagram_id, "bytes": len(sysml_code)},
            )

            # Reset cross-view state for this attempt (avoid leaking from prior attempts).
            last_cross_errors = []
            last_cross_scope_ids = []

            # Parse tree + rule validation.
            packages: list[Any] = []
            summary: dict[str, Any] = {}
            parse_error: Optional[str] = None
            if not sysml_code.strip():
                parse_error = "no SysML code extracted (empty ```sysml``` block)"
                summary = {"error": parse_error}
                events.log(
                    event_type="sysml.parse_tree",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "error": parse_error},
                )
            else:
                try:
                    from ..sysml.package_tree import parse_packages

                    package_dict: dict = {}
                    packages = parse_packages(sysml_code, package_dict)
                    summary = _summarize_packages(packages)
                    events.log(
                        event_type="sysml.parse_tree",
                        parent_span_id=diag_span,
                        payload={"diagram_id": item.diagram_id, "summary": summary},
                    )
                except Exception as e:
                    parse_error = str(e)
                    packages = []
                    summary = {"error": parse_error}
                    events.log(
                        event_type="sysml.parse_tree",
                        parent_span_id=diag_span,
                        payload={"diagram_id": item.diagram_id, "error": parse_error},
                    )

            last_errors = []
            last_warnings = []
            rule_summary: Optional[dict[str, Any]] = None

            if parse_error is not None:
                last_errors = [{"error": f"sysml.parse_tree failed: {parse_error}"}]
            elif not packages:
                last_errors = [{"error": "sysml.parse_tree returned no packages (output must define at least 1 package)"}]
            elif validate_rules:
                try:
                    from ..verification.engine import Rules

                    bdd_errors: list[Any] = []
                    bdd_warnings: list[Any] = []
                    cross_errors: list[Any] = []
                    cross_warnings: list[Any] = []
                    cross_errors_enforced: list[Any] = []
                    cross_scope = [d for d in dependency_closure(item.diagram_id) if state.status_by_id.get(d) == "done"]
                    cross_scope_ids = cross_scope + [item.diagram_id]

                    if requested_rule_type in {"bdd", "all"}:
                        rules_bdd = Rules(packages)
                        bdd_errors, bdd_warnings = rules_bdd.validate_by_type("bdd")
                        events.log(
                            event_type="rules.validate",
                            parent_span_id=diag_span,
                            payload={
                                "diagram_id": item.diagram_id,
                                "rule_type": "bdd",
                                "rule_type_requested": str(rule_type or ""),
                                "errors": len(bdd_errors),
                                "warnings": len(bdd_warnings),
                            },
                        )

                    if requested_rule_type in {"cross", "all"} and cross_view_mode != "off":
                        from ..sysml.package_tree import parse_packages as parse_packages_for_cross

                        combined: list[Any] = []
                        for dep_id in cross_scope:
                            meta = index.get(dep_id) or {}
                            dep_path_raw = meta.get("path")
                            if not dep_path_raw:
                                continue
                            dep_path = Path(str(dep_path_raw)).expanduser()
                            if not dep_path.is_file():
                                continue
                            dep_text = dep_path.read_text(encoding="utf-8", errors="replace")
                            dep_packages = parse_packages_for_cross(dep_text, {})
                            combined.extend(_flatten_packages(dep_packages))

                        combined.extend(_flatten_packages(packages))
                        if combined:
                            rules_cross = Rules(combined)
                            cross_errors, cross_warnings = rules_cross.validate_by_type("cross")
                            # RD-1/RD-2 are global traceability rules; do not block per-diagram generation retries.
                            cross_errors_enforced = _filter_cross_errors_for_stage(
                                [e for e in cross_errors if not _is_rd_cross_error(e)],
                                stage=stage,
                            )

                        events.log(
                            event_type="rules.validate",
                            parent_span_id=diag_span,
                            payload={
                                "diagram_id": item.diagram_id,
                                "rule_type": "cross",
                                "rule_type_requested": str(rule_type or ""),
                                "cross_view_mode": cross_view_mode,
                                "scope": cross_scope_ids,
                                "errors": len(cross_errors),
                                "errors_enforced": len(cross_errors_enforced),
                                "warnings": len(cross_warnings),
                            },
                        )

                    last_errors = list(bdd_errors)
                    last_warnings = list(bdd_warnings)
                    if cross_errors_enforced and cross_view_mode == "strict":
                        last_errors.extend(cross_errors_enforced)

                    last_cross_errors = list(cross_errors_enforced)
                    last_cross_scope_ids = list(cross_scope_ids)
                    rule_summary = {
                        "requested": requested_rule_type,
                        "cross_view_mode": cross_view_mode,
                        "bdd": {"errors": len(bdd_errors), "warnings": len(bdd_warnings)},
                        "cross": {
                            "errors": len(cross_errors_enforced),
                            "warnings": len(cross_warnings),
                            "errors_total": len(cross_errors),
                            "enforced_stage": stage,
                            "scope": cross_scope_ids,
                            "errors_sample": cross_errors_enforced[:5],
                        },
                    }
                except Exception as e:
                    last_errors = [{"error": str(e)}]
                    last_cross_errors = []
                    last_cross_scope_ids = []
                    events.log(
                        event_type="rules.validate",
                        parent_span_id=diag_span,
                        payload={
                            "diagram_id": item.diagram_id,
                            "rule_type": requested_rule_type,
                            "rule_type_requested": str(rule_type or ""),
                            "error": str(e),
                        },
                    )

            if not last_errors:
                # Success.
                confidence_payload: Optional[dict[str, Any]] = None
                try:
                    from .semantic_entropy import semantic_entropy_confidence

                    entropy_candidates: list[str] = []
                    for entry in attempt_history:
                        if not isinstance(entry, dict):
                            continue
                        code = entry.get("sysml_code")
                        if isinstance(code, str) and code.strip():
                            entropy_candidates.append(code.strip())
                    entropy_candidates.append(sysml_code.strip())

                    target_samples = int(semantic_entropy_samples or 0)
                    if target_samples < 1:
                        target_samples = 1

                    remaining = max(0, target_samples - len(entropy_candidates))
                    for sample_idx in range(remaining):
                        sample_text, _sample_tokens = llm.chat(
                            task=base_task,
                            context=context,
                            system_prompt=system_prompt,
                            parent_span_id=diag_span,
                            temperature=float(semantic_entropy_temperature),
                            tags={
                                "kind": "diagram.semantic_entropy.sample",
                                "diagram_id": item.diagram_id,
                                "type": item.type,
                                "attempt": attempt,
                                "sample_idx": sample_idx,
                                **ctx_meta,
                            },
                        )
                        entropy_candidates.append(extract_sysml_code(sample_text))

                    se = semantic_entropy_confidence(
                        entropy_candidates,
                        similarity_threshold=float(semantic_entropy_similarity_threshold),
                    )
                    confidence_payload = {"method": "semantic_entropy", **se.to_dict()}
                    events.log(
                        event_type="confidence.semantic_entropy",
                        parent_span_id=diag_span,
                        payload={"diagram_id": item.diagram_id, **confidence_payload},
                    )
                except Exception as e:
                    events.log(
                        event_type="confidence.semantic_entropy",
                        parent_span_id=diag_span,
                        payload={"diagram_id": item.diagram_id, "error": str(e)},
                    )

                safe_id = _sanitize_id(item.diagram_id)
                sysml_path = run_dir / "diagrams" / f"{safe_id}.sysml"
                sysml_path.write_text(sysml_code.strip() + "\n", encoding="utf-8")
                sha = _sha256_text(sysml_code.strip())
                events.log(
                    event_type="artifact.write",
                    parent_span_id=diag_span,
                    payload={
                        "diagram_id": item.diagram_id,
                        "path": str(sysml_path),
                        "sha256": sha,
                        "bytes": len(sysml_code.strip()),
                    },
                )
                prior_meta = index.get(item.diagram_id) if isinstance(index.get(item.diagram_id), dict) else {}
                prior_history = prior_meta.get("history") if isinstance(prior_meta, dict) else None
                entry = {
                    "diagram_id": item.diagram_id,
                    "status": "done",
                    "type": item.type,
                    "title": item.title,
                    "goal": item.goal,
                    "dependencies": list(item.dependencies),
                    "path": str(sysml_path),
                    "sha256": sha,
                    "summary": summary,
                    "rule_summary": rule_summary,
                    "confidence": confidence_payload,
                    "updated_at": _utc_ts(),
                }
                if prior_history:
                    entry["history"] = prior_history
                index[item.diagram_id] = entry
                index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                prev = state.status_by_id.get(item.diagram_id)
                state.status_by_id[item.diagram_id] = "done"
                state.updated_at = _utc_ts()
                state.last_error_by_id.pop(item.diagram_id, None)
                state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                built.append(item.diagram_id)
                events.log(
                    event_type="state.save",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "status": "done"},
                )
                events.log(
                    event_type="diagram.status",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "from": prev, "to": "done"},
                )
                events.log(
                    event_type="diagram.attempt.end",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "attempt": attempt, "success": True},
                )
                break

            # Retry on errors.
            attempt_history.append(
                {
                    "attempt": attempt,
                    "errors": list(last_errors),
                    "warnings": list(last_warnings),
                    "cross_errors": list(last_cross_errors),
                    "cross_scope": list(last_cross_scope_ids),
                    "sysml_code": sysml_code.strip(),
                }
            )
            events.log(
                event_type="diagram.attempt.end",
                parent_span_id=diag_span,
                payload={
                    "diagram_id": item.diagram_id,
                    "attempt": attempt,
                    "success": False,
                    "errors": len(last_errors),
                    "warnings": len(last_warnings),
                },
            )
            if attempt > builder_max_retries:
                break

        if state.status_by_id.get(item.diagram_id) != "done":
            if last_cross_errors and cross_view_mode == "strict" and int(state.replans_by_stage.get(stage) or 0) < MAX_STAGE_REPLANS_PER_STAGE:
                events.log(
                    event_type="planner.replan.request",
                    parent_span_id=diag_span,
                    payload={
                        "diagram_id": item.diagram_id,
                        "stage": stage,
                        "cross_errors": len(last_cross_errors),
                        "cross_scope": list(last_cross_scope_ids),
                        "replans_so_far": int(state.replans_by_stage.get(stage) or 0),
                        "max_replans": MAX_STAGE_REPLANS_PER_STAGE,
                    },
                )
                item_replan_request = StageReplanRequested(
                    stage=stage,
                    diagram_id=item.diagram_id,
                    cross_errors=last_cross_errors,
                    cross_scope=last_cross_scope_ids,
                    last_errors=last_errors,
                    last_warnings=last_warnings,
                )

            if item_replan_request is not None:
                # Replanning is handled by the outer loop. Leave this diagram in-progress for now.
                state.last_error_by_id[item.diagram_id] = json.dumps(
                    {"errors": last_errors, "warnings": last_warnings}, ensure_ascii=False
                )[:4000]
                state.updated_at = _utc_ts()
                state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                prev = state.status_by_id.get(item.diagram_id)
                state.status_by_id[item.diagram_id] = "failed"
                state.last_error_by_id[item.diagram_id] = json.dumps(
                    {"errors": last_errors, "warnings": last_warnings}, ensure_ascii=False
                )[:4000]
                state.updated_at = _utc_ts()
                state_path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                failed.append(item.diagram_id)
                events.log(
                    event_type="state.save",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "status": "failed"},
                )
                events.log(
                    event_type="diagram.status",
                    parent_span_id=diag_span,
                    payload={"diagram_id": item.diagram_id, "from": prev, "to": "failed"},
                )

        if item_replan_request is not None:
            _handle_stage_replan(req=item_replan_request, parent_span_id=diag_span)
            continue

    blocked = [
        i.diagram_id
        for i in state.plan
        if (state.status_by_id.get(i.diagram_id) in {"pending", "failed"}) and (not deps_satisfied(i))
    ]
    events.log(
        event_type="queue.done",
        payload={
            "built": len(built),
            "failed": len(failed),
            "blocked": len(blocked),
            "counts": _status_counts_for_plan(state),
        },
    )
    summary = {
        "run_id": chosen_run_id,
        "run_dir": str(run_dir),
        "built": built,
        "failed": failed,
        "blocked": blocked,
        "pending": [i.diagram_id for i in state.plan if state.status_by_id.get(i.diagram_id) == "pending"],
        "in_progress": [i.diagram_id for i in state.plan if state.status_by_id.get(i.diagram_id) == "in_progress"],
        "counts": _status_counts_for_plan(state),
        "processed_in_run": processed_total,
        "max_diagrams": max_to_process,
        "max_context_tokens": max_context_tokens,
        "max_tokens": max_tokens,
        "model": model_name,
        "rag_mode": rag_mode_norm,
    }
    events.log(event_type="artifact.write", payload={"kind": "summary", "summary": summary})
    return summary


__all__ = ["run_long_period", "extract_sysml_code"]
