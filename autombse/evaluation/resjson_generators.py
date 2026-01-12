from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def _default_system_prompt() -> str:
    return (
        "As a systems engineer working on complex projects, I am looking to adopt the SysML v2 methodology to enhance our "
        "Model-Based Systems Engineering (MBSE) capabilities. Please provide detailed SysML v2 code for creating robust MBSE "
        "models."
    )


@dataclass(frozen=True)
class LLMClientConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str


def resolve_llm_client_config(config: dict[str, Any]) -> LLMClientConfig:
    llm_cfg = config.get("llm") or {}

    api_key = llm_cfg.get("api_key") or os.environ.get("AUTOMBSE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("missing LLM api key (set llm.api_key or OPENAI_API_KEY/AUTOMBSE_API_KEY)")

    base_url = llm_cfg.get("base_url") or os.environ.get("AUTOMBSE_BASE_URL")
    model = llm_cfg.get("model") or os.environ.get("AUTOMBSE_MODEL")
    if not model:
        raise RuntimeError("missing LLM model id (set llm.model or AUTOMBSE_MODEL)")

    max_tokens = int(llm_cfg.get("max_tokens") or 2048)
    temperature = float(llm_cfg.get("temperature") or 0.2)
    system_prompt = str(llm_cfg.get("system_prompt") or _default_system_prompt()).strip()

    return LLMClientConfig(
        api_key=str(api_key),
        base_url=str(base_url).strip() if base_url else None,
        model=str(model),
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )


def _openai_client(cfg: LLMClientConfig):
    from openai import OpenAI

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url) if cfg.base_url else OpenAI(api_key=cfg.api_key)


def _extract_json_blob(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1].strip()
    return text.strip()


def _generate_sysml_once(cfg: LLMClientConfig, *, description: str, temperature: Optional[float] = None) -> str:
    client = _openai_client(cfg)
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {
            "role": "user",
            "content": (
                f"{description}\n\n"
                "Output exactly one SysML v2 model as a fenced code block:\n"
                "```sysml\n"
                "<your SysML v2 code>\n"
                "```\n"
                "Do not output anything outside the code block."
            ),
        },
    ]
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature if temperature is None else float(temperature),
        stream=False,
    )
    return completion.choices[0].message.content or ""


def _judge_sysml(cfg: LLMClientConfig, *, description: str, candidate: str) -> dict[str, Any]:
    client = _openai_client(cfg)
    judge_system = (
        "You are a strict SysML v2 reviewer. Your task is to judge whether the candidate SysML model satisfies the "
        "natural-language requirement description and follows basic SysML v2 modeling discipline.\n"
        "You must respond with JSON only."
    )
    messages = [
        {"role": "system", "content": judge_system},
        {
            "role": "user",
            "content": (
                f"Requirement description:\n{description}\n\n"
                "Candidate SysML:\n"
                f"{candidate}\n\n"
                "Return JSON with keys:\n"
                '- \"pass\": boolean\n'
                '- \"issues\": array of short strings (empty if pass)\n'
                "- \"fix_hint\": short instruction for the generator\n"
            ),
        },
    ]
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=min(512, cfg.max_tokens),
        temperature=0.0,
        stream=False,
    )
    text = completion.choices[0].message.content or ""
    blob = _extract_json_blob(text)
    try:
        obj = json.loads(blob)
    except Exception:
        return {"pass": False, "issues": ["judge returned non-JSON"], "fix_hint": "Regenerate SysML v2 as requested."}
    if not isinstance(obj, dict):
        return {"pass": False, "issues": ["judge returned non-object JSON"], "fix_hint": "Regenerate SysML v2 as requested."}
    return obj


def generate_method(
    method: str,
    *,
    description: str,
    cfg: LLMClientConfig,
    max_rounds: int = 2,
) -> Tuple[str, float]:
    """
    Generate a method output text for one `res.json` example.

    Returns (text, elapsed_seconds).
    """

    start = time.time()
    if method == "MBSE_wo_cache":
        text = _generate_sysml_once(cfg, description=description)
        return text, time.time() - start

    if method == "MBSE_wo_rules":
        text = _generate_sysml_once(cfg, description=description)
        for _ in range(max(1, int(max_rounds))):
            verdict = _judge_sysml(cfg, description=description, candidate=text)
            if bool(verdict.get("pass")):
                return text, time.time() - start

            issues = verdict.get("issues") if isinstance(verdict, dict) else None
            fix_hint = str(verdict.get("fix_hint") or "").strip() if isinstance(verdict, dict) else ""
            issues_text = ""
            if isinstance(issues, list):
                issues_text = "\n".join([f"- {str(i).strip()}" for i in issues if str(i).strip()][:12])

            issues_block = f"Issues:\n{issues_text}\n" if issues_text else ""
            fix_block = f"Fix hint: {fix_hint}\n" if fix_hint else ""
            prompt = (
                f"{description}\n\n"
                "The previous candidate failed review. Fix it with minimal changes.\n"
                f"{issues_block}"
                f"{fix_block}"
                "Output exactly one SysML v2 model as a fenced code block:\n"
                "```sysml\n"
                "<your SysML v2 code>\n"
                "```\n"
                "Do not output anything outside the code block."
            )
            client = _openai_client(cfg)
            completion = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                stream=False,
            )
            text = completion.choices[0].message.content or text

        return text, time.time() - start

    raise KeyError(f"unsupported method generator: {method}")


__all__ = ["LLMClientConfig", "generate_method", "resolve_llm_client_config"]
