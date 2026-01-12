from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager
from unittest.mock import patch

from autombse.pipeline.long_period import EventLogger, LLMClient, RunState, TokenCounter, TokenLedger


class _FakeUsage:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class _FakeOpenAICompletion:
    def __init__(self, *, content: str, usage: dict[str, Any]) -> None:
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        self.choices = [choice]
        self.usage = _FakeUsage(usage)


class _FakeOpenAI:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        completions = types.SimpleNamespace(create=self._create)
        self.chat = types.SimpleNamespace(completions=completions)

    def _create(self, *_args: Any, **_kwargs: Any) -> _FakeOpenAICompletion:
        return _FakeOpenAICompletion(
            content="FAKE_RESPONSE",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        )


def _patch_openai() -> ContextManager[None]:
    if "openai" not in sys.modules:
        try:
            import openai  # noqa: F401
        except ModuleNotFoundError:
            fake_module = types.ModuleType("openai")
            fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
            sys.modules["openai"] = fake_module
            return nullcontext()
    return patch("openai.OpenAI", _FakeOpenAI)


class TestLongPeriodEventLlmPromptResponse(unittest.TestCase):
    def test_llm_call_logs_prompt_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, _patch_openai():
            tmp_path = Path(tmpdir)
            events = EventLogger(path=tmp_path / "events.jsonl")
            ledger = TokenLedger(csv_path=tmp_path / "token_trace.csv")
            token_counter = TokenCounter(model="test-model")
            token_counter._encoder = None  # force deterministic fallback
            state = RunState(llm_round=0)

            llm = LLMClient(
                api_key="test",
                base_url=None,
                model="test-model",
                temperature=0.0,
                max_tokens=16,
                replay=None,
                events=events,
                ledger=ledger,
                token_counter=token_counter,
                budget_total=10_000,
                safety_margin_tokens=0,
                state=state,
            )

            response_text, _token_payload = llm.chat(
                task="TASK",
                context="CTX",
                system_prompt="SYS",
                parent_span_id=None,
                tags={"kind": "test"},
            )
            self.assertEqual(response_text, "FAKE_RESPONSE")

            lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj.get("event_type"), "llm.call")
            payload = obj.get("payload") or {}
            self.assertEqual(payload.get("response"), "FAKE_RESPONSE")

            prompt = payload.get("prompt") or {}
            messages = prompt.get("messages") or []
            self.assertEqual(
                messages,
                [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "TASK\n\nCTX"},
                ],
            )


if __name__ == "__main__":
    unittest.main()

