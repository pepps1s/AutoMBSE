from __future__ import annotations

import json
import io
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class _FakeMaxResult:
    def __init__(self, values):
        self.values = values


class _FakeTensor:
    def __init__(self, arr, np):
        self._arr = arr
        self._np = np

    @property
    def T(self):
        return _FakeTensor(self._arr.T, self._np)

    def __matmul__(self, other):
        return _FakeTensor(self._np.matmul(self._arr, other._arr), self._np)

    def __getitem__(self, item):
        return _FakeTensor(self._arr[item], self._np)

    def __gt__(self, other):
        return _FakeTensor(self._arr > other, self._np)

    def max(self, *, dim: int):
        return _FakeMaxResult(_FakeTensor(self._np.max(self._arr, axis=dim), self._np))

    def sum(self):
        return _FakeTensor(self._np.sum(self._arr), self._np)

    def item(self):
        return self._arr.item()


class _DummyEmbeddingCache:
    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self._cache: dict[str, "object"] = {}
        self.stats = types.SimpleNamespace(added=0)

    def _embed(self, text: str):
        np = self._np
        size = 8
        vec = np.zeros((size,), dtype=np.float32)
        vec[sum(ord(c) for c in text) % size] = 1.0
        return vec

    def ensure(self, texts) -> None:
        for t in texts:
            if not isinstance(t, str):
                continue
            s = t.strip()
            if not s or s in self._cache:
                continue
            self._cache[s] = self._embed(s)
            self.stats.added += 1

    def vectors(self, texts):
        np = self._np
        self.ensure(texts)
        if not texts:
            return _FakeTensor(np.zeros((0, 8), dtype=np.float32), np)
        vecs = []
        for t in texts:
            s = t.strip()
            v = self._cache.get(s)
            if v is None:
                v = self._embed(s)
            vecs.append(v)
        return _FakeTensor(np.stack(vecs, axis=0), np)

    def save(self) -> None:
        return


def _fake_generate_method(method: str, *, description: str, cfg, max_rounds: int = 2):
    _ = description
    _ = cfg
    _ = max_rounds
    text = f"```sysml\npackage Test {{ part {method}; }}\n```"
    return text, 0.01


def _fake_resolve_llm_client_config(config):
    _ = config
    return object()


class TestEvalIncrementalGeneration(unittest.TestCase):
    def test_generate_missing_methods_not_in_input_and_resume(self) -> None:
        from autombse.cli.commands.eval import eval_views_cmd

        with TemporaryDirectory() as td:
            tmp_dir = Path(td)
            res_path = tmp_dir / "res.json"
            examples = [
                {"description": "d1", "code": "package P1 { part A; }"},
                {"description": "d2", "code": "package P2 { part B; }"},
            ]
            res_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            args = types.SimpleNamespace(
                input=str(res_path),
                method=["MBSE_wo_cache", "MBSE_wo_rules"],
                threshold=0.9,
                update=True,
                generate_missing=True,
                regenerate=False,
                force=False,
            )

            with patch("autombse.cli.commands.eval.BertCLOSEmbeddingCache", _DummyEmbeddingCache), \
                patch("autombse.evaluation.resjson_generators.resolve_llm_client_config", _fake_resolve_llm_client_config), \
                patch("autombse.evaluation.resjson_generators.generate_method", side_effect=_fake_generate_method) as gen_mock:
                with redirect_stdout(io.StringIO()):
                    rc = eval_views_cmd(args=args, config={}, repo_root=tmp_dir)
                self.assertEqual(rc, 0)
                self.assertEqual(gen_mock.call_count, 4)

            updated = json.loads(res_path.read_text(encoding="utf-8"))
            for ex in updated:
                for m in ("MBSE_wo_cache", "MBSE_wo_rules"):
                    self.assertIn(m, ex)
                    self.assertIn(f"{m}_time", ex)
                    self.assertIn(f"{m}_similarity", ex)

            args2 = types.SimpleNamespace(
                input=str(res_path),
                method=["MBSE_wo_cache", "MBSE_wo_rules"],
                threshold=0.9,
                update=True,
                generate_missing=True,
                regenerate=False,
                force=False,
            )

            def _should_not_run(*_a, **_kw):
                raise AssertionError("generate_method should not be called when outputs exist")

            with patch("autombse.cli.commands.eval.BertCLOSEmbeddingCache", _DummyEmbeddingCache), \
                patch("autombse.evaluation.resjson_generators.resolve_llm_client_config", _fake_resolve_llm_client_config), \
                patch("autombse.evaluation.resjson_generators.generate_method", _should_not_run):
                with redirect_stdout(io.StringIO()):
                    rc2 = eval_views_cmd(args=args2, config={}, repo_root=tmp_dir)
                self.assertEqual(rc2, 0)
