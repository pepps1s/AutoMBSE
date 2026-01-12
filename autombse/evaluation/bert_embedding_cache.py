from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _cache_dir() -> Path:
    return Path(os.environ.get("AUTOMBSE_CACHE_DIR") or "AutoMBSE/cache").expanduser()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingCacheStats:
    hits: int = 0
    misses: int = 0
    added: int = 0
    loaded: int = 0


class BertCLOSEmbeddingCache:
    """
    CPU-friendly CLS embedding cache for `bert-base-uncased`.

    - Stores L2-normalized CLS embeddings (cosine similarity via dot product).
    - Persists to a pickle under AUTOMBSE_CACHE_DIR.
    """

    def __init__(
        self,
        *,
        model_name: str = "bert-base-uncased",
        cache_path: Optional[Path] = None,
        batch_size: int = 128,
        max_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.cache_path = cache_path or (
            _cache_dir() / f"bert_cls_norm.{_sha256_text(model_name)[:10]}.len{self.max_length}.pkl"
        )
        self._cache: Dict[str, "object"] = {}
        self.stats = EmbeddingCacheStats()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load()

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import BertModel, BertTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("missing optional dependencies for BERT embeddings (torch, transformers)") from e
        self._torch = torch
        self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self._model = BertModel.from_pretrained(self.model_name)
        self._model.eval()

    def _load(self) -> None:
        if not self.cache_path.is_file():
            return
        try:
            with self.cache_path.open("rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                self._cache = obj
                self.stats.loaded = len(self._cache)
        except Exception:
            self._cache = {}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _key(self, text: str) -> str:
        return _sha256_text(text)

    def has(self, text: str) -> bool:
        return self._key(text) in self._cache

    def get(self, text: str):
        key = self._key(text)
        if key in self._cache:
            self.stats.hits += 1
            return self._cache[key]
        self.stats.misses += 1
        return None

    def ensure(self, texts: Iterable[str]) -> None:
        unique: List[str] = []
        seen: set[str] = set()
        for t in texts:
            if not isinstance(t, str):
                continue
            s = t.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            key = self._key(s)
            if key in self._cache:
                continue
            unique.append(s)

        if not unique:
            return

        self._ensure_model()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        torch = self._torch
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            norm = torch.norm(emb, dim=1, keepdim=True).clamp(min=1e-12)
            emb_norm = emb / norm
            for idx, text in enumerate(batch):
                self._cache[self._key(text)] = emb_norm[idx].cpu()
                self.stats.added += 1

    def vectors(self, texts: Sequence[str]):
        self._ensure_model()
        assert self._torch is not None
        torch = self._torch
        vecs = []
        for t in texts:
            v = self.get(t)
            if v is None:
                # Ensure missing is computed (fallback to single).
                self.ensure([t])
                v = self.get(t)
            if v is None:
                v = torch.zeros((768,), dtype=torch.float32)
            vecs.append(v)
        return torch.stack(vecs, dim=0)


__all__ = ["BertCLOSEmbeddingCache", "EmbeddingCacheStats"]
