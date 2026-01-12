from __future__ import annotations

from typing import Any, Optional


class BERTStringSimilarity:
    def __init__(self):
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._torch: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return
        try:
            from transformers import BertTokenizer, BertModel  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("missing optional dependencies for BERT similarity (torch, transformers)") from e

        self._torch = torch
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._model = BertModel.from_pretrained("bert-base-uncased")

    def get_bert_embedding(self, text):
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return cls_embedding

    def calculate_similarity(self, s1, s2):
        from scipy.spatial.distance import cosine

        embedding1 = self.get_bert_embedding(s1)
        embedding2 = self.get_bert_embedding(s2)
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity


__all__ = ["BERTStringSimilarity"]

