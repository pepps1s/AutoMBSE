from __future__ import annotations

from typing import Any, List


class searchVec:
    """
    Legacy-compatible Qdrant retriever using BERT embeddings.

    This class intentionally mirrors the historical implementation to ease migration.
    """

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        try:
            from transformers import BertTokenizer, BertModel  # type: ignore
            import torch  # type: ignore
            from qdrant_client import QdrantClient  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "missing optional dependencies for RAG (install torch, transformers, qdrant-client)"
            ) from e

        self._torch = torch
        self.client = QdrantClient(host, port=port)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer_ch = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model_ch = BertModel.from_pretrained("bert-base-chinese")

    def encode_input(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector

    def encode_input_chinese(self, input_text):
        inputs = self.tokenizer_ch(input_text, return_tensors="pt", truncation=True, padding=True)
        with self._torch.no_grad():
            outputs = self.model_ch(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return vector

    def search_top_k(self, input_vector: list, k: int, collection_name: str) -> list:
        search_result = self.client.search(collection_name=collection_name, query_vector=input_vector, limit=k)

        top_k_results: List[Any] = []
        for result in search_result:
            res = result.payload
            res["score"] = result.score
            top_k_results.append(res)

        return top_k_results

    def find_top_k_similar(self, input_text, k: int = 3, collection: str = "collection", chinese: bool = False):
        if chinese:
            input_vector = self.encode_input_chinese(input_text)
        else:
            input_vector = self.encode_input(input_text)
        top_k_results = self.search_top_k(input_vector, k, collection_name=collection)
        return top_k_results
