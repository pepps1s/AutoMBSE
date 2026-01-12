from __future__ import annotations

__all__ = ["BERTStringSimilarity", "exampleComparison"]


def __getattr__(name: str):  # pragma: no cover
    if name == "BERTStringSimilarity":
        from .bert_string_similarity import BERTStringSimilarity

        return BERTStringSimilarity
    if name == "exampleComparison":
        from .metrics_parts import exampleComparison

        return exampleComparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
