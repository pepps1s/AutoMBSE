from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .bert_embedding_cache import BertCLOSEmbeddingCache


PartElementExtractor = Callable[[str], List[str]]


def _unique_stable(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_element_text(text: str, *, max_chars: int = 1200) -> str:
    """
    Normalize an element fragment for embedding/similarity.

    - collapse whitespace for stability
    - truncate very long fragments (keeps embedding cache bounded)
    """

    raw = (text or "").strip()
    if not raw:
        return ""
    collapsed = re.sub(r"\\s+", " ", raw)
    if len(collapsed) > max_chars:
        collapsed = collapsed[: max_chars - 1] + "…"
    return collapsed.strip()


def part_elements_strict(text: str) -> List[str]:
    """
    Default element extractor for view-level element metrics.

    Uses `autombse.sysml.parts.partComponentDepose`, which extracts `part ... ;` and
    `part ... { ... }` fragments.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    in_code = re.sub(r"//.*?$|/\\*.*?\\*/", "", text, flags=re.MULTILINE)
    pattern = r"part(.*?)(;|{)"
    matches = re.finditer(pattern, in_code, re.DOTALL)

    results: List[str] = []
    for match in matches:
        delimiter = match.group(2)
        if delimiter == ";":
            end_index = match.end() - 1
            results.append(in_code[match.start() : end_index + 1].strip())
            continue

        if delimiter != "{":
            continue

        brace_count = 1
        end_index = match.end()
        while brace_count > 0 and end_index < len(in_code):
            next_brace = in_code.find("{", end_index + 1)
            next_close_brace = in_code.find("}", end_index + 1)

            if next_close_brace == -1:
                break

            if next_brace != -1 and next_brace < next_close_brace:
                brace_count += 1
                end_index = next_brace
            else:
                brace_count -= 1
                end_index = next_close_brace

        if brace_count == 0:
            results.append(in_code[match.start() : end_index + 1].strip())

    return results


def part_elements_legacy_semicolon(text: str) -> List[str]:
    """
    Compatibility extractor that mirrors a historical `partComponentDepose` behavior.

    For `part ... ;` matches, it captures until the *next* semicolon after the match.
    This can merge adjacent statements into one fragment and is kept only for
    reproducing past evaluation numbers.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    in_code = re.sub(r"//.*?$|/\\*.*?\\*/", "", text, flags=re.MULTILINE)
    pattern = r"part(.*?)(;|{)"
    matches = re.finditer(pattern, in_code, re.DOTALL)

    results: List[str] = []
    for match in matches:
        delimiter = match.group(2)
        if delimiter == ";":
            end_index = in_code.find(";", match.end())
            if end_index != -1:
                results.append(in_code[match.start() : end_index + 1].strip())
            continue

        if delimiter != "{":
            continue

        brace_count = 1
        end_index = match.end()
        while brace_count > 0 and end_index < len(in_code):
            next_brace = in_code.find("{", end_index + 1)
            next_close_brace = in_code.find("}", end_index + 1)

            if next_close_brace == -1:
                break

            if next_brace != -1 and next_brace < next_close_brace:
                brace_count += 1
                end_index = next_brace
            else:
                brace_count -= 1
                end_index = next_close_brace

        if brace_count == 0:
            results.append(in_code[match.start() : end_index + 1].strip())

    return results


def extract_sysml_fenced(text: str) -> str:
    """
    Extract SysML fenced blocks for BLEU-style metrics.

    Note: This intentionally requires a trailing newline before the closing fence
    to keep consistency with existing *_similarity fields in res.json.
    """

    blocks = re.findall(r"```sysml\n(.*?)\n```", text or "", re.DOTALL | re.IGNORECASE)
    if not blocks:
        return ""
    return "\n".join(blocks)


def _get_ngrams(text: str, n: int) -> List[Tuple[str, ...]]:
    words = (text or "").split()
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def bleu_score(
    candidate: str,
    reference: str,
    *,
    ngram_weights: Sequence[float] = (0.7, 0.2, 0.1),
    max_ngram: int = 3,
) -> float:
    """
    Lightweight BLEU variant used by this repo (tokenized by whitespace).
    """

    cand = candidate or ""
    ref = reference or ""
    if not cand.strip() or not ref.strip():
        return 0.0

    def precision_n(n: int) -> float:
        cand_ngrams = _get_ngrams(cand, n)
        if not cand_ngrams:
            return 0.0
        ref_ngrams = set(_get_ngrams(ref, n))
        matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
        return float(matches) / float(len(cand_ngrams))

    def brevity_penalty() -> float:
        c = len(cand.split())
        r = len(ref.split())
        if c <= 0:
            return 0.0
        if c > r:
            return 1.0
        return math.exp(1 - float(r) / float(c))

    precisions: List[float] = []
    for n in range(1, max_ngram + 1):
        precisions.append(precision_n(n))

    bp = brevity_penalty()
    weighted = 0.0
    for w, p in zip(list(ngram_weights)[:max_ngram], precisions):
        if p <= 0:
            return 0.0
        weighted += float(w) * math.log(p)
    return float(bp) * math.exp(weighted)


@dataclass(frozen=True)
class ElementPRF:
    recall: float
    precision: float
    f1: float
    valid_examples: int
    invalid_examples: int


def element_prf_semantic(
    examples: Sequence[dict[str, Any]],
    *,
    method: str,
    threshold: float,
    embed_cache: BertCLOSEmbeddingCache,
    element_extractor: PartElementExtractor = part_elements_strict,
) -> ElementPRF:
    """
    Element-level P/R/F1 using BERT CLS cosine similarity + a threshold.

    This mirrors the intent of `exampleComparison.partCompare_sematicSimilarity`,
    but accelerates by caching embeddings and computing cosine similarities in
    vectorized batches.
    """

    thr = float(threshold)
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    invalid = 0

    prepared: List[Tuple[List[str], List[str]]] = []
    texts_to_embed: List[str] = []

    for example in examples:
        if not isinstance(example, dict):
            invalid += 1
            continue
        pred_raw = example.get(method)
        gt_raw = example.get("code")
        if not isinstance(pred_raw, str) or not isinstance(gt_raw, str):
            invalid += 1
            continue

        predicted_raw = element_extractor(pred_raw)
        ground_truth_raw = element_extractor(gt_raw)
        predicted = sorted(set([_normalize_element_text(p) for p in predicted_raw if p.strip()]))
        ground_truth = sorted(set([_normalize_element_text(p) for p in ground_truth_raw if p.strip()]))
        if not predicted or not ground_truth:
            invalid += 1
            continue

        prepared.append((predicted, ground_truth))
        texts_to_embed.extend(predicted)
        texts_to_embed.extend(ground_truth)

    if not prepared:
        return ElementPRF(recall=0.0, precision=0.0, f1=0.0, valid_examples=0, invalid_examples=invalid)

    # Pre-warm embeddings in large batches (avoid per-example model calls).
    embed_cache.ensure(texts_to_embed)

    for predicted, ground_truth in prepared:
        pred_vec = embed_cache.vectors(predicted)
        gt_vec = embed_cache.vectors(ground_truth)

        sims = pred_vec @ gt_vec.T

        max_sims = sims.max(dim=1).values
        true_positives = int((max_sims > thr).sum().item())
        false_positives = max(0, len(predicted) - true_positives)
        false_negatives = max(0, len(ground_truth) - true_positives) if true_positives < len(ground_truth) else 0

        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        total_recall += recall
        total_precision += precision
        total_f1 += f1

    valid = len(prepared)
    return ElementPRF(
        recall=total_recall / valid,
        precision=total_precision / valid,
        f1=total_f1 / valid,
        valid_examples=valid,
        invalid_examples=invalid,
    )


def avg_semantic_similarity(examples: Sequence[dict[str, Any]], *, method: str, ignore_zero: bool = True) -> float:
    key = f"{method}_similarity"
    values: List[float] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        v = ex.get(key)
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if ignore_zero and fv == 0.0:
            continue
        values.append(fv)
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def avg_elements_per_example(
    examples: Sequence[dict[str, Any]],
    *,
    method: str,
    element_extractor: PartElementExtractor = part_elements_strict,
) -> float:
    total = 0
    n = 0
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        v = ex.get(method)
        if not isinstance(v, str):
            continue
        fragments = element_extractor(v)
        total += len([p for p in fragments if isinstance(p, str) and p.strip()])
        n += 1
    if n <= 0:
        return 0.0
    return float(total) / float(n)


__all__ = [
    "ElementPRF",
    "PartElementExtractor",
    "avg_elements_per_example",
    "avg_semantic_similarity",
    "bleu_score",
    "element_prf_semantic",
    "extract_sysml_fenced",
    "part_elements_legacy_semicolon",
    "part_elements_strict",
]
