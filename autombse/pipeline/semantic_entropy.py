from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class SemanticEntropyResult:
    samples: int
    clusters: int
    cluster_sizes: List[int]
    entropy_bits: float
    normalized_entropy: float
    confidence: float
    similarity_threshold: float
    signature_kind: str
    invalid_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _cluster_signatures(signatures: Sequence[Set[str]], *, threshold: float) -> List[List[int]]:
    clusters: List[List[int]] = []
    representatives: List[Set[str]] = []

    for idx, sig in enumerate(signatures):
        best = -1
        best_sim = -1.0
        for c_idx, rep in enumerate(representatives):
            sim = _jaccard(sig, rep)
            if sim > best_sim:
                best_sim = sim
                best = c_idx
        if best >= 0 and best_sim >= threshold:
            clusters[best].append(idx)
        else:
            clusters.append([idx])
            representatives.append(sig)
    return clusters


def _shannon_entropy_bits(cluster_sizes: Sequence[int]) -> float:
    total = sum(int(s) for s in cluster_sizes)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for size in cluster_sizes:
        if size <= 0:
            continue
        p = float(size) / float(total)
        entropy -= p * math.log2(p)
    return entropy


def _normalize_entropy(entropy_bits: float, *, n_samples: int) -> float:
    if n_samples <= 1:
        return 0.0
    denom = math.log2(float(n_samples))
    if denom <= 0:
        return 0.0
    return float(entropy_bits) / float(denom)


def _clip01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def _extract_sysml_signature(sysml_code: str) -> Tuple[Set[str], bool]:
    """
    Extract a structural "semantic signature" from SysML-v2 code.

    Signature definition:
    - parse SysML into the project Package tree (autombse.sysml.package_tree.parse_packages)
    - flatten into a set of element-level facts: (type, name, belongsto)

    Returns (signature, ok).
    """

    code = (sysml_code or "").strip()
    if not code:
        return set(), False

    try:
        from ..sysml.package_tree import Package, parse_packages

        package_dict: dict = {}
        packages = parse_packages(code, package_dict)
    except Exception:
        return set(), False

    items: Set[str] = set()

    def walk(node: "Package") -> None:
        t = str(getattr(node, "type", "") or "").strip()
        name = str(getattr(node, "name", "") or "").strip()
        belongsto = str(getattr(node, "belongsto", "") or "").strip()
        if t or name or belongsto:
            items.add(f"{t}|{name}|{belongsto}")
        for child in getattr(node, "children", []) or []:
            walk(child)

    for pkg in packages:
        if pkg is None:
            continue
        walk(pkg)

    return items, True


def semantic_entropy_confidence(
    sysml_candidates: Sequence[str],
    *,
    similarity_threshold: float = 0.85,
    signature_kind: str = "sysml.package_tree.signature(type|name|belongsto)",
) -> SemanticEntropyResult:
    """
    Compute confidence from "semantic entropy" over multiple candidate SysML generations.

    Procedure:
    1) Convert each candidate into a semantic signature set S_i.
    2) Cluster candidates by Jaccard(S_i, S_rep) >= threshold.
    3) Let p_k be the fraction of samples in cluster k, compute entropy H = -sum p_k log2 p_k.
    4) Normalize by log2(N): H_norm = H / log2(N), and define confidence = 1 - H_norm.
    """

    threshold = float(similarity_threshold)
    signatures: List[Set[str]] = []
    invalid = 0
    for candidate in sysml_candidates:
        sig, ok = _extract_sysml_signature(candidate)
        signatures.append(sig)
        if not ok:
            invalid += 1

    clusters = _cluster_signatures(signatures, threshold=threshold)
    sizes = [len(c) for c in clusters]
    n = len(signatures)
    entropy_bits = _shannon_entropy_bits(sizes)
    norm = _normalize_entropy(entropy_bits, n_samples=n)
    confidence = _clip01(1.0 - norm)

    return SemanticEntropyResult(
        samples=n,
        clusters=len(sizes),
        cluster_sizes=sizes,
        entropy_bits=float(entropy_bits),
        normalized_entropy=float(norm),
        confidence=float(confidence),
        similarity_threshold=threshold,
        signature_kind=str(signature_kind),
        invalid_samples=int(invalid),
    )


__all__ = ["SemanticEntropyResult", "semantic_entropy_confidence"]

