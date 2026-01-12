from __future__ import annotations

from typing import Optional

from .retriever import searchVec


def bert_embedding(input_text):
    try:
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("missing optional dependencies for BERT embedding (torch, transformers)") from e

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def search_similar_chunks(input_text: str, k: int):
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("missing optional dependency qdrant-client") from e

    embedding = bert_embedding(input_text)

    client = QdrantClient(host="localhost", port=6333)
    search_results = client.search(
        collection_name="<collection_name>",
        query_vector=embedding.flatten().tolist(),
        query_filter=None,
        limit=k,
        score_threshold=None,
        with_payload=True,
    )
    return [result.payload["chunk_content"] for result in search_results]


def domainKnowledge(input: str) -> str:
    return domainKnowledge_with_params(input)


def domainKnowledge_with_params(
    input: str,
    *,
    k: int = 2,
    collection: str = "knowledge_chunk",
    retriever: Optional[searchVec] = None,
) -> str:
    searchVec_util = retriever or searchVec()
    res = searchVec_util.find_top_k_similar(input, k, collection, True)
    return (
        """Waterjet propulsion is a special propulsion method that uses the reaction force produced by the momentum difference between the ingested and ejected water flow to propel ships or offshore platforms.
A typical waterjet propulsion device usually includes a waterjet propulsor composed of an inlet duct, impeller, stator, nozzle, shafting, etc., and a steering/reversing mechanism composed of a steering nozzle/deflector and a reverse bucket, along with control and hydraulic systems to operate the propulsor and actuate the mechanisms.
The working principle is that the rotating impeller creates a pressure difference, draws water into the pump, accelerates it and ejects it aft; the momentum difference between inlet and outlet flow generates thrust to move the ship forward. The steering/reversing mechanism changes the jet direction by swinging and opening/closing, generating forces in different directions for steering or reversing. The impeller operates in the flow field inside a closed duct and can be considered a ducted pump system consisting of the impeller, stator, and inlet duct. If the water flowing into the propulsor per unit time is treated as a control volume, the forces acting on the control volume as it passes through the propulsor can be expressed via the momentum theorem using inlet/outlet velocities and the mass of the control volume.
"""
        + ",\n".join([item["text"] for item in res])
    )


__all__ = ["domainKnowledge", "domainKnowledge_with_params", "bert_embedding", "search_similar_chunks"]
