from __future__ import annotations

from pathlib import Path
from typing import Any


def _collection_exists(client: Any, collection_name: str) -> bool:
    try:
        if hasattr(client, "collection_exists"):
            try:
                return bool(client.collection_exists(collection_name))  # type: ignore[misc]
            except TypeError:
                return bool(client.collection_exists(collection_name=collection_name))  # type: ignore[misc]
        client.get_collection(collection_name)  # type: ignore[misc]
        return True
    except Exception:
        return False


def ingest_examples_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    import json

    try:
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch  # type: ignore
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("missing deps for ingest examples (torch, transformers, qdrant-client)") from e

    qdrant_cfg = config.get("qdrant") or {}
    host = qdrant_cfg.get("host") or "localhost"
    port = int(qdrant_cfg.get("port") or 6333)
    collections = qdrant_cfg.get("collections") or {}
    collection_name = collections.get("examples") or "examples_vec"

    paths_cfg = config.get("paths") or {}
    default_input = paths_cfg.get("examples_json") or "AutoMBSE/resource/examples/example.json"
    input_path = Path(getattr(args, "input", None) or default_input)
    input_path = input_path.expanduser()
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    data = json.loads(input_path.read_text(encoding="utf-8"))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    client = QdrantClient(host=host, port=port)
    if not _collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 768, "distance": "Cosine"},
        )

    points = []
    for i, item in enumerate(data):
        description = item["description"]
        inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        points.append(
            {
                "id": i,
                "vector": vector.tolist(),
                "payload": {"code": item["code"], "description": description},
            }
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"upserted {len(points)} points into {collection_name} ({host}:{port})")
    return 0


def ingest_knowledge_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    try:
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch  # type: ignore
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("missing deps for ingest knowledge (torch, transformers, qdrant-client)") from e

    qdrant_cfg = config.get("qdrant") or {}
    host = qdrant_cfg.get("host") or "localhost"
    port = int(qdrant_cfg.get("port") or 6333)
    collections = qdrant_cfg.get("collections") or {}
    collection_name = collections.get("knowledge") or "knowledge_chunk"

    paths_cfg = config.get("paths") or {}
    default_input = (
        paths_cfg.get("knowledge_txt")
        or paths_cfg.get("knowledge_text")
        or "AutoMBSE/resource/knowledge/knowledge.txt"
    )
    input_path = Path(getattr(args, "input", None) or default_input).expanduser()
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    text_corpus = input_path.read_text(encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    client = QdrantClient(host=host, port=port)
    if not _collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 768, "distance": "Cosine"},
        )

    chunk_size = 256
    text_len = len(text_corpus)
    chunks = [
        text_corpus[i : i + chunk_size + 32] if i + chunk_size + 32 < text_len else text_corpus[i:]
        for i in range(0, len(text_corpus), chunk_size)
    ]

    for index, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        client.upsert(
            collection_name=collection_name,
            points=[{"id": index, "vector": embeddings.tolist(), "payload": {"text": chunk}}],
        )

    print(f"ingested {len(chunks)} chunks into {collection_name} ({host}:{port})")
    return 0


def ingest_pump_parts_cmd(*, args: Any, config: dict[str, Any], repo_root: Path) -> int:
    try:
        import pandas as pd  # type: ignore
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch  # type: ignore
        from qdrant_client import QdrantClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit("missing deps for ingest pump-parts (pandas, torch, transformers, qdrant-client)") from e

    qdrant_cfg = config.get("qdrant") or {}
    host = qdrant_cfg.get("host") or "localhost"
    port = int(qdrant_cfg.get("port") or 6333)
    collections = qdrant_cfg.get("collections") or {}
    collection_name = collections.get("pump_parts") or "pump_parts_vec"

    input_path = Path(getattr(args, "input", None) or "pump_parts_out.csv").expanduser()
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    df = pd.read_csv(input_path, encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")

    client = QdrantClient(host=host, port=port)
    if not _collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 768, "distance": "Cosine"},
        )

    for index, row in df.iterrows():
        data_dict = row.to_dict()
        text = ", ".join(str(value) for value in data_dict.values())

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        client.upsert(
            collection_name=collection_name,
            points=[{"id": int(index), "vector": embeddings.tolist(), "payload": data_dict}],
        )

    print(f"ingested {len(df)} rows into {collection_name} ({host}:{port})")
    return 0
