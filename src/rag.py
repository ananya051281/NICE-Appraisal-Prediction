"""Sentence-transformer + FAISS retrieval with persisted, source-preserving metadata."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import DATA_DIR
from .data_loader import load_ml_data

VECTOR_STORE_DIR = DATA_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "nice_chunks.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "nice_chunks.json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text on word boundaries with an overlapping contextual window."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    words = str(text or "").split()
    return [" ".join(words[start:start + chunk_size]) for start in range(0, len(words), chunk_size - overlap)
            if words[start:start + chunk_size]]


def chunk_documents(documents: Iterable[dict[str, Any]], chunk_size: int = 500, overlap: int = 100) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = document.get("text", "")
        metadata = {key: value for key, value in document.items() if key != "text"}
        for number, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append({"text": chunk, "metadata": {**metadata, "chunk_number": number}})
    return chunks


def _embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required. Install requirements.txt first.") from exc
    # Keep model-cache writes inside the project instead of relying on a user home cache.
    cache_dir = DATA_DIR / ".embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(model_name, cache_folder=str(cache_dir))


def build_vector_store(documents: Iterable[dict[str, Any]], store_dir: Path = VECTOR_STORE_DIR,
                       model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, Any]:
    """Embed chunk records, build a cosine-similarity FAISS index, and persist metadata."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Install requirements.txt first.") from exc
    chunks = list(documents)
    if not chunks or any(not item.get("text", "").strip() for item in chunks):
        raise ValueError("At least one non-empty chunk document is required")
    encoder = _embedder(model_name)
    vectors = np.asarray(encoder.encode([item["text"] for item in chunks], normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_dir / INDEX_PATH.name))
    payload = {"embedding_model": model_name, "metric": "cosine_similarity", "chunks": chunks}
    (store_dir / METADATA_PATH.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks": len(chunks), "dimensions": int(vectors.shape[1]), "store_dir": str(store_dir)}


def search_similar_cases(query: str, top_k: int = 5, store_dir: Path = VECTOR_STORE_DIR) -> list[dict[str, Any]]:
    """Search a persisted store and return actual cosine similarities and metadata."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    index_file, metadata_file = store_dir / INDEX_PATH.name, store_dir / METADATA_PATH.name
    if not index_file.exists() or not metadata_file.exists():
        return []
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required. Install requirements.txt first.") from exc
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    encoder = _embedder(payload["embedding_model"])
    vector = np.asarray(encoder.encode([query], normalize_embeddings=True), dtype="float32")
    index = faiss.read_index(str(index_file))
    scores, positions = index.search(vector, min(top_k, index.ntotal))
    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        record = payload["chunks"][int(position)]
        results.append({"similarity": float(score), "text": record["text"], "metadata": record["metadata"],
                        "ta_id": record["metadata"].get("ta_id")})
    return results


def supplied_qa_documents(jsonl_path: Path = DATA_DIR / "NICE_LLM_TRAINING.jsonl") -> list[dict[str, Any]]:
    """Build clearly-labelled retrieval documents from supplied source-grounded Q&A, not decisions."""
    labels = {row.appraisal_id: {"outcome": row.decision_status, "guidance_type": row.guidance_type}
              for row in load_ml_data().itertuples()}
    documents = []
    pattern = re.compile(r"\b(?:TA|HST|HTG|NG|QS)\d+\b", re.I)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        messages = item.get("messages", [])
        question = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        answer = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")
        match = pattern.search(f"{question} {answer}")
        ta_id = match.group(0).upper() if match else None
        documents.append({"text": f"Question: {question}\nAnswer: {answer}", "ta_id": ta_id,
                          "source": "supplied NICE Q&A evidence", "record_type": "retrieval evidence",
                          **labels.get(ta_id, {})})
    return documents
