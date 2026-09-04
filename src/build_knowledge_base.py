"""Reproducible knowledge-base builder using supplied evidence or official NICE HTML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATA_DIR
from .nice_scraper import scrape_guidance
from .rag import build_vector_store, chunk_documents, supplied_qa_documents


def build_demo_store() -> dict:
    """Index supplied source-grounded Q&A as retrieval evidence, never as independent decisions."""
    return build_vector_store(chunk_documents(supplied_qa_documents(), chunk_size=180, overlap=30))


def build_from_guidance_ids(ids: list[str]) -> dict:
    documents = []
    extracted_dir = DATA_DIR / "extracted_documents"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for ta_id in ids:
        guidance = scrape_guidance(ta_id)
        for source in ("recommendations", "history"):
            text = guidance[source]
            if text:
                documents.append({"text": text, "ta_id": guidance["ta_id"], "source": f"NICE {source} HTML"})
                (extracted_dir / f"{guidance['ta_id']}_{source}.txt").write_text(text, encoding="utf-8")
    if not documents:
        return {"chunks": 0, "message": "No official NICE HTML content was available."}
    return build_vector_store(chunk_documents(documents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta-id", action="append", default=[], help="Official NICE ID, repeatable")
    parser.add_argument("--demo-from-supplied-qa", action="store_true")
    args = parser.parse_args()
    result = build_from_guidance_ids(args.ta_id) if args.ta_id else build_demo_store()
    print(json.dumps(result, indent=2))
