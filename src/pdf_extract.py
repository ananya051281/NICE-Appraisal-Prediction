"""Text extraction for text-based HTA PDFs; OCR is deliberately not attempted."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def extract_pdf_text(pdf_path: str | Path) -> dict[str, Any]:
    """Return extractable text and transparent status without claiming OCR."""
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        return {"status": "error", "text": "", "pages": 0, "error": f"File not found: {path}"}
    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except Exception as exc:  # pypdf exposes several version-specific exceptions
        return {"status": "error", "text": "", "pages": 0, "error": f"PDF extraction failed: {exc}"}
    if not text:
        return {"status": "no_extractable_text", "text": "", "pages": len(reader.pages),
                "error": "No embedded text was found; OCR was not performed."}
    return {"status": "ok", "text": text, "pages": len(reader.pages), "error": None}
