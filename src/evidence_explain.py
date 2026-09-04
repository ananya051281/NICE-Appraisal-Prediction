"""Evidence-grounded explanation assembly; optional LLM use never fabricates sources."""
from __future__ import annotations

import os
from typing import Any, Mapping

from .predict import predict_appraisal
from .shap_explain import explain_prediction
from .rag import search_similar_cases


def build_evidence_explanation(features: Mapping[str, Any], query: str, top_k: int = 5) -> dict[str, Any]:
    """Combine the existing ML interfaces with retrieved evidence, without an LLM."""
    prediction = predict_appraisal(features)
    drivers = explain_prediction(features)
    cases = search_similar_cases(query, top_k=top_k)
    return {
        "prediction": prediction,
        "model_drivers": drivers,
        "historical_evidence": cases,
        "narrative": None,
        "narrative_status": "No LLM narrative generated. Display model outputs and retrieved evidence directly.",
    }


def generate_grounded_narrative(bundle: Mapping[str, Any]) -> dict[str, str]:
    """Optionally use OpenAI only with supplied evidence; gracefully degrades without a key."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "unavailable", "text": "LLM narrative generation is unavailable because OPENAI_API_KEY is not configured."}
    evidence = bundle.get("historical_evidence", [])
    if not evidence:
        return {"status": "insufficient_evidence", "text": "No retrieved historical evidence is available for an LLM-grounded narrative."}
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[{"role": "system", "content": (
                "Summarise only the supplied model prediction, SHAP drivers, and retrieved evidence. "
                "Never invent NICE cases, recommendations, ICERs, or clinical evidence. "
                "Distinguish pilot model output from historical evidence, cite TA/HST identifiers, "
                "and state when evidence is insufficient.")},
                   {"role": "user", "content": str(bundle)}],
        )
        return {"status": "ok", "text": response.output_text}
    except Exception as exc:
        return {"status": "error", "text": f"LLM narrative unavailable: {exc}"}
