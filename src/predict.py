"""Importable prediction interface for the ML component or a Streamlit UI."""
from __future__ import annotations

from typing import Any, Mapping

import joblib

from .config import MODEL_PATH, PREPROCESSOR_PATH
from .features import prediction_frame, prepare_features


def _load_artifacts():
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Model artifacts are absent. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)


def predict_appraisal(features: Mapping[str, Any]) -> dict[str, Any]:
    """Return pilot-class probabilities for one structured appraisal record."""
    model, preprocessor = _load_artifacts()
    raw = prediction_frame(features)
    prepared = prepare_features(raw)
    probabilities = model.predict_proba(preprocessor.transform(prepared))[0]
    probability_by_class = dict(zip(model.classes_, probabilities))
    favourable = float(probability_by_class.get(1, 0.0))
    unfavourable = float(probability_by_class.get(0, 0.0))
    predicted = "favourable" if favourable >= unfavourable else "unfavourable"
    return {
        "predicted_class": predicted,
        "favourable_probability": favourable,
        "unfavourable_probability": unfavourable,
        "model_status": "Pilot model probability — hackathon prototype, not a NICE decision probability.",
    }


if __name__ == "__main__":
    example = {
        "guidance_type": "TA", "publication_date": "2024-01-01",
        "randomized_trial_evidence": True, "single_arm_evidence": False,
        "economic_evaluation_present": True, "icer_band": "20k_30k",
        "qaly_evidence_present": True, "substantial_uncertainty_present": True,
        "safety_evidence_discussion_present": True, "commercial_arrangement_present": True,
    }
    print(predict_appraisal(example))
