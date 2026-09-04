"""Non-causal model scenario comparisons for a single appraisal."""
from __future__ import annotations

from typing import Any, Mapping

from .predict import predict_appraisal


def compare_scenario(baseline: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Compare an allowed feature edit under the same saved pilot model."""
    allowed = set(baseline)
    unsupported = sorted(set(changes) - allowed)
    if unsupported:
        raise ValueError(f"Changes contain unsupported features: {unsupported}")
    scenario = dict(baseline)
    scenario.update(changes)
    before = predict_appraisal(baseline)
    after = predict_appraisal(scenario)
    return {
        "baseline_favourable_probability": before["favourable_probability"],
        "scenario_favourable_probability": after["favourable_probability"],
        "difference": after["favourable_probability"] - before["favourable_probability"],
        "changed_features": dict(changes),
        "model_status": "Model-based what-if scenario — not a causal prediction or guarantee.",
    }


if __name__ == "__main__":
    baseline = {
        "guidance_type": "TA", "publication_date": "2024-01-01",
        "randomized_trial_evidence": True, "single_arm_evidence": False,
        "economic_evaluation_present": True, "icer_band": "20k_30k",
        "qaly_evidence_present": True, "substantial_uncertainty_present": True,
        "safety_evidence_discussion_present": True, "commercial_arrangement_present": False,
    }
    print(compare_scenario(baseline, {"commercial_arrangement_present": True}))
