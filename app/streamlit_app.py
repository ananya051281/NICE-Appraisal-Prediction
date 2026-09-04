"""NICE HTA Decision Intelligence dashboard using the existing Person 1 ML artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_explain import build_evidence_explanation, generate_grounded_narrative
from src.what_if import compare_scenario

st.set_page_config(page_title="NICE HTA Decision Intelligence", layout="wide")
st.title("NICE HTA Decision Intelligence")
st.caption("Evidence-grounded prediction, historical case retrieval and strategy simulation")


def asset_inputs(prefix: str = "") -> dict:
    with st.sidebar:
        st.subheader("Theoretical asset evidence")
        guidance_type = st.selectbox("Guidance type", ["TA", "HST"], key=f"{prefix}guidance")
        publication_date = st.date_input("Indicative publication date", key=f"{prefix}date")
        randomized = st.checkbox("Randomized trial evidence", value=True, key=f"{prefix}randomized")
        single_arm = st.checkbox("Single-arm evidence", key=f"{prefix}single_arm")
        economic = st.checkbox("Economic evaluation present", value=True, key=f"{prefix}economic")
        icer = st.selectbox("ICER band", ["20k_30k", "30k_50k", "above_100k", "unknown"], key=f"{prefix}icer")
        qaly = st.checkbox("QALY evidence present", value=True, key=f"{prefix}qaly")
        uncertainty = st.checkbox("Substantial uncertainty present", value=True, key=f"{prefix}uncertainty")
        safety = st.checkbox("Safety evidence discussed", value=True, key=f"{prefix}safety")
        arrangement = st.checkbox("Commercial arrangement present", key=f"{prefix}arrangement")
    return {"guidance_type": guidance_type, "publication_date": publication_date.isoformat(),
            "randomized_trial_evidence": randomized, "single_arm_evidence": single_arm,
            "economic_evaluation_present": economic, "icer_band": None if icer == "unknown" else icer,
            "qaly_evidence_present": qaly, "substantial_uncertainty_present": uncertainty,
            "safety_evidence_discussion_present": safety, "commercial_arrangement_present": arrangement}


features = asset_inputs()
query = st.text_input("Historical evidence query", "cost effectiveness clinical evidence uncertainty")
if st.button("ANALYZE HTA PROBABILITY", type="primary"):
    try:
        bundle = build_evidence_explanation(features, query)
        st.session_state["bundle"] = bundle
        st.session_state["features"] = features
    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")

bundle = st.session_state.get("bundle")
if bundle:
    prediction = bundle["prediction"]
    left, right = st.columns(2)
    left.metric("HTA ACCEPTANCE PROBABILITY", f"{prediction['favourable_probability']:.1%}")
    right.metric("Predicted status", prediction["predicted_class"].title())
    st.info(prediction["model_status"])
    st.header("Why?")
    drivers = bundle["model_drivers"]
    positive, negative = st.columns(2)
    positive.subheader("Positive drivers")
    for item in [driver for driver in drivers if driver["direction"] == "positive"]:
        positive.write(f"+ {item['feature']} ({item['contribution']:.3f})")
    negative.subheader("Negative drivers")
    for item in [driver for driver in drivers if driver["direction"] == "negative"]:
        negative.write(f"- {item['feature']} ({item['contribution']:.3f})")
    st.header("Similar historical cases")
    cases = bundle["historical_evidence"]
    if not cases:
        st.warning("No vector store is available yet. Run `python -m src.build_knowledge_base` to build the supplied-evidence demo store.")
    for case in cases:
        metadata = case["metadata"]
        with st.expander(f"{case.get('ta_id') or 'Unidentified source'} — similarity {case['similarity']:.1%}"):
            st.caption(f"Source: {metadata.get('source', 'unknown')} | Outcome: {metadata.get('outcome', 'not available')}")
            st.write(case["text"])
    st.header("Why this prediction?")
    st.markdown("**MODEL PREDICTION:** Pilot XGBoost probability shown above.")
    st.markdown("**HISTORICAL EVIDENCE:** Retrieved snippets are shown above with their source identifiers.")
    if st.button("Generate grounded LLM summary (optional)"):
        narrative = generate_grounded_narrative(bundle)
        st.markdown("**LLM-GENERATED SUMMARY:**")
        st.write(narrative["text"])
    st.warning("Historical similarity does not establish causality. The ML model is a pilot/prototype, not clinical or regulatory certainty.")
    st.header("What-if strategy simulator")
    candidate_icer = st.selectbox("Scenario ICER band", ["20k_30k", "30k_50k", "above_100k"], key="scenario_icer")
    candidate_uncertainty = st.checkbox("Scenario has substantial uncertainty", value=features["substantial_uncertainty_present"], key="scenario_uncertainty")
    if st.button("Compare scenario"):
        scenario = compare_scenario(features, {"icer_band": candidate_icer, "substantial_uncertainty_present": candidate_uncertainty})
        one, two, three = st.columns(3)
        one.metric("Current probability", f"{scenario['baseline_favourable_probability']:.1%}")
        two.metric("Scenario probability", f"{scenario['scenario_favourable_probability']:.1%}")
        three.metric("Change", f"{scenario['difference']:+.1%}")
        st.caption(scenario["model_status"])
