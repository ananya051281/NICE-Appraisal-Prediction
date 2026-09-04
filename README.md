# Novo Nordisk Hackathon — NICE ML Foundation

This repository contains the Person 1 ML/data foundation for a hackathon prototype that predicts a **pilot model probability** of a favourable NICE appraisal from structured, pre-decision evidence fields. It is not a clinical, economic, or NICE decision system.

## Person 1 responsibilities

Person 1 owns dataset inspection and cleaning, leakage-safe feature engineering, the `decision_label` target, conservative XGBoost training, probability prediction, exploratory SHAP outputs, model scenario (what-if) comparisons, and saved model artifacts.

## Data

`data/NICE_LLM_TRAINING.jsonl` is the 371-record LLM/RAG Q&A dataset for Person 2. It is not used as ML appraisal decisions. `data/NICE_ML_APPRAISAL_TRAINING_PILOT.jsonl` is the appraisal-level ML source: 9 independent TA/HST records. `data/nice_guidance_inventory.jsonl` inventories supplied guidance and eligibility.

The ML target is `decision_label`: `0` = unfavourable, `1` = favourable, and `2` = no_decision. Binary training excludes label `2`; it is never recoded as unfavourable.

## Leakage prevention

The model excludes `decision_label`, `decision_status`, `group_id`, identifiers, intervention name, and any final recommendation/decision/conclusion/answer text. `group_id` is reserved for grouped evaluation logic. Features are guidance type, structured evidence flags, ICER band, and publication year.

## Setup and use

Install the dependencies, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.inspect_dataset
python -m src.train
python -m src.predict
python -m src.shap_explain
python -m src.what_if
```

`src.predict.predict_appraisal(features)` loads `models/xgboost_appraisal_model.joblib` and `models/appraisal_preprocessor.joblib` and returns predicted class, favourable probability, unfavourable probability, and the pilot-model status. This is designed for import by Person 2’s Streamlit application.

`src.what_if.compare_scenario(baseline, changes)` applies supported feature edits through exactly the same preprocessing/model and returns baseline and scenario favourable probabilities, their difference, and the changes. It is explicitly a **model-based what-if scenario — not a causal prediction or guarantee.**

`src.shap_explain` creates `outputs/shap_values.csv` and `outputs/shap_summary.png`. SHAP results are exploratory.

## Model and evaluation

The final prototype is a conservative `XGBClassifier` (30 estimators, depth 2, learning rate 0.05, fixed seed). Preprocessing uses median/mode imputation and one-hot encoding with unknown categories ignored, fitted only on the training dataset and saved separately.

Group-aware evaluation was assessed, but no metric is reported: the data has 7 binary records across 7 groups and only one unfavourable example. Any grouped split leaves a one-class or non-interpretable training/evaluation set. After this smoke-test decision, the final prototype trains on all seven valid binary records.

## Limitations

- Only 9 independent appraisal records currently exist; only 7 are usable for binary modelling.
- This is a hackathon pilot/prototype and is not production-ready.
- Probabilities are not validated NICE acceptance probabilities and must not be treated as guarantees.
- SHAP outputs are exploratory because of the extremely small training dataset.
- What-if outputs are model scenarios, not causal effects.
