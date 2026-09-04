"""Train the deliberately small, leakage-safe XGBoost pilot prototype."""
from __future__ import annotations

import json

import joblib
from xgboost import XGBClassifier

from .config import MODEL_DIR, MODEL_PATH, PREPROCESSOR_PATH, RANDOM_SEED
from .data_loader import load_binary_ml_data
from .features import build_preprocessor, prepare_features


def train_final_model() -> dict[str, object]:
    """Train on all valid binary pilot records after documenting evaluation limits."""
    data = load_binary_ml_data()
    if data.decision_label.nunique() != 2:
        raise ValueError("Binary training requires both favourable and unfavourable records")
    # With one negative example, every group-wise fold either has a one-class training
    # set or is too small to interpret. Do not manufacture a validation metric.
    evaluation = {
        "approach": "group-aware evaluation assessed",
        "metrics": None,
        "reason": ("Skipped: 7 binary records across 7 groups include only one "
                   "unfavourable example; meaningful group-aware metrics are impossible."),
    }
    features = prepare_features(data)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=30,
        max_depth=2, learning_rate=0.05, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED,
        n_jobs=1, verbosity=0,
    )
    model.fit(transformed, data.decision_label)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    metadata = {
        "records": int(len(data)), "groups": int(data.group_id.nunique()),
        "class_counts": {str(key): int(value) for key, value in data.decision_label.value_counts().items()},
        "xgboost_parameters": model.get_params(), "evaluation": evaluation,
    }
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train_final_model()
    print("Pilot XGBoost model trained on all binary records.")
    print(result["evaluation"]["reason"])
    print(f"Model: {MODEL_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
