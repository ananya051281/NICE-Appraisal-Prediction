"""Exploratory SHAP outputs for the final tiny-pilot model."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import joblib

from .config import MODEL_PATH, OUTPUT_DIR, PREPROCESSOR_PATH

# The module must run in headless CI/desktops as well as interactive notebooks.
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from .data_loader import load_binary_ml_data
from .features import feature_names, prepare_features


def generate_shap_outputs() -> tuple[str, str]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Model artifacts are absent. Run `python -m src.train` first.")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    data = load_binary_ml_data()
    transformed = preprocessor.transform(prepare_features(data))
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    names = feature_names(preprocessor)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values_path = OUTPUT_DIR / "shap_values.csv"
    pd.DataFrame(values, columns=names).assign(appraisal_id=data.appraisal_id.values).to_csv(values_path, index=False)
    plt.figure(figsize=(9, 5))
    shap.summary_plot(values, transformed, feature_names=names, plot_type="bar", show=False)
    plt.title("Exploratory SHAP feature importance (tiny pilot)")
    plt.tight_layout()
    image_path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close()
    return str(values_path), str(image_path)


if __name__ == "__main__":
    csv_file, image_file = generate_shap_outputs()
    print("Exploratory only: model trained on 7 binary pilot records.")
    print(f"SHAP values: {csv_file}\nSHAP summary: {image_file}")
