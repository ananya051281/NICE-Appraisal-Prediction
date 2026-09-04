"""Leakage-safe feature preparation and reproducible preprocessing."""
from __future__ import annotations

from typing import Mapping, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_loader import BOOLEAN_COLUMNS, _as_bool

CATEGORICAL_FEATURES = ["guidance_type", "icer_band"]
NUMERIC_FEATURES = [*BOOLEAN_COLUMNS, "publication_year"]
FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
LEAKAGE_COLUMNS = {"decision_label", "decision_status", "group_id", "appraisal_id", "intervention_name"}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only structured evidence features; outcome and identifier fields are excluded."""
    data = frame.copy()
    for column in CATEGORICAL_FEATURES:
        if column not in data:
            data[column] = pd.NA
        data[column] = data[column].astype("string").str.strip()
    for column in BOOLEAN_COLUMNS:
        if column not in data:
            data[column] = pd.NA
        data[column] = data[column].map(_as_bool).astype("float64")
    dates = pd.to_datetime(data.get("publication_date"), errors="coerce")
    data["publication_year"] = dates.dt.year.astype("float64")
    return data[FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
            ("numeric", Pipeline([( "imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [name.replace("categorical__", "").replace("numeric__", "")
            for name in preprocessor.get_feature_names_out()]


def prediction_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """Validate a user-supplied appraisal feature mapping for prediction/scenarios."""
    required = set(CATEGORICAL_FEATURES + BOOLEAN_COLUMNS + ["publication_date"])
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Missing required appraisal features: {missing}")
    return pd.DataFrame([dict(values)])
