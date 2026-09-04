"""Loading and validation for the appraisal-level pilot dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ML_DATASET_PATH

EXPECTED_COLUMNS = [
    "appraisal_id", "guidance_type", "intervention_name", "publication_date",
    "randomized_trial_evidence", "single_arm_evidence", "economic_evaluation_present",
    "icer_band", "qaly_evidence_present", "substantial_uncertainty_present",
    "safety_evidence_discussion_present", "commercial_arrangement_present",
    "decision_label", "decision_status", "group_id",
]
BOOLEAN_COLUMNS = [
    "randomized_trial_evidence", "single_arm_evidence", "economic_evaluation_present",
    "qaly_evidence_present", "substantial_uncertainty_present",
    "safety_evidence_discussion_present", "commercial_arrangement_present",
]
ALIASES = {"decision": "decision_label", "label": "decision_label", "date": "publication_date"}
LABEL_NAMES = {"unfavourable": 0, "unfavorable": 0, "favourable": 1, "favorable": 1,
               "no_decision": 2, "no decision": 2, "terminated": 2}


def _as_bool(value: Any) -> object:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    mapped = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
    value = str(value).strip().lower()
    if value in mapped:
        return mapped[value]
    raise ValueError(f"Invalid boolean value: {value!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(record)
    if not records:
        raise ValueError("The ML JSONL file contains no records")
    return records


def load_ml_data(path: Path | str = ML_DATASET_PATH) -> pd.DataFrame:
    """Load, validate, and safely normalize the pilot appraisal records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ML dataset not found: {path}")
    frame = pd.DataFrame(_read_jsonl(path)).rename(columns=ALIASES)
    missing_critical = {"appraisal_id", "decision_label", "group_id"} - set(frame.columns)
    if missing_critical:
        raise ValueError(f"Dataset is missing critical columns: {sorted(missing_critical)}")
    for column in EXPECTED_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[EXPECTED_COLUMNS].copy()
    if frame["appraisal_id"].isna().any() or frame["group_id"].isna().any():
        raise ValueError("appraisal_id and group_id cannot be missing")
    for column in BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(_as_bool).astype("boolean")
    raw_label = frame["decision_label"]
    normalized = raw_label.astype(str).str.strip().str.lower().map(LABEL_NAMES)
    numeric = pd.to_numeric(raw_label, errors="coerce")
    frame["decision_label"] = normalized.fillna(numeric)
    if frame["decision_label"].isna().any() or not frame["decision_label"].isin([0, 1, 2]).all():
        bad = raw_label[~frame["decision_label"].isin([0, 1, 2])].tolist()
        raise ValueError(f"decision_label must contain only 0, 1, or 2; invalid values: {bad}")
    frame["decision_label"] = frame["decision_label"].astype(int)
    frame["publication_date"] = pd.to_datetime(frame["publication_date"], errors="coerce")
    for column in ["guidance_type", "intervention_name", "icer_band", "decision_status"]:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def load_binary_ml_data(path: Path | str = ML_DATASET_PATH) -> pd.DataFrame:
    """Return only favourable/unfavourable records; no_decision is never negative."""
    return load_ml_data(path).query("decision_label in [0, 1]").copy()
