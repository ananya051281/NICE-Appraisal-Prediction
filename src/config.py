"""Portable project paths and shared constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ML_DATASET_PATH = DATA_DIR / "NICE_ML_APPRAISAL_TRAINING_PILOT.jsonl"
MODEL_PATH = MODEL_DIR / "xgboost_appraisal_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "appraisal_preprocessor.joblib"
RANDOM_SEED = 42
