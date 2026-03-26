from __future__ import annotations

from typing import Any

import joblib

from .config import MODEL_BUNDLE_PATH


def save_bundle(model_bundle: dict[str, Any]) -> str:
    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)
    return str(MODEL_BUNDLE_PATH)


def load_bundle() -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError("No saved model bundle was found. Train a model first.")
    return joblib.load(MODEL_BUNDLE_PATH)
