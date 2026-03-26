from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from datadrift_ai.cleaning import clean_dataset
from datadrift_ai.config import MODEL_BUNDLE_PATH
from datadrift_ai.modeling import predict_dataframe
from datadrift_ai.storage import load_bundle

app = FastAPI(title="DataDrift Protection AI API", version="1.0.0")


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": MODEL_BUNDLE_PATH.exists(),
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "target_column": bundle["target_column"],
        "problem_type": bundle["problem_type"],
        "best_model_name": bundle["best_model_name"],
        "metrics": bundle["metrics"],
        "feature_columns": bundle["feature_columns"],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="Request must include at least one record.")

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    frame = pd.DataFrame(request.records)
    target_column = bundle["target_column"] if bundle["target_column"] in frame.columns else None
    cleaned_frame, _ = clean_dataset(frame, target_column=target_column)
    predictions = predict_dataframe(bundle, cleaned_frame)

    return {
        "rows": len(predictions),
        "predictions": predictions.to_dict(orient="records"),
    }

