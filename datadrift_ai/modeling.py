from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import clean_dataset
from .drift import build_reference_profile
from .drift import monitor_batch


def infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique(dropna=True)
        if unique_values <= 10:
            return "classification"
        threshold = max(15, int(len(target) * 0.05))
        return "regression" if unique_values > threshold else "classification"
    return "classification"


def _build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def _candidate_models(problem_type: str) -> dict[str, Any]:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
            ),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


def _evaluate_predictions(problem_type: str, target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    if problem_type == "classification":
        return {
            "accuracy": round(float(accuracy_score(target, predictions)), 4),
            "precision": round(float(precision_score(target, predictions, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(target, predictions, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(target, predictions, average="weighted", zero_division=0)), 4),
        }

    return {
        "r2": round(float(r2_score(target, predictions)), 4),
        "rmse": round(float(mean_squared_error(target, predictions, squared=False)), 4),
        "mae": round(float(mean_absolute_error(target, predictions)), 4),
    }


def train_best_model(cleaned_dataset: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in cleaned_dataset.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the cleaned dataset.")

    dataset = cleaned_dataset.copy()
    features = dataset.drop(columns=[target_column])
    target = dataset[target_column]
    problem_type = infer_problem_type(target)

    preprocessor, numeric_columns, categorical_columns = _build_preprocessor(features)
    models = _candidate_models(problem_type)
    primary_metric = "accuracy" if problem_type == "classification" else "r2"

    stratify = None
    if problem_type == "classification":
        class_counts = target.value_counts()
        if target.nunique() > 1 and class_counts.min() >= 2:
            stratify = target

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    leaderboard: list[dict[str, Any]] = []
    best_pipeline = None
    best_name = ""
    best_metrics: dict[str, float] = {}
    best_score = -np.inf
    baseline_confidence = None
    baseline_prediction_std = None

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_valid)
        metrics = _evaluate_predictions(problem_type, y_valid, predictions)
        score = metrics[primary_metric]

        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(x_valid)
            confidence = float(np.max(probabilities, axis=1).mean())
        else:
            confidence = None

        if problem_type == "regression":
            prediction_std = float(np.std(predictions))
        else:
            prediction_std = None

        leaderboard.append(
            {
                "model": name,
                "score": round(float(score), 4),
                **metrics,
            }
        )

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = name
            best_metrics = metrics
            baseline_confidence = confidence
            baseline_prediction_std = prediction_std

    if best_pipeline is None:
        raise RuntimeError("Training failed. No candidate model produced a valid result.")

    reference_profile = build_reference_profile(features)
    bundle = {
        "model": best_pipeline,
        "target_column": target_column,
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "best_model_name": best_name,
        "metrics": best_metrics,
        "leaderboard": sorted(leaderboard, key=lambda row: row["score"], reverse=True),
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "reference_profile": reference_profile,
        "cleaned_training_data": dataset,
        "baseline_confidence": baseline_confidence,
        "baseline_prediction_std": baseline_prediction_std,
    }
    return bundle


def predict_dataframe(model_bundle: dict[str, Any], incoming_dataset: pd.DataFrame) -> pd.DataFrame:
    features = incoming_dataset.drop(columns=[model_bundle["target_column"]], errors="ignore")
    aligned = features.reindex(columns=model_bundle["feature_columns"])

    predictions = model_bundle["model"].predict(aligned)
    output = incoming_dataset.copy()
    output["prediction"] = predictions

    if model_bundle["problem_type"] == "classification" and hasattr(model_bundle["model"], "predict_proba"):
        probabilities = model_bundle["model"].predict_proba(aligned)
        output["prediction_confidence"] = np.max(probabilities, axis=1).round(4)

    return output


def auto_retrain(model_bundle: dict[str, Any], incoming_labeled_dataset: pd.DataFrame) -> dict[str, Any]:
    target_column = model_bundle["target_column"]
    if target_column not in incoming_labeled_dataset.columns:
        raise ValueError("Auto retraining requires the target column to be present in the incoming batch.")

    combined_dataset = pd.concat(
        [model_bundle["cleaned_training_data"], incoming_labeled_dataset],
        ignore_index=True,
    )
    return train_best_model(combined_dataset, target_column=target_column)


def run_automated_workflow(
    raw_dataset: pd.DataFrame,
    target_column: str,
    incoming_dataset: pd.DataFrame | None = None,
    *,
    run_auto_retrain: bool = True,
) -> dict[str, Any]:
    cleaned_df, quality_report = clean_dataset(raw_dataset, target_column=target_column)
    bundle = train_best_model(cleaned_df, target_column=target_column)

    result: dict[str, Any] = {
        "cleaned_df": cleaned_df,
        "quality_report": quality_report.to_dict(),
        "bundle": bundle,
        "retrained": False,
    }

    if incoming_dataset is None:
        return result

    target_for_cleaning = target_column if target_column in incoming_dataset.columns else None
    incoming_cleaned_df, incoming_quality_report = clean_dataset(
        incoming_dataset,
        target_column=target_for_cleaning,
    )
    predictions_df = predict_dataframe(bundle, incoming_cleaned_df)
    monitor_report = monitor_batch(bundle, incoming_cleaned_df)

    result.update(
        {
            "incoming_cleaned_df": incoming_cleaned_df,
            "incoming_quality_report": incoming_quality_report.to_dict(),
            "predictions_df": predictions_df,
            "monitor_report": monitor_report,
        }
    )

    if run_auto_retrain and monitor_report["should_retrain"] and target_column in incoming_cleaned_df.columns:
        result["bundle"] = auto_retrain(bundle, incoming_cleaned_df)
        result["retrained"] = True

    return result
