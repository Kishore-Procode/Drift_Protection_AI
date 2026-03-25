from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def _stable_ratio(values: np.ndarray) -> np.ndarray:
    adjusted = values.astype(float) + 1e-6
    return adjusted / adjusted.sum()


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected_ratio = _stable_ratio(expected)
    actual_ratio = _stable_ratio(actual)
    return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def build_reference_profile(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    profile: dict[str, dict[str, Any]] = {}

    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").fillna(series.median())
            quantiles = np.unique(numeric.quantile(np.linspace(0, 1, 11)).to_numpy())
            inner_edges = quantiles[1:-1].tolist() if len(quantiles) > 2 else []
            bins = [-np.inf, *inner_edges, np.inf]
            bucketed = pd.cut(numeric, bins=bins, include_lowest=True)
            expected = bucketed.value_counts(sort=False).to_numpy()
            profile[column] = {
                "type": "numeric",
                "bins": bins,
                "expected": expected.tolist(),
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0) or 1.0),
            }
        else:
            categorical = series.astype("string").fillna("Unknown")
            expected = categorical.value_counts(normalize=True).to_dict()
            profile[column] = {
                "type": "categorical",
                "expected": expected,
            }

    return profile


def detect_drift(reference_profile: dict[str, dict[str, Any]], incoming_features: pd.DataFrame) -> dict[str, Any]:
    feature_results: list[dict[str, Any]] = []

    for column, spec in reference_profile.items():
        if column not in incoming_features.columns:
            continue

        if spec["type"] == "numeric":
            series = pd.to_numeric(incoming_features[column], errors="coerce").fillna(spec["mean"])
            bucketed = pd.cut(series, bins=spec["bins"], include_lowest=True)
            actual = bucketed.value_counts(sort=False).to_numpy()
            score = _psi(np.array(spec["expected"]), actual)
            mean_shift = abs(float(series.mean()) - float(spec["mean"])) / max(float(spec["std"]), 1e-6)
        else:
            series = incoming_features[column].astype("string").fillna("Unknown")
            expected_map = spec["expected"]
            all_categories = sorted(set(expected_map) | set(series.astype(str).unique()))
            expected = np.array([expected_map.get(category, 0.0) for category in all_categories])
            actual_map = series.value_counts(normalize=True).to_dict()
            actual = np.array([actual_map.get(category, 0.0) for category in all_categories])
            score = _psi(expected, actual)
            mean_shift = None

        severity = "stable"
        if score >= 0.25:
            severity = "critical"
        elif score >= 0.10:
            severity = "warning"

        feature_results.append(
            {
                "feature": column,
                "score": round(score, 4),
                "severity": severity,
                "mean_shift": round(mean_shift, 4) if mean_shift is not None else None,
            }
        )

    feature_scores = [item["score"] for item in feature_results]
    overall_drift_score = float(np.mean(feature_scores)) if feature_scores else 0.0
    drift_detected = overall_drift_score >= 0.12 or any(item["score"] >= 0.25 for item in feature_results)

    return {
        "overall_drift_score": round(overall_drift_score, 4),
        "drift_detected": drift_detected,
        "feature_scores": sorted(feature_results, key=lambda item: item["score"], reverse=True),
    }


def assess_prediction_stability(model_bundle: dict[str, Any], incoming_features: pd.DataFrame) -> dict[str, Any]:
    model = model_bundle["model"]
    aligned = incoming_features.reindex(columns=model_bundle["feature_columns"])

    if model_bundle["problem_type"] == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(aligned)
        average_confidence = float(np.max(probabilities, axis=1).mean())
        baseline = float(model_bundle.get("baseline_confidence", average_confidence))
        delta = baseline - average_confidence
        return {
            "stability_metric": "confidence",
            "baseline": round(baseline, 4),
            "current": round(average_confidence, 4),
            "delta": round(delta, 4),
            "unstable": delta >= 0.08,
        }

    predictions = model.predict(aligned)
    prediction_std = float(np.std(predictions))
    baseline = float(model_bundle.get("baseline_prediction_std", prediction_std))
    ratio_delta = abs(prediction_std - baseline) / max(abs(baseline), 1e-6)
    return {
        "stability_metric": "prediction_std",
        "baseline": round(baseline, 4),
        "current": round(prediction_std, 4),
        "delta": round(ratio_delta, 4),
        "unstable": ratio_delta >= 0.35,
    }


def evaluate_batch_performance(model_bundle: dict[str, Any], incoming_dataset: pd.DataFrame) -> dict[str, Any] | None:
    target_column = model_bundle["target_column"]
    if target_column not in incoming_dataset.columns:
        return None

    batch = incoming_dataset.dropna(subset=[target_column]).copy()
    if batch.empty:
        return None

    features = batch.reindex(columns=model_bundle["feature_columns"])
    target = batch[target_column]
    predictions = model_bundle["model"].predict(features)

    if model_bundle["problem_type"] == "classification":
        current_score = accuracy_score(target, predictions)
        f1 = f1_score(target, predictions, average="weighted", zero_division=0)
        baseline_score = float(model_bundle["metrics"]["accuracy"])
        drop = baseline_score - current_score
        return {
            "primary_metric": "accuracy",
            "baseline_score": round(baseline_score, 4),
            "current_score": round(float(current_score), 4),
            "secondary_metric": round(float(f1), 4),
            "performance_drop": round(float(drop), 4),
            "degraded": drop >= 0.08,
        }

    current_score = r2_score(target, predictions)
    rmse = mean_squared_error(target, predictions, squared=False)
    mae = mean_absolute_error(target, predictions)
    baseline_score = float(model_bundle["metrics"]["r2"])
    drop = baseline_score - current_score
    return {
        "primary_metric": "r2",
        "baseline_score": round(baseline_score, 4),
        "current_score": round(float(current_score), 4),
        "secondary_metric": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "performance_drop": round(float(drop), 4),
        "degraded": drop >= 0.12,
    }


def monitor_batch(model_bundle: dict[str, Any], incoming_dataset: pd.DataFrame) -> dict[str, Any]:
    features = incoming_dataset.drop(columns=[model_bundle["target_column"]], errors="ignore")
    drift_summary = detect_drift(model_bundle["reference_profile"], features)
    stability_summary = assess_prediction_stability(model_bundle, features)
    performance_summary = evaluate_batch_performance(model_bundle, incoming_dataset)

    should_retrain = drift_summary["drift_detected"] or stability_summary["unstable"]
    if performance_summary:
        should_retrain = should_retrain or performance_summary["degraded"]

    return {
        **drift_summary,
        "stability": stability_summary,
        "performance": performance_summary,
        "should_retrain": should_retrain,
    }

