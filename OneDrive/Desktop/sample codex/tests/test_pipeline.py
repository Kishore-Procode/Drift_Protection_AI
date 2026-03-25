from __future__ import annotations

from datadrift_ai.cleaning import clean_dataset
from datadrift_ai.demo_data import generate_demo_training_data, generate_drifted_batch
from datadrift_ai.drift import monitor_batch
from datadrift_ai.modeling import auto_retrain, predict_dataframe, run_automated_workflow, train_best_model


def test_end_to_end_pipeline_smoke() -> None:
    raw_df = generate_demo_training_data(rows=260, seed=12)
    cleaned_df, quality_report = clean_dataset(raw_df, target_column="churn_risk")

    assert quality_report.quality_score > 0
    assert "churn_risk" in cleaned_df.columns

    bundle = train_best_model(cleaned_df, target_column="churn_risk")
    scored_df = predict_dataframe(bundle, cleaned_df.head(8))
    assert "prediction" in scored_df.columns

    incoming_df = generate_drifted_batch(rows=90, seed=18)
    cleaned_incoming_df, _ = clean_dataset(incoming_df, target_column="churn_risk")
    monitor_report = monitor_batch(bundle, cleaned_incoming_df)

    assert "overall_drift_score" in monitor_report
    assert "should_retrain" in monitor_report

    updated_bundle = auto_retrain(bundle, cleaned_incoming_df)
    assert updated_bundle["best_model_name"]


def test_automated_workflow_runs_with_less_manual_steps() -> None:
    raw_df = generate_demo_training_data(rows=220, seed=21)
    incoming_df = generate_drifted_batch(rows=80, seed=5)

    result = run_automated_workflow(
        raw_dataset=raw_df,
        target_column="churn_risk",
        incoming_dataset=incoming_df,
        run_auto_retrain=False,
    )

    assert "cleaned_df" in result
    assert "quality_report" in result
    assert "bundle" in result
    assert "predictions_df" in result
    assert "monitor_report" in result
    assert "prediction" in result["predictions_df"].columns
    assert isinstance(result["monitor_report"]["should_retrain"], bool)
    assert result["retrained"] is False
