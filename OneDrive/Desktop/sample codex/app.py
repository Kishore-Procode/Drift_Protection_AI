from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datadrift_ai.cleaning import clean_dataset
from datadrift_ai.demo_data import generate_demo_training_data, generate_drifted_batch
from datadrift_ai.drift import monitor_batch
from datadrift_ai.modeling import auto_retrain, predict_dataframe, run_automated_workflow, train_best_model
from datadrift_ai.storage import save_bundle

st.set_page_config(
    page_title="DataDrift Protection AI",
    page_icon="ML",
    layout="wide",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(227, 167, 95, 0.20), transparent 30%),
                    radial-gradient(circle at top right, rgba(16, 88, 93, 0.16), transparent 28%),
                    linear-gradient(180deg, #f7efe3 0%, #fffaf2 42%, #ffffff 100%);
                color: #17212b;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2.5rem;
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.02em;
            }

            p, li, label, div[data-testid="stMarkdownContainer"] {
                font-family: 'IBM Plex Sans', sans-serif;
            }

            .hero-card {
                border-radius: 28px;
                padding: 1.8rem 2rem;
                background: linear-gradient(135deg, #124e52 0%, #0f7c82 46%, #e09b47 100%);
                color: #fff9f1;
                box-shadow: 0 22px 60px rgba(18, 78, 82, 0.18);
                margin-bottom: 1.25rem;
            }

            .hero-chip {
                display: inline-block;
                margin-right: 0.5rem;
                margin-top: 0.6rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.16);
                font-size: 0.9rem;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(18, 78, 82, 0.08);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 10px 28px rgba(18, 78, 82, 0.08);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _reset_runtime() -> None:
    for key in [
        "cleaned_df",
        "quality_report",
        "bundle",
        "incoming_raw_df",
        "incoming_cleaned_df",
        "incoming_quality_report",
        "monitor_report",
        "predictions_df",
        "saved_model_path",
    ]:
        st.session_state.pop(key, None)


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">DataDrift Protection AI</h1>
            <p style="font-size:1.08rem; max-width: 880px;">
                Real-time protection for ML systems: clean the data, train the best model, monitor live batches,
                detect drift early, and automatically retrain before performance collapses.
            </p>
            <span class="hero-chip">Upload</span>
            <span class="hero-chip">Clean</span>
            <span class="hero-chip">AutoML</span>
            <span class="hero-chip">Predict</span>
            <span class="hero-chip">Detect Drift</span>
            <span class="hero-chip">Auto Fix</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dataframe_preview(label: str, frame: pd.DataFrame) -> None:
    st.markdown(f"**{label}**")
    st.dataframe(frame.head(12), use_container_width=True, hide_index=True)


def _metric_columns(metrics: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        column.metric(label=label, value=value, delta=delta)


def _quality_chart(report: dict[str, object]):
    chart_data = pd.DataFrame(
        {
            "metric": [
                "Duplicates Removed",
                "Missing Values Filled",
                "Outliers Clipped",
                "Target Rows Dropped",
            ],
            "count": [
                report["duplicates_removed"],
                report["missing_values_filled"],
                report["outliers_clipped"],
                report["target_rows_dropped"],
            ],
        }
    )
    return px.bar(
        chart_data,
        x="metric",
        y="count",
        color="metric",
        color_discrete_sequence=["#124e52", "#e09b47", "#7a9e7e", "#c8613f"],
    )


def _leaderboard_chart(bundle: dict[str, object]):
    leaderboard = pd.DataFrame(bundle["leaderboard"])
    return px.bar(
        leaderboard,
        x="model",
        y="score",
        text="score",
        color="model",
        color_discrete_sequence=["#124e52", "#e09b47", "#6b9080"],
    )


def _drift_indicator(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " PSI"},
            gauge={
                "axis": {"range": [0, 0.5]},
                "bar": {"color": "#c8613f"},
                "steps": [
                    {"range": [0, 0.1], "color": "#cfe7cf"},
                    {"range": [0.1, 0.25], "color": "#f6d8a9"},
                    {"range": [0.25, 0.5], "color": "#f3b3a8"},
                ],
                "threshold": {"line": {"color": "#7b271a", "width": 4}, "value": 0.12},
            },
        )
    )
    figure.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return figure


def _drift_chart(monitor_report: dict[str, object]):
    features = pd.DataFrame(monitor_report["feature_scores"])
    return px.bar(
        features.head(8),
        x="feature",
        y="score",
        color="severity",
        color_discrete_map={"stable": "#6b9080", "warning": "#e09b47", "critical": "#c8613f"},
    )


def _prepare_prediction_download(predictions: pd.DataFrame) -> bytes:
    buffer = StringIO()
    predictions.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _apply_workflow_result(result: dict[str, object]) -> None:
    st.session_state["cleaned_df"] = result["cleaned_df"]
    st.session_state["quality_report"] = result["quality_report"]
    st.session_state["bundle"] = result["bundle"]

    if "incoming_cleaned_df" in result:
        st.session_state["incoming_cleaned_df"] = result["incoming_cleaned_df"]
    if "incoming_quality_report" in result:
        st.session_state["incoming_quality_report"] = result["incoming_quality_report"]
    if "predictions_df" in result:
        st.session_state["predictions_df"] = result["predictions_df"]
    if "monitor_report" in result:
        st.session_state["monitor_report"] = result["monitor_report"]


_inject_styles()
_render_header()

st.sidebar.header("Workflow Controls")
st.sidebar.caption("Run the full ML lifecycle from raw CSV to self-healing model updates.")

if st.sidebar.button("Load Demo Dataset", use_container_width=True):
    st.session_state["raw_df"] = generate_demo_training_data()
    st.session_state["dataset_name"] = "Built-in churn demo"
    st.session_state.pop("training_upload_signature", None)
    _reset_runtime()

uploaded_training_file = st.sidebar.file_uploader("Upload training CSV", type=["csv"], key="training_upload")
if uploaded_training_file is not None:
    signature = (uploaded_training_file.name, uploaded_training_file.size)
    if st.session_state.get("training_upload_signature") != signature:
        st.session_state["raw_df"] = _load_dataframe(uploaded_training_file)
        st.session_state["dataset_name"] = uploaded_training_file.name
        st.session_state["training_upload_signature"] = signature
        _reset_runtime()

if st.sidebar.button("Quickstart: Demo + Automated Run", use_container_width=True):
    demo_df = generate_demo_training_data()
    st.session_state["raw_df"] = demo_df
    st.session_state["dataset_name"] = "Built-in churn demo"
    st.session_state["target_column"] = "churn_risk"
    _reset_runtime()
    incoming_demo_df = generate_drifted_batch()
    st.session_state["incoming_raw_df"] = incoming_demo_df

    result = run_automated_workflow(
        raw_dataset=demo_df,
        target_column="churn_risk",
        incoming_dataset=incoming_demo_df,
        run_auto_retrain=True,
    )
    _apply_workflow_result(result)
    st.session_state["saved_model_path"] = save_bundle(st.session_state["bundle"])
    st.sidebar.success("Quickstart completed. Model trained, monitored, and saved.")

raw_df = st.session_state.get("raw_df")

if raw_df is None:
    st.info("Load the demo dataset or upload a CSV to start the pipeline.")
    st.stop()

dataset_name = st.session_state.get("dataset_name", "Uploaded dataset")
st.caption(f"Active dataset: `{dataset_name}`")

if "target_column" not in st.session_state or st.session_state["target_column"] not in raw_df.columns:
    default_target = "churn_risk" if "churn_risk" in raw_df.columns else raw_df.columns[-1]
    st.session_state["target_column"] = default_target

selected_target = st.selectbox(
    "Target column",
    options=raw_df.columns.tolist(),
    index=raw_df.columns.tolist().index(st.session_state["target_column"]),
)
if selected_target != st.session_state.get("target_column"):
    st.session_state["target_column"] = selected_target
    _reset_runtime()

st.sidebar.markdown("---")
st.sidebar.subheader("Automation")
autopilot_use_incoming = st.sidebar.checkbox(
    "Use incoming batch for monitoring",
    value=True,
    help="If enabled, automation will monitor an incoming batch and generate predictions.",
)
autopilot_use_drifted_demo = st.sidebar.checkbox(
    "Auto-load drifted demo batch",
    value=True,
    help="If no incoming file is uploaded, use a synthetic drifted batch automatically.",
)
autopilot_auto_retrain = st.sidebar.checkbox(
    "Auto-retrain if needed",
    value=True,
    help="If drift/performance checks fail and labels exist, retrain automatically.",
)

if st.sidebar.button("Run Automated Pipeline", use_container_width=True):
    incoming_for_automation = None
    if autopilot_use_incoming:
        incoming_for_automation = st.session_state.get("incoming_raw_df")
        if incoming_for_automation is None and autopilot_use_drifted_demo:
            incoming_for_automation = generate_drifted_batch()
            st.session_state["incoming_raw_df"] = incoming_for_automation

    result = run_automated_workflow(
        raw_dataset=raw_df,
        target_column=st.session_state["target_column"],
        incoming_dataset=incoming_for_automation,
        run_auto_retrain=autopilot_auto_retrain,
    )
    _apply_workflow_result(result)
    st.session_state["saved_model_path"] = save_bundle(st.session_state["bundle"])

    if result.get("retrained"):
        st.sidebar.success("Automation complete: model retrained and redeployed.")
    else:
        st.sidebar.success("Automation complete: cleaning, training, and monitoring finished.")

st.markdown("## 1. Upload and Understand the Data")
_metric_columns(
    [
        ("Rows", f"{len(raw_df):,}", "raw"),
        ("Columns", str(raw_df.shape[1]), "features"),
        ("Missing Cells", f"{int(raw_df.isna().sum().sum()):,}", "needs cleaning"),
        ("Duplicate Rows", f"{int(raw_df.duplicated().sum()):,}", "candidate removals"),
    ]
)
_render_dataframe_preview("Raw dataset preview", raw_df)

clean_left, clean_right = st.columns([1, 1.2], gap="large")
with clean_left:
    st.markdown("## 2. Smart Cleaner")
    if st.button("Run Smart Cleaning", type="primary", use_container_width=True):
        cleaned_df, quality_report = clean_dataset(raw_df, target_column=st.session_state["target_column"])
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["quality_report"] = quality_report.to_dict()
    if "quality_report" in st.session_state:
        report = st.session_state["quality_report"]
        st.metric("Data Quality Score", f"{report['quality_score']:.1f}%")
        for note in report["notes"]:
            st.write(f"- {note}")

with clean_right:
    if "quality_report" in st.session_state:
        st.plotly_chart(_quality_chart(st.session_state["quality_report"]), use_container_width=True)
        _render_dataframe_preview("Cleaned dataset preview", st.session_state["cleaned_df"])

st.markdown("## 3. Train the Best Model")
train_col, leaderboard_col = st.columns([0.9, 1.1], gap="large")
with train_col:
    if st.button("Train AutoML Stack", use_container_width=True):
        cleaned_df = st.session_state.get("cleaned_df")
        if cleaned_df is None:
            cleaned_df, quality_report = clean_dataset(raw_df, target_column=st.session_state["target_column"])
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["quality_report"] = quality_report.to_dict()
        bundle = train_best_model(st.session_state["cleaned_df"], st.session_state["target_column"])
        st.session_state["bundle"] = bundle
        st.session_state["saved_model_path"] = save_bundle(bundle)

    bundle = st.session_state.get("bundle")
    if bundle:
        primary_metric = bundle["primary_metric"]
        st.metric("Selected Model", bundle["best_model_name"], f"best by {primary_metric}")
        metric_lines = [f"`{name}`: `{value}`" for name, value in bundle["metrics"].items()]
        st.write(" | ".join(metric_lines))
        st.caption(f"Saved artifact: {st.session_state['saved_model_path']}")

with leaderboard_col:
    bundle = st.session_state.get("bundle")
    if bundle:
        st.plotly_chart(_leaderboard_chart(bundle), use_container_width=True)
        st.dataframe(pd.DataFrame(bundle["leaderboard"]), use_container_width=True, hide_index=True)

st.markdown("## 4. Deploy-Like Predictions")
bundle = st.session_state.get("bundle")
if bundle is None:
    st.info("Train a model to unlock predictions, monitoring, and auto-retraining.")
    st.stop()

pred_left, pred_right = st.columns([0.95, 1.05], gap="large")
with pred_left:
    st.caption("You can also serve this bundle with `uvicorn api:app --reload`.")
    if st.button("Load Drifted Demo Batch", use_container_width=True):
        st.session_state["incoming_raw_df"] = generate_drifted_batch()
        st.session_state.pop("incoming_upload_signature", None)
    uploaded_batch_file = st.file_uploader("Upload incoming batch CSV", type=["csv"], key="incoming_upload")
    if uploaded_batch_file is not None:
        signature = (uploaded_batch_file.name, uploaded_batch_file.size)
        if st.session_state.get("incoming_upload_signature") != signature:
            st.session_state["incoming_raw_df"] = _load_dataframe(uploaded_batch_file)
            st.session_state["incoming_upload_signature"] = signature

    incoming_raw_df = st.session_state.get("incoming_raw_df")
    if incoming_raw_df is not None:
        _render_dataframe_preview("Incoming batch preview", incoming_raw_df)

        if st.button("Score and Monitor Incoming Batch", type="primary", use_container_width=True):
            target_for_cleaning = bundle["target_column"] if bundle["target_column"] in incoming_raw_df.columns else None
            cleaned_incoming, incoming_quality = clean_dataset(incoming_raw_df, target_column=target_for_cleaning)
            predictions_df = predict_dataframe(bundle, cleaned_incoming)
            monitor_report = monitor_batch(bundle, cleaned_incoming)
            st.session_state["incoming_cleaned_df"] = cleaned_incoming
            st.session_state["incoming_quality_report"] = incoming_quality.to_dict()
            st.session_state["predictions_df"] = predictions_df
            st.session_state["monitor_report"] = monitor_report

        if "predictions_df" in st.session_state:
            st.download_button(
                label="Download Predictions CSV",
                data=_prepare_prediction_download(st.session_state["predictions_df"]),
                file_name="driftguard_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

with pred_right:
    if "predictions_df" in st.session_state:
        st.dataframe(st.session_state["predictions_df"].head(15), use_container_width=True, hide_index=True)

st.markdown("## 5. Drift Detection and 6. Auto Fix")
monitor_report = st.session_state.get("monitor_report")
if monitor_report:
    monitor_left, monitor_mid, monitor_right = st.columns([0.9, 1.1, 1.0], gap="large")
    with monitor_left:
        st.plotly_chart(_drift_indicator(monitor_report["overall_drift_score"]), use_container_width=True)
        if monitor_report["drift_detected"]:
            st.warning("Drift detected in the incoming batch.")
        else:
            st.success("Incoming data is stable against the training baseline.")

        stability = monitor_report["stability"]
        st.metric(
            f"Prediction {stability['stability_metric']}",
            f"{stability['current']}",
            f"delta {stability['delta']}",
        )

        performance = monitor_report.get("performance")
        if performance:
            st.metric(
                f"Batch {performance['primary_metric']}",
                f"{performance['current_score']}",
                f"drop {performance['performance_drop']}",
            )
        else:
            st.caption("No target labels were provided in the incoming batch, so performance drop could not be measured.")

    with monitor_mid:
        st.plotly_chart(_drift_chart(monitor_report), use_container_width=True)
        st.dataframe(pd.DataFrame(monitor_report["feature_scores"]), use_container_width=True, hide_index=True)

    with monitor_right:
        st.markdown("### Auto-Retrain Decision")
        if monitor_report["should_retrain"]:
            st.error("The system recommends retraining and replacing the current model.")
        else:
            st.success("Current production model can remain active.")

        incoming_cleaned_df = st.session_state.get("incoming_cleaned_df")
        can_retrain = incoming_cleaned_df is not None and bundle["target_column"] in incoming_cleaned_df.columns

        if st.button("Run Auto Fix Now", use_container_width=True, disabled=not can_retrain):
            updated_bundle = auto_retrain(bundle, incoming_cleaned_df)
            st.session_state["bundle"] = updated_bundle
            st.session_state["saved_model_path"] = save_bundle(updated_bundle)
            st.success("Model updated automatically with the drifted batch.")
            st.rerun()

        if not can_retrain:
            st.caption("Auto retraining needs labeled incoming data that includes the target column.")

st.markdown("## 7. Executive Dashboard")
dashboard_cols = st.columns(4)
quality_report = st.session_state.get("quality_report", {})
monitor_report = st.session_state.get("monitor_report", {})
dashboard_cols[0].metric("Quality Score", f"{quality_report.get('quality_score', 0):.1f}%")
dashboard_cols[1].metric("Best Model", bundle["best_model_name"])
dashboard_cols[2].metric("Primary Metric", f"{bundle['metrics'][bundle['primary_metric']]:.4f}")
dashboard_cols[3].metric("Drift Score", f"{monitor_report.get('overall_drift_score', 0):.4f}")

with st.expander("Narrative summary for presentation", expanded=True):
    summary_lines = [
        "1. Raw data enters the system through CSV upload or a live batch.",
        "2. The smart cleaner removes duplicates, fills gaps, and clips extreme outliers.",
        "3. AutoML benchmarks multiple models and keeps the top performer automatically.",
        "4. The saved bundle acts like a deployed model for predictions and API scoring.",
        "5. Incoming batches are compared with the training baseline to detect drift.",
        "6. If data drift or performance degradation appears, the system recommends retraining.",
        "7. With labeled batch data available, the model is retrained and replaced automatically.",
    ]
    for line in summary_lines:
        st.write(line)
