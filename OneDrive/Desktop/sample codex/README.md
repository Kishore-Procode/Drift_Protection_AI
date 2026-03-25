# DataDrift Protection AI

An end-to-end MVP for the idea you described:

`Upload -> Clean -> Train -> Deploy -> Monitor -> Detect Drift -> Auto Retrain`

## What it does

- Upload a CSV or load a built-in demo dataset.
- Clean duplicates, missing values, and numeric outliers.
- Train multiple sklearn models and automatically keep the best one.
- Serve predictions through a Streamlit dashboard or a FastAPI endpoint.
- Compare incoming batches against the training baseline to detect drift.
- Retrain and replace the deployed model when drift or performance drop is detected.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the API in a second terminal if you want a deployment-style demo:

```powershell
.venv\Scripts\Activate.ps1
uvicorn api:app --reload
```

## API endpoints

- `GET /health`
- `GET /metadata`
- `POST /predict`

Example request body:

```json
{
  "records": [
    {
      "usage_hours": 38.2,
      "ticket_count": 2,
      "payment_delay_days": 3.0,
      "satisfaction_score": 81.0,
      "plan_value": 64.0,
      "tenure_months": 14,
      "region": "North",
      "device_type": "Mobile",
      "contract_type": "Annual",
      "is_premium": "Yes"
    }
  ]
}
```

## Project layout

- `app.py`: Streamlit product dashboard
- `api.py`: FastAPI prediction service
- `datadrift_ai/cleaning.py`: smart cleaning and quality scoring
- `datadrift_ai/modeling.py`: AutoML-style model training and predictions
- `datadrift_ai/drift.py`: drift and stability monitoring
- `datadrift_ai/demo_data.py`: built-in dataset generator for demos
- `tests/test_pipeline.py`: smoke test for the end-to-end flow

