from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_demo_training_data(rows: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    usage_hours = rng.normal(37, 9, size=rows).clip(6, 70)
    ticket_count = rng.poisson(2.2, size=rows)
    payment_delay_days = rng.normal(3.8, 2.2, size=rows).clip(0, 14)
    satisfaction_score = rng.normal(78, 10, size=rows).clip(35, 99)
    plan_value = rng.normal(68, 16, size=rows).clip(15, 140)
    tenure_months = rng.gamma(5.0, 4.0, size=rows).clip(1, 72)
    region = rng.choice(["North", "South", "East", "West"], size=rows, p=[0.28, 0.22, 0.26, 0.24])
    device_type = rng.choice(["Mobile", "Laptop", "Tablet"], size=rows, p=[0.52, 0.34, 0.14])
    contract_type = rng.choice(["Monthly", "Quarterly", "Annual"], size=rows, p=[0.47, 0.31, 0.22])
    is_premium = rng.choice(["Yes", "No"], size=rows, p=[0.4, 0.6])

    logit = (
        -3.2
        + 0.12 * ticket_count
        + 0.08 * payment_delay_days
        - 0.05 * (satisfaction_score - 70)
        - 0.02 * tenure_months
        + 0.04 * (usage_hours - 35)
        + np.where(contract_type == "Monthly", 0.45, -0.1)
        + np.where(is_premium == "No", 0.35, -0.2)
        + np.where(region == "South", 0.18, 0.0)
    )
    churn_risk = rng.binomial(1, _sigmoid(logit))

    frame = pd.DataFrame(
        {
            "usage_hours": usage_hours.round(2),
            "ticket_count": ticket_count,
            "payment_delay_days": payment_delay_days.round(2),
            "satisfaction_score": satisfaction_score.round(2),
            "plan_value": plan_value.round(2),
            "tenure_months": tenure_months.round(0).astype(int),
            "region": region,
            "device_type": device_type,
            "contract_type": contract_type,
            "is_premium": is_premium,
            "churn_risk": churn_risk,
        }
    )

    for column in ["usage_hours", "payment_delay_days", "satisfaction_score", "region", "device_type"]:
        missing_indices = rng.choice(frame.index, size=max(4, rows // 28), replace=False)
        frame.loc[missing_indices, column] = np.nan

    outlier_indices = rng.choice(frame.index, size=max(5, rows // 40), replace=False)
    frame.loc[outlier_indices, "payment_delay_days"] *= 3.2
    frame.loc[outlier_indices, "usage_hours"] *= 1.7

    duplicate_rows = frame.sample(max(10, rows // 35), random_state=seed)
    frame = pd.concat([frame, duplicate_rows], ignore_index=True)
    return frame


def generate_drifted_batch(rows: int = 220, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    usage_hours = rng.normal(52, 11, size=rows).clip(8, 85)
    ticket_count = rng.poisson(4.5, size=rows)
    payment_delay_days = rng.normal(8.5, 3.2, size=rows).clip(0, 25)
    satisfaction_score = rng.normal(62, 12, size=rows).clip(15, 95)
    plan_value = rng.normal(86, 21, size=rows).clip(20, 180)
    tenure_months = rng.gamma(3.2, 5.0, size=rows).clip(1, 48)
    region = rng.choice(["North", "South", "East", "West"], size=rows, p=[0.12, 0.38, 0.18, 0.32])
    device_type = rng.choice(["Mobile", "Laptop", "Tablet"], size=rows, p=[0.22, 0.18, 0.60])
    contract_type = rng.choice(["Monthly", "Quarterly", "Annual"], size=rows, p=[0.70, 0.18, 0.12])
    is_premium = rng.choice(["Yes", "No"], size=rows, p=[0.22, 0.78])

    logit = (
        -2.1
        + 0.16 * ticket_count
        + 0.09 * payment_delay_days
        - 0.04 * (satisfaction_score - 60)
        - 0.018 * tenure_months
        + 0.03 * (usage_hours - 45)
        + np.where(contract_type == "Monthly", 0.55, -0.15)
        + np.where(is_premium == "No", 0.42, -0.12)
        + np.where(device_type == "Tablet", 0.3, 0.0)
    )
    churn_risk = rng.binomial(1, _sigmoid(logit))

    frame = pd.DataFrame(
        {
            "usage_hours": usage_hours.round(2),
            "ticket_count": ticket_count,
            "payment_delay_days": payment_delay_days.round(2),
            "satisfaction_score": satisfaction_score.round(2),
            "plan_value": plan_value.round(2),
            "tenure_months": tenure_months.round(0).astype(int),
            "region": region,
            "device_type": device_type,
            "contract_type": contract_type,
            "is_premium": is_premium,
            "churn_risk": churn_risk,
        }
    )

    for column in ["payment_delay_days", "satisfaction_score", "contract_type"]:
        missing_indices = rng.choice(frame.index, size=max(6, rows // 18), replace=False)
        frame.loc[missing_indices, column] = np.nan

    return frame
