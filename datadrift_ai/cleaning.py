from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_values_filled: int
    outliers_clipped: int
    target_rows_dropped: int
    quality_score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["quality_score"] = round(self.quality_score, 2)
        return payload


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned_names: list[str] = []
    seen: dict[str, int] = {}

    for index, column in enumerate(frame.columns):
        candidate = str(column).strip() or f"column_{index + 1}"
        seen[candidate] = seen.get(candidate, 0) + 1
        if seen[candidate] > 1:
            candidate = f"{candidate}_{seen[candidate]}"
        cleaned_names.append(candidate)

    normalized = frame.copy()
    normalized.columns = cleaned_names
    return normalized


def clean_dataset(
    dataset: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    if dataset.empty:
        raise ValueError("The dataset is empty. Upload at least one row to continue.")

    frame = _normalize_columns(dataset)
    rows_before = len(frame)
    duplicates_removed = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)

    normalized_target = target_column.strip() if isinstance(target_column, str) else None
    if normalized_target and normalized_target not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found after normalization.")

    missing_values_filled = 0
    outliers_clipped = 0
    target_rows_dropped = 0
    notes: list[str] = []

    for column in frame.select_dtypes(include=["object", "string", "category"]).columns:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if normalized_target:
        target_missing = int(frame[normalized_target].isna().sum())
        if target_missing:
            frame = frame.dropna(subset=[normalized_target]).reset_index(drop=True)
            target_rows_dropped += target_missing
            notes.append(f"Dropped {target_missing} rows with missing target labels.")

    feature_columns = [column for column in frame.columns if column != normalized_target]
    numeric_columns = frame[feature_columns].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]

    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        missing_before = int(series.isna().sum())
        fill_value = float(series.median()) if not series.dropna().empty else 0.0
        series = series.fillna(fill_value)
        missing_values_filled += missing_before

        if series.nunique() > 3:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                clipped_mask = (series < lower) | (series > upper)
                outliers_clipped += int(clipped_mask.sum())
                series = series.clip(lower=lower, upper=upper)

        frame[column] = series

    for column in categorical_columns:
        series = frame[column].astype("string")
        missing_before = int(series.isna().sum())
        fill_value = series.mode(dropna=True).iloc[0] if not series.dropna().empty else "Unknown"
        frame[column] = series.fillna(fill_value)
        missing_values_filled += missing_before

    total_feature_cells = max(len(frame) * max(len(feature_columns), 1), 1)
    duplicate_rate = duplicates_removed / max(rows_before, 1)
    missing_rate = missing_values_filled / total_feature_cells
    outlier_rate = outliers_clipped / total_feature_cells
    target_drop_rate = target_rows_dropped / max(rows_before, 1)

    quality_score = 100.0
    quality_score -= min(25.0, duplicate_rate * 120)
    quality_score -= min(35.0, missing_rate * 160)
    quality_score -= min(20.0, outlier_rate * 200)
    quality_score -= min(20.0, target_drop_rate * 100)
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    if duplicates_removed:
        notes.append(f"Removed {duplicates_removed} duplicate rows.")
    if missing_values_filled:
        notes.append(f"Filled {missing_values_filled} missing values.")
    if outliers_clipped:
        notes.append(f"Clipped {outliers_clipped} outlier values using the IQR rule.")
    if not notes:
        notes.append("Dataset was already in strong shape with minimal cleaning needed.")

    report = DataQualityReport(
        rows_before=rows_before,
        rows_after=len(frame),
        duplicates_removed=duplicates_removed,
        missing_values_filled=missing_values_filled,
        outliers_clipped=outliers_clipped,
        target_rows_dropped=target_rows_dropped,
        quality_score=quality_score,
        notes=notes,
    )
    return frame.reset_index(drop=True), report

