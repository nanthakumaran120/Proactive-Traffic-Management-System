"""Strict dataset loading and validation for the Gandhipuram, Coimbatore CSV."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "gandhipuram_coimbatore_traffic_synthetic.csv"
VALID_TRAFFIC_LABELS = ("Low", "Medium", "High")
REQUIRED_COLUMNS = (
    "Timestamp",
    "Location",
    "Latitude",
    "Longitude",
    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_pct",
    "Traffic_Light_State",
    "Weather_Condition",
    "Accident_Report",
    "Traffic_Condition",
)


@dataclass
class ValidationReport:
    """Structured summary of Gandhipuram dataset validation checks."""

    dataset_path: str
    raw_row_count: int
    empty_rows_removed: int
    valid_row_count: int
    duplicate_rows: int
    duplicate_location_timestamp: int
    missing_values_after_cleaning: dict[str, int]
    invalid_timestamp_count: int
    invalid_numeric_counts: dict[str, int]
    invalid_label_count: int
    valid_labels_found: list[str]
    junction_count: int
    junction_names: list[str]
    timestamp_start: str
    timestamp_end: str
    unique_timestamps: int
    condition_distribution: dict[str, int]
    weather_values: list[str]
    signal_values: list[str]
    accident_values: list[str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


def load_and_validate_gandhipuram_data(
    dataset_path: Path | str | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load, clean (in-memory without mutating the source CSV), and validate the dataset.

    Args:
        dataset_path: Optional override path. Defaults to
            ``datasets/gandhipuram_coimbatore_traffic_synthetic.csv``.

    Returns:
        Tuple of ``(cleaned_dataframe, ValidationReport)``.
    """
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Required Gandhipuram dataset not found: {path}")

    raw_df = pd.read_csv(path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    raw_row_count = int(len(raw_df))

    # Drop completely empty rows (such as the 22 trailing blank CSV rows).
    df = raw_df.dropna(how="all").copy()
    empty_rows_removed = raw_row_count - int(len(df))

    # Strip whitespace in string columns safely.
    str_cols = [
        "Timestamp",
        "Location",
        "Traffic_Light_State",
        "Weather_Condition",
        "Accident_Report",
        "Traffic_Condition",
    ]
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()

    # Parse timestamps strictly in day-first format (DD-MM-YYYY HH:MM).
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    invalid_timestamp_count = int(df["Timestamp_dt"].isna().sum())

    # Validate numerical columns.
    numeric_cols = [
        "Latitude",
        "Longitude",
        "Vehicle_Count",
        "Traffic_Speed_kmh",
        "Road_Occupancy_pct",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_numeric_counts = {
        "Latitude": int((df["Latitude"].isna() | ~np.isfinite(df["Latitude"])).sum()),
        "Longitude": int((df["Longitude"].isna() | ~np.isfinite(df["Longitude"])).sum()),
        "Vehicle_Count": int((df["Vehicle_Count"].isna() | (df["Vehicle_Count"] <= 0)).sum()),
        "Traffic_Speed_kmh": int((df["Traffic_Speed_kmh"].isna() | (df["Traffic_Speed_kmh"] < 0)).sum()),
        "Road_Occupancy_pct": int(
            (df["Road_Occupancy_pct"].isna() | (df["Road_Occupancy_pct"] < 0) | (df["Road_Occupancy_pct"] > 100)).sum()
        ),
    }

    # Check missing values and duplicates before dropping invalid records.
    missing_values_after_cleaning = {
        col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS
    }
    duplicate_rows = int(df.duplicated(subset=list(REQUIRED_COLUMNS)).sum())
    duplicate_location_timestamp = int(df.duplicated(subset=["Timestamp_dt", "Location"]).sum())

    # Validate traffic condition labels.
    valid_label_mask = df["Traffic_Condition"].isin(VALID_TRAFFIC_LABELS)
    invalid_label_count = int((~valid_label_mask).sum())
    valid_labels_found = sorted(df.loc[valid_label_mask, "Traffic_Condition"].dropna().unique().tolist())

    # Filter down to strictly valid rows for downstream analytics/ML.
    valid_mask = (
        df["Timestamp_dt"].notna()
        & valid_label_mask
        & df["Location"].notna()
        & (df["Vehicle_Count"] > 0)
        & (df["Traffic_Speed_kmh"] >= 0)
        & (df["Road_Occupancy_pct"] >= 0)
        & (df["Road_Occupancy_pct"] <= 100)
    )
    cleaned_df = df.loc[valid_mask].drop_duplicates().copy()
    cleaned_df = cleaned_df.sort_values(["Timestamp_dt", "Location"]).reset_index(drop=True)

    # Convenience time features (derived without leakage).
    cleaned_df["Hour"] = cleaned_df["Timestamp_dt"].dt.hour
    cleaned_df["Minute"] = cleaned_df["Timestamp_dt"].dt.minute
    cleaned_df["Hour_Float"] = cleaned_df["Hour"] + cleaned_df["Minute"] / 60.0
    cleaned_df["DayOfWeek"] = cleaned_df["Timestamp_dt"].dt.dayofweek
    cleaned_df["DayName"] = cleaned_df["Timestamp_dt"].dt.day_name()

    junction_names = sorted(cleaned_df["Location"].unique().tolist())
    condition_dist = {
        str(k): int(v)
        for k, v in cleaned_df["Traffic_Condition"].value_counts().to_dict().items()
    }

    passed = (
        len(cleaned_df) > 0
        and invalid_timestamp_count == 0
        and invalid_label_count == 0
        and sum(invalid_numeric_counts.values()) == 0
        and sum(missing_values_after_cleaning.values()) == 0
        and duplicate_rows == 0
        and duplicate_location_timestamp == 0
    )

    report = ValidationReport(
        dataset_path=str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        raw_row_count=raw_row_count,
        empty_rows_removed=empty_rows_removed,
        valid_row_count=int(len(cleaned_df)),
        duplicate_rows=duplicate_rows,
        duplicate_location_timestamp=duplicate_location_timestamp,
        missing_values_after_cleaning=missing_values_after_cleaning,
        invalid_timestamp_count=invalid_timestamp_count,
        invalid_numeric_counts=invalid_numeric_counts,
        invalid_label_count=invalid_label_count,
        valid_labels_found=valid_labels_found,
        junction_count=len(junction_names),
        junction_names=junction_names,
        timestamp_start=cleaned_df["Timestamp_dt"].min().strftime("%d-%m-%Y %H:%M"),
        timestamp_end=cleaned_df["Timestamp_dt"].max().strftime("%d-%m-%Y %H:%M"),
        unique_timestamps=int(cleaned_df["Timestamp_dt"].nunique()),
        condition_distribution=condition_dist,
        weather_values=sorted(cleaned_df["Weather_Condition"].unique().tolist()),
        signal_values=sorted(cleaned_df["Traffic_Light_State"].unique().tolist()),
        accident_values=sorted(cleaned_df["Accident_Report"].unique().tolist()),
        passed=passed,
    )
    return cleaned_df, report


def get_junction_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract canonical coordinate table for the six Gandhipuram junctions."""
    coords = (
        df.groupby("Location", as_index=False)[["Latitude", "Longitude"]]
        .mean()
        .sort_values("Location")
        .reset_index(drop=True)
    )
    return coords

