"""Module 1: Future Traffic Prediction, Chronological Validation, and Proactive Alerts.

Uses ONLY datasets/gandhipuram_coimbatore_traffic_synthetic.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from src.validation import PROJECT_ROOT, load_and_validate_gandhipuram_data


MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "gandhipuram_classifier.pkl"
REG_COUNT_PATH = MODELS_DIR / "gandhipuram_reg_vehicle_count.pkl"
REG_SPEED_PATH = MODELS_DIR / "gandhipuram_reg_speed.pkl"
REG_OCC_PATH = MODELS_DIR / "gandhipuram_reg_occupancy.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "gandhipuram_label_encoder.pkl"
METADATA_PATH = MODELS_DIR / "gandhipuram_training_metadata.json"

CONDITION_TO_NUM = {"Low": 0, "Medium": 1, "High": 2}
CATEGORICAL_FEATURES = [
    "Location",
    "Origin_Weather",
    "Origin_Signal",
    "Origin_Accident",
    "Target_Weather",
    "Target_Accident",
]
NUMERIC_FEATURES = [
    "Origin_Hour",
    "Origin_DayOfWeek",
    "Origin_Vehicle_Count",
    "Origin_Traffic_Speed_kmh",
    "Origin_Road_Occupancy_pct",
    "Origin_Cond_Num",
    "Target_Hour",
    "Target_Hour_Sin",
    "Target_Hour_Cos",
    "Target_DayOfWeek",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DATASET_LIMITATION_NOTE = (
    "Dataset Limitation Notice: Gandhipuram synthetic dataset covers 5 weekdays "
    "(12-10-2026 to 16-10-2026, 07:30-20:00, 218 valid snapshots across 6 junctions, "
    "with only 3 recorded accident events). Forecasts are statistical ML projections, "
    "not live ground-truth sensor feeds; predictions outside 07:30-20:00 extrapolate "
    "diurnal trends with reduced statistical confidence."
)


@dataclass
class JunctionForecast:
    """Forecast output for a single junction at a specific future horizon."""

    junction: str
    latitude: float
    longitude: float
    base_timestamp: str
    target_timestamp: str
    horizon_hours: int
    current_condition: str
    current_vehicle_count: float
    current_speed_kmh: float
    current_occupancy_pct: float
    predicted_condition: str
    predicted_vehicle_count: float
    predicted_speed_kmh: float
    predicted_occupancy_pct: float
    congestion_probability: float
    class_probabilities: dict[str, float]
    congestion_risk_score: float
    prediction_confidence: float
    weather_scenario: str
    accident_scenario: str
    is_outside_observed_hours: bool
    scenario_adjusted: bool
    forecast_note: str
    is_model_forecast: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProactiveAlert:
    """Proactive congestion alert generated from multi-horizon forecasts."""

    alert_level: str  # CRITICAL | WARNING | NORMAL
    junction: str
    predicted_time: str
    horizon_hours: int
    predicted_condition: str
    congestion_probability: float
    congestion_risk_score: float
    occupancy_change_pct: float
    recommended_action: str
    reason: str
    is_predicted_alert: bool = True
    label_tag: str = "MODEL FORECAST ALERT (Not a live incident)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_horizon_pairs(df: pd.DataFrame, max_horizon_hours: float = 6.5) -> pd.DataFrame:
    """Create leak-free chronological (origin_t -> target_t+h) training pairs per junction.

    Every feature in a row is observable either at ``Origin_Timestamp`` or is a known
    calendar/scenario attribute of ``Target_Timestamp``. Future traffic signals or target
    observations are strictly excluded from the feature columns.
    """
    records: list[dict[str, Any]] = []
    sorted_df = df.sort_values(["Location", "Timestamp_dt"]).reset_index(drop=True)

    for loc, grp in sorted_df.groupby("Location"):
        grp = grp.sort_values("Timestamp_dt").reset_index(drop=True)
        n = len(grp)
        for i in range(n):
            t_org = grp.loc[i, "Timestamp_dt"]
            oh = float(t_org.hour + t_org.minute / 60.0)
            for j in range(i + 1, n):
                t_tgt = grp.loc[j, "Timestamp_dt"]
                dt_h = (t_tgt - t_org).total_seconds() / 3600.0
                if 0.5 <= dt_h <= max_horizon_hours:
                    th = float(t_tgt.hour + t_tgt.minute / 60.0)
                    records.append(
                        {
                            "Location": str(loc),
                            "Origin_Timestamp": t_org,
                            "Target_Timestamp": t_tgt,
                            "Target_Date": str(t_tgt.date()),
                            "Horizon_Hours": round(dt_h, 2),
                            "Origin_Hour": oh,
                            "Origin_DayOfWeek": int(t_org.dayofweek),
                            "Origin_Vehicle_Count": float(grp.loc[i, "Vehicle_Count"]),
                            "Origin_Traffic_Speed_kmh": float(grp.loc[i, "Traffic_Speed_kmh"]),
                            "Origin_Road_Occupancy_pct": float(grp.loc[i, "Road_Occupancy_pct"]),
                            "Origin_Cond_Num": int(CONDITION_TO_NUM[str(grp.loc[i, "Traffic_Condition"])]),
                            "Origin_Condition": str(grp.loc[i, "Traffic_Condition"]),
                            "Origin_Weather": str(grp.loc[i, "Weather_Condition"]),
                            "Origin_Signal": str(grp.loc[i, "Traffic_Light_State"]),
                            "Origin_Accident": str(grp.loc[i, "Accident_Report"]),
                            "Target_Hour": th,
                            "Target_Hour_Sin": float(np.sin(2.0 * np.pi * th / 24.0)),
                            "Target_Hour_Cos": float(np.cos(2.0 * np.pi * th / 24.0)),
                            "Target_DayOfWeek": int(t_tgt.dayofweek),
                            "Target_Weather": str(grp.loc[j, "Weather_Condition"]),
                            "Target_Accident": str(grp.loc[j, "Accident_Report"]),
                            "Target_Vehicle_Count": float(grp.loc[j, "Vehicle_Count"]),
                            "Target_Traffic_Speed_kmh": float(grp.loc[j, "Traffic_Speed_kmh"]),
                            "Target_Road_Occupancy_pct": float(grp.loc[j, "Road_Occupancy_pct"]),
                            "Target_Traffic_Condition": str(grp.loc[j, "Traffic_Condition"]),
                        }
                    )

    pdf = pd.DataFrame(records).sort_values(["Target_Timestamp", "Location"]).reset_index(drop=True)
    return pdf


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _classifier_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGBOOST:
        candidates["XGBoost"] = lambda: XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.06,
            random_state=42,
            eval_metric="mlogloss",
        )
    return candidates


def _regressor_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "Ridge Regression": lambda: Ridge(alpha=1.0),
    }
    if HAS_XGBOOST:
        candidates["XGBoost Regressor"] = lambda: XGBRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.06, random_state=42
        )
        candidates["Ensemble (RF + XGBoost)"] = lambda: VotingRegressor(
            [
                ("rf", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)),
                ("xgb", XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.06, random_state=42)),
            ]
        )
    return candidates


def train_and_evaluate_models(
    dataset_path: Path | str | None = None,
    models_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train classifiers and regressors using chronological expanding-window validation."""
    df, validation_report = load_and_validate_gandhipuram_data(dataset_path)
    pairs_df = build_horizon_pairs(df)
    if pairs_df.empty:
        raise RuntimeError("No chronological forecasting pairs could be constructed from the dataset.")

    out_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    le.fit(["High", "Low", "Medium"])

    dates = sorted(pairs_df["Target_Date"].unique().tolist())
    # Hold out Days 3..N chronologically (expanding window: train < date, test == date)
    eval_dates = dates[2:] if len(dates) >= 3 else dates[-1:]

    # --- 1. Classification benchmark (chronological expanding window) ---
    clf_candidates = _classifier_candidates()
    clf_results: list[dict[str, Any]] = []
    clf_oof: dict[str, tuple[list[int], list[int], pd.DataFrame]] = {}

    for name, factory in clf_candidates.items():
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        test_frames: list[pd.DataFrame] = []
        for eval_date in eval_dates:
            tr = pairs_df[pairs_df["Target_Date"] < eval_date]
            te = pairs_df[pairs_df["Target_Date"] == eval_date]
            if tr.empty or te.empty:
                continue
            pipe = Pipeline([("pre", _make_preprocessor()), ("clf", factory())])
            pipe.fit(tr[ALL_FEATURES], le.transform(tr["Target_Traffic_Condition"]))
            preds = pipe.predict(te[ALL_FEATURES])
            y_true_all.extend(le.transform(te["Target_Traffic_Condition"]).tolist())
            y_pred_all.extend(np.asarray(preds, dtype=int).tolist())
            test_frames.append(te)

        eval_df = pd.concat(test_frames, ignore_index=True)
        acc = float(accuracy_score(y_true_all, y_pred_all))
        prec = float(precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
        rec = float(recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0))
        clf_results.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "evaluated_samples": len(y_true_all),
            }
        )
        clf_oof[name] = (y_true_all, y_pred_all, eval_df)

    # Persistence baseline on exact same chronological folds
    pers_true: list[int] = []
    pers_pred: list[int] = []
    for eval_date in eval_dates:
        te = pairs_df[pairs_df["Target_Date"] == eval_date]
        if te.empty:
            continue
        pers_true.extend(le.transform(te["Target_Traffic_Condition"]).tolist())
        pers_pred.extend(le.transform(te["Origin_Condition"]).tolist())
    persistence_acc = float(accuracy_score(pers_true, pers_pred))
    persistence_f1 = float(f1_score(pers_true, pers_pred, average="weighted", zero_division=0))

    clf_results_sorted = sorted(clf_results, key=lambda r: (r["f1_score"], r["accuracy"]), reverse=True)
    best_clf_name = clf_results_sorted[0]["model"]
    best_y_true, best_y_pred, best_eval_df = clf_oof[best_clf_name]
    cm = confusion_matrix(best_y_true, best_y_pred, labels=[0, 1, 2]).tolist()
    clf_report_dict = classification_report(
        best_y_true,
        best_y_pred,
        labels=[0, 1, 2],
        target_names=le.classes_.tolist(),
        output_dict=True,
        zero_division=0,
    )

    # --- 2. Regression benchmark (count, speed, occupancy) ---
    reg_targets = {
        "vehicle_count": "Target_Vehicle_Count",
        "speed_kmh": "Target_Traffic_Speed_kmh",
        "occupancy_pct": "Target_Road_Occupancy_pct",
    }
    reg_candidates = _regressor_candidates()
    reg_metrics: dict[str, list[dict[str, Any]]] = {}
    best_reg_names: dict[str, str] = {}
    reg_oof_preds: dict[str, list[float]] = {}

    for key, tgt_col in reg_targets.items():
        target_rows: list[dict[str, Any]] = []
        preds_by_model: dict[str, list[float]] = {}
        for rname, rfactory in reg_candidates.items():
            yt_all: list[float] = []
            yp_all: list[float] = []
            for eval_date in eval_dates:
                tr = pairs_df[pairs_df["Target_Date"] < eval_date]
                te = pairs_df[pairs_df["Target_Date"] == eval_date]
                if tr.empty or te.empty:
                    continue
                pipe = Pipeline([("pre", _make_preprocessor()), ("reg", rfactory())])
                pipe.fit(tr[ALL_FEATURES], tr[tgt_col])
                yp = pipe.predict(te[ALL_FEATURES])
                yt_all.extend(te[tgt_col].astype(float).tolist())
                yp_all.extend(np.asarray(yp, dtype=float).tolist())
            mae = float(mean_absolute_error(yt_all, yp_all))
            rmse = float(np.sqrt(mean_squared_error(yt_all, yp_all)))
            target_rows.append(
                {
                    "target": key,
                    "model": rname,
                    "mae": round(mae, 3),
                    "rmse": round(rmse, 3),
                    "evaluated_samples": len(yt_all),
                }
            )
            preds_by_model[rname] = yp_all

        target_rows_sorted = sorted(target_rows, key=lambda r: (r["mae"], r["rmse"]))
        reg_metrics[key] = target_rows_sorted
        best_reg_name = target_rows_sorted[0]["model"]
        best_reg_names[key] = best_reg_name
        reg_oof_preds[key] = preds_by_model[best_reg_name]

    # Sample actual vs predicted records from chronological evaluation
    eval_preview = pd.DataFrame(
        {
            "Target_Timestamp": best_eval_df["Target_Timestamp"].dt.strftime("%d-%m-%Y %H:%M"),
            "Location": best_eval_df["Location"],
            "Horizon_Hours": best_eval_df["Horizon_Hours"],
            "Actual_Condition": le.inverse_transform(best_y_true),
            "Predicted_Condition": le.inverse_transform(best_y_pred),
            "Actual_Vehicle_Count": best_eval_df["Target_Vehicle_Count"].round(1),
            "Predicted_Vehicle_Count": np.round(reg_oof_preds["vehicle_count"], 1),
            "Actual_Speed_kmh": best_eval_df["Target_Traffic_Speed_kmh"].round(2),
            "Predicted_Speed_kmh": np.round(reg_oof_preds["speed_kmh"], 2),
            "Actual_Occupancy_pct": best_eval_df["Target_Road_Occupancy_pct"].round(2),
            "Predicted_Occupancy_pct": np.round(reg_oof_preds["occupancy_pct"], 2),
        }
    )

    # --- 3. Refit best models on full validated horizon-pair dataset ---
    final_clf = Pipeline([("pre", _make_preprocessor()), ("clf", clf_candidates[best_clf_name]())])
    final_clf.fit(pairs_df[ALL_FEATURES], le.transform(pairs_df["Target_Traffic_Condition"]))

    final_reg_count = Pipeline(
        [("pre", _make_preprocessor()), ("reg", reg_candidates[best_reg_names["vehicle_count"]]())]
    )
    final_reg_count.fit(pairs_df[ALL_FEATURES], pairs_df["Target_Vehicle_Count"])

    final_reg_speed = Pipeline(
        [("pre", _make_preprocessor()), ("reg", reg_candidates[best_reg_names["speed_kmh"]]())]
    )
    final_reg_speed.fit(pairs_df[ALL_FEATURES], pairs_df["Target_Traffic_Speed_kmh"])

    final_reg_occ = Pipeline(
        [("pre", _make_preprocessor()), ("reg", reg_candidates[best_reg_names["occupancy_pct"]]())]
    )
    final_reg_occ.fit(pairs_df[ALL_FEATURES], pairs_df["Target_Road_Occupancy_pct"])

    joblib.dump(final_clf, out_dir / CLASSIFIER_PATH.name)
    joblib.dump(final_reg_count, out_dir / REG_COUNT_PATH.name)
    joblib.dump(final_reg_speed, out_dir / REG_SPEED_PATH.name)
    joblib.dump(final_reg_occ, out_dir / REG_OCC_PATH.name)
    joblib.dump(le, out_dir / LABEL_ENCODER_PATH.name)

    metadata: dict[str, Any] = {
        "validation_report": validation_report.to_dict(),
        "total_horizon_pairs": int(len(pairs_df)),
        "chronological_eval_dates": eval_dates,
        "features": ALL_FEATURES,
        "best_classifier": best_clf_name,
        "classifier_comparison": clf_results_sorted,
        "persistence_baseline": {
            "model": "Persistence Baseline (Copy Current Condition)",
            "accuracy": round(persistence_acc, 4),
            "f1_score": round(persistence_f1, 4),
            "evaluated_samples": len(pers_true),
        },
        "confusion_matrix": {
            "labels": le.classes_.tolist(),
            "matrix": cm,
        },
        "classification_report": clf_report_dict,
        "best_regressors": best_reg_names,
        "regression_comparison": reg_metrics,
        "chronological_actual_vs_predicted": eval_preview.to_dict(orient="records"),
        "limitation_note": DATASET_LIMITATION_NOTE,
    }

    with open(out_dir / METADATA_PATH.name, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class FutureTrafficPredictor:
    """Loads trained Gandhipuram models and generates future forecasts and proactive alerts."""

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        models_dir: Path | str | None = None,
        auto_train: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.df, self.validation_report = load_and_validate_gandhipuram_data(self.dataset_path)

        required_files = [
            self.models_dir / CLASSIFIER_PATH.name,
            self.models_dir / REG_COUNT_PATH.name,
            self.models_dir / REG_SPEED_PATH.name,
            self.models_dir / REG_OCC_PATH.name,
            self.models_dir / LABEL_ENCODER_PATH.name,
            self.models_dir / METADATA_PATH.name,
        ]
        if not all(p.exists() for p in required_files):
            if not auto_train:
                raise FileNotFoundError(f"Trained models missing under {self.models_dir}")
            train_and_evaluate_models(self.dataset_path, self.models_dir)

        self.classifier = joblib.load(self.models_dir / CLASSIFIER_PATH.name)
        self.reg_count = joblib.load(self.models_dir / REG_COUNT_PATH.name)
        self.reg_speed = joblib.load(self.models_dir / REG_SPEED_PATH.name)
        self.reg_occ = joblib.load(self.models_dir / REG_OCC_PATH.name)
        self.label_encoder: LabelEncoder = joblib.load(self.models_dir / LABEL_ENCODER_PATH.name)
        with open(self.models_dir / METADATA_PATH.name, encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.junctions = sorted(self.df["Location"].unique().tolist())

    def get_feature_importances(self, top_n: int = 12) -> list[dict[str, Any]]:
        """Extract feature importances from the trained classifier when supported."""
        pre = self.classifier.named_steps.get("pre")
        clf = self.classifier.named_steps.get("clf")
        if pre is None or clf is None or not hasattr(clf, "feature_importances_"):
            return []
        raw_names = pre.get_feature_names_out()
        importances = clf.feature_importances_
        pairs = []
        for raw_name, score in zip(raw_names, importances):
            clean = (
                str(raw_name)
                .replace("num__", "")
                .replace("cat__", "")
                .replace("_", " ")
            )
            pairs.append({"feature": clean, "importance": round(float(score), 4)})
        pairs.sort(key=lambda item: item["importance"], reverse=True)
        return pairs[:top_n]

    def get_full_network_timestamps(self) -> list[pd.Timestamp]:
        """Return timestamps where all six junctions have simultaneous observations."""
        counts = self.df.groupby("Timestamp_dt")["Location"].nunique()
        full_ts = sorted(counts[counts == len(self.junctions)].index.tolist())
        return full_ts

    def get_latest_common_snapshot(self, base_timestamp: pd.Timestamp | str | None = None) -> pd.DataFrame:
        """Return the 6-junction snapshot at ``base_timestamp`` (default: latest full-network snapshot)."""
        if base_timestamp is not None:
            ts = pd.to_datetime(base_timestamp, dayfirst=True)
            snap = self.df[self.df["Timestamp_dt"] == ts].copy()
            if len(snap) == len(self.junctions):
                return snap.sort_values("Location").reset_index(drop=True)

        full_ts = self.get_full_network_timestamps()
        latest_full = full_ts[-1] if full_ts else self.df["Timestamp_dt"].max()
        return (
            self.df[self.df["Timestamp_dt"] == latest_full]
            .sort_values("Location")
            .reset_index(drop=True)
        )

    def get_latest_per_junction_snapshot(self) -> pd.DataFrame:
        """Return each junction's single most recent observation in the dataset."""
        idx = self.df.groupby("Location")["Timestamp_dt"].idxmax()
        return self.df.loc[idx].sort_values("Location").reset_index(drop=True)

    @staticmethod
    def _compute_risk_score(
        probs: dict[str, float],
        pred_speed: float,
        pred_occ: float,
    ) -> float:
        prob_component = 100.0 * (
            0.0 * probs.get("Low", 0.0)
            + 0.50 * probs.get("Medium", 0.0)
            + 1.00 * probs.get("High", 0.0)
        )
        occ_norm = float(np.clip(pred_occ / 85.0, 0.0, 1.0))
        speed_norm = float(np.clip((35.0 - pred_speed) / 30.0, 0.0, 1.0))
        pressure_component = 100.0 * (0.5 * occ_norm + 0.5 * speed_norm)
        return float(np.clip(round(0.65 * prob_component + 0.35 * pressure_component, 1), 0.0, 100.0))

    @staticmethod
    def _apply_scenario_adjustment(
        probs: dict[str, float],
        pred_count: float,
        pred_speed: float,
        pred_occ: float,
        weather_scenario: str,
        accident_scenario: str,
    ) -> tuple[dict[str, float], float, float, float, bool]:
        """Apply transparent scenario stress factors when rare incidents/rain are simulated.

        The historical CSV contains only 3 accident records (all at rush hour). To let
        operators simulate off-peak incidents honestly without fabricating training rows,
        explicit incident/rain penalties are applied and flagged as ``scenario_adjusted``.
        """
        adjusted = False
        p_low = probs.get("Low", 0.0)
        p_med = probs.get("Medium", 0.0)
        p_high = probs.get("High", 0.0)

        if weather_scenario == "Light Rain":
            pred_speed = max(5.0, pred_speed * 0.88)
            pred_occ = min(98.0, pred_occ + 5.5)
            pred_count = min(110.0, pred_count * 1.05)
            shift = 0.20 * p_low + 0.15 * p_med
            p_low = max(0.0, p_low - 0.20 * p_low)
            p_med = max(0.0, p_med + 0.20 * probs.get("Low", 0.0) - 0.15 * p_med)
            p_high = min(1.0, p_high + 0.15 * probs.get("Medium", 0.0))
            adjusted = True

        if accident_scenario == "Yes":
            pred_speed = max(5.0, pred_speed * 0.55)
            pred_occ = min(99.0, pred_occ + 22.0)
            pred_count = min(120.0, pred_count * 1.18)
            # Incident strongly shifts congestion risk toward High
            p_high = max(p_high, 0.78)
            rem = max(0.0, 1.0 - p_high)
            p_med = rem * 0.85
            p_low = rem * 0.15
            adjusted = True

        total = p_low + p_med + p_high
        if total > 0:
            probs_out = {
                "Low": round(p_low / total, 4),
                "Medium": round(p_med / total, 4),
                "High": round(p_high / total, 4),
            }
        else:
            probs_out = {"Low": 0.0, "Medium": 1.0, "High": 0.0}

        return probs_out, round(pred_count, 1), round(pred_speed, 2), round(pred_occ, 2), adjusted

    def predict_horizon(
        self,
        horizon_hours: int = 1,
        base_timestamp: pd.Timestamp | str | None = None,
        use_per_junction_latest: bool = False,
        weather_override: str | None = None,
        accident_junctions: set[str] | list[str] | None = None,
        junctions: list[str] | None = None,
    ) -> list[JunctionForecast]:
        """Predict future traffic metrics for selected (or all 6) junctions at +horizon_hours."""
        if horizon_hours < 0 or horizon_hours > 6:
            raise ValueError("horizon_hours must be between 0 and 6.")

        snap = (
            self.get_latest_per_junction_snapshot()
            if use_per_junction_latest and base_timestamp is None
            else self.get_latest_common_snapshot(base_timestamp)
        )
        if junctions is not None:
            snap = snap[snap["Location"].isin(junctions)].copy()

        accident_set = set(accident_junctions or [])
        classes = self.label_encoder.classes_.tolist()

        if horizon_hours == 0:
            # Horizon 0 represents the selected observed dataset snapshot + optional scenario
            out_zero: list[JunctionForecast] = []
            for _, row in snap.iterrows():
                loc = str(row["Location"])
                w = weather_override if weather_override else str(row["Weather_Condition"])
                acc = "Yes" if loc in accident_set else str(row["Accident_Report"])
                cond = str(row["Traffic_Condition"])
                probs = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
                probs[cond] = 1.0
                vc = float(row["Vehicle_Count"])
                sp = float(row["Traffic_Speed_kmh"])
                oc = float(row["Road_Occupancy_pct"])
                scenario_adj = False
                if (weather_override and weather_override != str(row["Weather_Condition"])) or (loc in accident_set):
                    probs, vc, sp, oc, scenario_adj = self._apply_scenario_adjustment(
                        probs, vc, sp, oc, w, acc
                    )
                    cond = max(probs.items(), key=lambda kv: kv[1])[0]
                risk = self._compute_risk_score(probs, sp, oc)
                out_zero.append(
                    JunctionForecast(
                        junction=loc,
                        latitude=float(row["Latitude"]),
                        longitude=float(row["Longitude"]),
                        base_timestamp=row["Timestamp_dt"].strftime("%d-%m-%Y %H:%M"),
                        target_timestamp=row["Timestamp_dt"].strftime("%d-%m-%Y %H:%M"),
                        horizon_hours=0,
                        current_condition=str(row["Traffic_Condition"]),
                        current_vehicle_count=float(row["Vehicle_Count"]),
                        current_speed_kmh=float(row["Traffic_Speed_kmh"]),
                        current_occupancy_pct=float(row["Road_Occupancy_pct"]),
                        predicted_condition=cond,
                        predicted_vehicle_count=round(vc, 1),
                        predicted_speed_kmh=round(sp, 2),
                        predicted_occupancy_pct=round(oc, 2),
                        congestion_probability=round(probs["High"], 4),
                        class_probabilities=probs,
                        congestion_risk_score=risk,
                        prediction_confidence=95.0 if not scenario_adj else 78.0,
                        weather_scenario=w,
                        accident_scenario=acc,
                        is_outside_observed_hours=False,
                        scenario_adjusted=scenario_adj,
                        forecast_note="Dataset Observation Snapshot" + (" (Scenario Adjusted)" if scenario_adj else ""),
                        is_model_forecast=scenario_adj,
                    )
                )
            return out_zero

        feat_rows: list[dict[str, Any]] = []
        meta_rows: list[tuple[pd.Series, pd.Timestamp, str, str, float]] = []

        for _, row in snap.iterrows():
            loc = str(row["Location"])
            t_org = row["Timestamp_dt"]
            t_tgt = t_org + pd.Timedelta(hours=horizon_hours)
            th = float(t_tgt.hour + t_tgt.minute / 60.0)
            oh = float(t_org.hour + t_org.minute / 60.0)
            w = weather_override if weather_override else str(row["Weather_Condition"])
            acc = "Yes" if loc in accident_set else "No"

            feat_rows.append(
                {
                    "Location": loc,
                    "Origin_Weather": str(row["Weather_Condition"]),
                    "Origin_Signal": str(row["Traffic_Light_State"]),
                    "Origin_Accident": str(row["Accident_Report"]),
                    "Target_Weather": w,
                    "Target_Accident": acc,
                    "Origin_Hour": oh,
                    "Origin_DayOfWeek": int(t_org.dayofweek),
                    "Origin_Vehicle_Count": float(row["Vehicle_Count"]),
                    "Origin_Traffic_Speed_kmh": float(row["Traffic_Speed_kmh"]),
                    "Origin_Road_Occupancy_pct": float(row["Road_Occupancy_pct"]),
                    "Origin_Cond_Num": int(CONDITION_TO_NUM[str(row["Traffic_Condition"])]),
                    "Target_Hour": th,
                    "Target_Hour_Sin": float(np.sin(2.0 * np.pi * th / 24.0)),
                    "Target_Hour_Cos": float(np.cos(2.0 * np.pi * th / 24.0)),
                    "Target_DayOfWeek": int(t_tgt.dayofweek),
                }
            )
            meta_rows.append((row, t_tgt, w, acc, th))

        X_pred = pd.DataFrame(feat_rows)[ALL_FEATURES]
        proba_matrix = self.classifier.predict_proba(X_pred)
        vc_preds = self.reg_count.predict(X_pred)
        sp_preds = self.reg_speed.predict(X_pred)
        oc_preds = self.reg_occ.predict(X_pred)

        forecasts: list[JunctionForecast] = []
        for idx, (row, t_tgt, w, acc, th) in enumerate(meta_rows):
            probs = {
                cls_name: round(float(proba_matrix[idx][cls_i]), 4)
                for cls_i, cls_name in enumerate(classes)
            }
            vc = float(np.clip(vc_preds[idx], 10.0, 120.0))
            sp = float(np.clip(sp_preds[idx], 5.0, 45.0))
            oc = float(np.clip(oc_preds[idx], 8.0, 99.0))

            probs, vc, sp, oc, scenario_adj = self._apply_scenario_adjustment(
                probs, vc, sp, oc, w, acc
            )
            pred_label = max(probs.items(), key=lambda kv: kv[1])[0]
            risk_score = self._compute_risk_score(probs, sp, oc)

            outside_hours = bool(th < 7.5 or th > 20.0)
            raw_conf = float(max(probs.values())) * 100.0
            if outside_hours:
                raw_conf *= 0.78
            if scenario_adj and acc == "Yes":
                raw_conf *= 0.80
            confidence = round(float(np.clip(raw_conf, 35.0, 99.0)), 1)

            notes = [f"ML Forecast (+{horizon_hours}h from {row['Timestamp_dt'].strftime('%d-%m %H:%M')})"]
            if outside_hours:
                notes.append("extrapolated outside dataset 07:30-20:00 window")
            if scenario_adj:
                notes.append("includes scenario stress adjustment")

            forecasts.append(
                JunctionForecast(
                    junction=str(row["Location"]),
                    latitude=float(row["Latitude"]),
                    longitude=float(row["Longitude"]),
                    base_timestamp=row["Timestamp_dt"].strftime("%d-%m-%Y %H:%M"),
                    target_timestamp=t_tgt.strftime("%d-%m-%Y %H:%M"),
                    horizon_hours=int(horizon_hours),
                    current_condition=str(row["Traffic_Condition"]),
                    current_vehicle_count=float(row["Vehicle_Count"]),
                    current_speed_kmh=float(row["Traffic_Speed_kmh"]),
                    current_occupancy_pct=float(row["Road_Occupancy_pct"]),
                    predicted_condition=pred_label,
                    predicted_vehicle_count=round(vc, 1),
                    predicted_speed_kmh=round(sp, 2),
                    predicted_occupancy_pct=round(oc, 2),
                    congestion_probability=round(probs.get("High", 0.0), 4),
                    class_probabilities=probs,
                    congestion_risk_score=risk_score,
                    prediction_confidence=confidence,
                    weather_scenario=w,
                    accident_scenario=acc,
                    is_outside_observed_hours=outside_hours,
                    scenario_adjusted=scenario_adj,
                    forecast_note=" | ".join(notes),
                    is_model_forecast=True,
                )
            )
        return forecasts

    def forecast_multi_horizon(
        self,
        max_horizon: int = 6,
        base_timestamp: pd.Timestamp | str | None = None,
        use_per_junction_latest: bool = False,
        weather_override: str | None = None,
        accident_junctions: set[str] | list[str] | None = None,
        junctions: list[str] | None = None,
    ) -> list[JunctionForecast]:
        """Generate forecasts across horizons 1..max_horizon for all selected junctions."""
        all_forecasts: list[JunctionForecast] = []
        for h in range(1, max_horizon + 1):
            all_forecasts.extend(
                self.predict_horizon(
                    horizon_hours=h,
                    base_timestamp=base_timestamp,
                    use_per_junction_latest=use_per_junction_latest,
                    weather_override=weather_override,
                    accident_junctions=accident_junctions,
                    junctions=junctions,
                )
            )
        return all_forecasts

    def generate_proactive_alerts(
        self,
        forecasts: list[JunctionForecast],
    ) -> dict[str, Any]:
        """Analyze multi-horizon forecasts and produce proactive alerts & earliest bottleneck."""
        alerts: list[ProactiveAlert] = []
        earliest_high: JunctionForecast | None = None

        by_junction: dict[str, list[JunctionForecast]] = {}
        for fc in forecasts:
            by_junction.setdefault(fc.junction, []).append(fc)

        for junc, fc_list in sorted(by_junction.items()):
            fc_sorted = sorted(fc_list, key=lambda x: x.horizon_hours)
            high_hits = [f for f in fc_sorted if f.predicted_condition == "High"]

            if high_hits:
                first_high = high_hits[0]
                if earliest_high is None or (
                    first_high.horizon_hours < earliest_high.horizon_hours
                    or (
                        first_high.horizon_hours == earliest_high.horizon_hours
                        and first_high.congestion_risk_score > earliest_high.congestion_risk_score
                    )
                ):
                    earliest_high = first_high

                occ_delta = round(first_high.predicted_occupancy_pct - first_high.current_occupancy_pct, 1)
                if first_high.accident_scenario == "Yes":
                    action = (
                        "Divert approaching traffic via alternate corridor, dispatch incident clearance, "
                        "and activate adaptive signal flushing."
                    )
                    reason = (
                        f"Simulated accident scenario triggers High congestion ({first_high.predicted_speed_kmh:.1f} km/h, "
                        f"{first_high.predicted_occupancy_pct:.1f}% occupancy)."
                    )
                elif first_high.horizon_hours <= 2:
                    action = (
                        "Trigger immediate preventive signal green-wave extension and pre-route traffic "
                        "to alternate Gandhipuram corridors."
                    )
                    reason = (
                        f"Near-term High congestion forecasted within +{first_high.horizon_hours}h "
                        f"(P(High)={first_high.congestion_probability*100:.0f}%, risk {first_high.congestion_risk_score:.0f}/100)."
                    )
                else:
                    action = (
                        "Schedule pre-peak adaptive signal plan and display corridor advisory before "
                        f"{first_high.target_timestamp[-5:]}."
                    )
                    reason = (
                        f"Peak buildup projected to cross High congestion threshold at +{first_high.horizon_hours}h "
                        f"({first_high.predicted_vehicle_count:.0f} vehicles, {first_high.predicted_speed_kmh:.1f} km/h)."
                    )

                alerts.append(
                    ProactiveAlert(
                        alert_level="CRITICAL",
                        junction=junc,
                        predicted_time=first_high.target_timestamp,
                        horizon_hours=first_high.horizon_hours,
                        predicted_condition=first_high.predicted_condition,
                        congestion_probability=first_high.congestion_probability,
                        congestion_risk_score=first_high.congestion_risk_score,
                        occupancy_change_pct=occ_delta,
                        recommended_action=action,
                        reason=reason,
                    )
                )
            else:
                peak_fc = max(fc_sorted, key=lambda x: x.congestion_risk_score)
                occ_delta = round(peak_fc.predicted_occupancy_pct - peak_fc.current_occupancy_pct, 1)
                if occ_delta >= 8.0 or peak_fc.congestion_risk_score >= 52.0:
                    alerts.append(
                        ProactiveAlert(
                            alert_level="WARNING",
                            junction=junc,
                            predicted_time=peak_fc.target_timestamp,
                            horizon_hours=peak_fc.horizon_hours,
                            predicted_condition=peak_fc.predicted_condition,
                            congestion_probability=peak_fc.congestion_probability,
                            congestion_risk_score=peak_fc.congestion_risk_score,
                            occupancy_change_pct=occ_delta,
                            recommended_action=(
                                "Monitor queue growth and prepare balanced green-phase extension if occupancy continues rising."
                            ),
                            reason=(
                                f"Significant occupancy / risk climb projected (+{occ_delta:.1f}% occupancy, "
                                f"risk {peak_fc.congestion_risk_score:.0f}/100)."
                            ),
                        )
                    )
                else:
                    alerts.append(
                        ProactiveAlert(
                            alert_level="NORMAL",
                            junction=junc,
                            predicted_time=peak_fc.target_timestamp,
                            horizon_hours=peak_fc.horizon_hours,
                            predicted_condition=peak_fc.predicted_condition,
                            congestion_probability=peak_fc.congestion_probability,
                            congestion_risk_score=peak_fc.congestion_risk_score,
                            occupancy_change_pct=occ_delta,
                            recommended_action="Maintain standard adaptive signal cycle; suitable for corridor diversion.",
                            reason=(
                                f"Forecast remains stable ({peak_fc.predicted_condition}, "
                                f"risk {peak_fc.congestion_risk_score:.0f}/100)."
                            ),
                        )
                    )

        severity_order = {"CRITICAL": 0, "WARNING": 1, "NORMAL": 2}
        alerts.sort(key=lambda a: (severity_order[a.alert_level], a.horizon_hours, -a.congestion_risk_score))

        return {
            "alerts": alerts,
            "critical_count": sum(1 for a in alerts if a.alert_level == "CRITICAL"),
            "warning_count": sum(1 for a in alerts if a.alert_level == "WARNING"),
            "normal_count": sum(1 for a in alerts if a.alert_level == "NORMAL"),
            "earliest_congested_junction": earliest_high.junction if earliest_high else None,
            "earliest_congestion_time": earliest_high.target_timestamp if earliest_high else None,
            "earliest_horizon_hours": earliest_high.horizon_hours if earliest_high else None,
        }
