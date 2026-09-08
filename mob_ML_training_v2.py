"""
Member 2 — ML Traffic Prediction Training Script (v2)
=====================================================
Proactive Traffic Management System

This script validates the existing model, addresses data leakage concerns,
and trains both a diagnostic model (real-time) and a predictive model (future).

Outputs:
    cleaned_data/traffic_prediction_model.pkl    — Diagnostic model (full features)
    cleaned_data/predictive_model.pkl            — Predictive model (time/historical features)
    cleaned_data/label_encoder.pkl               — Label encoder for traffic conditions
    cleaned_data/model_features.pkl              — Feature list for diagnostic model
    cleaned_data/predictive_model_features.pkl   — Feature list for predictive model
    cleaned_data/confusion_matrix_*.png          — Confusion matrix plots
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
DATA_FILE = os.path.join(DATA_DIR, "mobility_cleaned.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load and inspect data
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading dataset")
print("=" * 70)

df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Verify target and check for data leakage
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Target variable analysis")
print("=" * 70)

target_col = "Traffic_Condition"
print(f"\nTarget: {target_col}")
print(f"\nClass distribution:")
print(df[target_col].value_counts())
print(f"\nClass proportions:")
print(df[target_col].value_counts(normalize=True).round(4))

# Data leakage documentation
print("\n" + "-" * 50)
print("[WARNING] DATA LEAKAGE ASSESSMENT")
print("-" * 50)
print("""
The 'Traffic_Condition' column in the Smart Mobility dataset is a SYNTHETIC
label derived deterministically from contemporaneous features:
  - Vehicle_Count
  - Traffic_Speed_kmh
  - Road_Occupancy_%
  - Accident_Report

Using these features to predict Traffic_Condition is circular.

STRATEGY:
  1. DIAGNOSTIC MODEL — Uses all features including contemporaneous ones.
     This is valid for REAL-TIME classification (where CV provides live data).
     Expected accuracy: ~99-100% (learning the derivation rule).

  2. PREDICTIVE MODEL — Uses only time-based and historical features.
     This is valid for FUTURE traffic prediction (Date+Time feature).
     Expected accuracy: ~60-80% (honest predictive power).
""")

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Feature engineering
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 3 — Feature engineering")
print("=" * 70)

# Parse timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

print(f"Extracted time features: Hour, Day, Month, DayOfWeek")
print(f"Timestamp range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# Keep timestamp for time-aware split, drop later
df_with_ts = df.copy()

# Drop Timestamp for model training
df = df.drop("Timestamp", axis=1)

# Separate target
y_raw = df[target_col]
X_raw = df.drop(target_col, axis=1)

# One-hot encode categoricals AFTER defining X (prevent leakage into y)
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

print(f"\nLabel encoder classes: {label_encoder.classes_}")
print(f"Feature count after encoding: {X_encoded.shape[1]}")
print(f"Features: {list(X_encoded.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# Define feature sets
# ──────────────────────────────────────────────────────────────────────────────

# DIAGNOSTIC features — all available (for real-time classification)
diagnostic_features = list(X_encoded.columns)

# PREDICTIVE features — only time/context features (for future prediction)
# Exclude contemporaneous traffic measurements that won't be available in the future
contemporaneous_cols = [
    "Vehicle_Count",
    "Traffic_Speed_kmh",
    "Road_Occupancy_%",
    "Accident_Report",
    "Emission_Levels_g_km",
    "Energy_Consumption_L_h",
]
predictive_features = [f for f in diagnostic_features if f not in contemporaneous_cols]

print(f"\nDiagnostic features ({len(diagnostic_features)}): {diagnostic_features}")
print(f"\nPredictive features ({len(predictive_features)}): {predictive_features}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper — train and evaluate a set of models
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    split_name, feature_set_name, label_enc,
    save_prefix=None
):
    """Train 4 models, print reports, plot confusion matrices, return results."""

    # Try to import xgboost
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False
        print("  [!] XGBoost not installed, skipping XGBoost model")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    if has_xgb:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    best_model = None
    best_f1 = -1
    best_model_name = ""

    for name, model in models.items():
        print(f"\n  Training: {name}")

        # Use scaled data for Logistic Regression
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "model": model,
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")

        # Classification report
        print(f"\n    Classification Report ({name}):")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=label_enc.classes_,
                zero_division=0,
            )
        )

        # Confusion matrix plot
        if save_prefix:
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=label_enc.classes_,
                ax=ax,
                cmap="Blues",
            )
            safe_name = name.replace(" ", "_").lower()
            title = f"{name} — {feature_set_name} ({split_name})"
            ax.set_title(title)
            plt.tight_layout()
            fname = os.path.join(
                DATA_DIR,
                f"confusion_matrix_{save_prefix}_{safe_name}.png",
            )
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"    Saved: {fname}")

        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\n  [BEST] Best model: {best_model_name} (F1 = {best_f1:.4f})")
    return results, best_model, best_model_name


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — DIAGNOSTIC MODEL (Random Stratified Split)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — DIAGNOSTIC MODEL (full features, random stratified split)")
print("=" * 70)

X_diag = X_encoded[diagnostic_features]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diag, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print(f"Train size: {X_train_d.shape[0]}, Test size: {X_test_d.shape[0]}")

diag_results, diag_best_model, diag_best_name = train_and_evaluate(
    X_train_d, X_test_d, y_train_d, y_test_d,
    split_name="Random Stratified",
    feature_set_name="Diagnostic (All Features)",
    label_enc=label_encoder,
    save_prefix="diagnostic_random",
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — DIAGNOSTIC MODEL (Time-Aware Split)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — DIAGNOSTIC MODEL (full features, time-aware split)")
print("=" * 70)

# Split: Train on March 1–14, Test on March 15–18
split_date = pd.Timestamp("2024-03-15")
train_mask = df_with_ts["Timestamp"] < split_date
test_mask = df_with_ts["Timestamp"] >= split_date

# Need to rebuild X_encoded aligned with df_with_ts
df_time = df_with_ts.drop(["Timestamp", target_col], axis=1)
X_time_encoded = pd.get_dummies(df_time, drop_first=True)

# Ensure same columns as diagnostic features (add missing, remove extra)
for col in diagnostic_features:
    if col not in X_time_encoded.columns:
        X_time_encoded[col] = 0
X_time_encoded = X_time_encoded[diagnostic_features]

y_time_encoded = label_encoder.transform(df_with_ts[target_col])

X_train_t = X_time_encoded[train_mask].values
X_test_t = X_time_encoded[test_mask].values
y_train_t = y_time_encoded[train_mask]
y_test_t = y_time_encoded[test_mask]

print(f"Train (before {split_date.date()}): {X_train_t.shape[0]} samples")
print(f"Test  (from {split_date.date()}):   {X_test_t.shape[0]} samples")

if X_test_t.shape[0] > 0:
    diag_time_results, _, _ = train_and_evaluate(
        X_train_t, X_test_t, y_train_t, y_test_t,
        split_name="Time-Aware",
        feature_set_name="Diagnostic (All Features)",
        label_enc=label_encoder,
        save_prefix="diagnostic_time",
    )
else:
    print("  [!]  Time-aware split has empty test set — skipping")

# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — PREDICTIVE MODEL (Random Stratified Split)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — PREDICTIVE MODEL (time/context features, random stratified split)")
print("=" * 70)

X_pred = X_encoded[predictive_features]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pred, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print(f"Train size: {X_train_p.shape[0]}, Test size: {X_test_p.shape[0]}")
print(f"Using {len(predictive_features)} features (no contemporaneous traffic data)")

pred_results, pred_best_model, pred_best_name = train_and_evaluate(
    X_train_p, X_test_p, y_train_p, y_test_p,
    split_name="Random Stratified",
    feature_set_name="Predictive (Time/Context Only)",
    label_enc=label_encoder,
    save_prefix="predictive_random",
)

# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — PREDICTIVE MODEL (Time-Aware Split)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — PREDICTIVE MODEL (time/context features, time-aware split)")
print("=" * 70)

X_pred_time = X_time_encoded[predictive_features]

X_train_pt = X_pred_time[train_mask].values
X_test_pt = X_pred_time[test_mask].values
y_train_pt = y_time_encoded[train_mask]
y_test_pt = y_time_encoded[test_mask]

print(f"Train (before {split_date.date()}): {X_train_pt.shape[0]} samples")
print(f"Test  (from {split_date.date()}):   {X_test_pt.shape[0]} samples")

if X_test_pt.shape[0] > 0:
    pred_time_results, pred_time_best_model, pred_time_best_name = train_and_evaluate(
        X_train_pt, X_test_pt, y_train_pt, y_test_pt,
        split_name="Time-Aware",
        feature_set_name="Predictive (Time/Context Only)",
        label_enc=label_encoder,
        save_prefix="predictive_time",
    )
else:
    print("  [!]  Time-aware split has empty test set — skipping")

# ──────────────────────────────────────────────────────────────────────────────
# Step 8 — Model Comparison Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8 — Model comparison summary")
print("=" * 70)

print("\n[CHART] DIAGNOSTIC MODEL (Random Stratified Split):")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 70)
for name, r in diag_results.items():
    print(f"{name:<25} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} {r['Recall']:>10.4f} {r['F1-Score']:>10.4f}")

print("\n[CHART] PREDICTIVE MODEL (Random Stratified Split):")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 70)
for name, r in pred_results.items():
    print(f"{name:<25} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} {r['Recall']:>10.4f} {r['F1-Score']:>10.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 9 — Save artifacts
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9 — Saving model artifacts")
print("=" * 70)

# Diagnostic model
diag_path = os.path.join(DATA_DIR, "traffic_prediction_model.pkl")
joblib.dump(diag_best_model, diag_path)
print(f"[OK] Saved diagnostic model ({diag_best_name}): {diag_path}")

# Predictive model
pred_path = os.path.join(DATA_DIR, "predictive_model.pkl")
joblib.dump(pred_best_model, pred_path)
print(f"[OK] Saved predictive model ({pred_best_name}): {pred_path}")

# Label encoder
le_path = os.path.join(DATA_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, le_path)
print(f"[OK] Saved label encoder: {le_path}")

# Feature lists
diag_feat_path = os.path.join(DATA_DIR, "model_features.pkl")
joblib.dump(diagnostic_features, diag_feat_path)
print(f"[OK] Saved diagnostic features ({len(diagnostic_features)}): {diag_feat_path}")

pred_feat_path = os.path.join(DATA_DIR, "predictive_model_features.pkl")
joblib.dump(predictive_features, pred_feat_path)
print(f"[OK] Saved predictive features ({len(predictive_features)}): {pred_feat_path}")

# Also save the scaler for Logistic Regression if it was best
scaler_full = StandardScaler()
scaler_full.fit(X_encoded[diagnostic_features])
scaler_path = os.path.join(DATA_DIR, "feature_scaler.pkl")
joblib.dump(scaler_full, scaler_path)
print(f"[OK] Saved feature scaler: {scaler_path}")

print("\n" + "=" * 70)
print("[OK] Training complete! All artifacts saved to cleaned_data/")
print("=" * 70)
