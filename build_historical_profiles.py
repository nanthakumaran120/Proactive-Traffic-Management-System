"""
Member 2 — Build Historical Traffic Profiles
=============================================
Proactive Traffic Management System

Aggregates the mobility dataset by DayOfWeek + Hour to create historical
traffic profiles. These profiles power the predict_future_traffic() function.

Output:
    cleaned_data/historical_traffic_profile.csv
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
DATA_FILE = os.path.join(DATA_DIR, "mobility_cleaned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "historical_traffic_profile.csv")


def build_profiles():
    """Build historical traffic profiles grouped by DayOfWeek and Hour."""

    print("=" * 60)
    print("Building Historical Traffic Profiles")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Extract time features
    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["DayName"] = df["Timestamp"].dt.day_name()

    print(f"Dataset: {df.shape[0]} records")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Days of week present: {sorted(df['DayOfWeek'].unique())}")
    print(f"Hours present: {sorted(df['Hour'].unique())}")

    # ──────────────────────────────────────────────────────────────────
    # Aggregate by DayOfWeek + Hour
    # ──────────────────────────────────────────────────────────────────
    profile = df.groupby(["DayOfWeek", "Hour"]).agg(
        Avg_Vehicle_Count=("Vehicle_Count", "mean"),
        Std_Vehicle_Count=("Vehicle_Count", "std"),
        Avg_Speed_kmh=("Traffic_Speed_kmh", "mean"),
        Std_Speed_kmh=("Traffic_Speed_kmh", "std"),
        Avg_Road_Occupancy=("Road_Occupancy_%", "mean"),
        Std_Road_Occupancy=("Road_Occupancy_%", "std"),
        Avg_Sentiment=("Sentiment_Score", "mean"),
        Avg_Ride_Sharing=("Ride_Sharing_Demand", "mean"),
        Avg_Parking=("Parking_Availability", "mean"),
        Sample_Count=("Vehicle_Count", "count"),
    ).reset_index()

    # Fill NaN std values (happens when only 1 sample in group)
    profile = profile.fillna(0)

    # Compute traffic density as a derived feature
    # Density = Vehicle_Count / (Speed + 1) — higher density when many cars, low speed
    profile["Avg_Traffic_Density"] = (
        profile["Avg_Vehicle_Count"] / (profile["Avg_Speed_kmh"] + 1)
    ).round(4)

    # Add mode of Traffic_Condition for each group
    mode_condition = df.groupby(["DayOfWeek", "Hour"])["Traffic_Condition"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Medium"
    ).reset_index()
    mode_condition.columns = ["DayOfWeek", "Hour", "Mode_Condition"]

    profile = profile.merge(mode_condition, on=["DayOfWeek", "Hour"])

    # Compute congestion probability (fraction of "High" traffic)
    congestion_prob = df.groupby(["DayOfWeek", "Hour"]).apply(
        lambda g: (g["Traffic_Condition"] == "High").mean(), include_groups=False
    ).reset_index()
    congestion_prob.columns = ["DayOfWeek", "Hour", "Congestion_Probability"]

    profile = profile.merge(congestion_prob, on=["DayOfWeek", "Hour"])

    # Add day name for readability
    day_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday",
    }
    profile["DayName"] = profile["DayOfWeek"].map(day_names)

    # Round floats
    float_cols = profile.select_dtypes(include=[np.floating]).columns
    profile[float_cols] = profile[float_cols].round(4)

    # Sort
    profile = profile.sort_values(["DayOfWeek", "Hour"]).reset_index(drop=True)

    # Save
    profile.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Saved historical profile: {OUTPUT_FILE}")
    print(f"   Profile shape: {profile.shape}")

    # Display summary
    print(f"\n{'DayName':<12} {'Hour':>4} {'Avg Count':>10} {'Avg Speed':>10} "
          f"{'Avg Occ':>8} {'Density':>8} {'Mode':>8} {'Cong Prob':>10}")
    print("-" * 82)
    for _, row in profile.iterrows():
        print(
            f"{row['DayName']:<12} {int(row['Hour']):>4} "
            f"{row['Avg_Vehicle_Count']:>10.1f} {row['Avg_Speed_kmh']:>10.1f} "
            f"{row['Avg_Road_Occupancy']:>8.1f} {row['Avg_Traffic_Density']:>8.2f} "
            f"{row['Mode_Condition']:>8} {row['Congestion_Probability']:>10.3f}"
        )

    return profile


if __name__ == "__main__":
    build_profiles()
