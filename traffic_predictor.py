"""
Member 2 — Traffic Predictor Module
====================================
Proactive Traffic Management System

Clean integration module for Members 1 and 3.
Provides two main prediction functions:
    1. predict_current_traffic()  — Real-time prediction from CV output
    2. predict_future_traffic()   — Future prediction using historical profiles
    3. predict_route_traffic()    — Batch prediction for route segments

Usage:
    from traffic_predictor import TrafficPredictor

    tp = TrafficPredictor()

    # Real-time (from Member 1 CV output)
    result = tp.predict_current_traffic(
        vehicle_count=150, average_speed=25.0,
        traffic_density=0.75, hour=8, day_of_week=0
    )
    # Returns: "HIGH"

    # Future prediction (for Member 3 routing)
    result = tp.predict_future_traffic(
        road_id="A", date="2026-09-10", time="13:00"
    )
    # Returns: {"road_id": "A", "date": "2026-09-10", "time": "13:00",
    #           "predicted_traffic": "MEDIUM", "confidence": 0.72, ...}
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")

# Default road network definition (Member 1 CV assigns these IDs)
DEFAULT_ROAD_NETWORK = {
    "A": {"name": "Main Street North", "lat": 40.85, "lon": -73.85},
    "B": {"name": "Highway 101 East", "lat": 40.80, "lon": -73.90},
    "C": {"name": "Downtown Avenue", "lat": 40.75, "lon": -73.95},
    "D": {"name": "Industrial Road", "lat": 40.70, "lon": -73.80},
    "E": {"name": "Residential Drive", "lat": 40.82, "lon": -73.75},
}


class TrafficPredictor:
    """
    Single entry point for all traffic ML predictions.

    Loads pre-trained models and historical profiles to provide:
    - Real-time traffic classification from CV sensor data
    - Future traffic prediction from date/time inputs
    - Batch route traffic prediction for routing optimization
    """

    def __init__(self, data_dir=None, road_network=None):
        """
        Load model artifacts and historical profiles.

        Parameters
        ----------
        data_dir : str, optional
            Path to directory containing model artifacts.
            Defaults to cleaned_data/ in the project root.
        road_network : dict, optional
            Road network definition mapping road_id to metadata.
        """
        self.data_dir = data_dir or DATA_DIR
        self.road_network = road_network or DEFAULT_ROAD_NETWORK

        # Load diagnostic model (for real-time prediction)
        diag_model_path = os.path.join(self.data_dir, "traffic_prediction_model.pkl")
        self.diagnostic_model = joblib.load(diag_model_path)

        # Load predictive model (for future prediction)
        pred_model_path = os.path.join(self.data_dir, "predictive_model.pkl")
        if os.path.exists(pred_model_path):
            self.predictive_model = joblib.load(pred_model_path)
        else:
            # Fallback to diagnostic model if predictive not available
            self.predictive_model = self.diagnostic_model
            print("[!]  Predictive model not found, using diagnostic model as fallback")

        # Load label encoder
        le_path = os.path.join(self.data_dir, "label_encoder.pkl")
        self.label_encoder = joblib.load(le_path)

        # Load feature lists
        diag_feat_path = os.path.join(self.data_dir, "model_features.pkl")
        self.diagnostic_features = joblib.load(diag_feat_path)

        pred_feat_path = os.path.join(self.data_dir, "predictive_model_features.pkl")
        if os.path.exists(pred_feat_path):
            self.predictive_features = joblib.load(pred_feat_path)
        else:
            self.predictive_features = self.diagnostic_features

        # Load historical profiles
        profile_path = os.path.join(self.data_dir, "historical_traffic_profile.csv")
        if os.path.exists(profile_path):
            self.historical_profile = pd.read_csv(profile_path)
        else:
            self.historical_profile = None
            print("[!]  Historical profile not found. Run build_historical_profiles.py first.")

        print("[OK] TrafficPredictor initialized successfully")
        print(f"   Diagnostic model: {type(self.diagnostic_model).__name__}")
        print(f"   Predictive model: {type(self.predictive_model).__name__}")
        print(f"   Label classes: {list(self.label_encoder.classes_)}")
        print(f"   Diagnostic features: {len(self.diagnostic_features)}")
        print(f"   Predictive features: {len(self.predictive_features)}")
        if self.historical_profile is not None:
            print(f"   Historical profile: {self.historical_profile.shape[0]} entries")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def predict_current_traffic(
        self,
        vehicle_count: int,
        average_speed: float,
        traffic_density: float,
        hour: int,
        day_of_week: int,
        weather: str = "Clear",
        traffic_light: str = "Green",
        accident: int = 0,
    ) -> str:
        """
        Real-time traffic prediction from CV output.

        This uses the diagnostic model with full features. Valid for real-time
        classification because Member 1's CV system provides live sensor data.

        Parameters
        ----------
        vehicle_count : int
            Number of vehicles detected by CV.
        average_speed : float
            Average vehicle speed in km/h.
        traffic_density : float
            Traffic density (0-1 scale or road occupancy %).
            If value <= 1, it's treated as fraction and converted to percentage.
        hour : int
            Hour of day (0-23).
        day_of_week : int
            Day of week (0=Monday, 6=Sunday).
        weather : str
            Weather condition: "Clear", "Rain", "Fog", or "Snow".
        traffic_light : str
            Traffic light state: "Green", "Red", or "Yellow".
        accident : int
            Whether an accident is reported (0 or 1).

        Returns
        -------
        str
            Predicted traffic condition: "HIGH", "MEDIUM", or "LOW".
        """
        # Convert density fraction to percentage if needed
        road_occupancy = traffic_density * 100 if traffic_density <= 1.0 else traffic_density

        # Build feature dictionary
        features = {
            "Latitude": 40.80,  # Default NYC area
            "Longitude": -73.85,
            "Vehicle_Count": vehicle_count,
            "Traffic_Speed_kmh": average_speed,
            "Road_Occupancy_%": road_occupancy,
            "Accident_Report": accident,
            "Sentiment_Score": 0.0,  # Default neutral
            "Ride_Sharing_Demand": 50,  # Default medium
            "Parking_Availability": 25,  # Default medium
            "Emission_Levels_g_km": 250.0,  # Default medium
            "Energy_Consumption_L_h": 15.0,  # Default medium
            "Hour": hour,
            "Day": 15,  # Default mid-month
            "Month": 3,  # Default March (training data period)
            "DayOfWeek": day_of_week,
            # One-hot encoded features
            "Traffic_Light_State_Red": 1 if traffic_light == "Red" else 0,
            "Traffic_Light_State_Yellow": 1 if traffic_light == "Yellow" else 0,
            "Weather_Condition_Fog": 1 if weather == "Fog" else 0,
            "Weather_Condition_Rain": 1 if weather == "Rain" else 0,
            "Weather_Condition_Snow": 1 if weather == "Snow" else 0,
        }

        # Create DataFrame with correct feature order
        df_input = pd.DataFrame([features])

        # Ensure all diagnostic features are present
        for feat in self.diagnostic_features:
            if feat not in df_input.columns:
                df_input[feat] = 0

        df_input = df_input[self.diagnostic_features]

        # Predict
        prediction = self.diagnostic_model.predict(df_input)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]

        return label.upper()

    def predict_future_traffic(
        self,
        road_id: str,
        date: str,
        time: str,
    ) -> dict:
        """
        Future traffic prediction using historical profiles.

        Uses the predictive model (time/context features only) combined with
        historical traffic averages for the given day-of-week and hour.

        Parameters
        ----------
        road_id : str
            Road segment identifier (e.g., "A", "B", "C").
        date : str
            Future date in format "YYYY-MM-DD" (e.g., "2026-09-10").
        time : str
            Future time in format "HH:MM" (e.g., "13:00").

        Returns
        -------
        dict
            Prediction result containing:
            - road_id: The road identifier
            - date: The requested date
            - time: The requested time
            - predicted_traffic: "HIGH", "MEDIUM", or "LOW"
            - confidence: Prediction confidence (0-1)
            - historical_avg_speed: Historical average speed for this slot
            - historical_avg_count: Historical average vehicle count
            - congestion_probability: Historical probability of congestion
        """
        # Parse date and time
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        day = dt.day
        month = dt.month

        # Look up historical profile for this DayOfWeek + Hour
        hist_data = self._get_historical_profile(day_of_week, hour)

        # Get road info
        road_info = self.road_network.get(road_id, {"lat": 40.80, "lon": -73.85})
        lat = road_info.get("lat", 40.80)
        lon = road_info.get("lon", -73.85)

        # Build predictive feature set (NO contemporaneous traffic data)
        features = {
            "Latitude": lat,
            "Longitude": lon,
            "Sentiment_Score": hist_data.get("Avg_Sentiment", 0.0),
            "Ride_Sharing_Demand": hist_data.get("Avg_Ride_Sharing", 50),
            "Parking_Availability": hist_data.get("Avg_Parking", 25),
            "Hour": hour,
            "Day": day,
            "Month": month,
            "DayOfWeek": day_of_week,
            # One-hot encoded (default to clear/green for future prediction)
            "Traffic_Light_State_Red": 0,
            "Traffic_Light_State_Yellow": 0,
            "Weather_Condition_Fog": 0,
            "Weather_Condition_Rain": 0,
            "Weather_Condition_Snow": 0,
        }

        # Create DataFrame
        df_input = pd.DataFrame([features])

        # Ensure all predictive features are present
        for feat in self.predictive_features:
            if feat not in df_input.columns:
                df_input[feat] = 0
        df_input = df_input[self.predictive_features]

        # Predict
        prediction = self.predictive_model.predict(df_input)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]

        # Get confidence (probability of predicted class)
        confidence = 0.0
        if hasattr(self.predictive_model, "predict_proba"):
            proba = self.predictive_model.predict_proba(df_input)[0]
            confidence = float(max(proba))

        return {
            "road_id": road_id,
            "date": date,
            "time": time,
            "predicted_traffic": label.upper(),
            "confidence": round(confidence, 4),
            "historical_avg_speed": hist_data.get("Avg_Speed_kmh", 0.0),
            "historical_avg_count": hist_data.get("Avg_Vehicle_Count", 0.0),
            "historical_avg_occupancy": hist_data.get("Avg_Road_Occupancy", 0.0),
            "congestion_probability": hist_data.get("Congestion_Probability", 0.0),
        }

    def predict_route_traffic(
        self,
        route_road_ids: list,
        date: str,
        time: str,
    ) -> list:
        """
        Predict traffic for all road segments in a route.

        Used by Member 3 for route optimization — predicts the traffic
        condition on each segment of a proposed route.

        Parameters
        ----------
        route_road_ids : list
            List of road IDs forming the route (e.g., ["A", "B", "C"]).
        date : str
            Future date in format "YYYY-MM-DD".
        time : str
            Future time in format "HH:MM".

        Returns
        -------
        list
            List of prediction dicts, one per road segment.
        """
        results = []
        for road_id in route_road_ids:
            result = self.predict_future_traffic(road_id, date, time)
            results.append(result)
        return results

    def predict_congestion(
        self,
        vehicle_count: int,
        average_speed: float,
        traffic_density: float,
        hour: int,
        day_of_week: int,
    ) -> str:
        """
        Simplified interface matching the spec exactly.

        Parameters
        ----------
        vehicle_count : int
        average_speed : float
        traffic_density : float
        hour : int
        day_of_week : int

        Returns
        -------
        str
            "HIGH", "MEDIUM", or "LOW"
        """
        return self.predict_current_traffic(
            vehicle_count=vehicle_count,
            average_speed=average_speed,
            traffic_density=traffic_density,
            hour=hour,
            day_of_week=day_of_week,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_historical_profile(self, day_of_week: int, hour: int) -> dict:
        """
        Look up historical averages for a given DayOfWeek + Hour.

        Falls back to hour-only average if exact combo not found,
        then to global averages as last resort.
        """
        if self.historical_profile is None:
            return {
                "Avg_Vehicle_Count": 150.0,
                "Avg_Speed_kmh": 35.0,
                "Avg_Road_Occupancy": 55.0,
                "Avg_Traffic_Density": 4.0,
                "Avg_Sentiment": 0.0,
                "Avg_Ride_Sharing": 50,
                "Avg_Parking": 25,
                "Mode_Condition": "Medium",
                "Congestion_Probability": 0.5,
            }

        hp = self.historical_profile

        # Exact match: DayOfWeek + Hour
        match = hp[(hp["DayOfWeek"] == day_of_week) & (hp["Hour"] == hour)]
        if len(match) > 0:
            return match.iloc[0].to_dict()

        # Fallback: same hour, any day
        match = hp[hp["Hour"] == hour]
        if len(match) > 0:
            return match.mean(numeric_only=True).to_dict()

        # Last resort: global average
        return hp.mean(numeric_only=True).to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Standalone convenience functions (for simple imports)
# ──────────────────────────────────────────────────────────────────────────────

_predictor_instance = None


def _get_predictor():
    """Lazy-load singleton predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = TrafficPredictor()
    return _predictor_instance


def predict_current_traffic(
    vehicle_count, average_speed, traffic_density, hour, day_of_week
):
    """Convenience function — see TrafficPredictor.predict_current_traffic()."""
    return _get_predictor().predict_current_traffic(
        vehicle_count=vehicle_count,
        average_speed=average_speed,
        traffic_density=traffic_density,
        hour=hour,
        day_of_week=day_of_week,
    )


def predict_future_traffic(road_id, date, time):
    """Convenience function — see TrafficPredictor.predict_future_traffic()."""
    return _get_predictor().predict_future_traffic(
        road_id=road_id, date=date, time=time,
    )


def predict_congestion(
    vehicle_count, average_speed, traffic_density, hour, day_of_week
):
    """Convenience function — see TrafficPredictor.predict_congestion()."""
    return _get_predictor().predict_congestion(
        vehicle_count=vehicle_count,
        average_speed=average_speed,
        traffic_density=traffic_density,
        hour=hour,
        day_of_week=day_of_week,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Self-test when run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Traffic Predictor — Integration Test")
    print("=" * 60)

    tp = TrafficPredictor()

    # Test 1: Real-time prediction
    print("\n" + "-" * 50)
    print("TEST 1 — Real-time prediction (from CV output)")
    print("-" * 50)

    test_cases = [
        {"vehicle_count": 250, "average_speed": 15.0, "traffic_density": 0.90,
         "hour": 8, "day_of_week": 0, "desc": "Monday 8AM, heavy traffic"},
        {"vehicle_count": 50, "average_speed": 65.0, "traffic_density": 0.20,
         "hour": 14, "day_of_week": 6, "desc": "Sunday 2PM, light traffic"},
        {"vehicle_count": 150, "average_speed": 35.0, "traffic_density": 0.55,
         "hour": 12, "day_of_week": 3, "desc": "Thursday noon, moderate traffic"},
    ]

    for tc in test_cases:
        desc = tc.pop("desc")
        result = tp.predict_current_traffic(**tc)
        print(f"  {desc}: {result}")
        tc["desc"] = desc  # Restore for display

    # Test 2: Future prediction
    print("\n" + "-" * 50)
    print("TEST 2 — Future prediction (Date + Time)")
    print("-" * 50)

    future_tests = [
        {"road_id": "A", "date": "2026-09-10", "time": "08:00"},
        {"road_id": "A", "date": "2026-09-10", "time": "13:00"},
        {"road_id": "B", "date": "2026-09-10", "time": "18:00"},
        {"road_id": "C", "date": "2026-09-13", "time": "13:00"},  # Sunday
    ]

    for ft in future_tests:
        result = tp.predict_future_traffic(**ft)
        print(f"\n  Road {result['road_id']} on {result['date']} at {result['time']}:")
        print(f"    Predicted traffic : {result['predicted_traffic']}")
        print(f"    Confidence        : {result['confidence']:.2%}")
        print(f"    Hist avg speed    : {result['historical_avg_speed']:.1f} km/h")
        print(f"    Hist avg count    : {result['historical_avg_count']:.0f} vehicles")
        print(f"    Congestion prob   : {result['congestion_probability']:.2%}")

    # Test 3: Route prediction
    print("\n" + "-" * 50)
    print("TEST 3 — Route prediction (for Member 3)")
    print("-" * 50)

    route = ["A", "B", "C"]
    route_results = tp.predict_route_traffic(route, "2026-09-10", "13:00")

    print(f"\n  Route: {' → '.join(route)}")
    print(f"  Date: 2026-09-10, Time: 13:00\n")
    for r in route_results:
        print(f"    Road {r['road_id']}: {r['predicted_traffic']} "
              f"(confidence: {r['confidence']:.2%})")

    # Test 4: Simplified interface
    print("\n" + "-" * 50)
    print("TEST 4 — Simplified predict_congestion() interface")
    print("-" * 50)

    result = tp.predict_congestion(
        vehicle_count=200,
        average_speed=20.0,
        traffic_density=0.80,
        hour=17,
        day_of_week=4,
    )
    print(f"  Friday 5PM, heavy traffic: {result}")

    print("\n" + "=" * 60)
    print("[OK] All integration tests passed!")
    print("=" * 60)
