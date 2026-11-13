"""Software-only adaptive traffic signal & emergency preemption simulation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


SIMULATION_DISCLAIMER = (
    "Software Simulation Only: proposed signal timings are advisory simulation outputs "
    "and do not control real traffic signal hardware."
)

EMERGENCY_PREEMPTION_SECONDS = {
    "Ambulance": {"green_window_s": 90, "lead_time_s": 20, "clearance_s": 10},
    "Fire Engine": {"green_window_s": 95, "lead_time_s": 25, "clearance_s": 12},
    "Police Vehicle": {"green_window_s": 80, "lead_time_s": 15, "clearance_s": 8},
}


@dataclass
class SignalPlan:
    """Normal vs adaptive / emergency signal timing proposal for a single junction."""

    junction: str
    traffic_condition: str
    normal_signal_state: str
    normal_green_s: int
    normal_yellow_s: int
    normal_red_s: int
    proposed_signal_state: str
    proposed_green_s: int
    proposed_yellow_s: int
    proposed_red_s: int
    priority_activated: bool
    emergency_vehicle_type: str | None
    preemption_lead_s: int
    clearance_buffer_s: int
    estimated_wait_reduction_pct: float
    rationale: str
    simulation_only: bool = True
    disclaimer: str = SIMULATION_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_adaptive_signal(
    junction: str,
    traffic_condition: str,
    current_signal_state: str = "Red",
    vehicle_count: float = 50.0,
    road_occupancy_pct: float = 45.0,
) -> SignalPlan:
    """Compute fixed-baseline vs congestion-adaptive signal split for a junction."""
    cycle_s = 120
    normal_green = 45
    normal_yellow = 5
    normal_red = cycle_s - normal_green - normal_yellow

    cond = str(traffic_condition).title()
    if cond == "High":
        # Extend main-corridor green phase proportionally to occupancy/volume
        extra_green = min(25, max(10, int(round((road_occupancy_pct - 45.0) * 0.4 + (vehicle_count - 60.0) * 0.25))))
        proposed_green = normal_green + extra_green
        proposed_state = "Green (Extended Adaptive Phase)"
        wait_reduction = round(min(32.0, 12.0 + extra_green * 0.7), 1)
        rationale = (
            f"High congestion predicted ({vehicle_count:.0f} vehicles, {road_occupancy_pct:.1f}% occupancy); "
            f"extend main green phase by +{extra_green}s to clear queue buildup."
        )
    elif cond == "Medium":
        proposed_green = 52
        proposed_state = "Green (Balanced Adaptive Phase)"
        wait_reduction = 11.5
        rationale = (
            f"Medium congestion predicted ({vehicle_count:.0f} vehicles); "
            "apply modest +7s green extension to prevent saturation."
        )
    else:
        proposed_green = 35
        proposed_state = "Green (Shorter Off-Peak Cycle)"
        wait_reduction = 8.0
        rationale = (
            f"Low congestion predicted ({vehicle_count:.0f} vehicles); "
            "shorten main phase to reduce cross-street idle delay."
        )

    proposed_red = max(25, cycle_s - proposed_green - normal_yellow)
    return SignalPlan(
        junction=junction,
        traffic_condition=cond,
        normal_signal_state=current_signal_state,
        normal_green_s=normal_green,
        normal_yellow_s=normal_yellow,
        normal_red_s=normal_red,
        proposed_signal_state=proposed_state,
        proposed_green_s=proposed_green,
        proposed_yellow_s=normal_yellow,
        proposed_red_s=proposed_red,
        priority_activated=False,
        emergency_vehicle_type=None,
        preemption_lead_s=0,
        clearance_buffer_s=0,
        estimated_wait_reduction_pct=wait_reduction,
        rationale=rationale,
    )


def simulate_emergency_signal_priority(
    route_junctions: list[str],
    vehicle_type: str,
    junction_states: dict[str, dict[str, Any]],
) -> list[SignalPlan]:
    """Simulate green-wave signal preemption across every junction on an emergency route."""
    params = EMERGENCY_PREEMPTION_SECONDS.get(
        vehicle_type,
        EMERGENCY_PREEMPTION_SECONDS["Ambulance"],
    )
    cycle_s = 120
    normal_green = 45
    normal_yellow = 5
    normal_red = cycle_s - normal_green - normal_yellow

    plans: list[SignalPlan] = []
    for idx, junc in enumerate(route_junctions):
        info = junction_states.get(junc, {})
        cond = str(info.get("predicted_condition", info.get("Traffic_Condition", "Medium"))).title()
        normal_state = str(info.get("Traffic_Light_State", "Red"))
        vc = float(info.get("predicted_vehicle_count", info.get("Vehicle_Count", 55.0)))
        occ = float(info.get("predicted_occupancy_pct", info.get("Road_Occupancy_pct", 45.0)))

        # Higher congestion ahead benefits from slightly longer queue flush lead-time
        extra_flush = 10 if cond == "High" else (5 if cond == "Medium" else 0)
        proposed_green = min(105, params["green_window_s"] + extra_flush)
        proposed_red = max(10, cycle_s - proposed_green - normal_yellow)
        lead_s = params["lead_time_s"] + extra_flush
        clearance_s = params["clearance_s"]
        wait_reduction = round(65.0 if cond == "High" else (52.0 if cond == "Medium" else 38.0), 1)

        role_tag = "Origin Dispatch" if idx == 0 else ("Destination Arrival" if idx == len(route_junctions) - 1 else f"Corridor Hop {idx}")
        rationale = (
            f"[{role_tag}] Preempt cross-traffic to Red and force corridor Green for {vehicle_type} "
            f"({lead_s}s flush lead time before arrival; predicted {cond} congestion, "
            f"{vc:.0f} vehicles, {occ:.1f}% occupancy)."
        )

        plans.append(
            SignalPlan(
                junction=junc,
                traffic_condition=cond,
                normal_signal_state=normal_state,
                normal_green_s=normal_green,
                normal_yellow_s=normal_yellow,
                normal_red_s=normal_red,
                proposed_signal_state=f"GREEN PRIORITY ({vehicle_type.upper()} PREEMPTION)",
                proposed_green_s=proposed_green,
                proposed_yellow_s=normal_yellow,
                proposed_red_s=proposed_red,
                priority_activated=True,
                emergency_vehicle_type=vehicle_type,
                preemption_lead_s=lead_s,
                clearance_buffer_s=clearance_s,
                estimated_wait_reduction_pct=wait_reduction,
                rationale=rationale,
            )
        )
    return plans

