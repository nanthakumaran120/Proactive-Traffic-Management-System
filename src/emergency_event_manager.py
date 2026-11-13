"""Emergency event management, operator verification workflow, and public advisory dispatch.

Manages the lifecycle of emergency detection events:
Candidate -> Operator Review -> Confirmed/Dismissed -> Junction Association -> Priority Routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.emergency_vehicle_detection import (
    EmergencyDetectionEvent,
    EmergencyDetectionStatus,
)


class EventReviewStatus(str, Enum):
    """Operator review status for emergency detection events."""

    PENDING_REVIEW = "Pending Operator Review"
    CONFIRMED = "Confirmed & Active"
    DISMISSED = "Dismissed by Operator"
    RESOLVED = "Corridor Cleared / Resolved"


@dataclass
class EmergencyEventRecord:
    """Persistent representation of an emergency priority incident."""

    event_id: str
    vehicle_type: str
    detected_timestamp_sec: float
    detected_time_str: str
    confidence: float
    track_id: int | None
    video_source: str
    detection_status: str
    review_status: EventReviewStatus
    origin_junction: str | None = None
    destination_junction: str | None = None
    operator_notes: str = ""
    created_at_iso: str = field(default_factory=lambda: datetime.now().isoformat())
    operator_confirmed_at_iso: str | None = None
    cleared_at_iso: str | None = None
    routing_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["review_status"] = self.review_status.value
        return d


class EmergencyEventManager:
    """Session-level coordinator for emergency detection events and operator review."""

    def __init__(self) -> None:
        self.events: list[EmergencyEventRecord] = []
        self._active_event_id: str | None = None

    def create_event_from_detection(
        self,
        detection: EmergencyDetectionEvent,
        video_source_label: str = "PRE-RECORDED VIDEO INPUT",
        origin_junction: str | None = None,
        destination_junction: str | None = None,
    ) -> EmergencyEventRecord:
        """Create an emergency event record from an emergency detection."""
        # Check if already registered
        for ev in self.events:
            if ev.event_id == detection.event_id:
                return ev

        is_confirmed = (detection.status == EmergencyDetectionStatus.CONFIRMED and detection.operator_confirmed)
        review_status = (
            EventReviewStatus.CONFIRMED if is_confirmed else EventReviewStatus.PENDING_REVIEW
        )

        record = EmergencyEventRecord(
            event_id=detection.event_id,
            vehicle_type=detection.emergency_type,
            detected_timestamp_sec=detection.timestamp_sec,
            detected_time_str=f"{detection.timestamp_sec:.1f}s into video",
            confidence=detection.confidence,
            track_id=detection.track_id,
            video_source=video_source_label,
            detection_status=detection.status.value,
            review_status=review_status,
            origin_junction=origin_junction or detection.assigned_origin_junction,
            destination_junction=destination_junction or detection.assigned_destination_junction,
            operator_notes=detection.operator_notes,
        )
        self.events.insert(0, record)
        if review_status == EventReviewStatus.CONFIRMED:
            self._active_event_id = record.event_id
        return record

    def confirm_and_activate_event(
        self,
        event_id: str,
        origin_junction: str,
        destination_junction: str,
        operator_notes: str = "Verified by Control Centre Operator",
    ) -> EmergencyEventRecord | None:
        """Confirm an emergency event, associate Gandhipuram junctions, and activate priority routing."""
        for ev in self.events:
            if ev.event_id == event_id:
                ev.review_status = EventReviewStatus.CONFIRMED
                ev.origin_junction = origin_junction
                ev.destination_junction = destination_junction
                ev.operator_notes = operator_notes
                ev.operator_confirmed_at_iso = datetime.now().isoformat()
                self._active_event_id = event_id
                return ev
        return None

    def dismiss_event(self, event_id: str, operator_notes: str = "Dismissed by Operator") -> bool:
        """Dismiss an unconfirmed or false emergency detection."""
        for ev in self.events:
            if ev.event_id == event_id:
                ev.review_status = EventReviewStatus.DISMISSED
                ev.operator_notes = operator_notes
                if self._active_event_id == event_id:
                    self._active_event_id = None
                return True
        return False

    def clear_active_corridor(self, event_id: str) -> bool:
        """Mark an active emergency corridor as resolved."""
        for ev in self.events:
            if ev.event_id == event_id:
                ev.review_status = EventReviewStatus.RESOLVED
                ev.cleared_at_iso = datetime.now().isoformat()
                if self._active_event_id == event_id:
                    self._active_event_id = None
                return True
        return False

    def get_active_event(self) -> EmergencyEventRecord | None:
        """Return the currently active confirmed emergency event, if any."""
        if not self._active_event_id:
            return None
        for ev in self.events:
            if ev.event_id == self._active_event_id and ev.review_status == EventReviewStatus.CONFIRMED:
                return ev
        return None

    def get_all_events(self) -> list[EmergencyEventRecord]:
        return list(self.events)

    def get_pending_review_events(self) -> list[EmergencyEventRecord]:
        return [ev for ev in self.events if ev.review_status == EventReviewStatus.PENDING_REVIEW]

    def get_public_traffic_advisory(self) -> str | None:
        """Generate an operator-approved public traffic advisory for normal users."""
        active = self.get_active_event()
        if not active or not active.origin_junction or not active.destination_junction:
            return None

        return (
            f"Active {active.vehicle_type} Priority Corridor in effect from "
            f"{active.origin_junction} to {active.destination_junction}. "
            "Traffic signals along this route are operating under simulated green-priority preemption. "
            "Normal traffic is advised to divert and use alternative corridors."
        )

