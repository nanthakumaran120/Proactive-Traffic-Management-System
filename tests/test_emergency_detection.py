"""Unit and integration tests for emergency vehicle recognition and event management."""

from __future__ import annotations

import unittest

from src.emergency_event_manager import (
    EmergencyEventManager,
    EmergencyEventRecord,
    EventReviewStatus,
)
from src.emergency_vehicle_detection import (
    EmergencyDetectionStatus,
    EmergencyModelAuditor,
    TemporalEmergencyConfirmationFilter,
    VideoEmergencyAnalyzer,
)


class TestEmergencyVehicleRecognition(unittest.TestCase):
    def test_model_auditor_reports_truth_about_coco_weights(self) -> None:
        """Verify model auditor honestly reports that standard YOLOv8n has no emergency classes."""
        report = EmergencyModelAuditor.audit_model("yolov8n.pt")
        self.assertEqual(report.model_name, "yolov8n.pt")
        self.assertEqual(report.total_classes, 80)
        self.assertFalse(report.has_ambulance_class)
        self.assertFalse(report.has_fire_engine_class)
        self.assertFalse(report.has_police_class)
        self.assertFalse(report.is_dedicated_emergency_model)
        self.assertIn("COCO General Vehicle Detector", report.validation_status)

    def test_video_analyzer_honestly_assesses_clip(self) -> None:
        """Verify Trafficvedio.mp4 is assessed as normal traffic with no emergency vehicles."""
        analyzer = VideoEmergencyAnalyzer("yolov8n.pt")
        truth = analyzer.analyze_clip_truthfulness("Trafficvedio.mp4", total_tracked_vehicles=89)
        self.assertEqual(truth["overall_status"], EmergencyDetectionStatus.NONE.value)
        self.assertFalse(truth["emergency_vehicles_found_in_clip"])
        self.assertIn("No ambulances, fire engines, or police vehicles are present", truth["summary"])

    def test_temporal_filter_requires_consecutive_frames_for_confirmation(self) -> None:
        """Verify single frame is a candidate and >= 3 frames becomes confirmed."""
        filt = TemporalEmergencyConfirmationFilter(
            min_consecutive_frames=3,
            conf_threshold=0.70,
            uncertain_threshold=0.45,
        )

        # Frame 1: Candidate
        ev1 = filt.evaluate_candidate(
            emergency_type="Ambulance",
            frame_idx=10,
            timestamp_sec=0.42,
            confidence=0.85,
            track_id=101,
            bbox=(100, 100, 200, 200),
        )
        self.assertEqual(ev1.status, EmergencyDetectionStatus.CANDIDATE)
        self.assertEqual(ev1.consecutive_frames_seen, 1)

        # Frame 2: Candidate
        ev2 = filt.evaluate_candidate(
            emergency_type="Ambulance",
            frame_idx=11,
            timestamp_sec=0.46,
            confidence=0.87,
            track_id=101,
            bbox=(102, 100, 202, 200),
        )
        self.assertEqual(ev2.status, EmergencyDetectionStatus.CANDIDATE)
        self.assertEqual(ev2.consecutive_frames_seen, 2)

        # Frame 3: Confirmed
        ev3 = filt.evaluate_candidate(
            emergency_type="Ambulance",
            frame_idx=12,
            timestamp_sec=0.50,
            confidence=0.88,
            track_id=101,
            bbox=(105, 100, 205, 200),
        )
        self.assertEqual(ev3.status, EmergencyDetectionStatus.CONFIRMED)
        self.assertEqual(ev3.consecutive_frames_seen, 3)

        # Low confidence -> Uncertain
        ev_low = filt.evaluate_candidate(
            emergency_type="Ambulance",
            frame_idx=50,
            timestamp_sec=2.08,
            confidence=0.35,
            track_id=202,
            bbox=(50, 50, 100, 100),
        )
        self.assertEqual(ev_low.status, EmergencyDetectionStatus.UNCERTAIN)

    def test_emergency_event_manager_operator_workflow(self) -> None:
        """Verify candidate event creation, operator confirmation, and public advisory."""
        mgr = EmergencyEventManager()
        analyzer = VideoEmergencyAnalyzer("yolov8n.pt")

        injected = analyzer.inject_simulated_test_detection(
            emergency_type="Ambulance",
            confidence=0.91,
            consecutive_frames=3,
            origin_junction="100 Feet Road Junction",
            destination_junction="Gandhipuram Bus Stand Approach",
        )

        record = mgr.create_event_from_detection(injected)
        self.assertIsNotNone(record)
        self.assertEqual(record.vehicle_type, "Ambulance")
        self.assertEqual(record.review_status, EventReviewStatus.PENDING_REVIEW)

        # Prior to confirmation, no active event
        self.assertIsNone(mgr.get_active_event())
        self.assertIsNone(mgr.get_public_traffic_advisory())

        # Operator confirms event and associates origin & destination
        confirmed_rec = mgr.confirm_and_activate_event(
            event_id=record.event_id,
            origin_junction="100 Feet Road Junction",
            destination_junction="Gandhipuram Bus Stand Approach",
            operator_notes="Confirmed by Control Centre Lead",
        )
        self.assertIsNotNone(confirmed_rec)
        self.assertEqual(confirmed_rec.review_status, EventReviewStatus.CONFIRMED)

        # Active event and public advisory available
        active = mgr.get_active_event()
        self.assertIsNotNone(active)
        self.assertEqual(active.event_id, record.event_id)

        advisory = mgr.get_public_traffic_advisory()
        self.assertIsNotNone(advisory)
        self.assertIn("Active Ambulance Priority Corridor", advisory)
        self.assertIn("100 Feet Road Junction", advisory)

        # Clearing active corridor
        mgr.clear_active_corridor(record.event_id)
        self.assertIsNone(mgr.get_active_event())


if __name__ == "__main__":
    unittest.main()

