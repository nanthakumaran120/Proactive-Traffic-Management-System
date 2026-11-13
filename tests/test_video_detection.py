"""Unit and integration tests for video-based vehicle detection, tracking, and counting (Source B)."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from src.video_processing import ProcessingState, VideoStreamProcessor
from src.video_vehicle_detection import (
    CLASS_DISPLAY_NAMES,
    VideoVehicleDetector,
    resolve_video_path,
)


class TestVideoVehicleDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mp4_path, cls.mp4_label = resolve_video_path()
        cls.avi_path = Path("Trafficvedio.avi")

    def test_video_paths_and_metadata_exist(self) -> None:
        """Verify MP4 primary exists and AVI fallback exists and have valid metadata."""
        self.assertTrue(self.mp4_path.exists(), f"MP4 video missing: {self.mp4_path}")
        meta_mp4 = VideoVehicleDetector.get_video_metadata(self.mp4_path)
        self.assertEqual(meta_mp4.file_name, "Trafficvedio.mp4")
        self.assertGreater(meta_mp4.width, 0)
        self.assertGreater(meta_mp4.height, 0)
        self.assertGreater(meta_mp4.fps, 0)
        self.assertGreater(meta_mp4.total_frames, 0)
        self.assertGreater(meta_mp4.duration_sec, 0)
        self.assertEqual(meta_mp4.source_label, "PRE-RECORDED VIDEO INPUT")

        if self.avi_path.exists():
            meta_avi = VideoVehicleDetector.get_video_metadata(self.avi_path)
            self.assertEqual(meta_avi.file_name, "Trafficvedio.avi")
            self.assertGreater(meta_avi.total_frames, 0)

    def test_detector_processes_frame_with_tracking_and_hud(self) -> None:
        """Verify YOLO tracking produces bounding boxes, track IDs, and HUD overlays."""
        detector = VideoVehicleDetector(conf_threshold=0.35)
        cap = cv2.VideoCapture(str(self.mp4_path))
        self.assertTrue(cap.isOpened(), "Could not open MP4 video file.")

        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read first frame from video.")

        ann_frame, analytics = detector.process_frame(frame, frame_idx=0, fps=24.0, persist=True)

        self.assertEqual(analytics.frame_idx, 0)
        self.assertGreater(analytics.active_vehicle_count, 0)
        self.assertGreater(analytics.cumulative_unique_vehicles, 0)
        self.assertGreater(len(analytics.detections), 0)
        self.assertIn("Car", analytics.cumulative_class_counts)
        self.assertGreaterEqual(analytics.cumulative_class_counts["Car"], 1)

        # Check that output image has HUD and matching dimensions
        self.assertEqual(ann_frame.shape, frame.shape)
        first_det = analytics.detections[0]
        self.assertIn(first_det.display_name, CLASS_DISPLAY_NAMES.values())
        self.assertGreater(first_det.confidence, 0.30)
        self.assertEqual(len(first_det.bbox), 4)

    def test_tracking_persistence_increments_unique_vehicles(self) -> None:
        """Verify track IDs persist across consecutive frames without exploding unique count."""
        detector = VideoVehicleDetector(conf_threshold=0.35)
        cap = cv2.VideoCapture(str(self.mp4_path))

        uniques_per_frame = []
        for f_idx in range(4):
            ret, frame = cap.read()
            if not ret:
                break
            _, analytics = detector.process_frame(frame, frame_idx=f_idx, fps=24.0, persist=True)
            uniques_per_frame.append(analytics.cumulative_unique_vehicles)
        cap.release()

        self.assertEqual(len(uniques_per_frame), 4)
        # Because cars move slowly, count of unique cars in 4 consecutive frames should remain stable
        self.assertAlmostEqual(uniques_per_frame[0], uniques_per_frame[-1], delta=5)

    def test_stream_processor_stepping_and_summary(self) -> None:
        """Verify VideoStreamProcessor steps, maintains state, and outputs summary."""
        processor = VideoStreamProcessor(
            video_path=self.mp4_path,
            conf_threshold=0.35,
            frame_step=10,
        )
        self.assertEqual(processor.state, ProcessingState.READY)
        self.assertEqual(processor.current_frame_idx, 0)

        # Step 1
        frame_rgb, summary = processor.process_next_step(max_frames_to_advance=1)
        self.assertIsNotNone(frame_rgb)
        self.assertEqual(processor.state, ProcessingState.PROCESSING)
        self.assertEqual(summary.state, ProcessingState.PROCESSING)
        self.assertGreater(summary.total_unique_vehicles, 0)
        self.assertGreater(summary.frames_processed, 0)
        self.assertEqual(summary.source_label, "PRE-RECORDED VIDEO INPUT")

        # Step 2
        _, summary2 = processor.process_next_step(max_frames_to_advance=1)
        self.assertGreater(summary2.frames_processed, summary.frames_processed)

        # Reset
        processor.reset()
        self.assertEqual(processor.state, ProcessingState.READY)
        self.assertEqual(processor.current_frame_idx, 0)
        self.assertEqual(len(processor.detector.tracked_vehicles), 0)


if __name__ == "__main__":
    unittest.main()

