"""Video processing controller, session state management, and playback orchestration.

Provides non-blocking, cached, and interactive video playback for Streamlit,
avoiding repetitive re-decoding of frames on every UI interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import time
from typing import Any, Generator

import cv2
import numpy as np

from src.video_vehicle_detection import (
    CLASS_DISPLAY_NAMES,
    FrameAnalytics,
    VehicleDetection,
    VideoMetadata,
    VideoVehicleDetector,
    resolve_video_path,
)
from src.emergency_vehicle_detection import (
    EmergencyDetectionEvent,
    EmergencyDetectionStatus,
    VideoEmergencyAnalyzer,
)


class ProcessingState(str, Enum):
    """Lifecycle states of the video analytics pipeline."""

    READY = "Ready"
    PROCESSING = "Processing"
    PAUSED = "Paused"
    COMPLETED = "Completed"
    ERROR = "Error"


@dataclass
class VideoSessionSummary:
    """Consolidated metrics of a video processing session."""

    state: ProcessingState
    video_metadata: VideoMetadata
    frames_processed: int
    percent_complete: float
    total_unique_vehicles: int
    active_vehicles_current: int
    class_breakdown: dict[str, int]
    current_frame_class_breakdown: dict[str, int]
    mean_confidence: float
    emergency_status: str
    confirmed_emergency_count: int
    processing_fps: float
    detection_log: list[dict[str, Any]]
    source_label: str = "PRE-RECORDED VIDEO INPUT"
    mapped_location: str = "Unmapped Video Source"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "video_file": self.video_metadata.file_name,
            "frames_processed": self.frames_processed,
            "percent_complete": round(self.percent_complete, 1),
            "total_unique_vehicles": self.total_unique_vehicles,
            "active_vehicles_current": self.active_vehicles_current,
            "class_breakdown": self.class_breakdown,
            "mean_confidence": round(self.mean_confidence, 3),
            "emergency_status": self.emergency_status,
            "confirmed_emergency_count": self.confirmed_emergency_count,
            "processing_fps": round(self.processing_fps, 1),
            "source_label": self.source_label,
            "mapped_location": self.mapped_location,
        }


class VideoStreamProcessor:
    """Stateful video processor that controls stepping, playback, and metrics caching."""

    def __init__(
        self,
        video_path: Path | str | None = None,
        conf_threshold: float = 0.35,
        frame_step: int = 5,
        target_preview_width: int = 960,
    ) -> None:
        if video_path is None:
            resolved_p, _ = resolve_video_path()
            self.video_path = resolved_p
        else:
            self.video_path = Path(video_path)

        self.conf_threshold = conf_threshold
        self.frame_step = max(1, frame_step)
        self.target_preview_width = target_preview_width

        self.detector = VideoVehicleDetector(conf_threshold=conf_threshold)
        self.emergency_analyzer = VideoEmergencyAnalyzer(self.detector.model_path)

        self.state: ProcessingState = ProcessingState.READY
        self.metadata = VideoVehicleDetector.get_video_metadata(self.video_path)

        # Playback tracking
        self.current_frame_idx: int = 0
        self.latest_annotated_frame_rgb: np.ndarray | None = None
        self.latest_analytics: FrameAnalytics | None = None
        self.processing_start_time: float | None = None
        self.processed_frames_count: int = 0

        # Cached frame previews and event log
        self.detection_event_log: list[dict[str, Any]] = []
        self._seen_track_ids_for_log: set[int] = set()

        # Association with Gandhipuram network (default unmapped)
        self.mapped_junction: str | None = None

    def reset(self) -> None:
        """Reset video session back to start."""
        self.state = ProcessingState.READY
        self.current_frame_idx = 0
        self.latest_annotated_frame_rgb = None
        self.latest_analytics = None
        self.processed_frames_count = 0
        self.processing_start_time = None
        self.detection_event_log.clear()
        self._seen_track_ids_for_log.clear()
        self.detector.reset_tracking()
        self.emergency_analyzer.reset()

    def set_mapped_junction(self, junction_name: str | None) -> None:
        """Associate this camera with a specific Gandhipuram junction or keep unmapped."""
        self.mapped_junction = junction_name

    def process_next_step(self, max_frames_to_advance: int = 1) -> tuple[np.ndarray | None, VideoSessionSummary]:
        """Advance processing by one or more sampled frames.

        Returns:
            tuple of (annotated_frame_rgb, VideoSessionSummary)
        """
        if self.state in (ProcessingState.COMPLETED, ProcessingState.ERROR):
            return self.latest_annotated_frame_rgb, self.get_summary()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.state = ProcessingState.ERROR
            return None, self.get_summary()

        t_start = time.time()
        if self.processing_start_time is None:
            self.processing_start_time = t_start

        frames_run = 0
        while frames_run < max_frames_to_advance and self.current_frame_idx < self.metadata.total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                self.state = ProcessingState.COMPLETED
                break

            # Process frame through YOLO and tracking
            ann_bgr, analytics = self.detector.process_frame(
                frame=frame,
                frame_idx=self.current_frame_idx,
                fps=self.metadata.fps,
                persist=True,
                conf_override=self.conf_threshold,
            )

            # Resize for efficient web display preview
            h, w = ann_bgr.shape[:2]
            if w > self.target_preview_width:
                scale = self.target_preview_width / float(w)
                ann_bgr = cv2.resize(ann_bgr, (self.target_preview_width, int(h * scale)))

            # Convert to RGB for Streamlit st.image
            self.latest_annotated_frame_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
            self.latest_analytics = analytics
            self.processed_frames_count += 1

            # Log newly observed unique vehicle tracks in the detection log
            for det in analytics.detections:
                if det.track_id is not None and det.track_id not in self._seen_track_ids_for_log:
                    self._seen_track_ids_for_log.add(det.track_id)
                    self.detection_event_log.insert(
                        0,
                        {
                            "Timestamp": f"{analytics.timestamp_sec:.2f}s",
                            "Frame": self.current_frame_idx,
                            "Track ID": f"#{det.track_id}",
                            "Vehicle Class": det.display_name,
                            "Confidence": f"{det.confidence * 100:.1f}%",
                            "Detection Status": "Tracked",
                            "Emergency Candidate": "No",
                        },
                    )

            # Advance by sampling step
            self.current_frame_idx += self.frame_step
            frames_run += 1

        cap.release()

        if self.current_frame_idx >= self.metadata.total_frames:
            self.state = ProcessingState.COMPLETED
        else:
            self.state = ProcessingState.PROCESSING

        # Keep event log bounded (latest 100 entries)
        if len(self.detection_event_log) > 100:
            self.detection_event_log = self.detection_event_log[:100]

        return self.latest_annotated_frame_rgb, self.get_summary()

    def get_summary(self) -> VideoSessionSummary:
        """Compute the current summary metrics snapshot."""
        total_f = max(1, self.metadata.total_frames)
        pct = min(100.0, (self.current_frame_idx / float(total_f)) * 100.0)

        cum_classes = (
            dict(self.latest_analytics.cumulative_class_counts)
            if self.latest_analytics
            else {c: 0 for c in CLASS_DISPLAY_NAMES.values()}
        )
        cur_classes = (
            dict(self.latest_analytics.current_frame_class_counts)
            if self.latest_analytics
            else {c: 0 for c in CLASS_DISPLAY_NAMES.values()}
        )
        unique_v = len(self.detector.tracked_vehicles)
        active_v = self.latest_analytics.active_vehicle_count if self.latest_analytics else 0
        mean_c = self.latest_analytics.mean_confidence if self.latest_analytics else 0.0

        elapsed = time.time() - self.processing_start_time if self.processing_start_time else 0.0
        fps = (self.processed_frames_count / elapsed) if elapsed > 0 else 0.0

        # Location label
        loc_label = self.mapped_junction if self.mapped_junction else "Unmapped Video Source"

        # Emergency evaluation
        confirmed_emg = sum(
            1 for ev in self.emergency_analyzer.detection_history if ev.status == EmergencyDetectionStatus.CONFIRMED
        )
        emg_status = (
            f"Active Confirmed Event ({confirmed_emg})"
            if confirmed_emg > 0
            else EmergencyDetectionStatus.NONE.value
        )

        return VideoSessionSummary(
            state=self.state,
            video_metadata=self.metadata,
            frames_processed=self.current_frame_idx,
            percent_complete=pct,
            total_unique_vehicles=unique_v,
            active_vehicles_current=active_v,
            class_breakdown=cum_classes,
            current_frame_class_breakdown=cur_classes,
            mean_confidence=mean_c,
            emergency_status=emg_status,
            confirmed_emergency_count=confirmed_emg,
            processing_fps=fps,
            detection_log=list(self.detection_event_log),
            mapped_location=loc_label,
        )

