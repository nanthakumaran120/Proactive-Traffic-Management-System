"""Video-based vehicle detection, tracking, classification, and unique counting module.

SOURCE B: PRE-RECORDED VIDEO INPUT
Uses OpenCV and YOLOv8 to detect, classify, track, and count vehicles across frames
from 'Trafficvedio.mp4' (with 'Trafficvedio.avi' fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Generator

import cv2
import numpy as np
from ultralytics import YOLO


# Standard YOLO COCO vehicle class mappings
COCO_VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "bicycle": "Bicycle",
}

CLASS_COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "car": (59, 130, 246),        # Blue
    "motorcycle": (234, 179, 8),   # Yellow
    "bus": (168, 85, 247),        # Purple
    "truck": (249, 115, 22),       # Orange
    "bicycle": (16, 185, 129),     # Green
}

DEFAULT_MP4_NAME = "Trafficvedio.mp4"
DEFAULT_AVI_NAME = "Trafficvedio.avi"
DEFAULT_MODEL_NAME = "yolov8n.pt"


def resolve_video_path(workspace_dir: Path | str | None = None) -> tuple[Path, str]:
    """Resolve preferred MP4 video path or fallback AVI file.

    Returns:
        tuple of (resolved_path, source_format_note)
    """
    base = Path(workspace_dir) if workspace_dir is not None else Path.cwd()

    mp4_path = base / DEFAULT_MP4_NAME
    avi_path = base / DEFAULT_AVI_NAME

    if mp4_path.exists() and mp4_path.stat().st_size > 0:
        return mp4_path, "MP4 Primary Video"
    if avi_path.exists() and avi_path.stat().st_size > 0:
        return avi_path, "AVI Fallback Video"

    # Search for any other .mp4 or .avi in workspace
    for alt in base.glob("*.mp4"):
        if alt.stat().st_size > 0 and "checkpoint" not in alt.name.lower():
            return alt, f"Alternative MP4 ({alt.name})"
    for alt in base.glob("*.avi"):
        if alt.stat().st_size > 0:
            return alt, f"Alternative AVI ({alt.name})"

    # Return default expected path even if missing
    return mp4_path, "MP4 (File Pending)"


def resolve_model_path(workspace_dir: Path | str | None = None) -> Path:
    """Resolve YOLO weights path."""
    base = Path(workspace_dir) if workspace_dir is not None else Path.cwd()
    root_model = base / DEFAULT_MODEL_NAME
    if root_model.exists():
        return root_model
    models_dir = base / "models" / DEFAULT_MODEL_NAME
    if models_dir.exists():
        return models_dir
    return root_model


@dataclass
class VehicleDetection:
    """Single vehicle detection instance in a video frame."""

    track_id: int | None
    class_name: str
    display_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[int, int]          # (cx, cy)
    is_emergency_candidate: bool = False
    emergency_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "center": self.center,
            "is_emergency_candidate": self.is_emergency_candidate,
            "emergency_type": self.emergency_type,
        }


@dataclass
class FrameAnalytics:
    """Detection, tracking, and counting metrics for a single video frame."""

    frame_idx: int
    timestamp_sec: float
    detections: list[VehicleDetection]
    active_vehicle_count: int
    cumulative_unique_vehicles: int
    cumulative_class_counts: dict[str, int]
    current_frame_class_counts: dict[str, int]
    mean_confidence: float
    is_keyframe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "active_vehicle_count": self.active_vehicle_count,
            "cumulative_unique_vehicles": self.cumulative_unique_vehicles,
            "cumulative_class_counts": dict(self.cumulative_class_counts),
            "current_frame_class_counts": dict(self.current_frame_class_counts),
            "mean_confidence": round(self.mean_confidence, 3),
            "detection_count": len(self.detections),
        }


@dataclass
class VideoMetadata:
    """Metadata and technical specs of the traffic video."""

    file_path: str
    file_name: str
    file_size_mb: float
    format_label: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    source_label: str = "PRE-RECORDED VIDEO INPUT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size_mb": round(self.file_size_mb, 2),
            "format_label": self.format_label,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "source_label": self.source_label,
        }


class VideoVehicleDetector:
    """Loads YOLO model and executes vehicle detection, tracking, and unique counting."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        vehicle_class_ids: dict[int, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else resolve_model_path()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.vehicle_class_ids = vehicle_class_ids or COCO_VEHICLE_CLASSES
        self._model: YOLO | None = None

        # Cumulative tracking memory
        self.tracked_vehicles: dict[int, str] = {}  # track_id -> class_name
        self.vehicle_first_seen: dict[int, int] = {}  # track_id -> frame_idx
        self.vehicle_last_seen: dict[int, int] = {}   # track_id -> frame_idx
        self.vehicle_positions: dict[int, list[tuple[int, int]]] = {} # track_id -> [(cx, cy)]

    @property
    def model(self) -> YOLO:
        """Lazy-loaded YOLO model."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"YOLO model weights not found at: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def reset_tracking(self) -> None:
        """Reset internal tracking state for a new video run."""
        self.tracked_vehicles.clear()
        self.vehicle_first_seen.clear()
        self.vehicle_last_seen.clear()
        self.vehicle_positions.clear()

    @staticmethod
    def get_video_metadata(video_path: Path | str) -> VideoMetadata:
        """Inspect video technical parameters without decoding all frames."""
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found at: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise IOError(f"OpenCV failed to open video file: {vpath}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        size_mb = vpath.stat().st_size / (1024 * 1024)
        ext = vpath.suffix.upper().lstrip(".")
        format_label = f"{ext} Container"

        return VideoMetadata(
            file_path=str(vpath.resolve()),
            file_name=vpath.name,
            file_size_mb=size_mb,
            format_label=format_label,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float = 24.0,
        persist: bool = True,
        conf_override: float | None = None,
    ) -> tuple[np.ndarray, FrameAnalytics]:
        """Process a single video frame: run YOLO track, extract IDs, and render overlays.

        Args:
            frame: OpenCV BGR image
            frame_idx: integer index of the frame in the video
            fps: video frames per second
            persist: whether to maintain tracker state across consecutive frames
            conf_override: optional custom confidence threshold

        Returns:
            tuple of (annotated_frame_bgr, FrameAnalytics)
        """
        conf = conf_override if conf_override is not None else self.conf_threshold
        classes_filter = list(self.vehicle_class_ids.keys())

        # Execute tracking using Ultralytics ByteTrack / BoT-SORT
        results = self.model.track(
            source=frame,
            classes=classes_filter,
            persist=persist,
            conf=conf,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections: list[VehicleDetection] = []
        current_counts: dict[str, int] = {name: 0 for name in CLASS_DISPLAY_NAMES.values()}
        confidences: list[float] = []

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes_data = results.boxes

            # Extract track IDs if available
            track_ids = (
                boxes_data.id.cpu().numpy().astype(int)
                if boxes_data.id is not None
                else [None] * len(boxes_data)
            )

            coords = boxes_data.xyxy.cpu().numpy().astype(int)
            clss = boxes_data.cls.cpu().numpy().astype(int)
            confs = boxes_data.conf.cpu().numpy().astype(float)

            for bbox, cls_id, score, trk_id in zip(coords, clss, confs, track_ids):
                class_key = self.vehicle_class_ids.get(cls_id, "car")
                display_label = CLASS_DISPLAY_NAMES.get(class_key, class_key.capitalize())
                current_counts[display_label] = current_counts.get(display_label, 0) + 1
                confidences.append(float(score))

                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Update cumulative tracking dictionaries
                if trk_id is not None:
                    tid = int(trk_id)
                    if tid not in self.tracked_vehicles:
                        self.tracked_vehicles[tid] = display_label
                        self.vehicle_first_seen[tid] = frame_idx
                        self.vehicle_positions[tid] = []
                    self.vehicle_last_seen[tid] = frame_idx
                    self.vehicle_positions[tid].append((cx, cy))

                det = VehicleDetection(
                    track_id=int(trk_id) if trk_id is not None else None,
                    class_name=class_key,
                    display_name=display_label,
                    confidence=float(score),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(cx, cy),
                )
                detections.append(det)

                # Draw bounding box and label overlay
                color = CLASS_COLOR_PALETTE.get(class_key, (0, 255, 0))
                # Bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Label text
                tid_str = f"#{trk_id} " if trk_id is not None else ""
                label_txt = f"{tid_str}{display_label} {int(score * 100)}%"
                (lw, lh), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

                # Label background tag
                tag_y1 = max(0, y1 - lh - 8)
                tag_y2 = y1
                cv2.rectangle(annotated_frame, (x1, tag_y1), (x1 + lw + 8, tag_y2), color, -1)
                cv2.putText(
                    annotated_frame,
                    label_txt,
                    (x1 + 4, tag_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Center marker
                cv2.circle(annotated_frame, (cx, cy), 3, color, -1)

        # Build cumulative class breakdown
        cum_classes: dict[str, int] = {name: 0 for name in CLASS_DISPLAY_NAMES.values()}
        for _, c_name in self.tracked_vehicles.items():
            cum_classes[c_name] = cum_classes.get(c_name, 0) + 1

        # Render top status HUD banner
        timestamp_sec = frame_idx / fps if fps > 0 else 0.0
        self._render_hud_banner(
            annotated_frame,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            active_count=len(detections),
            unique_count=len(self.tracked_vehicles),
            classes_breakdown=cum_classes,
        )

        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        analytics = FrameAnalytics(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detections=detections,
            active_vehicle_count=len(detections),
            cumulative_unique_vehicles=len(self.tracked_vehicles),
            cumulative_class_counts=cum_classes,
            current_frame_class_counts=current_counts,
            mean_confidence=mean_conf,
        )

        return annotated_frame, analytics

    def _render_hud_banner(
        self,
        img: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        active_count: int,
        unique_count: int,
        classes_breakdown: dict[str, int],
    ) -> None:
        """Render professional dark semi-transparent HUD banner on top of video frame."""
        h, w = img.shape[:2]
        hud_h = 44

        # Semi-transparent bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

        # Border line
        cv2.line(img, (0, hud_h), (w, hud_h), (37, 99, 235), 2)

        # Left label: SOURCE B indicator
        source_txt = "SOURCE B: PRE-RECORDED VIDEO INPUT"
        cv2.putText(img, source_txt, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (56, 189, 248), 2, cv2.LINE_AA)

        # Middle metrics: Frame, Time, Active, Unique
        time_txt = (
            f"Frame: {frame_idx:04d} | Time: {timestamp_sec:.1f}s | "
            f"Active: {active_count} | Unique Total: {unique_count}"
        )
        cv2.putText(img, time_txt, (430, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (241, 245, 249), 1, cv2.LINE_AA)

        # Right stats: breakdown
        cars = classes_breakdown.get("Car", 0)
        trucks = classes_breakdown.get("Truck", 0)
        buses = classes_breakdown.get("Bus", 0)
        stats_txt = f"Cars: {cars} | Trucks: {trucks} | Buses: {buses}"
        (sw, _), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(img, stats_txt, (w - sw - 16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA)

