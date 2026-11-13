# Proactive Traffic Management System with Video-Based Emergency Vehicle Detection

AI-Powered Proactive Traffic Management and Intelligent Route Recommendation System for **Gandhipuram, Coimbatore** (`6` monitored junctions), featuring **two independent data input sources**, a **role-based authentication system** with an enterprise dark **Admin Traffic Control Centre** and a clean light **Normal User Navigation Portal**, integrated with real **OpenStreetMap** geographic street maps and **YOLOv8 video vehicle detection & tracking**.

---

## 1. Two Independent Data Sources

The platform explicitly maintains technical separation between two distinct data streams:

### Source A — Historical Traffic Dataset
- **File:** `datasets/gandhipuram_coimbatore_traffic_synthetic.csv`
- **Coverage:** 218 valid timestamped records (240 raw rows minus 22 trailing empty CSV rows) across 5 weekdays (`12-10-2026 07:30` to `16-10-2026 15:30`) and 6 Gandhipuram junctions:
  1. `100 Feet Road Junction` (`11.018874, 76.971176`)
  2. `Avarampalayam Road Approach` (`11.023995, 76.967988`)
  3. `Cross Cut Road` (`11.020062, 76.965988`)
  4. `Dr Nanjappa Road` (`11.015542, 76.966603`)
  5. `Gandhipuram Bus Stand Approach` (`11.017811, 76.967426`)
  6. `Sathy Road Junction` (`11.021250, 76.969984`)
- **Exclusive Usage:** Historical traffic analysis, chronological ML model training, 1–6 hour future congestion forecasting, proactive congestion alerts, and junction traffic statistics. Never mixed with unverified video counts.

### Source B — Pre-Recorded Traffic Video Input
- **Files:** `Trafficvedio.mp4` (Preferred primary, 1920x1080, 23.98 fps, 660 frames, 27.5s) and `Trafficvedio.avi` (Fallback container).
- **Labeling in UI:** `PRE-RECORDED VIDEO INPUT`
- **Functionality:** OpenCV frame decoding, YOLOv8 vehicle detection, ByteTrack multi-object tracking, unique vehicle counting across classes (Cars, Buses, Trucks, Motorcycles), and emergency vehicle candidate evaluation.
- **Separation Guarantee:** Video analytics are derived from pre-recorded highway footage and default to `Unmapped Video Source` unless explicitly associated with a monitored junction by the control operator. Video detections never overwrite CSV ground truth or corrupt model training weights.

---

## 2. Video Vehicle Detection & Tracking Architecture

Implemented in `src/video_vehicle_detection.py` and `src/video_processing.py`:
- **Model:** YOLOv8 (`yolov8n.pt`) with COCO-80 vehicle classes (`Car`, `Truck`, `Bus`, `Motorcycle`, `Bicycle`).
- **Tracking:** Multi-frame tracking with persistent IDs (`model.track(..., persist=True)`), storing trajectory positions and unique counts.
- **HUD Overlay:** Renders real-time telemetry banner on top of video frames (Source tag, Frame index, Timestamp, Active vehicles, Cumulative unique count, Class breakdown).
- **Session Controller:** Stateful `VideoStreamProcessor` supporting `Start / Step Processing`, `Fast 10 Steps`, `Pause`, `Stop`, and `Reset`, with configurable confidence thresholds (0.20–0.90) and frame sampling steps (1–15).
- **Memory Efficiency:** Avoids caching uncompressed 1080p frames; resizes preview frames dynamically for high-speed, non-blocking Streamlit rendering.

---

## 3. Emergency Vehicle Recognition & Operator Workflow

Implemented in `src/emergency_vehicle_detection.py` and `src/emergency_event_manager.py`:
- **Honest Model Auditing:** `EmergencyModelAuditor` inspects loaded weights and transparently verifies that standard COCO weights (`yolov8n.pt`) do **NOT** contain discrete classes for `Ambulance`, `Fire Engine`, or `Police Vehicle`. Rejects false re-labeling of civilian vehicles (e.g. white vans or red trucks).
- **Footage Truth Assessment:** Audits `Trafficvedio.mp4` and confirms it depicts standard civilian highway traffic with **no emergency vehicles present** (`Status: No emergency vehicle detected`).
- **Multi-Frame Temporal Confirmation Filter:** Enforces temporal persistence rules to eliminate false positives:
  - Single frame observation ➔ `Detection candidate` (awaiting temporal confirmation).
  - Persisted across $\ge 3$ consecutive frames with confidence $\ge 0.70$ ➔ `Confirmed detection`.
  - Intermittent or confidence between $0.40$ and $0.70$ ➔ `Uncertain detection` (disqualified from automatic routing).
- **Operator Review & Verification Workflow:** Confirmed detections require operator approval (Confirm / Dismiss) and junction association before activating emergency priority routing.
- **Simulation & Test Injection Suite:** Allows operators to inject verified test scenarios (Ambulance, Fire Engine, Police Vehicle) across consecutive frames to validate the entire end-to-end confirmation, junction association, green-wave signal preemption, and consumer advisory pipeline.

---

## 4. Emergency Priority Routing & Simulated Signal Preemption

Implemented in `src/emergency_routing.py` and `src/signal_controller.py`:
- **Corridor Optimization:** Calculates emergency-priority corridor route minimizing travel time and avoiding predicted High-congestion bottlenecks and accident locations.
- **Real OSM Geometry:** Overlays the emergency route (purple corridor) alongside standard routes on the real OpenStreetMap street network.
- **Simulated Signal Preemption:** Generates intersection signal timing adjustments (cycle duration, green phase extension, preemption lead time, and queue clearance) for all junctions along the emergency path.
- **Mandatory Disclaimer:** Displays prominent warning across Admin and User interfaces:
  > *Emergency route and signal priority are software simulations. No physical traffic signal or emergency service is controlled.*

---

## 5. Dual-Role User Interface

### Admin — Intelligent Traffic Control Centre (Dark Enterprise Theme)
Sidebar navigation with 13 functional pages + Logout:
1. **Overview:** Dual-stream KPI panels (Historical CSV metrics vs Video Analytics metrics), large interactive OpenStreetMap map, 9-field junction detail inspector, snapshot/forecast toggles, and full 6-junction table.
2. **Junction Monitoring:** Three-way monitoring source selector (`Historical Dataset`, `Video Analytics`, `Combined View`), junction history trends, speed graphs, and camera location mapping.
3. **Video Analytics:** Real-time video processing workspace with file selector, confidence slider, frame step control, annotated video playback with HUD, vehicle class metrics, and detection event log table.
4. **Emergency Vehicle Detection:** Model architecture inspector, video footage ground truth assessment, multi-frame temporal confirmation filter, event history, operator confirmation controls, and test injection suite.
5. **Future Traffic Prediction:** 1–6 hour horizon forecasting for vehicle volume, speed, road occupancy, and risk scores with forecast charts and map marker updates.
6. **Proactive Alerts:** Early warning system for anticipated High congestion 1–6 hours ahead with preventive action recommendations and click-to-zoom map focus.
7. **Intelligent Route Management:** OpenStreetMap shortest vs AI-recommended routes with mapped road distances, ETAs, and alternative routes.
8. **Emergency Priority Control:** Direct dispatch from confirmed video events, emergency corridor routing, time savings KPIs, and simulated signal preemption table.
9. **Signal Simulation:** Adaptive intersection signal timing simulation with green phase split adjustments and cycle time breakdown.
10. **Model Performance:** Random Forest classification accuracy, regression MAE/RMSE, feature importance, and confusion matrix heatmap.
11. **Dataset Validation:** Strict 218-record CSV integrity report with junction coordinates and trailing row removal stats.
12. **User Management:** RBAC user directory, password resets, account creation, and deactivation.
13. **Settings:** Offline road fallback toggle, tile failure simulation toggle, and OSM cache refresh.
14. **Logout:** Clean session teardown and return to login screen.

### Normal User — Smart Traffic and Route Navigation Portal (Light Theme)
Top-header navigation bar with 7 sections + Logout:
1. **Home:** Map-first interface with Gandhipuram street map, 6 junction status cards, `FIND BEST ROUTE` quick search, and active emergency public advisory banner.
2. **Traffic Map:** Real OpenStreetMap with 6 CSV junctions, observed speeds, road occupancy, and clickable junction detail spotlight.
3. **Find Best Route:** Origin & destination search, departure time selection, AI recommended route with alternative options, mapped road distances, and journey times.
4. **Future Traffic:** 1–6 hour forecast for any junction with clear human-readable explanations.
5. **Traffic Alerts:** Predicted High congestion warnings with click-to-focus map zoom.
6. **Emergency Information:** Read-only public traffic advisory for active emergency corridors, vehicle type, and safe detour recommendations (strictly prevents normal users from controlling signals or video).
7. **My Account:** User profile information and password change.
8. **Logout:** Clean logout.

---

## 6. Project File Structure

```
├── app.py                            # Streamlit entry point, RBAC router & theme injector
├── datasets/
│   └── gandhipuram_coimbatore_traffic_synthetic.csv  # Sole traffic dataset (218 valid rows)
├── data/
│   ├── traffic_app.db                # Local SQLite authentication & settings database
│   └── gandhipuram_osm_road_network.json  # Cached OpenStreetMap Gandhipuram road graph
├── models/                           # Trained ML models and metadata
│   ├── gandhipuram_classifier.pkl
│   ├── gandhipuram_reg_vehicle_count.pkl
│   ├── gandhipuram_reg_speed.pkl
│   └── gandhipuram_reg_occupancy.pkl
├── src/
│   ├── auth.py                       # PBKDF2 authentication, RBAC, sessions, and settings
│   ├── charts.py                     # Plotly chart builders with dual-mode theme support
│   ├── emergency_event_manager.py    # Emergency event lifecycle, review, and public advisories
│   ├── emergency_routing.py          # Emergency priority corridor routing & Dijkstra weighting
│   ├── emergency_vehicle_detection.py# Model auditor, temporal confirmation filter & simulation
│   ├── future_predictor.py           # Chronological multi-horizon forecasting & proactive alerts
│   ├── future_routing.py             # Route recommendation engine & corridor definitions
│   ├── map_component.py              # Plotly MapLibre reusable map component
│   ├── map_data.py                   # Map nodes, color palettes, and bounding box math
│   ├── road_network.py               # OpenStreetMap road graph parser & coordinate snapper
│   ├── route_visualization.py        # Unified route bundling across real roads and fallbacks
│   ├── signal_controller.py          # Adaptive signal controller & emergency preemption
│   ├── ui_admin.py                   # Admin Traffic Control Centre interface (13 pages)
│   ├── ui_login.py                   # Login & registration UI
│   ├── ui_theme.py                   # Dark & Light CSS injection and card renderers
│   ├── ui_user.py                    # Normal User consumer navigation portal (7 pages)
│   ├── validation.py                 # Strict CSV validation and cleaning
│   ├── video_processing.py           # Stateful video session processor & playback controller
│   └── video_vehicle_detection.py    # YOLOv8 video vehicle detection, tracking & counting
├── tests/
│   ├── test_app.py                   # End-to-end AppTest suite (Admin, User, RBAC, Settings)
│   ├── test_auth.py                  # PBKDF2 hashing, session tokens, and RBAC tests
│   ├── test_emergency_detection.py   # Model auditor, temporal filter, and event manager tests
│   ├── test_emergency_routing.py     # Emergency routing and signal priority unit tests
│   ├── test_future_predictor.py      # Forecasting, multi-horizon, and CSV validation tests
│   ├── test_future_routing.py        # Dijkstra routing and accident avoidance tests
│   ├── test_map_and_routing.py       # OpenStreetMap snapping, mapped distance & tile tests
│   └── test_video_detection.py       # Video decoding, YOLO tracking, and unique counting tests
├── Trafficvedio.mp4                  # Primary pre-recorded video input
├── Trafficvedio.avi                  # Fallback pre-recorded video container
└── yolov8n.pt                        # YOLOv8 nano object detection weights
```

---

## 7. Automated Testing & Verification

Run the complete test suite:
```bash
python -m pytest -v
```

**Verification Results:**
- **Total Test Cases:** 32 passed, 0 failed in `29.5s`
- **Coverage:**
  - `tests/test_app.py` (3 passed: Admin 13-page traversal, Normal User 7-page navigation, RBAC query param block, and User self-registration)
  - `tests/test_auth.py` (6 passed: PBKDF2 hashing, sessions, admin bootstrap, user creation, RBAC enforcement)
  - `tests/test_video_detection.py` (4 passed: MP4 & AVI metadata, YOLO tracking, unique vehicle counts, stream stepping)
  - `tests/test_emergency_detection.py` (4 passed: honest COCO audit, video truth assessment, multi-frame temporal filter, operator workflow)
  - `tests/test_emergency_routing.py` (2 passed: emergency corridor calculation, signal priority plans)
  - `tests/test_future_predictor.py` (5 passed: strict CSV validation, chronological forecasting, proactive alerts)
  - `tests/test_future_routing.py` (3 passed: Dijkstra shortest vs AI route, departure time sensitivity)
  - `tests/test_map_and_routing.py` (5 passed: OSM node snapping, real road polyline distances vs straight line, tile fallback)

---

## 8. Launching the Application

Run the application locally:
```bash
streamlit run app.py
```
*(If `streamlit` is not directly on Windows `PATH`, run `python -m streamlit run app.py`.)*

### Default Accounts & Setup
1. **Initial Admin Setup:** If `data/traffic_app.db` is fresh, navigate to `http://localhost:8501` to configure the initial administrator.
2. **CLI Account Provisioning:**
   ```bash
   python -m src.auth set-admin --username admin@example.com --replace-existing
   ```
3. **Consumer Self-Registration:** Any other email can register from the login screen (strictly assigned role `user`).
