# Edge-Optimized Real-Time Traffic & Pedestrian Anomaly Detector

A lightweight computer vision pipeline designed for **Intelligent Transportation Systems (ITS)** and edge deployment. Built using **YOLO11n**, **ONNX Runtime (FP16)**, and **ByteTrack**, this system tracks multi-lane vehicles and pedestrians in real time to flag high-risk road anomalies.

## Live Demo & Technical Walkthrough
> **[▶ Watch Full 4K Traffic & Pedestrian Anomaly Detection Demo](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

| Video Timeline | What You See in the Video | AI Technique & Implementation |
| :--- | :--- | :--- |
| **0:00 – 0:15** | Real-time bounding boxes on cars, buses, and pedestrians | **YOLO11n exported to FP16 ONNX** (~6.3 ms/frame inference) |
| **0:15 – 0:35** | Persistent ID numbers stay locked onto moving vehicles | **ByteTrack** multi-object cross-frame association |
| **0:35 – End** | Blue road polygon monitors lane intrusion & abrupt stops | **Ground-anchor projection** & sliding-window velocity smoothing |

## Key Features
- **Edge-Ready Inference:** Exported to FP16 ONNX format for low-latency execution (~6.3 ms/frame on T4 GPU).
- **Multi-Object Tracking:** Persistent cross-frame ID assignment using ByteTrack.
- **Spatial & Velocity Anomaly Detection:**
  - **Jaywalking / Zone Intrusion:** Flags pedestrians entering active traffic lane polygons using bottom-center ground projection.
  - **Sudden Vehicle Stops:** Uses sliding-window pixel velocity smoothing to detect abrupt deceleration in active lanes without single-frame jitter false positives.

## Repository Structure
- `src/mobility_anomaly_detector.py` — Standalone inference, tracking, and anomaly alert pipeline.
- `01_baseline_training_and_onnx_ipynb_.ipynb` — Training, FP16 ONNX export, and speed benchmarking notebook.

## Quickstart
```bash
pip install ultralytics opencv-python numpy onnxruntime-gpu
python src/mobility_anomaly_detector.py
```
Try without personalization

