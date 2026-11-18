# Edge-Optimized Real-Time Traffic & Pedestrian Anomaly Detector

A lightweight computer vision pipeline designed for **Intelligent Transportation Systems (ITS)** and edge deployment. Built using **YOLO11n**, **ONNX Runtime (FP16)**, and **ByteTrack**, this system tracks multi-lane vehicles and pedestrians in real time to flag high-risk road anomalies.

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
python src/mobility_anomaly_detector.
Try without personalization

