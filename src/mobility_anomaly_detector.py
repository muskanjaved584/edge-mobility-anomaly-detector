import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}

# Road zone polygon calibrated for active intersection lanes (normalized 0.0 to 1.0)
ROAD_ZONE_NORM = np.array(
    list((
        tuple((0.05, 0.22)),
        tuple((0.55, 0.22)),
        tuple((0.75, 0.55)),
        tuple((0.02, 0.55)),
    )),
    dtype=np.float32
)


def run_anomaly_detector(video_path, model_path="best.onnx", output_path="ajou_anomaly_demo.mp4"):
    model = YOLO(model_path)
    track_history = defaultdict(lambda: deque(maxlen=30))
    speed_history = defaultdict(lambda: deque(maxlen=15))

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    road_polygon = (ROAD_ZONE_NORM * np.array(tuple((width, height)))).astype(np.int32)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        overlay = frame.copy()
        cv2.fillPoly(overlay, list((road_polygon,)), (255, 200, 0))
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.polylines(frame, list((road_polygon,)), True, (255, 200, 0), 2)

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        res0 = results.__getitem__(0)
        active_alerts = list()

        if res0.boxes is not None and res0.boxes.id is not None:
            boxes = res0.boxes.xyxy.cpu().numpy()
            track_ids = res0.boxes.id.int().cpu().tolist()
            class_ids = res0.boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                anchor_x = int((x1 + x2) / 2)
                anchor_y = int(y2)
                track_history[track_id].append((anchor_x, anchor_y))

                color = (0, 255, 0)
                label = f"ID:{track_id}"
                in_road = cv2.pointPolygonTest(road_polygon, (float(anchor_x), float(anchor_y)), False) >= 0

                if cls_id == PERSON_CLASS and in_road:
                    color = (0, 0, 255)
                    label = f"ALERT: JAYWALKING (ID:{track_id})"
                    active_alerts.append(f"Frame {frame_idx}: Jaywalking ID {track_id}")

                elif cls_id in VEHICLE_CLASSES:
                    pts = list(track_history[track_id])
                    if len(pts) >= 2:
                        p_curr = np.array(pts.__getitem__(-1))
                        p_prev = np.array(pts.__getitem__(-2))
                        speed_history[track_id].append(np.linalg.norm(p_curr - p_prev))
                    speeds = list(speed_history[track_id])
                    if len(speeds) >= 10:
                        past_avg = np.mean(speeds[:-3])
                        curr_avg = np.mean(speeds[-3:])
                        if in_road and past_avg > 12.0 and curr_avg < 3.0 and (curr_avg / past_avg) < 0.25:
                            color = (0, 140, 255)
                            label = f"ALERT: SUDDEN STOP (ID:{track_id})"
                            active_alerts.append(f"Frame {frame_idx}: Sudden Stop ID {track_id}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.circle(frame, (anchor_x, anchor_y), 6, color, -1)
                cv2.putText(frame, label, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        if active_alerts:
            cv2.rectangle(frame, (20, 20), (900, 80), (0, 0, 200), -1)
            cv2.putText(frame, active_alerts.__getitem__(0), (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")


if __name__ == "__main__":
    run_anomaly_detector("traffic_sample.mp4")

