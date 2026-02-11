import cv2
import numpy as np
import time
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

# ECOLOGICAL IMPACT SCALE (0–100 normalized)
# RESEARCH EMISSION DATASET (Derived from DEFRA/EPA/IPCC/OpenLCA)
# Units: normalized CO2 intensity

eco_scores = {

    # Very high impact
    "car": 90,
    "truck": 100,
    "bus": 85,
    "motorcycle": 60,
    "cow": 95,

    # Electronics
    "cell phone": 40,
    "laptop": 45,
    "tv": 50,
    "keyboard": 25,
    "mouse": 20,

    # Plastics
    "bottle": 25,
    "cup": 15,
    "chair": 10,
    "bench": 12,

    # Low impact
    "person": 5,
    "dog": 3,
    "bicycle": 5,

    "__default__": 10
}

# Size + motion scaling
MIN_SIZE_FACTOR = 0.6
MAX_SIZE_FACTOR = 2.0
MOTION_SAT = 30

MAX_FRAME_IMPACT = 500

def get_score(cls):
    return eco_scores.get(cls, eco_scores["__default__"])

def size_factor(area_ratio):
    r = min(area_ratio, 0.25)
    return MIN_SIZE_FACTOR + (MAX_SIZE_FACTOR - MIN_SIZE_FACTOR) * (r / 0.25)

def motion_factor(speed):
    s = min(speed, MOTION_SAT)
    return 1.0 + (s / MOTION_SAT)

# COLOR MAP
def eco_color(score):

    if score < 20:
        return (0, 200, 0)      # green
    elif score < 50:
        return (0, 220, 220)    # yellow
    elif score < 75:
        return (0, 140, 255)    # orange
    else:
        return (0, 0, 255)      # red

def normalize(total):
    val = 100 * (1 - total / MAX_FRAME_IMPACT)
    return max(0, min(100, val))


# Simple motion tracker to estimate speed for motion factor (not object-specific, just class-based)
class Tracker:

    def __init__(self):
        self.prev = {}

    def speed(self, key, cx, cy):

        now = time.time()
        if key in self.prev:
            px, py, pt = self.prev[key]
            dt = now - pt
            if dt > 0:
                v = np.hypot(cx - px, cy - py) / dt
            else:
                v = 0
        else:
            v = 0

        self.prev[key] = (cx, cy, now)
        return v / 30


model = YOLO(MODEL_NAME)
tracker = Tracker()

names = model.model.names if hasattr(model, "model") else model.names

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]

    total_impact = 0

    if results.boxes is not None:

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = names[int(box.cls[0])]
            conf = float(box.conf[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            area = ((x2 - x1) * (y2 - y1)) / (w * h)

            base = get_score(cls)
            s = size_factor(area)
            m = motion_factor(tracker.speed(cls, cx, cy))

            impact = base * s * m
            total_impact += impact

            color = eco_color(base)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame,
                        f"{cls} {impact:.1f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

    eco_index = normalize(total_impact)

    cv2.rectangle(frame, (10, 10), (320, 90), (30, 30, 30), -1)

    cv2.putText(frame,
                f"ECO INDEX: {eco_index:.1f}/100",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if eco_index > 50 else (0, 0, 255),
                3)

    cv2.imshow("Ecological Indicator", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()