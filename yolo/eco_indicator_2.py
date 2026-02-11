import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

# -----------------------------
# RESEARCH EMISSION DATASET
# (Derived from DEFRA/EPA/IPCC/OpenLCA)
# Units: normalized CO2 intensity
# -----------------------------

RAW_EMISSIONS = {

    # Very high impact
    "car": 90,
    "truck": 100,
    "bus": 85,
    "motorcycle": 60,
    "cow": 95,

    # Electronics (mapped to closest YOLO classes)
    "cell phone": 45,
    "laptop": 50,
    "tv": 55,
    "keyboard": 25,
    "mouse": 20,

    # Plastics
    "bottle": 25,
    "cup": 15,
    "chair": 25,
    "bench": 30,

    # Low impact
    "person": 5,
    "dog": 3,
    "bicycle": 5,

    "__default__": 10
}

def normalize_emissions(raw):

    normalized = {}
    for k,v in raw.items():
        normalized[k] = math.log(v + 1)

    return normalized

EMISSIONS = normalize_emissions(RAW_EMISSIONS)

# -----------------------------
# HYBRID MODEL PARAMETERS
# -----------------------------

ALPHA = 1.2
BETA = 0.8
MAX_IMPACT = 25

def emission(cls):
    return EMISSIONS.get(cls, EMISSIONS["__default__"])

# -----------------------------
# TRACKER
# -----------------------------

class Tracker:

    def __init__(self):
        self.prev = {}

    def speed(self, key, cx, cy):

        now = time.time()

        if key in self.prev:
            px, py, pt = self.prev[key]
            dt = now - pt
            v = np.hypot(cx - px, cy - py) / max(dt, 1e-6)
        else:
            v = 0

        self.prev[key] = (cx, cy, now)
        return v

tracker = Tracker()

# -----------------------------
# ECO INDEX
# -----------------------------

def eco_index(total):
    val = 100 * (1 - total / MAX_IMPACT)
    return max(0, min(100, val))

# -----------------------------
# COLOR MAP
# -----------------------------

def eco_color(score):

    if score < 20:
        return (0, 200, 0)      # green
    elif score < 50:
        return (0, 220, 220)    # yellow
    elif score < 75:
        return (0, 140, 255)    # orange
    else:
        return (0, 0, 255)      # red

# def eco_color(score):

#     score = max(0, min(100, score))

#     r = int(255 * score / 100)
#     g = int(255 * (100 - score) / 100)

#     return (0, g, r)

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------

def sensitivity_report(total):

    print("Impact:", round(total,3))
    print("Eco Index:", round(eco_index(total),2))

# -----------------------------
# YOLO INIT
# -----------------------------

model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h,w = frame.shape[:2]

    results = model(frame)[0]

    total = 0

    if results.boxes is not None:

        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = names[int(box.cls[0])]

            cx = (x1+x2)//2
            cy = (y1+y2)//2

            area = ((x2-x1)*(y2-y1))/(w*h)
            speed = tracker.speed(cls, cx, cy)

            S = min(area*4,1)
            M = min(speed/600,1)

            impact = emission(cls)*(S**ALPHA)*(M**BETA)

            total += impact

            color = eco_color(impact*20)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            cv2.putText(frame,
                        f"{cls}:{impact:.2f}",
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

    eco = eco_index(total)

    cv2.rectangle(frame,(10,10),(350,90),(30,30,30),-1)

    cv2.putText(frame,
                f"ECO INDEX: {eco:.1f}",
                (20,55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                eco_color(100-eco),
                3)

    cv2.imshow("Research Eco Indicator",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()