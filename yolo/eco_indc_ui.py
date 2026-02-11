import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
from collections import defaultdict, deque

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.35

# ECOLOGICAL IMPACT SCALE (0–100 normalized)
# RESEARCH EMISSION DATASET (Derived from DEFRA/EPA/IPCC/OpenLCA)
# Units: normalized CO2 intensity

eco_scores = {
    "car": 90, "truck": 100, "bus": 85,
    "motorcycle": 60, "cow": 95,
    "cell phone": 40, "laptop": 45, "tv": 50,
    "keyboard": 25, "mouse": 20,
    "bottle": 25, "cup": 15,
    "chair": 10, "bench": 12,
    "person": 5, "dog": 3, "bicycle": 5,
    "__default__": 10
}

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

def eco_color(score):
    if score < 20: return (0,200,0)
    elif score < 50: return (0,220,220)
    elif score < 75: return (0,140,255)
    else: return (0,0,255)

def normalize(total):
    val = 100 * (1 - total / MAX_FRAME_IMPACT)
    return max(0, min(100, val))

# ---------- TRACKER ----------

class Tracker:
    def __init__(self):
        self.prev = {}

    def speed(self, key, cx, cy):
        now = time.time()
        if key in self.prev:
            px, py, pt = self.prev[key]
            dt = now - pt
            v = np.hypot(cx-px, cy-py) / dt if dt > 0 else 0
        else:
            v = 0
        self.prev[key] = (cx, cy, now)
        return v/30

# ---------- NEW FEATURES ----------

eco_history = deque(maxlen=120)
class_contrib = defaultdict(float)
session_total = 0
debug = False

def draw_gauge(img, value):
    center = (550, 100)
    radius = 60
    angle = int(180 * value / 100)

    cv2.ellipse(img, center, (radius, radius), 0, 180, 0, (60,60,60), 10)
    cv2.ellipse(img, center, (radius, radius), 0, 180, 180-angle, eco_color(100-value), 10)

    cv2.putText(img, f"{value:.1f}",
        (center[0]-30, center[1]+10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255,255,255), 2)

def draw_trend(img):
    if len(eco_history) < 2: return

    for i in range(1, len(eco_history)):
        x1 = 50 + (i-1)*2
        x2 = 50 + i*2
        y1 = 300 - int(eco_history[i-1])
        y2 = 300 - int(eco_history[i])
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

# ---------- MAIN ----------

model = YOLO(MODEL_NAME)
tracker = Tracker()
names = model.model.names
cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]

    total_impact = 0
    class_contrib.clear()

    if results.boxes is not None:
        for box in results.boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = names[int(box.cls[0])]

            cx, cy = (x1+x2)//2, (y1+y2)//2
            area = ((x2-x1)*(y2-y1))/(w*h)

            base = get_score(cls)
            s = size_factor(area)
            m = motion_factor(tracker.speed(cls,cx,cy))

            impact = base*s*m
            total_impact += impact
            class_contrib[cls] += impact

            color = eco_color(base)

            # Draw bounding box
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            # ---------- NEW: LABEL WITH OBJECT + SCORE ----------
            label = f"{cls} {impact:.1f}"

            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )

            # background box for text
            cv2.rectangle(
                frame,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                (0,0,0),
                -1
            )

            # draw text
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                2
            )

            if debug:
                cv2.putText(frame,
                    f"S:{s:.2f} M:{m:.2f}",
                    (x1,y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,(255,255,255),1)

    eco = normalize(total_impact)
    eco_history.append(eco)
    session_total += total_impact

    # ---------- UI ----------

    cv2.rectangle(frame,(10,10),(300,110),(30,30,30),-1)
    cv2.putText(frame,f"ECO INDEX: {eco:.1f}",
        (20,50),cv2.FONT_HERSHEY_SIMPLEX,1,
        eco_color(100-eco),3)

    draw_gauge(frame, eco)
    draw_trend(frame)

    # Top polluter
    if class_contrib:
        worst = max(class_contrib, key=class_contrib.get)
        cv2.putText(frame,f"Top impact: {worst}",
            (20,140),cv2.FONT_HERSHEY_SIMPLEX,
            0.6,(0,255,255),2)

    # FPS
    now = time.time()
    fps = 1/(now-prev_time)
    prev_time = now

    cv2.putText(frame,f"FPS: {fps:.1f}",
        (20,170),cv2.FONT_HERSHEY_SIMPLEX,
        0.5,(255,255,255),1)

    cv2.imshow("Ecological Indicator Pro",frame)

    key = cv2.waitKey(1)
    if key == 27: break
    if key == ord('d'): debug = not debug

cap.release()
cv2.destroyAllWindows()