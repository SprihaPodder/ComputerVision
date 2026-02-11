import cv2
import mediapipe as mp
import numpy as np
import glob
import os
import math
import time

# ---------------- LOAD DATASET ----------------

dataset_path = "dataset"
image_files = sorted(
    glob.glob(os.path.join(dataset_path, "*.[jJ][pP][gG]")) +
    glob.glob(os.path.join(dataset_path, "*.[jJ][pP][eE][gG]")) +
    glob.glob(os.path.join(dataset_path, "*.[pP][nN][gG]"))
)

if len(image_files) == 0:
    print("No images found in dataset folder!")
    exit()

index = len(image_files) // 2
zoom = 1.0

# ---------------- UI LAYOUT ----------------

CAM_W = 640
CAM_H = 480

VIEW_W = 640
VIEW_H = 480

THUMB_W = 120

WINDOW_W = CAM_W + VIEW_W + THUMB_W
WINDOW_H = max(CAM_H, VIEW_H)

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- GESTURE STATE ----------------

prev_wrist_x = None
prev_pinch = None
cooldown = 0

SWIPE_THRESHOLD = 40


# ---------------- THUMBNAILS ----------------

thumbs = []

for path in image_files:
    img = cv2.imread(path)
    if img is None:
        thumbs.append(None)
        continue

    thumb = cv2.resize(img, (THUMB_W-10, 80))
    thumbs.append(thumb)

# ---------------- MAIN LOOP ----------------

while True:

    ret, cam = cap.read()
    if not ret:
        break

    cam = cv2.flip(cam, 1)
    cam = cv2.resize(cam, (CAM_W, CAM_H))

    fingertip_data = None

    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                cam,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = cam.shape

            wrist = hand.landmark[0]
            thumb = hand.landmark[4]
            index_tip = hand.landmark[8]

            wx = int(wrist.x * w)
            tx = int(thumb.x * w)
            ty = int(thumb.y * h)
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            fingertip_data = (wx, tx, ty, ix, iy)

    # ---------------- GESTURES ----------------

    now = time.time()

    if fingertip_data and now > cooldown:

        wx, tx, ty, ix, iy = fingertip_data

        # swipe
        if prev_wrist_x is not None:
            dx = wx - prev_wrist_x

            if dx > SWIPE_THRESHOLD:
                index = max(0, index - 1)
                cooldown = now + 0.4

            elif dx < -SWIPE_THRESHOLD:
                index = min(len(image_files)-1, index + 1)
                cooldown = now + 0.4

        prev_wrist_x = wx

        # zoom
        pinch = math.hypot(tx-ix, ty-iy)

        if prev_pinch is not None:
            dz = pinch - prev_pinch
            zoom += dz * 0.005
            zoom = max(0.5, min(3.0, zoom))

        prev_pinch = pinch

    # ---------------- VIEW IMAGE ----------------

    img = cv2.imread(image_files[index])

    if img is None:
        continue

    h, w = img.shape[:2]

    new_w = int(w * zoom)
    new_h = int(h * zoom)

    img = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

    ih, iw = img.shape[:2]

    x = max(0, (iw - VIEW_W)//2)
    y = max(0, (ih - VIEW_H)//2)

    crop = img[y:y+VIEW_H, x:x+VIEW_W]

    ch, cw = crop.shape[:2]
    canvas[:ch, :cw] = crop

    # ---------------- THUMBNAIL STRIP ----------------

    thumb_panel = np.zeros((WINDOW_H, THUMB_W, 3), dtype=np.uint8)

    y = 10

    for i, thumb in enumerate(thumbs):

        if thumb is None:
            continue

        th, tw = thumb.shape[:2]

        thumb_panel[y:y+th, 5:5+tw] = thumb

        if i == index:
            cv2.rectangle(
                thumb_panel,
                (5, y),
                (5+tw, y+th),
                (0,255,0),
                3
            )

        y += th + 10

        if y > WINDOW_H - th:
            break

    # ---------------- COMBINE UI ----------------

    combined = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

    combined[:CAM_H, :CAM_W] = cam
    combined[:VIEW_H, CAM_W:CAM_W+VIEW_W] = canvas
    combined[:, CAM_W+VIEW_W:] = thumb_panel

    cv2.putText(
        combined,
        f"{index+1}/{len(image_files)}  Zoom:{zoom:.2f}",
        (CAM_W+20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Gesture Dataset Viewer", combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()