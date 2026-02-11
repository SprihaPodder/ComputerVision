import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# ---------------- SCREEN SIZE ----------------

screen_w, screen_h = pyautogui.size()

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

cam_w, cam_h = 640, 480

# ---------------- SMOOTHING ----------------

prev_x, prev_y = 0, 0
smoothening = 0.2

# ---------------- CLICK CONTROL ----------------

click_cooldown = 0
CLICK_DELAY = 0.5

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (cam_w, cam_h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    fingertip = None
    pinch_dist = None

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            # Index fingertip (cursor)
            ix = int(hand.landmark[8].x * w)
            iy = int(hand.landmark[8].y * h)

            # Thumb tip (for click)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)

            fingertip = (ix, iy)

            # draw cursor point
            cv2.circle(frame, fingertip, 10, (0,255,0), -1)

            # pinch distance
            pinch_dist = math.hypot(ix - tx, iy - ty)

    # ---------------- MOVE CURSOR ----------------

    if fingertip:

        # map to screen
        x = np.interp(fingertip[0], (0, cam_w), (0, screen_w))
        y = np.interp(fingertip[1], (0, cam_h), (0, screen_h))

        # smoothing
        curr_x = prev_x + (x - prev_x) * smoothening
        curr_y = prev_y + (y - prev_y) * smoothening

        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

    # ---------------- CLICK GESTURE ----------------

    now = time.time()

    if pinch_dist and pinch_dist < 30 and now > click_cooldown:

        pyautogui.click()
        click_cooldown = now + CLICK_DELAY

        cv2.putText(
            frame,
            "CLICK",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0,0,255),
            3
        )

    # ---------------- DISPLAY ----------------

    cv2.imshow("Hand Mouse Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()