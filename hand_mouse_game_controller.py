import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import webbrowser

# ---------------- OPEN GAME ----------------

GAME_URL = "https://poki.com"

webbrowser.open(GAME_URL)

time.sleep(5)  # give browser time to load

# ---------------- SCREEN ----------------

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
smooth = 0.2

# ---------------- CLICK ----------------

click_cd = 0
CLICK_DELAY = 0.4

# ---------------- DRAG ----------------

dragging = False

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (cam_w, cam_h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    index_tip = None
    thumb_tip = None
    middle_tip = None

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            def get_point(i):
                lm = hand.landmark[i]
                return int(lm.x * w), int(lm.y * h)

            index_tip = get_point(8)
            thumb_tip = get_point(4)
            middle_tip = get_point(12)

            cv2.circle(frame, index_tip, 10, (0,255,0), -1)

    # ---------------- CURSOR MOVE ----------------

    if index_tip:

        x = np.interp(index_tip[0], (0, cam_w), (0, screen_w))
        y = np.interp(index_tip[1], (0, cam_h), (0, screen_h))

        curr_x = prev_x + (x - prev_x) * smooth
        curr_y = prev_y + (y - prev_y) * smooth

        pyautogui.moveTo(curr_x, curr_y)

        prev_x, prev_y = curr_x, curr_y

    # ---------------- GESTURES ----------------

    now = time.time()

    if index_tip and thumb_tip:

        pinch = math.dist(index_tip, thumb_tip)

        # click
        if pinch < 30 and now > click_cd:
            pyautogui.click()
            click_cd = now + CLICK_DELAY
            cv2.putText(frame, "CLICK", (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0,0,255), 3)

    # drag gesture (index + middle pinch)

    if index_tip and middle_tip:

        pinch2 = math.dist(index_tip, middle_tip)

        if pinch2 < 30 and not dragging:
            pyautogui.mouseDown()
            dragging = True
            cv2.putText(frame, "DRAG", (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255,0,0), 3)

        elif pinch2 > 40 and dragging:
            pyautogui.mouseUp()
            dragging = False

    # ---------------- DISPLAY ----------------

    cv2.imshow("Gesture Game Controller", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()