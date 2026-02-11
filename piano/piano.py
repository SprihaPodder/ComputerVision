import cv2
import mediapipe as mp
import numpy as np
import os
import threading
from queue import Queue

# ---------------- SOUND ENGINE ----------------

sound_queue = Queue()

note_files = [
    "sounds/C.wav",
    "sounds/C#.wav",
    "sounds/D.wav",
    "sounds/D#.wav",
    "sounds/E.wav",
    "sounds/F.wav",
    "sounds/F#.wav",
    "sounds/G.wav",
    "sounds/G#.wav",
    "sounds/A.wav",
    "sounds/A#.wav",
    "sounds/B.wav",
    "sounds/C2.wav"
]

def sound_worker():
    while True:
        note = sound_queue.get()

        if 0 <= note < len(note_files):
            os.system(f"afplay -t 2 {note_files[note]} &")

        sound_queue.task_done()

threading.Thread(target=sound_worker, daemon=True).start()

def play_note(note_index):
    sound_queue.put(note_index)

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- PIANO SETUP ----------------

WIDTH = 1000
HEIGHT = 600

num_keys = 13
key_width = WIDTH // num_keys
key_height = 150

last_note_time = [0] * num_keys
cooldown = 0.2  # seconds

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    fingertips = []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape
            tip_ids = [4, 8, 12, 16, 20]  # thumb → pinky

            for tid in tip_ids:
                tip = hand.landmark[tid]
                x = int(tip.x * w)
                y = int(tip.y * h)

                fingertips.append((x, y))

                cv2.circle(frame, (x, y), 10, (0,255,0), 3)

    # ---------------- DRAW KEYS ----------------

    for i in range(num_keys):

        x1 = i * key_width
        x2 = x1 + key_width

        # alternate white/black style
        if i % 2 == 0:
            color = (255,255,255)
            text_color = (0,0,0)
        else:
            color = (50,50,50)
            text_color = (255,255,255)

        cv2.rectangle(
            frame,
            (x1, 0),
            (x2, key_height),
            color,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, 0),
            (x2, key_height),
            (0,0,0),
            2
        )

        note_name = note_files[i].split("/")[-1].replace(".wav","")

        cv2.putText(
            frame,
            note_name,
            (x1 + 10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        # ---------------- HIT DETECTION ----------------

        for fx, fy in fingertips:

            if x1 <= fx <= x2 and 0 <= fy <= key_height:

                now = cv2.getTickCount() / cv2.getTickFrequency()

                if now - last_note_time[i] > cooldown:

                    play_note(i)
                    last_note_time[i] = now

                    cv2.rectangle(
                        frame,
                        (x1, 0),
                        (x2, key_height),
                        (0,255,0),
                        4
                    )

    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()