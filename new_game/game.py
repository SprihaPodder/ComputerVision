import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import math
import threading
from queue import Queue

sound_queue = Queue()

# –––––––– SETTINGS ––––––––

GAME_WIDTH = 800
GAME_HEIGHT = 600
GAME_TIME = 30
CLICK_TIME = 0.25

MENU, PLAYING, GAMEOVER, LEADERBOARD = 0, 1, 2, 3

# –––––––– SOUND ––––––––

def play_pop():
    sound_queue.put("pop")

def sound_worker():
    while True:
        sound = sound_queue.get()

        if sound == "pop":
            os.system("afplay -v 1.5 pop.wav")

        sound_queue.task_done()

def play_bg_music():
    while True:
        os.system("afplay -v 1.0 bg.wav")

# –––––––– PARTICLES ––––––––

particles = []

def spawn_explosion(x, y):
    play_pop()
    for _ in range(20):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 6)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        particles.append([x, y, vx, vy, 20])

def update_particles(screen):
    global particles
    new_particles = []
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= 1

        if p[4] > 0:
            cv2.circle(screen, (int(p[0]), int(p[1])), 3, (0,255,255), -1)
            new_particles.append(p)

    particles = new_particles

# –––––––– LOAD SPRITES ––––––––

balloon = cv2.imread("balloon.png", cv2.IMREAD_UNCHANGED)
avatar = cv2.imread("avatar.png", cv2.IMREAD_UNCHANGED)

if balloon is not None:
    balloon = cv2.resize(balloon, (50,50))

if avatar is not None:
    avatar = cv2.resize(avatar, (70,70))

def overlay_png(bg, fg, x, y):
    if fg is None:
        return

    h, w = fg.shape[:2]

    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
        return

    alpha = fg[:,:,3] / 255.0
    alpha = alpha[:,:,None]

    bg[y:y+h, x:x+w] = (
        alpha * fg[:,:,:3]
        + (1-alpha) * bg[y:y+h, x:x+w]
    ).astype(np.uint8)

# –––––––– BACKGROUND ––––––––

landscape = cv2.imread("landscape.jpg")

if landscape is not None:
    landscape = cv2.resize(
        landscape,
        (GAME_WIDTH, GAME_HEIGHT)
    )

# –––––––– MEDIAPIPE ––––––––

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

player = input("Enter Player Name: ")

threading.Thread(target=play_bg_music, daemon=True).start()
# start background music in separate thread
threading.Thread(target=sound_worker, daemon=True).start()

state = MENU
score = 0
balloons = []
start_time = 0
hover_start = 0
hovering = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    fingertip = None

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            # draw full hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            h, w, _ = frame.shape
            tip = hand.landmark[8]  # index fingertip
            x = int(tip.x * w)
            y = int(tip.y * h)
            fingertip = (x, y)

            # highlight index finger
            cv2.circle(frame, fingertip, 12, (0,255,0), 3)

    if landscape is not None:
        game = landscape.copy()
    else:
        game = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        game[:] = (30,30,60)

    def hover_click(rect):
        global hovering, hover_start
        if fingertip is None:
            hovering = False
            return False

        fx = int(fingertip[0] * GAME_WIDTH / frame.shape[1])
        fy = int(fingertip[1] * GAME_HEIGHT / frame.shape[0])

        x,y,w,h = rect

        if x <= fx <= x+w and y <= fy <= y+h:
            if not hovering:
                hover_start = time.time()
                hovering = True

            cv2.rectangle(game,(x,y),(x+w,y+h),(0,255,255),3)

            return (time.time()-hover_start) > CLICK_TIME

        hovering = False
        return False

    # ===== MENU =====
    if state == MENU:

        cv2.putText(game,"BALLOON POP",(200,180),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4)

        start_btn = (300,280,200,80)
        x,y,w,h = start_btn
        cv2.rectangle(game,(x,y),(x+w,y+h),(0,200,0),-1)

        cv2.putText(game,"START",(330,335),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)

        if hover_click(start_btn):
            state = PLAYING
            score = 0
            balloons = []
            particles.clear()
            start_time = time.time()

    # ===== PLAYING =====
    elif state == PLAYING:

        elapsed = time.time() - start_time

        if elapsed > GAME_TIME:
            with open("scores.txt","a") as f:
                f.write(f"{player} {score}\n")
            state = GAMEOVER

        if random.randint(0,100) < 4:
            balloons.append([
                random.randint(40,GAME_WIDTH-40),
                GAME_HEIGHT,
                25,
                random.randint(2,5),
                True
            ])

        for b in balloons:
            if not b[4]:
                continue

            b[1] -= b[3]

            if balloon is not None:
                overlay_png(game, balloon, b[0]-25, b[1]-25)

            if fingertip:
                fx = int(fingertip[0] * GAME_WIDTH / frame.shape[1])
                fy = int(fingertip[1] * GAME_HEIGHT / frame.shape[0])

                if math.dist((fx,fy),(b[0],b[1])) < b[2]:
                    b[4] = False
                    score += 1
                    spawn_explosion(b[0],b[1])

        update_particles(game)

        if fingertip and avatar is not None:
            fx = int(fingertip[0] * GAME_WIDTH / frame.shape[1])
            fy = int(fingertip[1] * GAME_HEIGHT / frame.shape[0])
            overlay_png(game, avatar, fx-35, fy-35)

        cv2.putText(game,f"Score: {score}",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)

        cv2.putText(game,f"Time: {GAME_TIME-int(elapsed)}",(600,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)

    # ===== GAME OVER =====
    elif state == GAMEOVER:

        cv2.putText(game,"GAME OVER",(220,220),
                    cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

        cv2.putText(game,f"Score: {score}",(300,300),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3)

        lb_btn = (300,370,200,70)
        back_btn = (300,460,200,80)

        cv2.rectangle(game,(300,370),(500,440),(0,150,200),-1)
        cv2.putText(game,"LEADERBOARD",(305,415),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),3)

        cv2.rectangle(game,(300,460),(500,540),(0,200,0),-1)
        cv2.putText(game,"MENU",(350,515),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)

        if fingertip:
            fx = int(fingertip[0] * GAME_WIDTH / frame.shape[1])
            fy = int(fingertip[1] * GAME_HEIGHT / frame.shape[0])

            lx,ly,lw,lh = lb_btn
            bx,by,bw,bh = back_btn

            if lx <= fx <= lx+lw and ly <= fy <= ly+lh:
                if hover_click(lb_btn):
                    state = LEADERBOARD

            elif bx <= fx <= bx+bw and by <= fy <= by+bh:
                if hover_click(back_btn):
                    state = MENU

            else:
                hovering = False

    # ===== LEADERBOARD =====
    elif state == LEADERBOARD:

        cv2.putText(game,"LEADERBOARD",(200,120),
                    cv2.FONT_HERSHEY_SIMPLEX,1.8,(255,255,255),3)

        # read and sort leaderboard
        scores = []

        if os.path.exists("scores.txt"):
            with open("scores.txt") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) >= 2:
                        name = parts[0]
                        try:
                            sc = int(parts[1])
                            scores.append((name, sc))
                        except:
                            pass

        # sort high → low
        scores.sort(key=lambda x: x[1], reverse=True)

        # keep top 5
        top_scores = scores[:5]

        # display
        y = 200

        for i, (name, sc) in enumerate(top_scores):

            text = f"{i+1}. {name} : {sc}"

            cv2.putText(
                game,
                text,
                (240, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (200,200,200),
                2
            )

            y += 40

        back_btn = (300,480,200,80)

        cv2.rectangle(game,(300,480),(500,560),(0,200,0),-1)
        cv2.putText(game,"MENU",(350,535),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)

        if fingertip:
            fx = int(fingertip[0] * GAME_WIDTH / frame.shape[1])
            fy = int(fingertip[1] * GAME_HEIGHT / frame.shape[0])

            bx,by,bw,bh = back_btn

            if bx <= fx <= bx+bw and by <= fy <= by+bh:
                if hover_click(back_btn):
                    state = MENU
            else:
                hovering = False

    cv2.imshow("Webcam", frame)
    cv2.imshow("Game", game)

    if cv2.waitKey(16) == 27:
        break

cap.release()
cv2.destroyAllWindows()