
import argparse
import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import websockets

try:
    import joblib
except Exception:
    joblib = None


# -------------------- WS + modèle --------------------
DEFAULT_WS_URI = "ws://127.0.0.1:8765"
DEFAULT_MODEL_PATH = Path("gesture_model.pkl")

LABELS = ["LEFT", "RIGHT", "FIRE", "ENTER", "IDLE"]
CONFIDENCE_THRESHOLD = 0.65

# Mouvement: le serveur Node relâche les touches au bout de ~100ms,
# donc on "répète" LEFT/RIGHT toutes les 120ms pour simuler un maintien.
MOVE_REPEAT_S = 0.12

# Cooldowns events
FIRE_COOLDOWN_S = 0.28
ENTER_COOLDOWN_S = 0.80

# ENTER triggers 
TWO_HANDS_HOLD_S = 0.60
FIST_HOLD_S = 0.90

# Smoothing/hystérésis MOVE (wrist x)
X_SMOOTH_N = 7
LEFT_ON, LEFT_OFF = 0.40, 0.46
RIGHT_ON, RIGHT_OFF = 0.60, 0.54

# Pinch (ratio) + hystérésis
PINCH_ON, PINCH_OFF = 0.35, 0.45


# -------------------- MediaPipe landmarks --------------------
WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [1, 5, 9, 13, 17]


# -------------------- Utils --------------------
def l2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a, np.float32) - np.array(b, np.float32)))


def get_handedness_label(results, hand_index: int) -> Optional[str]:
    """Return 'Left' or 'Right' if available."""
    if not results.multi_handedness:
        return None
    try:
        return results.multi_handedness[hand_index].classification[0].label
    except Exception:
        return None


def count_raised_fingers(hand_landmarks, handedness: Optional[str], mirror: bool) -> int:
    """
    Compte doigts levés (heuristique) + correction main gauche/droite.
    (Assez robuste pour feature + fallback.)
    """
    lm = hand_landmarks.landmark

    # si on mirror la frame, on inverse la logique handedness pour le pouce
    hand = handedness
    if hand in ("Left", "Right") and mirror:
        hand = "Left" if hand == "Right" else "Right" 

    # pouce : dépend main
    if hand == "Right":
        thumb_up = 1 if lm[THUMB_TIP].x < lm[THUMB_IP].x else 0
    elif hand == "Left":
        thumb_up = 1 if lm[THUMB_TIP].x > lm[THUMB_IP].x else 0
    else:
        thumb_up = 1 if lm[THUMB_TIP].x < lm[THUMB_IP].x else 0

    count = thumb_up

    # index..auriculaire : tip plus haut que MCP => levé
    for tip, mcp in zip(FINGERTIPS[1:], MCP_JOINTS[1:]):
        if lm[tip].y < lm[mcp].y:
            count += 1

    return count


def extract_features(hand_landmarks, handedness: Optional[str], mirror: bool) -> np.ndarray:
    """
    Features simples & efficaces (M2):
      - 21 landmarks (x,y) relatifs au poignet (42)
      - distances thumb->fingertips (pinch-like) (5)
      - distances wrist->fingertips (openness) (5)
      - wrist absolu (x,y) (2)
      - pinch_ratio (1)
      - openness_sum (1)
      - fingers_up (1)
    Total = 58 features

    IMPORTANT: ton train_model.py doit utiliser EXACTEMENT la même fonction.
    """
    lm = hand_landmarks.landmark
    wx, wy = lm[WRIST].x, lm[WRIST].y

    coords = []
    for i in range(21):
        coords.extend([lm[i].x - wx, lm[i].y - wy])

    thumb = (lm[THUMB_TIP].x, lm[THUMB_TIP].y)

    pinch = [l2((lm[t].x, lm[t].y), thumb) for t in FINGERTIPS]  # 5
    openness = [l2((lm[t].x, lm[t].y), (wx, wy)) for t in FINGERTIPS]  # 5
    openness_sum = float(np.sum(openness))

    scale = l2((lm[WRIST].x, lm[WRIST].y), (lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y)) + 1e-6
    pinch_ratio = l2(
        (lm[THUMB_TIP].x, lm[THUMB_TIP].y),
        (lm[INDEX_TIP].x, lm[INDEX_TIP].y),
    ) / scale

    fingers_up = float(count_raised_fingers(hand_landmarks, handedness, mirror))

    feat = np.array(coords + pinch + openness + [wx, wy, pinch_ratio, openness_sum, fingers_up], dtype=np.float32)
    return feat.reshape(1, -1)


def heuristic_predict(hand_landmarks, handedness: Optional[str], mirror: bool) -> str:
    """
    Fallback sans modèle :
      - doigts>=4 => FIRE
      - doigts<=1 => ENTER (mais ENTER est re-filtré par hold+cooldown ensuite)
      - sinon LEFT/RIGHT via wrist-x
    """
    lm = hand_landmarks.landmark
    wx = lm[WRIST].x
    nf = count_raised_fingers(hand_landmarks, handedness, mirror)

    if nf >= 4:
        return "FIRE"
    if nf <= 1:
        return "ENTER"
    if wx < 0.35:
        return "LEFT"
    if wx > 0.65:
        return "RIGHT"
    return "IDLE"


# -------------------- Controllers --------------------
class MoveController:
    """Wrist-x smoothing + hystérésis -> LEFT/RIGHT/None (fallback move)."""
    def __init__(self):
        self.x_hist = deque(maxlen=X_SMOOTH_N)
        self.state: Optional[str] = None

    def reset(self):
        self.x_hist.clear()
        self.state = None

    def update(self, wrist_x: float) -> Optional[str]:
        self.x_hist.append(wrist_x)
        x = float(np.mean(self.x_hist))

        if self.state == "LEFT":
            if x > LEFT_OFF:
                self.state = None
            return self.state

        if self.state == "RIGHT":
            if x < RIGHT_OFF:
                self.state = None
            return self.state

        if x < LEFT_ON:
            self.state = "LEFT"
        elif x > RIGHT_ON:
            self.state = "RIGHT"
        else:
            self.state = None
        return self.state


class EventController:
    """FIRE/ENTER as events (edge/hold) with cooldowns."""
    def __init__(self):
        self.pinch_active = False
        self.last_fire_t = 0.0
        self.last_enter_t = 0.0

        self.two_hands_start: Optional[float] = None
        self.fist_start: Optional[float] = None

        self.last_pinch_ratio: Optional[float] = None

    def reset_when_no_hand(self):
        self.pinch_active = False
        self.two_hands_start = None
        self.fist_start = None
        self.last_pinch_ratio = None

    def _pinch_ratio(self, hand_landmarks) -> float:
        lm = hand_landmarks.landmark
        pinch_dist = l2((lm[THUMB_TIP].x, lm[THUMB_TIP].y),
                        (lm[INDEX_TIP].x, lm[INDEX_TIP].y))
        scale = l2((lm[WRIST].x, lm[WRIST].y),
                   (lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y)) + 1e-6
        return pinch_dist / scale

    def update(self,
               hands_count: int,
               primary_hand,
               fingers_up: int,
               ml_label: Optional[str],
               now: float) -> Tuple[bool, bool]:
        fire_evt = False
        enter_evt = False

        # ---- ENTER: 2 hands hold (plus robuste)
        if hands_count >= 2:
            if self.two_hands_start is None:
                self.two_hands_start = now
            elif (now - self.two_hands_start) >= TWO_HANDS_HOLD_S:
                if (now - self.last_enter_t) >= ENTER_COOLDOWN_S:
                    enter_evt = True
                    self.last_enter_t = now
                self.two_hands_start = None
        else:
            self.two_hands_start = None

        # ---- ENTER: ML intent
        if not enter_evt and ml_label == "ENTER":
            if (now - self.last_enter_t) >= ENTER_COOLDOWN_S:
                enter_evt = True
                self.last_enter_t = now

        # ---- ENTER: fist hold fallback
        if not enter_evt:
            if fingers_up <= 1:
                if self.fist_start is None:
                    self.fist_start = now
                elif (now - self.fist_start) >= FIST_HOLD_S:
                    if (now - self.last_enter_t) >= ENTER_COOLDOWN_S:
                        enter_evt = True
                        self.last_enter_t = now
                    self.fist_start = None
            else:
                self.fist_start = None

        # ---- FIRE: pinch edge (best)
        if primary_hand is not None:
            ratio = self._pinch_ratio(primary_hand)
            self.last_pinch_ratio = ratio

            if self.pinch_active:
                if ratio > PINCH_OFF:
                    self.pinch_active = False
            else:
                if ratio < PINCH_ON:
                    self.pinch_active = True
                    if (now - self.last_fire_t) >= FIRE_COOLDOWN_S:
                        fire_evt = True
                        self.last_fire_t = now

        # ---- FIRE: ML intent
        if not fire_evt and ml_label == "FIRE":
            if (now - self.last_fire_t) >= FIRE_COOLDOWN_S:
                fire_evt = True
                self.last_fire_t = now

        # ---- FIRE: heuristic fallback (doigts)
        if not fire_evt and fingers_up >= 4:
            if (now - self.last_fire_t) >= FIRE_COOLDOWN_S:
                fire_evt = True
                self.last_fire_t = now

        return fire_evt, enter_evt


class FPSTracker:
    def __init__(self, maxlen: int = 30):
        self.ts = deque(maxlen=maxlen)

    def tick(self):
        self.ts.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self.ts) < 2:
            return 0.0
        return (len(self.ts) - 1) / (self.ts[-1] - self.ts[0] + 1e-9)


# -------------------- Overlay --------------------
COL_WHITE = (240, 240, 240)
COL_GREY = (160, 160, 160)
COL_GREEN = (0, 210, 90)
COL_RED = (0, 60, 220)

def draw_overlay(frame, mode: str, move_label: str,
                 ml_label: Optional[str], ml_conf: Optional[float],
                 fingers: int, pinch_ratio: Optional[float],
                 fps: float, show_fps: bool, ws_ok: bool):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 95), (20, 20, 20), -1)

    cv2.putText(frame, f"Mode: {mode}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Move: {move_label}", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_WHITE, 2, cv2.LINE_AA)

    if ml_label is not None:
        txt = f"ML: {ml_label}"
        if ml_conf is not None:
            txt += f" ({ml_conf*100:.0f}%)"
        cv2.putText(frame, txt, (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ML: OFF", (12, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_GREY, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Fingers: {fingers}", (w - 190, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2, cv2.LINE_AA)
    if pinch_ratio is not None:
        cv2.putText(frame, f"Pinch: {pinch_ratio:.2f}", (w - 190, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2, cv2.LINE_AA)

    ws_col = COL_GREEN if ws_ok else COL_RED
    cv2.putText(frame, "WS OK" if ws_ok else "WS OFF", (w - 95, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ws_col, 2, cv2.LINE_AA)

    if show_fps:
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 190, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_WHITE, 2, cv2.LINE_AA)


# -------------------- Model load --------------------
def load_model(model_path: Path):
    if joblib is None:
        raise RuntimeError("joblib not installed. pip install joblib scikit-learn")

    payload = joblib.load(model_path)

    if isinstance(payload, dict) and "pipeline" in payload:
        pipe = payload["pipeline"]
        classes = payload.get("classes", None)
        if classes is None and "label_encoder" in payload:
            try:
                classes = list(payload["label_encoder"].classes_)
            except Exception:
                classes = None
        return pipe, classes

    # payload = pipeline direct
    pipe = payload
    classes = getattr(pipe, "classes_", None)
    if classes is not None:
        classes = list(classes)
    return pipe, classes


# -------------------- Async runner --------------------
async def run(args):
    mirror = not args.no_mirror

    # Mode + load model if possible
    model = None
    model_classes = None
    use_model = not args.heuristic

    if use_model:
        if not args.model.exists():
            print(f"[cv] Model not found: {args.model} -> fallback heuristic.")
            use_model = False
        else:
            try:
                model, model_classes = load_model(args.model)
                # must expose predict_proba for confidence
                if not hasattr(model, "predict_proba"):
                    print("[cv] Model has no predict_proba -> fallback heuristic.")
                    use_model = False
                    model = None
            except Exception as e:
                print(f"[cv] Model load failed ({e}) -> fallback heuristic.")
                use_model = False
                model = None

    mode_label = "ML" if use_model else "heuristic"
    print(f"[cv] Mode: {mode_label}")
    print("[cv] Press 'q' to quit.")

    # MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Controllers
    move_ctrl = MoveController()
    evt_ctrl = EventController()
    fps = FPSTracker()

    # WebSocket with reconnect
    ws = None
    ws_ok = False

    async def try_connect():
        nonlocal ws, ws_ok
        try:
            ws = await websockets.connect(args.ws, ping_interval=None, ping_timeout=None)
            ws_ok = True
            print(f"[cv] WS connected: {args.ws}")
        except Exception as e:
            ws = None
            ws_ok = False
            print(f"[cv] WS not available ({e}) -> display-only")

    await try_connect()

    async def send(cmd: str):
        nonlocal ws, ws_ok
        if not ws_ok or ws is None:
            return
        try:
            await ws.send(cmd)
        except Exception:
            ws_ok = False
            try:
                await ws.close()
            except Exception:
                pass
            ws = None
            print("[cv] WS disconnected -> reconnecting…")
            await try_connect()

    last_move_send_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        now = time.monotonic()

        move_label = "IDLE"
        ml_label = None
        ml_conf = None
        fingers = 0

        if not results.multi_hand_landmarks:
            move_ctrl.reset()
            evt_ctrl.reset_when_no_hand()
        else:
            # draw hands
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            hands_count = len(results.multi_hand_landmarks)
            hl0 = results.multi_hand_landmarks[0]
            handed0 = get_handedness_label(results, 0)
            fingers = count_raised_fingers(hl0, handed0, mirror)

            # ---- Model prediction (intent)
            intent_label = "IDLE"
            if use_model and model is not None:
                feat = extract_features(hl0, handed0, mirror)
                proba = model.predict_proba(feat)[0]
                best_i = int(np.argmax(proba))
                ml_conf = float(proba[best_i])

                classes = getattr(model, "classes_", None)
                if classes is None and model_classes is not None:
                    classes = model_classes
                if classes is not None:
                    raw = str(classes[best_i])
                else:
                    raw = str(best_i)

                ml_label = raw
                if ml_conf >= CONFIDENCE_THRESHOLD and raw in LABELS:
                    intent_label = raw
                else:
                    intent_label = "IDLE"
            else:
                intent_label = heuristic_predict(hl0, handed0, mirror)

            if intent_label in ("LEFT", "RIGHT"):
                move_label = intent_label
            else:
                mv = move_ctrl.update(hl0.landmark[WRIST].x)
                move_label = mv if mv in ("LEFT", "RIGHT") else "IDLE"

            fire_evt, enter_evt = evt_ctrl.update(
                hands_count=hands_count,
                primary_hand=hl0,
                fingers_up=fingers,
                ml_label=intent_label if intent_label in ("FIRE", "ENTER") else None,
                now=now,
            )

            if move_label in ("LEFT", "RIGHT") and (now - last_move_send_t) >= MOVE_REPEAT_S:
                await send(move_label)
                last_move_send_t = now

        
            if fire_evt:
                await send("FIRE")
            if enter_evt:
                await send("ENTER")

        fps.tick()
        draw_overlay(
            frame=frame,
            mode=("ML" if use_model else "heuristic"),
            move_label=move_label,
            ml_label=(ml_label if use_model else None),
            ml_conf=(ml_conf if use_model else None),
            fingers=fingers,
            pinch_ratio=evt_ctrl.last_pinch_ratio,
            fps=fps.fps,
            show_fps=args.show_fps,
            ws_ok=ws_ok,
        )

        cv2.imshow("Space Invaders — CV Control", frame)

        await asyncio.sleep(0.001)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    if ws is not None:
        try:
            await ws.close()
        except Exception:
            pass
    print("[cv] Stopped.")


def main():
    p = argparse.ArgumentParser(description="Space Invaders CV control module (ML + heuristics)")
    p.add_argument("--ws", default=DEFAULT_WS_URI, help=f"WebSocket URI (default {DEFAULT_WS_URI})")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Joblib model path")
    p.add_argument("--heuristic", action="store_true", help="Force heuristic mode (ignore ML model)")
    p.add_argument("--no-mirror", action="store_true", help="Disable mirroring")
    p.add_argument("--show-fps", action="store_true", help="Show FPS overlay")
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
