"""
Cámara detectora de gestos, versión Bjork.

Imspirado en Meowmeow cat cam meme detector de catherpiee

Repositorio: https://github.com/catherpiee/meowmeowcatcam

Dos ventanas, 'Camarita' un feed webcam con los landmarks (puntos de referencia) de las manos y 
'BJORK' que muestra la imagen que calza con el gesto que 'Camarita' identifique. 


Gestos:
  default (sin gesto)                              -> bjorkReact/bjork.jpg
  Shhh   (dedo indice sobre la boca + ojos entre abiertos)        -> bjorkReact/bjorkShhh*.jpg
  middleFinger (dedo del medio levantado)                    -> bjorkReact/bjorkMiddleFinger.jpg
  sixSeven (manos alternando de arriba a abajo) -> bjorkReact/bjork67.jpg
  debut  (manos juntas como rezando por sobre la boca)            -> bjorkReact/bjorkDebut*.jpg
  sus    (ceja izquierda levantada + cabeza girada a la derecha) -> bjorkReact/bjorkSuspicious.jpg
  kitty  (manos cerradas (fists) debajo del menton) -> bjorkReact/bjorkCAT*.jpg
  huh    (manos abiertas al lado del torso) -> bjorkReact/bjorkGrapes.jpeg


La ventana 'Camarita' muestra informacion de 'debug' en tiempo real (HUDs de apertura de los ojos,
asimetria de las cejas, rotacion de la cabeza, numero de movimientos de las manos) en la esquina 
superior izquierda, sirve para identificar cada umbral visualmente y luego modificarlo si es necesario.
 
Presiona la tecla q o ESC para salir


"""


import math
import random
import time
from pathlib import Path


import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat


ROOT = Path(__file__).parent
MODELS = ROOT / "models"
BJORKS = ROOT / "bjorkReact"


GESTURE_BJORKS = {
  "default": ["bjork.jpg"],
  "Shhh": ["bjorkShhh.jpg", "bjorkShhh2.jpg"],
  "middleFinger": ["bjorkMiddleFinger.jpg"],
  "sixSeven": ["bjork67.jpg"],
  "debut": ["bjorkDebutRacoon.jpg", "bjorkDebutHealing.jpg"],
  "sus": ["bjorkSuspicious.jpg"],
  "kitty": ["bjorkCAT.jpg", "bjorkCAT2.jpg"],
  "huh": ["bjorkGrapes.jpeg"],
}


STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200


# Shhh: ojos semi abiertos


# eye "openness" = espacio vertical parpados / ancho cara.  "eyeOpen" HUD
# entrecerrar los ojos para gatillarlo 


EYES_SEMI_CLOSED_MAX = 0.04


# sus: ceja izquierda levantada + cabeza girada a la derecha 
# que tan alta esta la ceja izquierda comparada a la derecha
# rangos normalizamos por el ancho de cara. Levanta una de las cejas para gatillarlo "browDiff" HUD.


EYEBROW_RAISE_DIFF = 0.01
# head-turn (giro, grados) rango = "girada hacia la derecha" -
# gira lo suficiente para detectarlo pero no tanto como estar de perfil


# Si quieres cambiar el lado al que se gira intercambia los signos de los limites de abajo 


# (p.e. -28 < giroGrados < -6).
SUS_TURN_MIN_DEG = 6
SUS_TURN_MAX_DEG = 28


# kitty: manos cerradas (fist) debajo de la mandibula, como patitas de gato


CAT_PAWS_TOGETHER_DIST = 1.8
CAT_PAWS_NEAR_FACE_DIST = 1.6


# debut: manos juntas como rezando por encima de la boca 


PRAYER_HANDS_TOGETHER_DIST = 1.6
PRAYER_NEAR_MOUTH_DIST = 2.0


# huh: manos abiertas al lado del torso
HUH_SPREAD_MIN_DIST = 2.2
HUH_BELOW_FACE_MIN = 1.0
HUH_HEIGHT_MATCH_MAX = 1.0


# sixSeven: manos alternado de arriba a abajo


SIXSEVEN_MIN_SWING = 0.35
SIXSEVEN_WINDOW_MS = 1500
SIXSEVEN_MIN_FLIPS = 2


# parametros MediaPipe 


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]




# ayudas de geometria  


def p3(lm):
    return np.array([lm.x, lm.y, lm.z])




def dist(a, b):
    return float(np.linalg.norm(a - b))




def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))




def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45




def yaw_from_transform_matrix(matrix):
    """Angulo giro a la derecha o a la izquierda (giro, grados) de la matrix de transformacion de MediaPipe"""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)




def roll_from_transform_matrix(matrix):
    """Cabeza ladeada (ladeo, grados - como acercando la oreja al hombro, la misma matrix que la del giro"""
    r = np.asarray(matrix)[:3, :3]
    roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(roll)




def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6


    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)


    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05


    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)


    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "wrist": pts[0],
        "palmCenter": pts[9],
    }




# pose de mano exclusiva - reglas
# cada regla por cada mano y la lectura del rostro (si es que se ve) 
# devuelve True (verdad) si se esta haciendo el gesto 
# se ve en orden en decide() primera coincidencia 'gana'
def _rule_middle_finger(h, face):
    return h["middleUp"] and not h["indexUp"] and not h["ringUp"] and not h["pinkyUp"]

def _rule_shhh(h, face):
    # dedo indice por encima de la boca (como en la Meowmeow
    #  cat cam meme detector de catherpiee) + los ojos entre abiertos
    if not (h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]):
        return False
    if face is None:
        return False
    d = dist(h["indexTip"], face["mouthCenter"]) / face["faceWidth"]
    return d < 0.55 and face["eyeOpenAvg"] < EYES_SEMI_CLOSED_MAX


SINGLE_HAND_RULES = [
    ("middleFinger", _rule_middle_finger),
    ("Shhh", _rule_shhh),
]


# Pose con las dos manos, cada regla clasifica las dos manos
# y la lectura del rostro, devuelve True (verdad). El orden importa,'kitty' 
# (puños) y 'debut' (manos como rezando) no pueden ser las dos 'verdad' al 
# mismo tiempo. 

def _rule_kitty(hands, face):
    # two fists held below the chin, like cat paws.
    if not all(h["curledCount"] == 4 for h in hands):
        return False
    avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
    paws_gap = dist(hands[0]["palmCenter"], hands[1]["palmCenter"]) / avg_scale
    below_mouth = all(h["palmCenter"][1] > face["mouthCenter"][1] for h in hands)
    near_face = all(
        dist(h["palmCenter"], face["mouthCenter"]) / face["faceWidth"] < CAT_PAWS_NEAR_FACE_DIST for h in hands
    )
    return paws_gap < CAT_PAWS_TOGETHER_DIST and below_mouth and near_face


def _rule_debut(hands, face):
    # palmas juntas apuntando hacia arriba como rezando
    # (como la portada de'Debut' de Björk).

    avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
    palm_gap = dist(hands[0]["palmCenter"], hands[1]["palmCenter"]) / avg_scale
    avg_palm = (hands[0]["palmCenter"] + hands[1]["palmCenter"]) / 2
    near_mouth = dist(avg_palm, face["mouthCenter"]) / face["faceWidth"] < PRAYER_NEAR_MOUTH_DIST
    fingers_up = all(h["indexTip"][1] < h["wrist"][1] + 0.15 * h["handScale"] for h in hands)
    return palm_gap < PRAYER_HANDS_TOGETHER_DIST and near_mouth and fingers_up


def _rule_huh(hands, face):
    # ambas manos abiertas a los lados del torso
    # pose 'no lo sé'
    if not all(h["curledCount"] == 0 for h in hands):
        return False
    x_gap = abs(hands[0]["palmCenter"][0] - hands[1]["palmCenter"][0]) / face["faceWidth"]
    height_diff = abs(hands[0]["palmCenter"][1] - hands[1]["palmCenter"][1]) / face["faceWidth"]
    avg_y = (hands[0]["palmCenter"][1] + hands[1]["palmCenter"][1]) / 2
    below_face = (avg_y - face["mouthCenter"][1]) / face["faceWidth"]
    return x_gap > HUH_SPREAD_MIN_DIST and height_diff < HUH_HEIGHT_MATCH_MAX and below_face > HUH_BELOW_FACE_MIN


TWO_HAND_RULES = [
    ("kitty", _rule_kitty),
    ("debut", _rule_debut),
    ("huh", _rule_huh),
]


class GestureState:
    def __init__(self):
        self.last_face = None  # dict (diccionario): mouthCenter, faceWidth, mouthOpen, yawDeg, rollDeg, eyeOpenAvg, browDiff, t
        self.face_seen_this_frame = False
        self.swing_history = []  # [(t, diff), ...] sixSeven
        self.last_debug = {"eyeOpen": 0.0, "browDiff": 0.0, "yaw": 0.0}

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            # ojos: espacio vertical de los parpados, normalizado, 
            # promedio entre los dos ojos
            right_eye_open = dist(p3(f[159]), p3(f[145])) / face_width
            left_eye_open = dist(p3(f[386]), p3(f[374])) / face_width
            eye_open_avg = (right_eye_open + left_eye_open) / 2

            # cejas: espacio entre las cejas y el parpado por cada lado
            # browDiff > 0 significa que tu ceja izq esta mas alta.
            right_brow_gap = dist(p3(f[105]), p3(f[159])) / face_width
            left_brow_gap = dist(p3(f[334]), p3(f[386])) / face_width
            brow_diff = left_brow_gap - right_brow_gap

            yaw_deg = 0.0
            roll_deg = 0.0
            if face_result.facial_transformation_matrixes:
                m = face_result.facial_transformation_matrixes[0]
                yaw_deg = yaw_from_transform_matrix(m)
                roll_deg = roll_from_transform_matrix(m)

            self.last_face = {
                "mouthCenter": mouth_center,
                "faceWidth": face_width,
                "mouthOpen": mouth_open,
                "yawDeg": yaw_deg,
                "rollDeg": roll_deg,
                "eyeOpenAvg": eye_open_avg,
                "browDiff": brow_diff,
                "t": now,
            }
            self.last_debug = {"eyeOpen": eye_open_avg, "browDiff": brow_diff, "yaw": yaw_deg}
        self.face_seen_this_frame = saw_face

    def is_sus_face(self):
        return (
            self.last_face is not None
            and self.last_face["browDiff"] < -EYEBROW_RAISE_DIFF
            and SUS_TURN_MIN_DEG < self.last_face["yawDeg"] < SUS_TURN_MAX_DEG
        )

    def update_swing_history(self, hands):
        # tracks the left/right hand height relationship over time to catch
        # a "weighing" up/down alternation (sixSeven). Hands are paired by x
        # position each frame (left = smaller x) rather than landmark order,
        # since MediaPipe doesn't guarantee stable left/right ordering.

        # rastrea la relacion de altura entre la mano izq y derecha para ver cuando se alternan
        # como el gesto sixSeven. las manos toman la posicion x en cada cuadro (izquierda = x
        # más pequeña) a excepcion del orden de landmark ya que MediaPipe no estabiliza un orden 
        # de izq/derecha

        now = time.time() * 1000
        if len(hands) == 2:
            sorted_hands = sorted(hands, key=lambda h: h["palmCenter"][0])
            left_hand, right_hand = sorted_hands
            scale = (left_hand["handScale"] + right_hand["handScale"]) / 2
            diff = (left_hand["palmCenter"][1] - right_hand["palmCenter"][1]) / scale
            self.swing_history.append((now, diff))
        self.swing_history = [(t, d) for t, d in self.swing_history if now - t < SIXSEVEN_WINDOW_MS]

    def is_six_seven(self):
        samples = [d for _, d in self.swing_history if abs(d) > SIXSEVEN_MIN_SWING]
        if len(samples) < 2:
            return False
        flips = sum(1 for i in range(1, len(samples)) if (samples[i] > 0) != (samples[i - 1] > 0))
        return flips >= SIXSEVEN_MIN_FLIPS

    def decide(self, hand_result):
        face_is_fresh = self.last_face is not None and time.time() * 1000 - self.last_face["t"] < FACE_STALE_MS
        face = self.last_face if face_is_fresh else None

        hands = [classify_hand(lm) for lm in (hand_result.hand_landmarks or [])]
        self.update_swing_history(hands)

        # sixSeven is a full-arm motion rather than a held pose, so it takes
        # priority over whatever static shape the hands are in mid-swing.
        # sixSeven es un gesto en movimiento a diferencia de las poses estaticas
        # así que se pone sobre la pose que este entre los movimientos.

        if self.is_six_seven():
            return "sixSeven"

        if len(hands) == 2 and face:
            for name, rule in TWO_HAND_RULES:
                if rule(hands, face):
                    return name

        if hands:
            h = hands[0]
            for name, rule in SINGLE_HAND_RULES:
                if rule(h, face):
                    return name

        # sus es una pose solo de rostro: ceja izq levantada + la cabeza girada
        # a la izq. Revisa el ultimo cuadro para que se sobreponga aunque las manos
        # no estén visibles / haciendo alguna seña
        if face and self.is_sus_face():
            return "sus"

        return "default"


def load_bjorks():
    cache = {}
    for gesture, files in GESTURE_BJORKS.items():
        imgs = []
        for name in files:
            img = cv2.imread(str(BJORKS / name))
            if img is None:
                raise FileNotFoundError(f"missing BJORK file: {BJORKS / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    d = state.last_debug
    samples = [df for _, df in state.swing_history if abs(df) > SIXSEVEN_MIN_SWING]
    flips = sum(1 for i in range(1, len(samples)) if (samples[i] > 0) != (samples[i - 1] > 0))
    lines = [
        f"gesture: {gesture}",
        f"eyeOpen: {d['eyeOpen']:.3f}  (Shhh thr < {EYES_SEMI_CLOSED_MAX})",
        f"browDiff: {d['browDiff']:.3f}  (sus thr < {-EYEBROW_RAISE_DIFF})",
        f"yaw: {d['yaw']:+.1f} deg  (sus range {SUS_TURN_MIN_DEG}-{SUS_TURN_MAX_DEG})",
        f"sixSeven flips: {flips}  (thr >= {SIXSEVEN_MIN_FLIPS})",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    bjorks = load_bjorks()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camarita")
    cv2.namedWindow("BJORK")
    cv2.moveWindow("Camarita", 40, 80)
    cv2.moveWindow("BJORK", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_bjork = random.choice(bjorks["default"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_bjork = random.choice(bjorks[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_bjork = random.choice(bjorks["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            bjork_view = fit_to_height(current_bjork, frame.shape[0])
            cv2.imshow("Camarita", frame)
            cv2.imshow("BJORK", bjork_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()