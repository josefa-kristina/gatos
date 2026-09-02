"""
Webcam gesture -> meme detector (desktop version), Bjork edition.

Opens two windows, side by side like the OBS/streamer setups:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the meme matching whatever gesture you're making

Gestures:
  default (no gesture)                              -> bjorkReact/bjork.jpg
  Shhh   (finger on mouth + eyes semi-closed)        -> bjorkReact/bjorkShhh*.jpg
  FuckOff (middle finger up)                         -> bjorkReact/bjorkFuckOff.jpg
  sixSeven (hands alternating up/down, like weighing) -> bjorkReact/bjork67.jpg
  debut  (prayer hands held at the mouth)            -> bjorkReact/bjorkDebut*.jpg
  sus    (left eyebrow raised + head slightly tilted) -> bjorkReact/bjorkSuspicious.jpg

The Camera window shows a live debug readout (eye openness, eyebrow
asymmetry, head roll, sixSeven swing-flip count) in the top-left corner so
each threshold can be tuned by eye - see the constants below.

Press q or ESC to quit.
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
MEMES = ROOT / "memes"

GESTURE_MEMES = {
  "default": ["bjork.jpg"],
  "Shhh": ["bjorkReact/bjorkShhh.jpg", "bjorkReact/bjorkShhh2.jpg", "bjorkReact/bjorkShhh3.jpg"],
  "FuckOff": ["bjorkReact/bjorkFuckOff.jpg"],
  "sixSeven": ["bjorkReact/bjork67.jpg"],
  "debut": ["bjorkReact/bjorkDebutRacoon.jpg", "bjorkReact/bjorkDebutHealing.jpg"],
  "sus": ["bjorkReact/bjorkSuspicious.jpg"],
  "bjorkCAT": ["bjorkReact/bjorkCAT.jpg", "bjorkReact/bjorkCAT2.jpg"],
  "rawr": ["bjorkReact/bjorkRAWR.gif"],
}

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# --- Shhh: eyes semi-closed ------------------------------------------------
# eye "openness" = vertical eyelid gap / faceWidth. Watch the "eyeOpen" HUD
# line while squinting to tune this.
EYES_SEMI_CLOSED_MAX = 0.02

# --- sus: raised eyebrow + slight head tilt --------------------------------
# how much higher the left eyebrow must sit than the right one (both
# normalized by faceWidth) to count as "raised". Watch "browDiff" in the HUD.
EYEBROW_RAISE_DIFF = 0.01
# head-tilt (roll, degrees) range that counts as "levemente ladeada" - big
# enough to be deliberate, small enough it isn't a full ear-to-shoulder tilt.
SUS_TILT_MIN_DEG = 6
SUS_TILT_MAX_DEG = 28

# --- bjorkCAT: two fists held below the chin, like cat paws ---------------
CAT_PAWS_TOGETHER_DIST = 1.8
CAT_PAWS_NEAR_FACE_DIST = 1.6

# --- debut: prayer hands at the mouth --------------------------------------
PRAYER_HANDS_TOGETHER_DIST = 0.9
PRAYER_NEAR_MOUTH_DIST = 1.4

# --- sixSeven: alternating hands (like weighing something) ----------------
SIXSEVEN_MIN_SWING = 0.35
SIXSEVEN_WINDOW_MS = 1500
SIXSEVEN_MIN_FLIPS = 2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
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
    """Head left/right turn angle (yaw, degrees) from MediaPipe's facial
    transformation matrix."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def roll_from_transform_matrix(matrix):
    """Head tilt sideways (roll, degrees - like tilting an ear toward a
    shoulder), from the same rotation matrix used for yaw."""
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


class GestureState:
    def __init__(self):
        self.last_face = None  # dict: mouthCenter, faceWidth, mouthOpen, yawDeg, rollDeg, eyeOpenAvg, browDiff, t
        self.face_seen_this_frame = False
        self.swing_history = []  # [(t, diff), ...] for sixSeven
        self.last_debug = {"eyeOpen": 0.0, "browDiff": 0.0, "roll": 0.0}

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

            # eyes: vertical eyelid gap, normalized, averaged across both eyes
            right_eye_open = dist(p3(f[159]), p3(f[145])) / face_width
            left_eye_open = dist(p3(f[386]), p3(f[374])) / face_width
            eye_open_avg = (right_eye_open + left_eye_open) / 2

            # eyebrows: gap between brow ridge and eyelid, per side. browDiff
            # > 0 means the left brow (subject's left) sits higher.
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
            self.last_debug = {"eyeOpen": eye_open_avg, "browDiff": brow_diff, "roll": roll_deg}
        self.face_seen_this_frame = saw_face

    def is_sus_face(self):
        return (
            self.last_face is not None
            and self.last_face["browDiff"] > EYEBROW_RAISE_DIFF
            and SUS_TILT_MIN_DEG < abs(self.last_face["rollDeg"]) < SUS_TILT_MAX_DEG
        )

    def update_swing_history(self, hands):
        # tracks the left/right hand height relationship over time to catch
        # a "weighing" up/down alternation (sixSeven). Hands are paired by x
        # position each frame (left = smaller x) rather than landmark order,
        # since MediaPipe doesn't guarantee stable left/right ordering.
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

        hands = [classify_hand(lm) for lm in (hand_result.hand_landmarks or [])]
        self.update_swing_history(hands)

        # sixSeven is a full-arm motion rather than a held pose, so it takes
        # priority over whatever static shape the hands are in mid-swing.
        if self.is_six_seven():
            return "sixSeven"

        if not hands:
            if face_is_fresh and self.is_sus_face():
                return "sus"
            return "default"

        if len(hands) == 2 and face_is_fresh:
            mouth_center, face_width = self.last_face["mouthCenter"], self.last_face["faceWidth"]

            # bjorkCAT: two fists held below the chin, like cat paws.
            both_fists = all(h["curledCount"] == 4 for h in hands)
            if both_fists:
                avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
                paws_gap = dist(hands[0]["palmCenter"], hands[1]["palmCenter"]) / avg_scale
                below_mouth = all(h["palmCenter"][1] > mouth_center[1] for h in hands)
                near_face = all(dist(h["palmCenter"], mouth_center) / face_width < CAT_PAWS_NEAR_FACE_DIST for h in hands)
                if paws_gap < CAT_PAWS_TOGETHER_DIST and below_mouth and near_face:
                    return "bjorkCAT"

            # debut: palms together, held up near the mouth, fingers
            # pointing up - a prayer-hands pose (nod to Bjork's "Debut").
            avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
            palm_gap = dist(hands[0]["palmCenter"], hands[1]["palmCenter"]) / avg_scale
            avg_palm = (hands[0]["palmCenter"] + hands[1]["palmCenter"]) / 2
            near_mouth = dist(avg_palm, mouth_center) / face_width < PRAYER_NEAR_MOUTH_DIST
            fingers_up = all(h["indexTip"][1] < h["wrist"][1] for h in hands)
            if palm_gap < PRAYER_HANDS_TOGETHER_DIST and near_mouth and fingers_up:
                return "debut"

        h = hands[0]

        # FuckOff: only the middle finger extended.
        if h["middleUp"] and not h["indexUp"] and not h["ringUp"] and not h["pinkyUp"]:
            return "FuckOff"

        # Shhh: index finger at the mouth (same shape as the original
        # exercise), plus eyes semi-closed.
        if h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"] and face_is_fresh:
            mouth_center, face_width = self.last_face["mouthCenter"], self.last_face["faceWidth"]
            d = dist(h["indexTip"], mouth_center) / face_width
            if d < 0.55 and self.last_face["eyeOpenAvg"] < EYES_SEMI_CLOSED_MAX:
                return "Shhh"

        # sus can also win with a hand up but no recognized shape yet.
        if face_is_fresh and self.is_sus_face():
            return "sus"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
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
        f"browDiff: {d['browDiff']:.3f}  (sus thr > {EYEBROW_RAISE_DIFF})",
        f"roll: {d['roll']:+.1f} deg  (sus range {SUS_TILT_MIN_DEG}-{SUS_TILT_MAX_DEG})",
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

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])

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
                current_meme = random.choice(memes[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

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