import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- meme mapping -----------------------------------------------------
// Each gesture maps to one or more meme images. When a gesture has more
// than one image, one is picked at random each time the gesture is newly
// (re)triggered, so repeated gestures don't always show the same frame.
const GESTURE_MEMES = {
  default: ["bjorkReact/bjork.jpg"],
  Shhh: ["bjorkReact/bjorkShhh.jpg", "bjorkReact/bjorkShhh2.jpg", "bjorkReact/bjorkShhh3.jpg"],
  FuckOff: ["bjorkReact/bjorkFuckOff.jpg"],
  sixSeven: ["bjorkReact/bjork67.jpg"],
  debut: ["bjorkReact/bjorkDebutRacoon.jpg", "bjorkReact/bjorkDebutHealing.jpg"],
  sus: ["bjorkReact/bjorkSuspicious.jpg"],
  bjorkCAT: ["bjorkReact/bjorkCAT.jpg", "bjorkReact/bjorkCAT2.jpg"],
  rawr: ["bjorkReact/bjorkRAWR.gif"],
};

// cuadros estables de la misma pose para poner la imagen
const STABLE_FRAMES_REQUIRED = 5;
// sino detecta un gesto en esta cantidad de ms, vuelve a la imagen default
const DEFAULT_FALLBACK_MS = 600;
// how long we trust a stale face box after the face detector loses the face
const FACE_STALE_MS = 1200;

// --- Shhh: eyes semi-closed ----------------------------------------------
// eye "openness" = vertical gap between eyelids / faceWidth. Watch the
// "eyeOpen" debug HUD line while squinting to tune this.
const EYES_SEMI_CLOSED_MAX = 0.02;

// --- sus: raised eyebrow + slight head tilt -------------------------------
// how much higher the left eyebrow must sit than the right one (both
// normalized by faceWidth) to count as "raised". Watch "browDiff" in the HUD.
const EYEBROW_RAISE_DIFF = 0.01;
// head-tilt (roll, degrees) range that counts as "levemente ladeada" - big
// enough to be deliberate, small enough it isn't a full ear-to-shoulder tilt.
const SUS_TILT_MIN_DEG = 6;
const SUS_TILT_MAX_DEG = 28;

// --- bjorkCAT: two fists held below the chin, like cat paws --------------
// how close the two fists must be to each other, normalized by avg hand scale
const CAT_PAWS_TOGETHER_DIST = 1.8;
// how close the fists must stay to the face (not held way out at the
// waist), normalized by faceWidth
const CAT_PAWS_NEAR_FACE_DIST = 1.6;

// --- debut: prayer hands at the mouth -------------------------------------
// how close the two palms must be to each other, normalized by avg hand scale
const PRAYER_HANDS_TOGETHER_DIST = 0.9;
// how close the joined hands must be to the mouth, normalized by faceWidth
const PRAYER_NEAR_MOUTH_DIST = 1.4;

// --- sixSeven: alternating hands (like weighing something) ---------------
// vertical gap between hands (normalized by faceWidth-scale hand distance)
// big enough to count as a real "swing" sample rather than noise
const SIXSEVEN_MIN_SWING = 0.35;
// trailing window we look for alternation in
const SIXSEVEN_WINDOW_MS = 1500;
// how many times the up/down relationship must flip inside that window
const SIXSEVEN_MIN_FLIPS = 2;

// --- rawr: hands opening and closing repeatedly, like squeezing the air --
// trailing window we look for the open/closed alternation in
const RAWR_WINDOW_MS = 1600;
// how many open<->closed transitions must happen inside that window
// (each full squeeze cycle = 2 transitions, so 3 asks for a bit more than
// one full open-close-open before it fires)
const RAWR_MIN_FLIPS = 3;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null; // { mouthCenter, faceWidth, mouthOpen, yawDeg, rollDeg, eyeOpenAvg, browDiff, t }
let lastFaceSeenThisFrame = false;
let swingHistory = []; // [{t, diff}] for sixSeven
let rawrHistory = []; // [{t, state}] for rawr (open/closed alternation)
let lastDebug = { eyeOpen: 0, browDiff: 0, roll: 0 };

async function init() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  requestAnimationFrame(loop);
}

// ---- 3D-aware geometry helpers -----------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}
function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

// a finger is "extended" if its two segments (mcp->pip, pip->tip) point in
// roughly the same direction; "curled" if it folds back sharply.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

// extract the head's left/right turn angle (yaw, degrees) from MediaPipe's
// facial transformation matrix.
function yawFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

// extract head tilt sideways (roll, degrees - like tilting an ear toward a
// shoulder) from the same rotation matrix used for yaw.
function rollFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  return (Math.atan2(r10, r00) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // wrist -> middle mcp

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.05;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    wrist: lm[0],
    palmCenter: lm[9],
  };
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek);
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    // eyes: vertical eyelid gap, normalized, averaged across both eyes.
    // landmark indices are the standard MediaPipe 468-point face mesh ones.
    const rightEyeOpen = dist(f[159], f[145]) / faceWidth;
    const leftEyeOpen = dist(f[386], f[374]) / faceWidth;
    const eyeOpenAvg = (rightEyeOpen + leftEyeOpen) / 2;

    // eyebrows: gap between the brow ridge and the eyelid, per side - a
    // raised brow increases this gap. browDiff > 0 means the left brow
    // (subject's left) sits higher than the right.
    const rightBrowGap = dist(f[105], f[159]) / faceWidth;
    const leftBrowGap = dist(f[334], f[386]) / faceWidth;
    const browDiff = leftBrowGap - rightBrowGap;

    let yawDeg = 0;
    let rollDeg = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const m = faceResult.facialTransformationMatrixes[0].data;
      yawDeg = yawFromTransformMatrix(m);
      rollDeg = rollFromTransformMatrix(m);
    }

    lastFace = { mouthCenter, faceWidth, mouthOpen, yawDeg, rollDeg, eyeOpenAvg, browDiff, t: now };
    lastDebug = { eyeOpen: eyeOpenAvg, browDiff, roll: rollDeg };
  }
  lastFaceSeenThisFrame = sawFace;
}

function isSusFace() {
  return (
    !!lastFace &&
    lastFace.browDiff > EYEBROW_RAISE_DIFF &&
    Math.abs(lastFace.rollDeg) > SUS_TILT_MIN_DEG &&
    Math.abs(lastFace.rollDeg) < SUS_TILT_MAX_DEG
  );
}

// tracks the left/right hand height relationship over time to catch a
// "weighing" up/down alternation (sixSeven). Hands are paired by x position
// each frame (left = smaller x) rather than landmark order, since MediaPipe
// doesn't guarantee stable left/right ordering frame to frame.
function updateSwingHistory(hands) {
  const now = performance.now();
  if (hands.length === 2) {
    const sorted = [...hands].sort((a, b) => a.palmCenter.x - b.palmCenter.x);
    const [leftHand, rightHand] = sorted;
    const scale = (leftHand.handScale + rightHand.handScale) / 2;
    const diff = (leftHand.palmCenter.y - rightHand.palmCenter.y) / scale;
    swingHistory.push({ t: now, diff });
  }
  swingHistory = swingHistory.filter((s) => now - s.t < SIXSEVEN_WINDOW_MS);
}

function isSixSeven() {
  const samples = swingHistory.filter((s) => Math.abs(s.diff) > SIXSEVEN_MIN_SWING);
  if (samples.length < 2) return false;
  let flips = 0;
  for (let i = 1; i < samples.length; i++) {
    if (Math.sign(samples[i].diff) !== Math.sign(samples[i - 1].diff)) flips++;
  }
  return flips >= SIXSEVEN_MIN_FLIPS;
}

// tracks whether the hand(s) are "open" or "closed" over time to catch a
// repeated squeezing motion (rawr). Uses however many hands are visible
// (1 or 2), averaging their curl so both hands squeezing together still
// reads as one clean open/closed signal.
function updateRawrHistory(hands) {
  const now = performance.now();
  if (hands.length >= 1) {
    const avgCurl = hands.reduce((sum, h) => sum + h.curledCount, 0) / hands.length;
    let state = null;
    if (avgCurl <= 1) state = "open";
    else if (avgCurl >= 3) state = "closed";
    if (state) rawrHistory.push({ t: now, state });
  }
  rawrHistory = rawrHistory.filter((s) => now - s.t < RAWR_WINDOW_MS);
}

function isRawr() {
  if (rawrHistory.length < 2) return false;
  let flips = 0;
  for (let i = 1; i < rawrHistory.length; i++) {
    if (rawrHistory[i].state !== rawrHistory[i - 1].state) flips++;
  }
  return flips >= RAWR_MIN_FLIPS;
}

function decideGesture(handResult) {
  const faceIsFresh = !!lastFace && performance.now() - lastFace.t < FACE_STALE_MS;
  const hands = (handResult.landmarks || []).map(classifyHand);

  updateSwingHistory(hands);
  updateRawrHistory(hands);

  // sixSeven and rawr are both full-motion gestures rather than held poses,
  // so they take priority over whatever static shape the hands happen to
  // be in mid-motion.
  if (isSixSeven()) {
    return "sixSeven";
  }
  if (isRawr()) {
    return "rawr";
  }

  if (hands.length === 0) {
    // sus is a face-only pose: raised eyebrow + slight head tilt.
    if (faceIsFresh && isSusFace()) {
      return "sus";
    }
    return "default";
  }

  if (hands.length === 2 && faceIsFresh) {
    // bjorkCAT: two fists held below the chin, like cat paws. Checked
    // before debut since a fist and an open prayer-hand shape can't
    // both be true, but checking this first keeps the intent explicit.
    const bothFists = hands.every((hh) => hh.curledCount === 4);
    if (bothFists) {
      const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
      const pawsGap = dist(hands[0].palmCenter, hands[1].palmCenter) / avgScale;
      const belowMouth = hands.every((hh) => hh.palmCenter.y > lastFace.mouthCenter.y);
      const nearFace = hands.every(
        (hh) => dist(hh.palmCenter, lastFace.mouthCenter) / lastFace.faceWidth < CAT_PAWS_NEAR_FACE_DIST
      );
      if (pawsGap < CAT_PAWS_TOGETHER_DIST && belowMouth && nearFace) {
        return "bjorkCAT";
      }
    }

    // debut: palms together, held up near the mouth, fingers pointing up -
    // a prayer-hands pose (nod to Bjork's "Debut" album art).
    const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
    const palmGap = dist(hands[0].palmCenter, hands[1].palmCenter) / avgScale;
    const avgPalm = {
      x: (hands[0].palmCenter.x + hands[1].palmCenter.x) / 2,
      y: (hands[0].palmCenter.y + hands[1].palmCenter.y) / 2,
      z: (hands[0].palmCenter.z + hands[1].palmCenter.z) / 2,
    };
    const nearMouth = dist(avgPalm, lastFace.mouthCenter) / lastFace.faceWidth < PRAYER_NEAR_MOUTH_DIST;
    const fingersUp = hands.every((h) => h.indexTip.y < h.wrist.y);
    if (palmGap < PRAYER_HANDS_TOGETHER_DIST && nearMouth && fingersUp) {
      return "debut";
    }
  }

  const h = hands[0];

  // FuckOff: only the middle finger extended.
  if (h.middleUp && !h.indexUp && !h.ringUp && !h.pinkyUp) {
    return "FuckOff";
  }

  // Shhh: index finger at the mouth (same shape as the original exercise),
  // plus eyes semi-closed.
  if (h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp && faceIsFresh) {
    const d = dist(h.indexTip, lastFace.mouthCenter) / lastFace.faceWidth;
    if (d < 0.55 && lastFace.eyeOpenAvg < EYES_SEMI_CLOSED_MAX) {
      return "Shhh";
    }
  }

  // sus can also win with a hand up but no specific shape recognized yet.
  if (faceIsFresh && isSusFace()) {
    return "sus";
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture];
  return images[Math.floor(Math.random() * images.length)];
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    const handResult = handLandmarker.detectForVideo(video, ts);
    const faceResult = faceLandmarker.detectForVideo(video, ts);
    updateFace(faceResult);

    const gesture = decideGesture(handResult);

    if (gesture === candidateGesture) {
      candidateStreak++;
    } else {
      candidateGesture = gesture;
      candidateStreak = 1;
    }

    if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
      applyGesture(gesture);
    }

    if (gesture !== "default") lastNonDefaultAt = now;
    if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
      applyGesture("default");
    }

    updateDebugHud();
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  const samples = swingHistory.filter((s) => Math.abs(s.diff) > SIXSEVEN_MIN_SWING);
  let flips = 0;
  for (let i = 1; i < samples.length; i++) {
    if (Math.sign(samples[i].diff) !== Math.sign(samples[i - 1].diff)) flips++;
  }
  let rawrFlips = 0;
  for (let i = 1; i < rawrHistory.length; i++) {
    if (rawrHistory[i].state !== rawrHistory[i - 1].state) rawrFlips++;
  }
  debugHud.textContent =
    `gesture: ${currentGesture}\n` +
    `eyeOpen: ${lastDebug.eyeOpen.toFixed(3)}  (Shhh thr < ${EYES_SEMI_CLOSED_MAX})\n` +
    `browDiff: ${lastDebug.browDiff.toFixed(3)}  (sus thr > ${EYEBROW_RAISE_DIFF})\n` +
    `roll: ${lastDebug.roll >= 0 ? "+" : ""}${lastDebug.roll.toFixed(1)} deg  (sus range ${SUS_TILT_MIN_DEG}-${SUS_TILT_MAX_DEG})\n` +
    `sixSeven flips: ${flips}  (thr >= ${SIXSEVEN_MIN_FLIPS})\n` +
    `rawr flips: ${rawrFlips}  (thr >= ${RAWR_MIN_FLIPS})`;
}

init().catch((err) => console.error(err));
