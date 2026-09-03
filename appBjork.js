import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// Cada gesto tiene una o mas imagenes. Cuando un gesto tiene mas de una,
// se elige una al azar cada vez que el gesto recien se gatilla, asi los
// gestos repetidos no siempre muestran el mismo cuadro.
const GESTURE_MEMES = {
  default: ["bjorkReact/bjork.jpg"],
  Shhh: ["bjorkReact/bjorkShhh.jpg", "bjorkReact/bjorkShhh2.jpg", "bjorkReact/bjorkShhh3.gif"],
  middleFinger: ["bjorkReact/bjorkMiddleFinger.jpg"],
  sixSeven: ["bjorkReact/bjork67.jpg"],
  debut: ["bjorkReact/bjorkDebutRacoon.jpg", "bjorkReact/bjorkDebutHealing.jpg"],
  sus: ["bjorkReact/bjorkSuspicious.jpg"],
  kitty: ["bjorkReact/bjorkCAT.jpg", "bjorkReact/bjorkCAT2.jpg"],
  rawr: ["bjorkReact/bjorkRAWR.gif"],
  huh: ["bjorkReact/bjorkGrapes.jpeg"],
};

// cuadros estables de la misma pose para poner la imagen
const STABLE_FRAMES_REQUIRED = 5;
// sino detecta un gesto en esta cantidad de ms, vuelve a la imagen default
const DEFAULT_FALLBACK_MS = 600;
// cuanto tiempo confiamos en el rostro (viejo) despues que el detector lo pierde
const FACE_STALE_MS = 1200;

// --- Shhh: ojos semi abiertos ---------------------------------------------
// eye "openness" = espacio vertical parpados / ancho cara. Observa la linea
// "eyeOpen" del HUD de debug mientras entrecierras los ojos para ajustar esto.
const EYES_SEMI_CLOSED_MAX = 0.04;

// --- sus: ceja levantada + cabeza levemente ladeada -----------------------
// que tan alta esta la ceja izquierda comparada con la derecha (ambas
// normalizadas por el ancho de cara) para contar como "levantada". Levanta
// una ceja para gatillarlo, mira "browDiff" en el HUD.
const EYEBROW_RAISE_DIFF = 0.01;
// ladeo de cabeza (roll, grados) rango que cuenta como "levemente ladeada" -
// suficiente para ser deliberado, pero no tanto como un ladeo completo de
// oreja a hombro.
const SUS_TILT_MIN_DEG = 6;
const SUS_TILT_MAX_DEG = 28;

// --- kitty: manos cerradas (fist) debajo de la mandibula, como patitas de gato ---
// que tan cerca deben estar los dos puños entre si, normalizado por el
// promedio de escala de mano
const CAT_PAWS_TOGETHER_DIST = 1.8;
// que tan cerca deben quedarse los puños de la cara (no estirados hacia la
// cintura), normalizado por el ancho de cara
const CAT_PAWS_NEAR_FACE_DIST = 1.6;

// --- debut: manos juntas como rezando por encima de la boca ---------------
// que tan cerca deben estar las dos palmas entre si, normalizado por el
// promedio de escala de mano
const PRAYER_HANDS_TOGETHER_DIST = 1.6;
// que tan cerca deben estar las manos juntas de la boca, normalizado por el
// ancho de cara
const PRAYER_NEAR_MOUTH_DIST = 2.0;

// --- huh: manos abiertas al lado del torso, como encogimiento de hombros -
// que tan separadas deben estar las manos horizontalmente, normalizado por
// el ancho de cara
const HUH_SPREAD_MIN_DIST = 2.2;
// que tan abajo de la boca deben estar las manos, normalizado por el ancho
// de cara (descarta manos levantadas cerca de la cara)
const HUH_BELOW_FACE_MIN = 1.0;
// que tan parecida debe ser la altura entre las dos manos, normalizado por
// el ancho de cara (descarta una mano arriba / una abajo)
const HUH_HEIGHT_MATCH_MAX = 1.0;

// --- sixSeven: manos alternando de arriba a abajo -------------------------
// espacio vertical entre manos (normalizado por la escala de mano) lo
// suficientemente grande para contar como un "swing" real y no ruido
const SIXSEVEN_MIN_SWING = 0.35;
// ventana de tiempo hacia atras en la que buscamos la alternancia
const SIXSEVEN_WINDOW_MS = 1500;
// cuantas veces debe cambiar la relacion arriba/abajo dentro de esa ventana
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
let lastFace = null; // diccionario: mouthCenter, faceWidth, mouthOpen, yawDeg, rollDeg, eyeOpenAvg, browDiff, t
let lastFaceSeenThisFrame = false;
let swingHistory = []; // [{t, diff}, ...] sixSeven
let rawrHistory = []; // [{t, state}] for rawr (open/closed alternation)
let lastDebug = { eyeOpen: 0, browDiff: 0, roll: 0, debutPalmGap: null, debutNearMouth: null, debutFingersUp: null };

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

// ---- ayudas de geometria (3D) --------------------------------------------
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

// un dedo esta "extendido" si sus dos segmentos (mcp->pip, pip->tip) apuntan
// mas o menos en la misma direccion; "curled" (doblado) si se pliega en seco.
function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

// angulo de giro a la derecha o a la izquierda (yaw, grados) de la matrix
// de transformacion de MediaPipe
function yawFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const sy = Math.hypot(r00, r10);
  if (sy < 1e-6) return 0;
  return (Math.atan2(-r20, sy) * 180) / Math.PI;
}

// cabeza ladeada (roll, grados - como acercando la oreja al hombro) de la
// misma matrix de rotacion que se usa para el yaw
function rollFromTransformMatrix(matrixData) {
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  return (Math.atan2(r10, r00) * 180) / Math.PI;
}

function classifyHand(lm) {
  const handScale = dist(lm[0], lm[9]) || 1e-6; // muñeca -> mcp del dedo medio

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

    // ojos: espacio vertical de los parpados, normalizado, promedio entre los
    // dos ojos. Los indices de landmark son los del face mesh estandar de
    // MediaPipe (468 puntos).
    const rightEyeOpen = dist(f[159], f[145]) / faceWidth;
    const leftEyeOpen = dist(f[386], f[374]) / faceWidth;
    const eyeOpenAvg = (rightEyeOpen + leftEyeOpen) / 2;

    // cejas: espacio entre las cejas y el parpado por cada lado - una ceja
    // levantada aumenta este espacio. browDiff > 0 significa que tu ceja izq
    // esta mas alta que la derecha.
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
    lastDebug.eyeOpen = eyeOpenAvg;
    lastDebug.browDiff = browDiff;
    lastDebug.roll = rollDeg;
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

// rastrea la relacion de altura entre la mano izq y derecha para ver cuando
// se alternan como el gesto sixSeven. las manos toman la posicion x en cada
// cuadro (izquierda = x mas pequeña) a excepcion del orden de landmark ya
// que MediaPipe no estabiliza un orden de izq/derecha
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

// --- pose de mano exclusiva - reglas ---------------------------------------
// cada regla recibe la mano clasificada y la lectura actual del rostro (o
// null si el detector de rostro no tiene una lectura fresca), y devuelve
// true (verdad) si se esta haciendo el gesto. se ven en orden - primera
// coincidencia 'gana'.
const SINGLE_HAND_RULES = [
  {
    name: "middleFinger",
    test: (h) => h.middleUp && !h.indexUp && !h.ringUp && !h.pinkyUp,
  },
  {
    name: "Shhh",
    // dedo indice por encima de la boca (como en la Meowmeow cat cam meme
    // detector de catherpiee) + los ojos entre abiertos
    test: (h, face) =>
      h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp &&
      !!face &&
      dist(h.indexTip, face.mouthCenter) / face.faceWidth < 0.55 &&
      face.eyeOpenAvg < EYES_SEMI_CLOSED_MAX,
  },
];

// --- pose con las dos manos ------------------------------------------------
// cada regla clasifica las dos manos y la lectura del rostro, devuelve true
// (verdad). El orden importa, 'kitty' (puños) y 'debut' (manos como
// rezando) no pueden ser las dos 'verdad' al mismo tiempo.
const TWO_HAND_RULES = [
  {
    name: "kitty",
    // two fists held below the chin, like cat paws.
    test: (hands, face) => {
      if (!hands.every((h) => h.curledCount === 4)) return false;
      const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
      const pawsGap = dist(hands[0].palmCenter, hands[1].palmCenter) / avgScale;
      const belowMouth = hands.every((h) => h.palmCenter.y > face.mouthCenter.y);
      const nearFace = hands.every(
        (h) => dist(h.palmCenter, face.mouthCenter) / face.faceWidth < CAT_PAWS_NEAR_FACE_DIST
      );
      return pawsGap < CAT_PAWS_TOGETHER_DIST && belowMouth && nearFace;
    },
  },
  {
    name: "debut",
    // palmas juntas apuntando hacia arriba, cerca de la boca - una pose de
    // manos rezando (como la portada de 'Debut' de Björk).
    test: (hands, face) => {
      const avgScale = (hands[0].handScale + hands[1].handScale) / 2;
      const palmGap = dist(hands[0].palmCenter, hands[1].palmCenter) / avgScale;
      const avgPalm = {
        x: (hands[0].palmCenter.x + hands[1].palmCenter.x) / 2,
        y: (hands[0].palmCenter.y + hands[1].palmCenter.y) / 2,
        z: (hands[0].palmCenter.z + hands[1].palmCenter.z) / 2,
      };
      const nearMouthDist = dist(avgPalm, face.mouthCenter) / face.faceWidth;
      // pequeña tolerancia hacia abajo (relativa al tamaño de mano) para que
      // dedos levemente inclinados sigan contando como "arriba".
      const fingersUp = hands.every((h) => h.indexTip.y < h.wrist.y + 0.15 * h.handScale);
      lastDebug.debutPalmGap = palmGap;
      lastDebug.debutNearMouth = nearMouthDist;
      lastDebug.debutFingersUp = fingersUp;
      return palmGap < PRAYER_HANDS_TOGETHER_DIST && nearMouthDist < PRAYER_NEAR_MOUTH_DIST && fingersUp;
    },
  },
  {
    name: "huh",
    // ambas manos abiertas a los lados del torso, como encogimiento de
    // hombros / pose 'no lo sé'.
    test: (hands, face) => {
      const bothOpen = hands.every((h) => h.curledCount === 0);
      if (!bothOpen) return false;
      const xGap = Math.abs(hands[0].palmCenter.x - hands[1].palmCenter.x) / face.faceWidth;
      const heightDiff = Math.abs(hands[0].palmCenter.y - hands[1].palmCenter.y) / face.faceWidth;
      const avgY = (hands[0].palmCenter.y + hands[1].palmCenter.y) / 2;
      const belowFace = (avgY - face.mouthCenter.y) / face.faceWidth;
      return xGap > HUH_SPREAD_MIN_DIST && heightDiff < HUH_HEIGHT_MATCH_MAX && belowFace > HUH_BELOW_FACE_MIN;
    },
  },
];

function decideGesture(handResult) {
  const faceIsFresh = !!lastFace && performance.now() - lastFace.t < FACE_STALE_MS;
  const face = faceIsFresh ? lastFace : null;
  const hands = (handResult.landmarks || []).map(classifyHand);

  updateSwingHistory(hands);
  updateRawrHistory(hands);

  // sixSeven y rawr son gestos en movimiento a diferencia de las poses
  // estaticas, asi que se ponen sobre la pose que tengan las manos en medio
  // del movimiento.
  if (isSixSeven()) {
    return "sixSeven";
  }
  if (isRawr()) {
    return "rawr";
  }

  if (hands.length === 2 && face) {
    for (const rule of TWO_HAND_RULES) {
      if (rule.test(hands, face)) return rule.name;
    }
  } else {
    lastDebug.debutPalmGap = null;
    lastDebug.debutNearMouth = null;
    lastDebug.debutFingersUp = null;
  }

  if (hands.length >= 1) {
    const h = hands[0];
    for (const rule of SINGLE_HAND_RULES) {
      if (rule.test(h, face)) return rule.name;
    }
  }

  // sus es una pose solo de rostro: ceja levantada + cabeza levemente
  // ladeada. Se revisa al final para que pueda ganar tengan o no las manos
  // visibles / haciendo alguna seña.
  if (face && isSusFace()) {
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
  const debutLine =
    lastDebug.debutPalmGap === null
      ? "debut: (need 2 hands)"
      : `debut: gap ${lastDebug.debutPalmGap.toFixed(2)} (thr < ${PRAYER_HANDS_TOGETHER_DIST}) ` +
        `mouth ${lastDebug.debutNearMouth.toFixed(2)} (thr < ${PRAYER_NEAR_MOUTH_DIST}) ` +
        `up ${lastDebug.debutFingersUp}`;
  debugHud.textContent =
    `gesture: ${currentGesture}\n` +
    `eyeOpen: ${lastDebug.eyeOpen.toFixed(3)}  (Shhh thr < ${EYES_SEMI_CLOSED_MAX})\n` +
    `browDiff: ${lastDebug.browDiff.toFixed(3)}  (sus thr > ${EYEBROW_RAISE_DIFF})\n` +
    `roll: ${lastDebug.roll >= 0 ? "+" : ""}${lastDebug.roll.toFixed(1)} deg  (sus range ${SUS_TILT_MIN_DEG}-${SUS_TILT_MAX_DEG})\n` +
    `sixSeven flips: ${flips}  (thr >= ${SIXSEVEN_MIN_FLIPS})\n` +
    `rawr flips: ${rawrFlips}  (thr >= ${RAWR_MIN_FLIPS})\n` +
    debutLine;
}

init().catch((err) => console.error(err));