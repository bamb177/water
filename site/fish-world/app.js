import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const $ = (id) => document.getElementById(id);

/* =========================
   CONFIG
========================= */
// 처리 해상도(속도/정확도 트레이드오프)
const PROC_W = 640;
const PROC_H = 360;

// 카메라 FOV 가정(대부분 폰에서 무난). 필요 시 55~70 범위 튜닝
const ASSUMED_FOV_DEG = 60;

// (ArUco) 마커 한 변 길이(미터). 실제 출력물 기준으로 맞추면 스케일이 정확해짐
const MARKER_LEN_M = 0.04; // 4cm

// (카드) 가상 카드 크기(미터). 오프셋 계산에 사용
const CARD_W = 0.20;       // 20cm
const CARD_H = 0.13;       // 13cm (대략)
// 코너에서 마커 중심이 떨어진 거리(미터) — “마커를 코너에 붙였다”는 가정
const CORNER_MARGIN = 0.01;

// ArUco Dictionary (대부분 출력/예제에서 DICT_4X4_50 사용)
const ARUCO_DICT = "DICT_4X4_50";

// 코너 ID 매핑(권장 배치)
// TL=왼쪽위, TR=오른쪽위, BR=오른쪽아래, BL=왼쪽아래
// 당신이 프린트/부착한 ID에 맞게 여기만 바꾸면 “카드 중심” 정렬이 정확해짐
const CORNER_IDS = {
  TL: 0,
  TR: 1,
  BR: 2,
  BL: 3,
};

// 추적 안정화(EMA)
const SMOOTH_ALPHA_POS = 0.22;
const SMOOTH_ALPHA_ROT = 0.22;
const LOST_GRACE_FRAMES = 20;

// (폴백) 사각 프레임 윤곽 검출 튜닝
const EXPECTED_ASPECT = 1.52;
const ASPECT_TOL = 0.38;
const MIN_QUAD_AREA = 12000;
const DETECT_EVERY_N_FRAMES = 2;

/* =========================
   STATE / UI
========================= */
const state = {
  facingMode: "environment",
  bgMode: "day",
  started: false,
  cvReady: false,
  debug: false,

  // aruco available?
  arucoReady: false,

  frameCount: 0,
  lostCount: 0,

  // smoothing
  smoothed: {
    pos: new THREE.Vector3(),
    quat: new THREE.Quaternion(),
    inited: false,
  },

  // last pose mats (for cleanup)
  lastPose: null, // { rvec: cv.Mat, tvec: cv.Mat }
};

const video = $("cam");
const cvStatus = $("cvStatus");
const trackStatus = $("trackStatus");
const fpsEl = $("fps");
const bgOverlay = $("bgOverlay");
const dbgCanvas = $("dbg");
const dbgCtx = dbgCanvas.getContext("2d");

/* =========================
   THREE
========================= */
let renderer, scene, camera;
let markerGroup, lobsterOffsetGroup, lobster;
let clock;

/* =========================
   OpenCV buffers
========================= */
let offCanvas, offCtx;
let matRGBA, matGray, matEq, matBlur, matEdges, matMorph;
let contours, hierarchy;

// intrinsics
let cameraMatrix = null;  // cv.Mat 3x3
let distCoeffs = null;    // cv.Mat 1x5
let projNeedsUpdate = true;

// aruco
let arucoDict = null;
let arucoParams = null;

/* =========================
   Helpers
========================= */
function setBg(mode){
  state.bgMode = mode;
  const path = mode === "day"
    ? "../assets/bg/underwater_bg_day.png"
    : "../assets/bg/underwater_bg_night.png";
  bgOverlay.style.backgroundImage = `url("${path}")`;
}

function setTracking(on, reason=""){
  trackStatus.textContent = on ? "Tracking: 인식중" : `Tracking: 대기${reason ? " ("+reason+")" : ""}`;
}

function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }

function emaVec3(out, target, a){
  out.x = out.x*(1-a) + target.x*a;
  out.y = out.y*(1-a) + target.y*a;
  out.z = out.z*(1-a) + target.z*a;
  return out;
}
function emaQuat(out, target, a){
  out.slerp(target, a);
  return out;
}

/* =========================
   Camera
========================= */
async function startCamera(){
  if(video.srcObject){
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: state.facingMode,
      width: { ideal: 1280 },
      height:{ ideal: 720 }
    },
    audio:false
  });
  video.srcObject = stream;
  await video.play();

  // 전면 카메라는 UX상 미러링이 자연스러움
  video.style.transform = (state.facingMode === "user") ? "scaleX(-1)" : "scaleX(1)";
  projNeedsUpdate = true;
}

/* =========================
   THREE init
========================= */
function initThree(){
  renderer = new THREE.WebGLRenderer({ canvas: $("gl"), alpha:true, antialias:true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(ASSUMED_FOV_DEG, window.innerWidth/window.innerHeight, 0.01, 50);
  camera.position.set(0,0,0);
  camera.lookAt(0,0,-1);

  scene.add(new THREE.AmbientLight(0xffffff, 0.95));
  const dir = new THREE.DirectionalLight(0xffffff, 1.05);
  dir.position.set(1.2, 1.6, 1.2);
  scene.add(dir);

  // markerGroup: ArUco pose를 그대로 적용하는 그룹(카메라 좌표계)
  markerGroup = new THREE.Group();
  markerGroup.visible = false;
  markerGroup.matrixAutoUpdate = false;
  scene.add(markerGroup);

  // lobsterOffsetGroup: “카드 중심”으로 옮기기 위한 로컬 오프셋 그룹
  lobsterOffsetGroup = new THREE.Group();
  markerGroup.add(lobsterOffsetGroup);

  lobster = createLobster3D();
  lobsterOffsetGroup.add(lobster);

  clock = new THREE.Clock();
}

function onResize(){
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);

/* =========================
   Lobster (procedural placeholder)
========================= */
function createLobster3D(){
  const g = new THREE.Group();

  const matBody = new THREE.MeshStandardMaterial({ color: 0xff4a3d, roughness: 0.55, metalness: 0.05 });
  const matDark = new THREE.MeshStandardMaterial({ color: 0xd63a30, roughness: 0.6, metalness: 0.05 });
  const matEye  = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.2, metalness: 0.1 });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.08, 8, 16), matBody);
  body.rotation.z = Math.PI/2;
  g.add(body);

  const shell = new THREE.Mesh(new THREE.CapsuleGeometry(0.055, 0.07, 6, 16), matDark);
  shell.rotation.z = Math.PI/2;
  shell.position.z = 0.01;
  g.add(shell);

  const eyeL = new THREE.Mesh(new THREE.SphereGeometry(0.012, 14, 14), matEye);
  const eyeR = eyeL.clone();
  eyeL.position.set(-0.045, 0.03, 0.04);
  eyeR.position.set(-0.045,-0.03, 0.04);
  g.add(eyeL, eyeR);

  const antMat = new THREE.MeshStandardMaterial({ color: 0xffb0a8, roughness: 0.6, metalness: 0.0 });
  const antenna1 = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.006, 0.16, 10), antMat);
  const antenna2 = antenna1.clone();
  antenna1.position.set(-0.09, 0.05, 0.04);
  antenna2.position.set(-0.09,-0.05, 0.04);
  antenna1.rotation.z = 0.25;
  antenna2.rotation.z = -0.25;
  g.add(antenna1, antenna2);

  const tail = new THREE.Group();
  tail.position.set(0.09, 0, 0);
  g.add(tail);

  for(let i=0;i<4;i++){
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.045 - i*0.006, 0.052 - i*0.006, 0.03, 16), matBody);
    seg.rotation.z = Math.PI/2;
    seg.position.x = i*0.03;
    tail.add(seg);
  }
  const fan = new THREE.Mesh(new THREE.CircleGeometry(0.045, 18), matDark);
  fan.position.set(0.13, 0, 0);
  fan.rotation.y = Math.PI/2;
  tail.add(fan);

  const clawL = createClaw(matBody, matDark);
  const clawR = createClaw(matBody, matDark);
  clawL.position.set(-0.01, 0.09, 0);
  clawR.position.set(-0.01,-0.09, 0);
  clawL.rotation.z = 0.25;
  clawR.rotation.z = -0.25;
  g.add(clawL, clawR);

  g.userData = { tail, clawL, clawR, antenna1, antenna2 };
  return g;
}

function createClaw(matBody, matDark){
  const claw = new THREE.Group();
  const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.016, 0.09, 14), matBody);
  arm.rotation.z = Math.PI/2;
  claw.add(arm);

  const base = new THREE.Mesh(new THREE.SphereGeometry(0.02, 14, 14), matDark);
  base.position.set(0.04, 0, 0);
  claw.add(base);

  const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.016, 0.07, 14), matBody);
  const lower = upper.clone();
  upper.position.set(0.07, 0.012, 0);
  lower.position.set(0.07,-0.012, 0);
  upper.rotation.z = Math.PI/2 + 0.35;
  lower.rotation.z = Math.PI/2 - 0.35;
  claw.add(upper, lower);

  claw.userData = { upper, lower };
  return claw;
}

/* =========================
   OpenCV init
========================= */
function initCVBuffers(){
  offCanvas = document.createElement("canvas");
  offCanvas.width = PROC_W;
  offCanvas.height = PROC_H;
  offCtx = offCanvas.getContext("2d", { willReadFrequently:true });

  dbgCanvas.width = PROC_W;
  dbgCanvas.height = PROC_H;

  matRGBA = new cv.Mat(PROC_H, PROC_W, cv.CV_8UC4);
  matGray = new cv.Mat(PROC_H, PROC_W, cv.CV_8UC1);
  matEq   = new cv.Mat(PROC_H, PROC_W, cv.CV_8UC1);
  matBlur = new cv.Mat(PROC_H, PROC_W, cv.CV_8UC1);
  matEdges= new cv.Mat(PROC_H, PROC_W, cv.CV_8UC1);
  matMorph= new cv.Mat(PROC_H, PROC_W, cv.CV_8UC1);

  contours = new cv.MatVector();
  hierarchy = new cv.Mat();

  // ArUco 준비(가능한 빌드에서만)
  state.arucoReady = !!(cv.aruco && cv.aruco.Dictionary_get && cv.aruco.DetectorParameters_create);
  if(state.arucoReady){
    arucoDict = cv.aruco.Dictionary_get(cv.aruco[ARUCO_DICT] ?? cv.aruco.DICT_4X4_50);
    arucoParams = cv.aruco.DetectorParameters_create();
    // 튜닝(오탐/미탐 줄이기)
    arucoParams.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX;
    arucoParams.adaptiveThreshWinSizeMin = 3;
    arucoParams.adaptiveThreshWinSizeMax = 23;
    arucoParams.adaptiveThreshWinSizeStep = 10;
    arucoParams.minCornerDistanceRate = 0.05;
    arucoParams.minMarkerDistanceRate = 0.05;
    arucoParams.polygonalApproxAccuracyRate = 0.03;
    arucoParams.minOtsuStdDev = 5.0;
    arucoParams.perspectiveRemovePixelPerCell = 8;
    arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13;

    cvStatus.textContent = "OpenCV: 준비완료 (ArUco ON)";
  }else{
    cvStatus.textContent = "OpenCV: 준비완료 (ArUco OFF → 폴백)";
  }
}

function buildIntrinsicsFor(videoW, videoH){
  const fov = ASSUMED_FOV_DEG * Math.PI/180;
  const fx = videoW / (2 * Math.tan(fov/2));
  const fy = fx;
  const cx = videoW * 0.5;
  const cy = videoH * 0.5;

  cameraMatrix = cv.matFromArray(3,3, cv.CV_64F, [
    fx, 0,  cx,
    0,  fy, cy,
    0,  0,  1
  ]);
  distCoeffs = cv.matFromArray(1,5, cv.CV_64F, [0,0,0,0,0]);

  updateThreeProjectionFromIntrinsics(fx, fy, cx, cy, videoW, videoH);
}

function updateThreeProjectionFromIntrinsics(fx, fy, cx, cy, w, h){
  const near = camera.near;
  const far  = camera.far;

  const m00 = 2*fx/w;
  const m11 = 2*fy/h;
  const m02 = 1 - 2*cx/w;
  const m12 = 2*cy/h - 1;
  const m22 = -(far+near)/(far-near);
  const m23 = -(2*far*near)/(far-near);

  const P = new THREE.Matrix4();
  P.set(
    m00, 0,   m02, 0,
    0,   m11, m12, 0,
    0,   0,   m22, m23,
    0,   0,   -1,  0
  );
  camera.projectionMatrix.copy(P);
  camera.projectionMatrixInverse.copy(P).invert();
}

/* =========================
   PREPROCESS
========================= */
function preprocessGray(){
  offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
  const imgData = offCtx.getImageData(0, 0, PROC_W, PROC_H);
  matRGBA.data.set(imgData.data);

  cv.cvtColor(matRGBA, matGray, cv.COLOR_RGBA2GRAY, 0);

  // CLAHE로 명암 보정 (반사/그림자 완화)
  const clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
  clahe.apply(matGray, matEq);
  clahe.delete();

  cv.GaussianBlur(matEq, matBlur, new cv.Size(5,5), 0, 0, cv.BORDER_DEFAULT);

  if(state.debug){
    const show = new ImageData(new Uint8ClampedArray(matBlur.data), PROC_W, PROC_H);
    dbgCtx.putImageData(show, 0, 0);
  }
}

/* =========================
   ARUCO DETECT + POSE
========================= */
function detectArucoPose(){
  preprocessGray();

  const corners = new cv.MatVector();
  const ids = new cv.Mat();
  const rejected = new cv.MatVector();

  cv.aruco.detectMarkers(matBlur, arucoDict, corners, ids, arucoParams, rejected);

  if(state.debug){
    // draw detected markers on debug canvas
    const dbg = new cv.Mat(PROC_H, PROC_W, cv.CV_8UC3);
    cv.cvtColor(matBlur, dbg, cv.COLOR_GRAY2RGB);
    if(ids.rows > 0){
      cv.aruco.drawDetectedMarkers(dbg, corners, ids);
    }
    const img = new ImageData(new Uint8ClampedArray(dbg.data), PROC_W, PROC_H);
    dbgCtx.putImageData(img, 0, 0);
    dbg.delete();
  }

  if(ids.rows <= 0){
    corners.delete(); ids.delete(); rejected.delete();
    return null;
  }

  // 가장 큰 마커(화면에서 큰 것) 선택
  let bestIdx = 0;
  let bestArea = -1;

  for(let i=0;i<ids.rows;i++){
    const c = corners.get(i); // 1x4x2 float
    const p0 = {x: c.data32F[0], y: c.data32F[1]};
    const p1 = {x: c.data32F[2], y: c.data32F[3]};
    const p2 = {x: c.data32F[4], y: c.data32F[5]};
    const p3 = {x: c.data32F[6], y: c.data32F[7]};
    const area = Math.abs(
      (p0.x*p1.y - p1.x*p0.y) +
      (p1.x*p2.y - p2.x*p1.y) +
      (p2.x*p3.y - p3.x*p2.y) +
      (p3.x*p0.y - p0.x*p3.y)
    ) * 0.5;

    if(area > bestArea){
      bestArea = area;
      bestIdx = i;
    }
    c.delete();
  }

  // estimatePoseSingleMarkers는 모든 마커에 대해 rvec/tvec를 반환
  const rvecs = new cv.Mat();
  const tvecs = new cv.Mat();

  cv.aruco.estimatePoseSingleMarkers(
    corners,
    MARKER_LEN_M,
    cameraMatrix,
    distCoeffs,
    rvecs,
    tvecs
  );

  // id
  const id = ids.intAt(bestIdx, 0);

  // rvec/tvec 추출(bestIdx)
  const rvec = new cv.Mat(3,1, cv.CV_64F);
  const tvec = new cv.Mat(3,1, cv.CV_64F);

  // rvecs: Nx1x3, tvecs: Nx1x3
  rvec.doublePtr(0,0)[0] = rvecs.doubleAt(bestIdx, 0);
  rvec.doublePtr(1,0)[0] = rvecs.doubleAt(bestIdx, 1);
  rvec.doublePtr(2,0)[0] = rvecs.doubleAt(bestIdx, 2);

  tvec.doublePtr(0,0)[0] = tvecs.doubleAt(bestIdx, 0);
  tvec.doublePtr(1,0)[0] = tvecs.doubleAt(bestIdx, 1);
  tvec.doublePtr(2,0)[0] = tvecs.doubleAt(bestIdx, 2);

  // cleanup
  corners.delete(); ids.delete(); rejected.delete();
  rvecs.delete(); tvecs.delete();

  return { id, rvec, tvec };
}

/* =========================
   Pose -> Three (marker pose)
========================= */
function applyMarkerPoseToThree(pose){
  // 이전 pose 메모리 정리
  if(state.lastPose){
    state.lastPose.rvec.delete();
    state.lastPose.tvec.delete();
    state.lastPose = null;
  }
  state.lastPose = pose;

  // Rodrigues
  const R = new cv.Mat();
  cv.Rodrigues(pose.rvec, R);

  // OpenCV: x right, y down, z forward
  // Three:  x right, y up,   z backward
  // 변환: (x, y, z) -> (x, -y, -z)
  const r00 = R.doubleAt(0,0), r01 = R.doubleAt(0,1), r02 = R.doubleAt(0,2);
  const r10 = R.doubleAt(1,0), r11 = R.doubleAt(1,1), r12 = R.doubleAt(1,2);
  const r20 = R.doubleAt(2,0), r21 = R.doubleAt(2,1), r22 = R.doubleAt(2,2);

  const tx = pose.tvec.doubleAt(0,0);
  const ty = pose.tvec.doubleAt(1,0);
  const tz = pose.tvec.doubleAt(2,0);

  R.delete();

  // marker->camera pose matrix (Three)
  // 축부호 반영
  const m = new THREE.Matrix4();
  m.set(
    r00, -r01, -r02,  tx,
   -r10,  r11,  r12, -ty,
   -r20,  r21,  r22, -tz,
    0,    0,    0,    1
  );

  // decompose
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  m.decompose(pos, quat, scl);

  // smoothing
  if(!state.smoothed.inited){
    state.smoothed.pos.copy(pos);
    state.smoothed.quat.copy(quat);
    state.smoothed.inited = true;
  }else{
    emaVec3(state.smoothed.pos, pos, SMOOTH_ALPHA_POS);
    emaQuat(state.smoothed.quat, quat, SMOOTH_ALPHA_ROT);
  }

  // markerGroup에 matrix로 반영
  markerGroup.matrix.identity();
  markerGroup.position.copy(state.smoothed.pos);
  markerGroup.quaternion.copy(state.smoothed.quat);
  markerGroup.scale.set(1,1,1);
  markerGroup.updateMatrix();
  markerGroup.visible = true;

  // “카드 중심” 오프셋(마커 ID에 따라)
  applyCardCenterOffset(pose.id);
}

/* =========================
   Marker ID -> Card Center Offset
   (marker 좌표계에서 card 중심으로 이동)
========================= */
function applyCardCenterOffset(markerId){
  // 기본: 마커 중심에 바닷가재를 띄움(마커 1개만 있어도 안정)
  let ox = 0, oy = 0;

  // 코너 마커를 카드 코너에 붙였다는 가정 하에, 카드 중심까지 오프셋
  // marker 좌표: x right, y down(OpenCV). Three에선 y up이므로 여기서는 “로컬 이동”으로만 사용
  // 아래 값은 “마커 중심이 코너에서 margin + marker/2 만큼 안쪽”에 있다고 가정한 중심 오프셋.
  const mx = (CARD_W/2) - (CORNER_MARGIN + MARKER_LEN_M/2);
  const my = (CARD_H/2) - (CORNER_MARGIN + MARKER_LEN_M/2);

  if(markerId === CORNER_IDS.TL){
    // TL 마커: 카드 중심은 +x, +y(아래) 방향
    ox = +mx; oy = +my;
  }else if(markerId === CORNER_IDS.TR){
    // TR 마커: 카드 중심은 -x, +y
    ox = -mx; oy = +my;
  }else if(markerId === CORNER_IDS.BR){
    // BR 마커: 카드 중심은 -x, -y(위)
    ox = -mx; oy = -my;
  }else if(markerId === CORNER_IDS.BL){
    // BL 마커: 카드 중심은 +x, -y
    ox = +mx; oy = -my;
  }

  // OpenCV y down -> Three y up : oy 부호 반전
  lobsterOffsetGroup.position.set(ox, -oy, 0);

  // 카드 위로 살짝 띄우기(법선방향 +z, Three에선 -z가 카메라 앞쪽이지만 marker pose 변환에서 이미 맞춰둠)
  lobsterOffsetGroup.position.z = 0.02;

  // 모델 스케일(원하면 여기서 조정)
  lobsterOffsetGroup.scale.set(1.0, 1.0, 1.0);
}

/* =========================
   Fallback: quad detect (기존)
========================= */
function autoCanny(srcGray, dstEdges){
  const data = srcGray.data;
  let vals = [];
  const step = Math.max(1, Math.floor(data.length / 8000));
  for(let i=0;i<data.length;i+=step) vals.push(data[i]);
  vals.sort((a,b)=>a-b);
  const med = vals[Math.floor(vals.length*0.5)] || 127;

  const lower = clamp(0.66*med, 20, 120);
  const upper = clamp(1.33*med, 80, 220);
  cv.Canny(srcGray, dstEdges, lower, upper);
}

function preprocessForQuad(){
  preprocessGray();

  autoCanny(matBlur, matEdges);

  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5,5));
  cv.morphologyEx(matEdges, matMorph, cv.MORPH_CLOSE, k);
  cv.morphologyEx(matMorph, matMorph, cv.MORPH_OPEN, k);
  k.delete();

  if(state.debug){
    const show = new ImageData(new Uint8ClampedArray(matMorph.data), PROC_W, PROC_H);
    dbgCtx.putImageData(show, 0, 0);
  }
}

function findBestQuad(){
  cv.findContours(matMorph, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  let best = null;
  let bestScore = -1;

  for(let i=0; i<contours.size(); i++){
    const cnt = contours.get(i);
    const area = cv.contourArea(cnt);
    if(area < MIN_QUAD_AREA){ cnt.delete(); continue; }

    const peri = cv.arcLength(cnt, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, 0.02 * peri, true);

    if(approx.rows === 4 && cv.isContourConvex(approx)){
      const pts = [];
      for(let k=0;k<4;k++){
        const x = approx.intPtr(k,0)[0];
        const y = approx.intPtr(k,0)[1];
        pts.push({x,y});
      }
      const ordered = orderCorners(pts);
      const sc = scoreQuad(ordered);
      if(sc > bestScore){
        bestScore = sc;
        best = ordered;
      }
    }

    approx.delete();
    cnt.delete();
  }

  return { quad: best, score: bestScore };
}

function orderCorners(pts){
  const sum = pts.map(p => p.x + p.y);
  const diff= pts.map(p => p.x - p.y);
  const tl = pts[sum.indexOf(Math.min(...sum))];
  const br = pts[sum.indexOf(Math.max(...sum))];
  const tr = pts[diff.indexOf(Math.min(...diff))];
  const bl = pts[diff.indexOf(Math.max(...diff))];
  return [tl, tr, br, bl];
}

function polygonArea(p){
  let a = 0;
  for(let i=0;i<p.length;i++){
    const j = (i+1)%p.length;
    a += p[i].x*p[j].y - p[j].x*p[i].y;
  }
  return Math.abs(a)*0.5;
}

function boundingRect(p){
  let minX=1e9,minY=1e9,maxX=-1e9,maxY=-1e9;
  for(const q of p){
    minX = Math.min(minX, q.x);
    minY = Math.min(minY, q.y);
    maxX = Math.max(maxX, q.x);
    maxY = Math.max(maxY, q.y);
  }
  return { x:minX, y:minY, w:(maxX-minX), h:(maxY-minY) };
}

function rightAngleScore(p){
  let total = 0;
  for(let i=0;i<4;i++){
    const a = p[(i+3)%4];
    const b = p[i];
    const c = p[(i+1)%4];
    const v1 = { x: a.x-b.x, y:a.y-b.y };
    const v2 = { x: c.x-b.x, y:c.y-b.y };
    const n1 = Math.hypot(v1.x,v1.y) + 1e-6;
    const n2 = Math.hypot(v2.x,v2.y) + 1e-6;
    const cos = (v1.x*v2.x + v1.y*v2.y) / (n1*n2);
    const sc = 1 - clamp(Math.abs(cos)/0.35, 0, 1);
    total += sc;
  }
  return total/4;
}

function scoreQuad(quad){
  const area = polygonArea(quad);
  if(area < MIN_QUAD_AREA) return -1;

  const rect = boundingRect(quad);
  const rectArea = rect.w * rect.h;
  const rectangularity = clamp(area / (rectArea + 1e-6), 0, 1);

  const aspect = rect.w / (rect.h + 1e-6);
  const aspectScore = 1 - clamp(Math.abs(aspect - EXPECTED_ASPECT) / ASPECT_TOL, 0, 1);

  const angScore = rightAngleScore(quad);
  const sizeScore = clamp(area / (PROC_W*PROC_H*0.55), 0, 1);

  return (0.30*rectangularity) + (0.30*angScore) + (0.25*aspectScore) + (0.15*sizeScore);
}

/* =========================
   Main Loop
========================= */
function animate(){
  requestAnimationFrame(animate);

  const dt = clock.getDelta();
  const t = clock.elapsedTime;

  // FPS
  const fps = Math.round(1/Math.max(dt, 1e-6));
  if(state.frameCount % 15 === 0) fpsEl.textContent = `FPS: ${fps}`;

  // Lobster idle anim
  if(lobster){
    const u = lobster.userData;
    if(u.tail) u.tail.rotation.z = Math.sin(t*3.2) * 0.18;

    for(const claw of [u.clawL, u.clawR]){
      if(claw && claw.userData){
        const open = (Math.sin(t*2.6) * 0.5 + 0.5) * 0.35 + 0.10;
        claw.userData.upper.rotation.z = Math.PI/2 + open;
        claw.userData.lower.rotation.z = Math.PI/2 - open;
      }
    }

    if(u.antenna1) u.antenna1.rotation.y = Math.sin(t*4.0) * 0.25;
    if(u.antenna2) u.antenna2.rotation.y = -Math.sin(t*4.0) * 0.25;

    lobster.position.z = 0.01 + Math.sin(t*1.4)*0.01;
  }

  if(state.cvReady && state.started && video.readyState >= 2){
    // intrinsics (1회)
    if(projNeedsUpdate && video.videoWidth && video.videoHeight){
      buildIntrinsicsFor(video.videoWidth, video.videoHeight);
      projNeedsUpdate = false;
    }

    const doDetect = (state.frameCount % DETECT_EVERY_N_FRAMES === 0);

    let poseFound = false;

    // 1) ArUco 우선
    if(state.arucoReady && doDetect){
      const pose = detectArucoPose();
      if(pose){
        applyMarkerPoseToThree(pose);
        poseFound = true;
      }
    }

    // 2) 폴백: 윤곽 기반(ArUco 불가/미검출)
    if(!poseFound && doDetect){
      preprocessForQuad();
      const { quad, score } = findBestQuad();
      const ok = quad && score >= 0.42;

      if(ok){
        // 윤곽 폴백에서는 “기울기/깊이”까지 완벽하지 않으므로,
        // markerGroup 대신 화면 중앙 기반 간이 배치(기존 버전 유지 목적)
        // (ArUco가 켜졌다면 대부분 여기로 안 옵니다.)
        markerGroup.visible = true;
        markerGroup.matrix.identity();
        markerGroup.position.set(0, 0, -0.6);
        markerGroup.quaternion.identity();
        markerGroup.scale.set(1,1,1);
        markerGroup.updateMatrix();

        lobsterOffsetGroup.position.set(0,0,0.02);
        poseFound = true;
      }
    }

    if(poseFound){
      state.lostCount = 0;
      setTracking(true);
    }else{
      state.lostCount++;
      if(state.lostCount > LOST_GRACE_FRAMES){
        markerGroup.visible = false;
        state.smoothed.inited = false;
        setTracking(false, "미검출");
      }
    }
  }

  renderer.render(scene, camera);
  state.frameCount++;
}

/* =========================
   UI
========================= */
$("btnStart").addEventListener("click", async () => {
  if(state.started) return;
  state.started = true;
  await startCamera();
});

$("btnFlip").addEventListener("click", async () => {
  state.facingMode = (state.facingMode === "environment") ? "user" : "environment";
  projNeedsUpdate = true;
  await startCamera();
});

$("btnBg").addEventListener("click", () => {
  setBg(state.bgMode === "day" ? "night" : "day");
});

$("btnDbg").addEventListener("click", () => {
  state.debug = !state.debug;
  dbgCanvas.style.display = state.debug ? "block" : "none";
});

/* =========================
   Boot
========================= */
function boot(){
  setBg("day");
  initThree();
  setTracking(false);

  const cvWait = setInterval(() => {
    const ready = (window.__cvReady === true) && (typeof cv !== "undefined") && cv.Mat;
    if(ready){
      clearInterval(cvWait);
      state.cvReady = true;
      initCVBuffers();
    }
  }, 200);

  animate();
}

boot();
