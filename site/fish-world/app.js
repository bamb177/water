import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const $ = (id) => document.getElementById(id);

/* =========================
   Config (필요시 조정)
========================= */
// 프레임 기대 비율(내부 큰 사각 프레임 기준). 모르면 1.45~1.60 범위가 무난합니다.
const EXPECTED_ASPECT = 1.52;      // width/height
const ASPECT_TOL = 0.38;           // 허용오차(클수록 관대)
const MIN_QUAD_AREA = 12000;       // downscale 기준 최소 면적
const DETECT_EVERY_N_FRAMES = 2;   // 1이면 매프레임 검출(무거움)
const LOST_GRACE_FRAMES = 18;      // 잠깐 놓쳐도 유지하는 프레임 수
const SMOOTH_ALPHA = 0.22;         // 포즈 EMA 스무딩(0~1, 높을수록 빠르게 반응)

// solvePnP용 “가상의 카드 크기”(단위: meter). 실제값이 달라도 “기울기/회전”은 충분히 자연스럽게 나옵니다.
const CARD_W = 0.20;               // 20cm
const CARD_H = CARD_W / EXPECTED_ASPECT;

// 카메라 FOV 가정(디바이스별 다름). 너무 작으면 깊이가 과장되고, 너무 크면 납작해 보입니다.
const ASSUMED_FOV_DEG = 60;

// OpenCV 처리 해상도(낮출수록 빠르지만 정확도 감소)
const PROC_W = 640;
const PROC_H = 360;

/* =========================
   State / UI
========================= */
const state = {
  facingMode: "environment",
  bgMode: "day",
  started: false,
  cvReady: false,
  debug: false,

  // tracking
  tracking: false,
  lostCount: 0,
  frameCount: 0,

  // pose smoothing cache
  pose: null,          // { rvec, tvec } (cv.Mat)
  smoothed: null,      // { pos: THREE.Vector3, quat: THREE.Quaternion }
};

const video = $("cam");
const cvStatus = $("cvStatus");
const trackStatus = $("trackStatus");
const fpsEl = $("fps");
const bgOverlay = $("bgOverlay");
const dbgCanvas = $("dbg");
const dbgCtx = dbgCanvas.getContext("2d");

/* =========================
   Three.js
========================= */
let renderer, scene, camera;
let lobsterRoot, lobster;
let clock;

/* =========================
   OpenCV buffers
========================= */
let offCanvas, offCtx;
let matRGBA, matGray, matEq, matBlur, matEdges, matMorph;
let contours, hierarchy;

let cameraMatrix = null;  // cv.Mat 3x3
let distCoeffs = null;    // cv.Mat 1x5 (0)
let projNeedsUpdate = true;

/* =========================
   Helpers
========================= */
function setTrackLabel(on, reason=""){
  state.tracking = on;
  trackStatus.textContent = on ? "Tracking: 인식중" : `Tracking: 대기${reason ? " ("+reason+")" : ""}`;
}

function setBg(mode){
  state.bgMode = mode;
  const path = mode === "day"
    ? "../assets/bg/underwater_bg_day.png"
    : "../assets/bg/underwater_bg_night.png";
  bgOverlay.style.backgroundImage = `url("${path}")`;
}

function clamp(v,min,max){ return Math.max(min, Math.min(max, v)); }

function emaVec3(out, target, a){
  out.x = out.x*(1-a) + target.x*a;
  out.y = out.y*(1-a) + target.y*a;
  out.z = out.z*(1-a) + target.z*a;
  return out;
}
function emaQuat(out, target, a){
  // slerp
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

  // front camera: mirror for natural UX
  video.style.transform = (state.facingMode === "user") ? "scaleX(-1)" : "scaleX(1)";

  projNeedsUpdate = true;
}

/* =========================
   Three init
========================= */
function initThree(){
  const gl = $("gl");
  renderer = new THREE.WebGLRenderer({ canvas: gl, alpha:true, antialias:true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(ASSUMED_FOV_DEG, window.innerWidth/window.innerHeight, 0.01, 50);
  camera.position.set(0,0,0);
  camera.lookAt(0,0,-1);

  const amb = new THREE.AmbientLight(0xffffff, 0.95);
  scene.add(amb);

  const dir = new THREE.DirectionalLight(0xffffff, 1.05);
  dir.position.set(1.2, 1.6, 1.2);
  scene.add(dir);

  lobsterRoot = new THREE.Group();
  lobsterRoot.visible = false;
  scene.add(lobsterRoot);

  lobster = createLobster3D();
  lobsterRoot.add(lobster);

  clock = new THREE.Clock();
}

function onResize(){
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);

/* =========================
   Lobster (procedural)
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

  const legMat = new THREE.MeshStandardMaterial({ color: 0xff6b60, roughness: 0.65, metalness: 0.0 });
  for(let s of [-1,1]){
    for(let i=0;i<3;i++){
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.006, 0.08, 10), legMat);
      leg.position.set(0.01 + i*0.025, 0.06*s, -0.02);
      leg.rotation.z = (0.55 + i*0.12) * s;
      g.add(leg);
    }
  }

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
}

function buildIntrinsicsFor(videoW, videoH){
  // fx = w/(2*tan(fov/2))
  const fov = ASSUMED_FOV_DEG * Math.PI/180;
  const fx = videoW / (2 * Math.tan(fov/2));
  const fy = fx; // 가정
  const cx = videoW * 0.5;
  const cy = videoH * 0.5;

  cameraMatrix = cv.matFromArray(3,3, cv.CV_64F, [
    fx, 0,  cx,
    0,  fy, cy,
    0,  0,  1
  ]);
  distCoeffs = cv.matFromArray(1,5, cv.CV_64F, [0,0,0,0,0]);

  // Three camera projection matrix를 intrinsics 기반으로 업데이트
  updateThreeProjectionFromIntrinsics(fx, fy, cx, cy, videoW, videoH);
}

function updateThreeProjectionFromIntrinsics(fx, fy, cx, cy, w, h){
  const near = camera.near;
  const far  = camera.far;

  // OpenCV(이미지 좌표) → WebGL(clip space) 매핑
  // 참고식:
  // [ 2fx/w,   0,    1 - 2cx/w,   0 ]
  // [   0,   2fy/h,  2cy/h - 1,   0 ]
  // [   0,     0,  -(f+n)/(f-n), -2fn/(f-n) ]
  // [   0,     0,           -1,   0 ]
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
   Detection tuning
========================= */
function autoCanny(srcGray, dstEdges){
  // median-based thresholds
  const data = srcGray.data;
  let vals = [];
  // 샘플링해서 median 근사(속도)
  const step = Math.max(1, Math.floor(data.length / 8000));
  for(let i=0;i<data.length;i+=step) vals.push(data[i]);
  vals.sort((a,b)=>a-b);
  const med = vals[Math.floor(vals.length*0.5)] || 127;

  const lower = clamp(0.66*med, 20, 120);
  const upper = clamp(1.33*med, 80, 220);
  cv.Canny(srcGray, dstEdges, lower, upper);
}

function preprocess(){
  offCtx.drawImage(video, 0, 0, PROC_W, PROC_H);
  const imgData = offCtx.getImageData(0, 0, PROC_W, PROC_H);
  matRGBA.data.set(imgData.data);

  cv.cvtColor(matRGBA, matGray, cv.COLOR_RGBA2GRAY, 0);

  // CLAHE로 명암 보정(반사/그림자 완화)
  const clahe = new cv.CLAHE(2.0, new cv.Size(8,8));
  clahe.apply(matGray, matEq);
  clahe.delete();

  // Blur(노이즈 억제)
  cv.GaussianBlur(matEq, matBlur, new cv.Size(5,5), 0, 0, cv.BORDER_DEFAULT);

  // edges
  autoCanny(matBlur, matEdges);

  // Morph close -> 선 연결, open -> 잡티 제거
  const k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5,5));
  cv.morphologyEx(matEdges, matMorph, cv.MORPH_CLOSE, k);
  cv.morphologyEx(matMorph, matMorph, cv.MORPH_OPEN, k);
  k.delete();

  if(state.debug){
    // debug draw edges
    const show = new ImageData(new Uint8ClampedArray(matMorph.data), PROC_W, PROC_H);
    dbgCtx.putImageData(show, 0, 0);
  }
}

function scoreQuad(quad){
  // quad: [{x,y}..4] in PROC space
  // 1) area
  const area = polygonArea(quad);
  if(area < MIN_QUAD_AREA) return -1;

  // 2) convexity + rectangularity + aspect
  const rect = boundingRect(quad);
  const rectArea = rect.w * rect.h;
  const rectangularity = clamp(area / (rectArea + 1e-6), 0, 1);

  const aspect = rect.w / (rect.h + 1e-6);
  const aspectScore = 1 - clamp(Math.abs(aspect - EXPECTED_ASPECT) / ASPECT_TOL, 0, 1);

  // 3) right-angle score
  const angScore = rightAngleScore(quad); // 0..1

  // 4) size score
  const sizeScore = clamp(area / (PROC_W*PROC_H*0.55), 0, 1);

  // 가중치
  const s = (0.30*rectangularity) + (0.30*angScore) + (0.25*aspectScore) + (0.15*sizeScore);
  return s;
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
  // tl: min(x+y), br: max(x+y), tr: min(x-y), bl: max(x-y)
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
  // 4개 각의 cos가 0에 가까우면 직각
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
    // 직각이면 cos≈0 -> score≈1
    const sc = 1 - clamp(Math.abs(cos)/0.35, 0, 1);
    total += sc;
  }
  return total/4;
}

/* =========================
   solvePnP → Three pose
========================= */
function computePoseFromQuad(quadProc){
  // quadProc is in PROC space. Convert to video space for intrinsics solvePnP.
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  // scale factors: PROC -> video
  const sx = vw / PROC_W;
  const sy = vh / PROC_H;

  // mirror compensation for user-facing camera (because video is CSS-mirrored)
  const mirrored = (state.facingMode === "user");

  const imgPts = quadProc.map(p => {
    let x = p.x * sx;
    let y = p.y * sy;
    if(mirrored){
      x = vw - x;
    }
    return {x,y};
  });

  // 2D points Mat (4x1x2)
  const imagePoints = cv.matFromArray(4,1, cv.CV_64FC2, [
    imgPts[0].x, imgPts[0].y,
    imgPts[1].x, imgPts[1].y,
    imgPts[2].x, imgPts[2].y,
    imgPts[3].x, imgPts[3].y
  ]);

  // 3D object points (z=0 plane). tl, tr, br, bl
  const objPts = [
    -CARD_W/2,  CARD_H/2, 0,
     CARD_W/2,  CARD_H/2, 0,
     CARD_W/2, -CARD_H/2, 0,
    -CARD_W/2, -CARD_H/2, 0,
  ];
  const objectPoints = cv.matFromArray(4,1, cv.CV_64FC3, objPts);

  const rvec = new cv.Mat();
  const tvec = new cv.Mat();

  // solvePnP: 안정적 옵션(SOLVEPNP_ITERATIVE)
  const ok = cv.solvePnP(
    objectPoints,
    imagePoints,
    cameraMatrix,
    distCoeffs,
    rvec,
    tvec,
    false,
    cv.SOLVEPNP_ITERATIVE
  );

  objectPoints.delete();
  imagePoints.delete();

  if(!ok){
    rvec.delete(); tvec.delete();
    return null;
  }
  return { rvec, tvec };
}

function applyPoseToThree(pose){
  // Rodrigues → R(3x3)
  const R = new cv.Mat();
  cv.Rodrigues(pose.rvec, R);

  // OpenCV: x right, y down, z forward
  // Three:  x right, y up,   z backward(카메라가 -Z 바라봄)
  // 변환 C = diag(1, -1, -1)
  const r00 = R.doubleAt(0,0), r01 = R.doubleAt(0,1), r02 = R.doubleAt(0,2);
  const r10 = R.doubleAt(1,0), r11 = R.doubleAt(1,1), r12 = R.doubleAt(1,2);
  const r20 = R.doubleAt(2,0), r21 = R.doubleAt(2,1), r22 = R.doubleAt(2,2);

  const tx = pose.tvec.doubleAt(0,0);
  const ty = pose.tvec.doubleAt(1,0);
  const tz = pose.tvec.doubleAt(2,0);

  R.delete();

  // Apply C on both R and t: (x, y, z) -> (x, -y, -z)
  const t3 = new THREE.Vector3(tx, -ty, -tz);

  // Build Three rotation matrix from converted R:
  // R' = C * R * C^-1  (C^-1=C). 여기서는 축 부호 반영을 간단히 적용:
  const m = new THREE.Matrix4();
  m.set(
    r00, -r01, -r02, t3.x,
   -r10,  r11,  r12, t3.y,
   -r20,  r21,  r22, t3.z,
    0,    0,    0,    1
  );

  // Decompose
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scl = new THREE.Vector3();
  m.decompose(pos, quat, scl);

  // Smoothing
  if(!state.smoothed){
    state.smoothed = {
      pos: pos.clone(),
      quat: quat.clone()
    };
  }else{
    emaVec3(state.smoothed.pos, pos, SMOOTH_ALPHA);
    emaQuat(state.smoothed.quat, quat, SMOOTH_ALPHA);
  }

  lobsterRoot.position.copy(state.smoothed.pos);
  lobsterRoot.quaternion.copy(state.smoothed.quat);

  // 실제 AR처럼 “카드 위에 올려놓는 느낌”을 위해 약간 띄움
  lobsterRoot.position.z += 0.02;

  // 모델 크기(카드 크기에 비례)
  const scale = 1.0; // 필요하면 0.8~1.3 조정
  lobsterRoot.scale.set(scale, scale, scale);

  lobsterRoot.visible = true;
}

/* =========================
   Main loop
========================= */
function animate(){
  requestAnimationFrame(animate);

  const dt = clock.getDelta();
  const t = clock.elapsedTime;

  // FPS 표시
  const fps = Math.round(1/Math.max(dt, 1e-6));
  if(state.frameCount % 15 === 0) fpsEl.textContent = `FPS: ${fps}`;

  // Lobster idle
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
    // 최초 1회 intrinsics 구성
    if(projNeedsUpdate && video.videoWidth && video.videoHeight){
      buildIntrinsicsFor(video.videoWidth, video.videoHeight);
      projNeedsUpdate = false;
    }

    // 검출 간격 조절
    const doDetect = (state.frameCount % DETECT_EVERY_N_FRAMES === 0);

    if(doDetect){
      preprocess();
      const { quad, score } = findBestQuad();

      // 점수 임계값: 너무 빡세면 놓치고, 너무 느슨하면 오탐
      const ok = quad && score >= 0.42;

      if(ok){
        const pose = computePoseFromQuad(quad);
        if(pose){
          // old pose free
          if(state.pose){
            state.pose.rvec.delete();
            state.pose.tvec.delete();
          }
          state.pose = pose;

          applyPoseToThree(pose);

          state.lostCount = 0;
          setTrackLabel(true);
        }else{
          state.lostCount++;
        }
      }else{
        state.lostCount++;
      }
    }else{
      // 검출 사이 프레임은 마지막 포즈 유지(시각적으로 훨씬 안정)
      if(state.pose){
        applyPoseToThree(state.pose);
      }
    }

    if(state.lostCount > LOST_GRACE_FRAMES){
      lobsterRoot.visible = false;
      state.smoothed = null;
      if(state.pose){
        state.pose.rvec.delete();
        state.pose.tvec.delete();
        state.pose = null;
      }
      setTrackLabel(false, "카드 미검출");
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
  setTrackLabel(false);

  // OpenCV ready wait
  const cvWait = setInterval(() => {
    const ready = (window.__cvReady === true) && (typeof cv !== "undefined") && cv.Mat;
    if(ready){
      clearInterval(cvWait);
      state.cvReady = true;
      cvStatus.textContent = "OpenCV: 준비완료";
      initCVBuffers();
    }
  }, 200);

  animate();
}

boot();
