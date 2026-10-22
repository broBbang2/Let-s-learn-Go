// 착수·따냄·불법수의 "손맛"을 한곳에 모은다.
//
// 지금(웹): 실제 바둑돌 녹음(public/sounds, KaTrain MIT) + 안드로이드 브라우저면 진동.
//          녹음을 아직 못 불러왔으면 짧은 잡음 충격음으로 대신한다.
// 나중(앱): React Native로 옮길 때 이 함수들 안에 햅틱 한 줄씩만 추가하면 된다.
// 화면 애니메이션은 GoBoard가 돌의 등장·퇴장으로 처리하고, 여기서는 감각 신호만 낸다.
//
// 바둑돌 소리는 음정 없는 "탁"이다. 사인파 공명을 쓰면 "띵·띠링"처럼 들리므로 쓰지 않는다.

const STORAGE_KEY = "adaptive-go:sound";

let ctx: AudioContext | null = null;
let enabled = readEnabled();

function readEnabled(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) !== "off";
  } catch {
    return true;
  }
}

export function soundEnabled() {
  return enabled;
}

export function setSoundEnabled(on: boolean) {
  enabled = on;
  try {
    localStorage.setItem(STORAGE_KEY, on ? "on" : "off");
  } catch {
    /* 저장 못 해도 이번 방문 동안은 유지 */
  }
}

function audio(): AudioContext | null {
  if (!enabled || typeof AudioContext === "undefined") return null;
  ctx ??= new AudioContext();
  if (ctx.state === "suspended") void ctx.resume();
  decodeSamples(ctx);
  return ctx;
}

const jitter = (v: number, r: number) => v * (1 + (Math.random() * 2 - 1) * r);

// ---------- 녹음 ----------

const STONE_FILES = ["stone1", "stone2", "stone3", "stone4", "stone5"];
const stones: AudioBuffer[] = [];
let decoding = false;

// 파일은 페이지를 열자마자 받아 두고(소리 허용 전에도 가능), 소리가 허용되면 바로 디코딩한다
const raw: Promise<ArrayBuffer | null>[] =
  typeof fetch === "undefined"
    ? []
    : STONE_FILES.map((name) =>
        fetch(`/sounds/${name}.wav`)
          .then((r) => (r.ok ? r.arrayBuffer() : null))
          .catch(() => null),
      );

function decodeSamples(a: AudioContext) {
  if (decoding) return;
  decoding = true;
  for (const p of raw) {
    p.then((buf) => buf && a.decodeAudioData(buf))
      .then((b) => b && stones.push(b))
      .catch(() => {
        /* 못 불러오면 대체음을 계속 쓴다 */
      });
  }
}

// 브라우저는 사용자가 한 번 누른 뒤에야 소리를 허용하므로, 첫 클릭 때 디코딩해 둔다 (첫 착수부터 녹음으로)
if (typeof window !== "undefined") window.addEventListener("pointerdown", () => audio(), { once: true });

let lastStone = -1;

/** 착수 녹음 하나를 재생. rate는 재생 속도(높을수록 작고 가벼운 소리) */
function playStone(at: number, gain: number, rate = 1): boolean {
  const a = audio();
  if (!a || !stones.length) return false;
  // 같은 녹음이 연달아 나오지 않게
  let i = Math.floor(Math.random() * stones.length);
  if (stones.length > 1 && i === lastStone) i = (i + 1) % stones.length;
  lastStone = i;
  const src = a.createBufferSource();
  src.buffer = stones[i];
  src.playbackRate.value = jitter(rate, 0.03);
  const g = a.createGain();
  g.gain.value = jitter(gain, 0.08);
  src.connect(g).connect(a.destination);
  src.start(a.currentTime + at);
  return true;
}

// ---------- 대체음: 음정 없는 짧은 잡음 충격 ----------

/** 대역을 거른 아주 짧은 잡음 하나 (len초, center Hz 근처) */
function click(at: number, gain: number, center: number, len: number) {
  const a = audio();
  if (!a) return;
  const buf = a.createBuffer(1, Math.ceil(a.sampleRate * len), a.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 4);
  const src = a.createBufferSource();
  src.buffer = buf;
  const band = a.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = center;
  band.Q.value = 0.7;
  const g = a.createGain();
  g.gain.value = gain;
  src.connect(band).connect(g).connect(a.destination);
  src.start(a.currentTime + at);
}

function vibrate(pattern: number | number[]) {
  // 데스크톱·iOS 사파리는 지원하지 않는다 (호출해도 무시됨)
  try {
    navigator.vibrate?.(pattern);
  } catch {
    /* 무시 */
  }
}

export const feedback = {
  place() {
    if (!playStone(0, 0.9)) {
      click(0, 0.9, 3500, 0.008);
      click(0, 0.4, 1200, 0.012);
    }
    vibrate(10);
  },
  /** n점을 따냈을 때: 돌이 튕겨 나가는 연출과 맞춰, 집어 든 돌끼리 부딪치는 "탁탁" (착수음을 작고 가볍게) */
  capture(n: number) {
    const clicks = Math.min(n, 5);
    for (let i = 0; i < clicks; i++) {
      const at = 0.09 + i * 0.05 + Math.random() * 0.02;
      if (!playStone(at, 0.45 - i * 0.05, 1.35)) click(at, 0.5 - i * 0.05, 4500, 0.005);
    }
    vibrate(n > 1 ? [12, 40, 12] : 15);
  },
  /** 둘 수 없는 자리: 손가락으로 판을 두드리듯 낮고 둔한 "톡" */
  illegal() {
    click(0, 0.6, 700, 0.015);
    vibrate([8, 30, 8]);
  },
};
