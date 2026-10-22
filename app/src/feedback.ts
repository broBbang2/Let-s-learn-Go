// 착수·따냄·불법수의 "손맛"을 한곳에 모은다.
//
// 지금(웹): 짧은 소리(Web Audio로 합성, 파일 없음) + 안드로이드 브라우저면 진동
// 나중(앱): React Native로 옮길 때 이 함수들 안에 햅틱 한 줄씩만 추가하면 된다.
// 화면 애니메이션은 GoBoard가 돌의 등장·퇴장으로 처리하고, 여기서는 감각 신호만 낸다.

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
  return ctx;
}

// ---------- 바둑돌 소리 (모달 합성) ----------
//
// 실제 착수음은 세 겹이다.
// 1. 돌이 판에 닿는 순간의 날카로운 "딱": 돌(조개·슬레이트)의 고음 공명, 20ms 안에 사라짐
// 2. 나무판 몸통의 울림: 비자나무 판의 저·중음 공명 여러 개가 100~200ms 동안 감쇠 → "나무 판" 느낌
// 3. 돌이 자리 잡으며 한 번 더 나는 작은 딱 (30~55ms 뒤)
// 공명마다 감쇠하는 사인파 하나씩(모드)을 만들어 합친다. 매번 주파수·세기를 조금씩 흔들어 같은 소리가 반복되지 않게.

/** 판 몸통 공명 [주파수 Hz, 세기, 감쇠 시간 s] — 두꺼운 원목 판의 비조화 모드 */
const BOARD_MODES: [number, number, number][] = [
  [185, 1.0, 0.17],
  [403, 0.62, 0.12],
  [688, 0.42, 0.085],
  [1046, 0.3, 0.06],
  [1530, 0.2, 0.045],
  [2210, 0.12, 0.03],
];
/** 돌 자체의 고음 공명 — 단단하고 작은 물체라 높고 짧다 */
const STONE_MODES: [number, number, number][] = [
  [2950, 1.0, 0.022],
  [4380, 0.7, 0.016],
  [5720, 0.45, 0.012],
  [7300, 0.25, 0.008],
];

const jitter = (v: number, r: number) => v * (1 + (Math.random() * 2 - 1) * r);

let master: GainNode | null = null;

/** 모든 소리가 지나가는 출력단: 아주 높은 잡음을 살짝 깎고, 여러 공명이 겹쳐도 찢어지지 않게 리미터 */
function out(a: AudioContext): AudioNode {
  if (!master || master.context !== a) {
    master = a.createGain();
    master.gain.value = 0.6;
    const lp = a.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 9000;
    const limit = a.createDynamicsCompressor();
    limit.threshold.value = -6;
    limit.ratio.value = 12;
    limit.attack.value = 0.001;
    limit.release.value = 0.05;
    master.connect(lp).connect(limit).connect(a.destination);
  }
  return master;
}

/** 감쇠하는 사인파 묶음 하나 (공명 모드들) */
function modes(a: AudioContext, t: number, list: [number, number, number][], gain: number, pitch: number, decayScale = 1) {
  for (const [f, amp, decay] of list) {
    const osc = a.createOscillator();
    osc.frequency.value = jitter(f * pitch, 0.025);
    const g = a.createGain();
    const d = jitter(decay * decayScale, 0.1);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain * amp, t + 0.0015); // 1.5ms 만에 최대 (딸깍하는 잡음 없이 빠른 어택)
    g.gain.exponentialRampToValueAtTime(0.0001, t + d * 5);
    osc.connect(g).connect(out(a));
    osc.start(t);
    osc.stop(t + d * 5 + 0.01);
  }
}

/** 충돌 순간의 짧은 잡음 (접촉음). center 근처 대역만 남긴다 */
function contact(a: AudioContext, t: number, gain: number, center: number, len = 0.006) {
  const buf = a.createBuffer(1, Math.ceil(a.sampleRate * len), a.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 3);
  const src = a.createBufferSource();
  src.buffer = buf;
  const band = a.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = center;
  band.Q.value = 0.9;
  const g = a.createGain();
  g.gain.value = gain;
  src.connect(band).connect(g).connect(out(a));
  src.start(t);
}

/** 돌을 판에 놓는 "딱 —". gain은 세기, pitch는 1이 보통 */
function stoneOnBoard(at: number, gain = 1, pitch = 1) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime + at;
  const g = jitter(gain, 0.12);
  contact(a, t, g * 0.9, 3800 * pitch);
  modes(a, t, STONE_MODES, g * 0.22, pitch);
  modes(a, t, BOARD_MODES, g * 0.34, pitch);
  // 자리 잡는 작은 두 번째 딱
  const settle = t + 0.03 + Math.random() * 0.025;
  contact(a, settle, g * 0.12, 4200 * pitch, 0.003);
  modes(a, settle, STONE_MODES, g * 0.04, pitch * 1.02, 0.6);
}

/** 돌끼리 부딪치는 "짤깍": 판 울림 없이 돌 공명만, 더 높고 짧게 */
function stoneOnStone(at: number, gain = 0.5) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime + at;
  const p = jitter(1.25, 0.08);
  contact(a, t, gain * 0.6, 5200 * p, 0.003);
  modes(a, t, STONE_MODES, gain * 0.2, p, 0.7);
}

/** 둘 수 없는 자리: 손가락으로 판을 톡 — 돌 소리 없이 낮고 둔한 나무 울림만 */
function boardTap(at: number) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime + at;
  contact(a, t, 0.25, 900, 0.004);
  modes(a, t, BOARD_MODES.slice(0, 3), 0.18, 0.8, 0.6);
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
    stoneOnBoard(0);
    vibrate(10);
  },
  /** n점을 따냈을 때: 돌이 튕겨 나가는 연출과 맞춰, 집어 든 돌끼리 부딪치는 짤깍을 여러 번. */
  capture(n: number) {
    const clicks = Math.min(n, 6);
    for (let i = 0; i < clicks; i++) stoneOnStone(0.09 + i * 0.045 + Math.random() * 0.015, 0.5 - i * 0.04);
    vibrate(n > 1 ? [12, 40, 12] : 15);
  },
  illegal() {
    boardTap(0);
    vibrate([8, 30, 8]);
  },
};
