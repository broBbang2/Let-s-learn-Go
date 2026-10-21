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

/** 나무판에 돌이 닿는 "딱": 아주 짧은 잡음을 대역 필터로 깎고, 낮은 음을 살짝 섞는다. */
function knock(at: number, gain = 0.5, pitch = 1) {
  const a = audio();
  if (!a) return;
  const t = a.currentTime + at;
  const len = 0.05;
  const buf = a.createBuffer(1, Math.ceil(a.sampleRate * len), a.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / data.length, 6);
  const noise = a.createBufferSource();
  noise.buffer = buf;
  const band = a.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = 2400 * pitch;
  band.Q.value = 1.4;
  const g = a.createGain();
  g.gain.value = gain;
  noise.connect(band).connect(g).connect(a.destination);
  noise.start(t);

  const body = a.createOscillator();
  body.frequency.setValueAtTime(320 * pitch, t);
  body.frequency.exponentialRampToValueAtTime(140 * pitch, t + 0.06);
  const bg = a.createGain();
  bg.gain.setValueAtTime(gain * 0.5, t);
  bg.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
  body.connect(bg).connect(a.destination);
  body.start(t);
  body.stop(t + 0.09);
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
    knock(0);
    vibrate(10);
  },
  /** n점을 따냈을 때: 돌이 튕겨 나가는 연출과 맞춰 짧은 딸깍을 여러 번. */
  capture(n: number) {
    const clicks = Math.min(n, 6);
    for (let i = 0; i < clicks; i++) knock(0.09 + i * 0.045, 0.22, 1.6 + Math.random() * 0.3);
    vibrate(n > 1 ? [12, 40, 12] : 15);
  },
  illegal() {
    knock(0, 0.18, 0.55);
    vibrate([8, 30, 8]);
  },
};
