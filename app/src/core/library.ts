// 기보 보관함의 데이터와 계산 (adaptive_go/library.py 응답 모양). DOM 없이 순수 함수만 둔다.
import { opponent, pointOf, type Color, type Stone } from "../go/rules";
import { apiUrl, pct, request } from "./http";

export interface LibraryItem {
  id: string;
  kind: "famous" | "mine";
  title: string;
  black: string;
  white: string;
  date: string;
  result: string;
  event: string;
  size: number;
  moves: number;
}

export interface NextMove {
  color: Color;
  move: string;
  loss: number;
  winrate_loss: number;
  human_p: number;
  human_rank: number | null;
  engine_rank: number | null;
}

export interface TurnAnalysis {
  turn: number;
  winrate_black: number;
  score_black: number;
  best: string;
  best_pv: string[];
  next?: NextMove;
}

export interface Moment {
  turn: number;
  move_number: number;
  text: string;
}

export type Analysis =
  | { state: "none" }
  | { state: "running"; done: number; total: number }
  | { state: "error" | "busy"; error: string }
  | {
      state: "done";
      profile: string;
      turns: TurnAnalysis[];
      turning_points: Moment[];
      surprises: Moment[];
      my_mistakes?: Moment[];
    };
export type DoneAnalysis = Extract<Analysis, { state: "done" }>;

export interface LibraryGame {
  id: string;
  size: number;
  komi: number;
  meta: Record<string, string>;
  moves: [Color, string][];
  comments?: Record<string, string>;
  notes?: Record<string, string>;
  warning?: string;
  user_color?: Color;
  analysis: Analysis;
}

const enc = encodeURIComponent;

export const libraryApi = {
  list: () => request<LibraryItem[]>("/api/library"),
  load: (id: string) => request<LibraryGame>(`/api/library/${enc(id)}`),
  analysis: (id: string) => request<Analysis>(`/api/library/${enc(id)}/analysis`),
  analyze: (id: string) => request<Analysis>(`/api/library/${enc(id)}/analyze`, {}),
  sgfUrl: (id: string) => apiUrl(`/api/library/${enc(id)}/sgf`),
};

// ---------- 목록·제목 문구 ----------

export function itemMeta(g: LibraryItem) {
  // 제목에 이미 대회명이 있으면 메타에서 반복하지 않는다
  const event = g.event && !g.title.startsWith(g.event) ? g.event : "";
  return [g.date, event, g.result, `${g.size}줄 ${g.moves}수`].filter(Boolean).join(" · ");
}

export function gameHeader(game: LibraryGame) {
  const m = game.meta;
  return {
    title: m.GN || (m.EV && m.RO ? `${m.EV} 제${m.RO}국` : m.EV) || "기보",
    players: `흑 ${m.PB || "?"}${m.BR ? ` (${m.BR})` : ""} · 백 ${m.PW || "?"}${m.WR ? ` (${m.WR})` : ""}`,
    meta: [m.DT, m.PC, m.RE && `결과 ${m.RE}`, `덤 ${game.komi}`].filter(Boolean).join(" · "),
  };
}

// ---------- 재생 ----------

/**
 * 국면별 돌 목록 (0수 ~ 끝). 기보는 이미 둔 수라 합법성 검사 없이 따냄만 처리한다.
 * 돌 id는 놓인 수 번호라서, 한 수씩 넘길 때 화면이 새 돌·따인 돌을 알아본다.
 */
export function replayStones(size: number, moves: [Color, string][]): Stone[][] {
  let grid: (Stone | null)[][] = Array.from({ length: size }, () => Array<Stone | null>(size).fill(null));
  const flat = () => grid.flat().filter((s): s is Stone => s !== null);
  const out = [flat()];
  const nbrs = (r: number, c: number) =>
    ([[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]] as [number, number][]).filter(
      ([a, b]) => a >= 0 && b >= 0 && a < size && b < size,
    );
  const group = (r: number, c: number) => {
    const color = grid[r][c]!.color;
    const seen = new Set([r * size + c]);
    const stones: [number, number][] = [[r, c]];
    let libs = 0;
    for (let i = 0; i < stones.length; i++) {
      for (const [a, b] of nbrs(...stones[i])) {
        const k = a * size + b;
        if (seen.has(k)) continue;
        const s = grid[a][b];
        if (!s) {
          libs++;
          seen.add(k);
        } else if (s.color === color) {
          seen.add(k);
          stones.push([a, b]);
        }
      }
    }
    return { stones, libs };
  };
  moves.forEach(([color, v], id) => {
    grid = grid.map((row) => row.slice());
    if (v !== "pass") {
      const [r, c] = pointOf(v);
      grid[r][c] = { id, color, row: r, col: c };
      for (const [a, b] of nbrs(r, c)) {
        if (grid[a][b] && grid[a][b]!.color === opponent(color)) {
          const g = group(a, b);
          if (!g.libs) for (const [y, x] of g.stones) grid[y][x] = null;
        }
      }
    }
    out.push(flat());
  });
  return out;
}

/** t수까지 둔 국면에서 다음에 둘 색 */
export function toPlayAt(moves: [Color, string][], t: number): Color {
  if (t < moves.length) return moves[t][0];
  return t > 0 ? opponent(moves[t - 1][0]) : "B";
}

/** 명장면 메모: 그 수를 두기 직전(어떤 수가 나올지 생각해 볼 때)과 둔 직후에 보여준다 */
export function noteAt(game: LibraryGame, t: number) {
  const notes = game.notes ?? {};
  if (notes[t + 1]) return `다음 ${t + 1}수: ${notes[t + 1]}`;
  if (notes[t]) return `${t}수: ${notes[t]}`;
  return "";
}

// ---------- 분석 문구 ----------

export const eraOf = (a: DoneAnalysis) => a.profile.replace("proyear_", "");

export function leadText(scoreBlack: number) {
  return scoreBlack >= 0 ? `흑 ${scoreBlack.toFixed(1)}집 우세` : `백 ${(-scoreBlack).toFixed(1)}집 우세`;
}

export function engineText(n: NextMove) {
  const engine = n.engine_rank ? `KataGo 후보 ${n.engine_rank}위` : "KataGo 후보 밖";
  const wr = n.winrate_loss < 0.005 ? "승률 그대로" : `승률 −${pct(n.winrate_loss)}`;
  return `${engine} · 손실 ${n.loss.toFixed(1)}집 · ${wr}`;
}

export function humanText(n: NextMove) {
  return n.human_rank ? `${pct(n.human_p)} (${n.human_rank}위)` : pct(n.human_p);
}

export type MomentKind = "mine" | "star" | "dot" | "diamond";
export interface MomentItem {
  turn: number;
  text: string;
  kind: MomentKind;
}

/** 주요 장면 목록: 내 실수 먼저, 그다음 수 순서 (명장면 메모·결정적 수·예상 밖의 명수) */
export function buildMoments(game: LibraryGame, a: DoneAnalysis): MomentItem[] {
  const era = eraOf(a);
  const notes = Object.entries(game.notes ?? {}).map(([num, text]): MomentItem => {
    const t = Number(num) - 1;
    const nx = a.turns[t]?.next;
    const view = nx
      ? ` KataGo: 손실 ${nx.loss.toFixed(1)}집(후보 ${nx.engine_rank ? `${nx.engine_rank}위` : "밖"}), ${era}년 프로 확률 ${pct(nx.human_p)}.`
      : "";
    return { turn: t, text: `${num}수 명장면 — ${text}${view}`, kind: "star" };
  });
  // 내 대국이면 "내가 놓친 장면"(내 수 중 승률을 가장 많이 잃은 수)을 맨 앞에
  const mine = (a.my_mistakes ?? []).map((m): MomentItem => ({ turn: m.turn, text: `내 실수 — ${m.text}`, kind: "mine" }));
  return [
    ...mine,
    ...notes,
    ...a.turning_points.map((m): MomentItem => ({ turn: m.turn, text: m.text, kind: "dot" })),
    ...a.surprises.map((m): MomentItem => ({ turn: m.turn, text: m.text, kind: "diamond" })),
  ].sort((x, y) => Number(y.kind === "mine") - Number(x.kind === "mine") || x.turn - y.turn);
}

// ---------- 흑 승률 그래프 좌표 ----------

export interface ChartLayout {
  width: number;
  height: number;
  padL: number;
  padR: number;
  padT: number;
  padB: number;
  n: number;
  x: (t: number) => number;
  y: (p: number) => number;
  path: string;
  /** 그 수를 둔 뒤(turn+1)에 승률이 바뀌므로 그 지점에 표시 */
  turning: { x: number; y: number }[];
  surprises: { x: number; y: number }[];
  /** 가로 픽셀 → 가장 가까운 수 */
  turnAtX: (px: number) => number;
}

export function chartLayout(a: DoneAnalysis, width: number, height = 150): ChartLayout {
  const padL = 30, padR = 6, padT = 8, padB = 18;
  const turns = a.turns;
  const n = turns.length - 1 || 1;
  const x = (t: number) => padL + (t / n) * (width - padL - padR);
  const y = (p: number) => padT + (1 - p) * (height - padT - padB);
  const path = turns.map((t, i) => `${i ? "L" : "M"}${x(t.turn).toFixed(1)},${y(t.winrate_black).toFixed(1)}`).join("");
  const mark = (ms: Moment[]) =>
    ms.flatMap((m) => {
      const t = turns[m.turn + 1];
      return t ? [{ x: x(t.turn), y: y(t.winrate_black) }] : [];
    });
  return {
    width, height, padL, padR, padT, padB, n, x, y, path,
    turning: mark(a.turning_points),
    surprises: mark(a.surprises),
    turnAtX: (px) => Math.max(0, Math.min(n, Math.round(((px - padL) / (width - padL - padR)) * n))),
  };
}
