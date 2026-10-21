// /api/state 응답 모양 (adaptive_go/server.py Game.state())
import type { Cell, Color } from "./go/rules";

export type Mode = "human" | "adaptive" | "best";
export type Ownership = number[][]; // [row][col], 흑 +1 ~ 백 -1

export interface Candidate {
  move: string;
  loss: number;
  winrate: number;
  human_p: number | null;
  visits: number;
  chosen: boolean;
  pv: string[];
}

export interface TerritoryDelta {
  move: string;
  mine: number;
  theirs: number;
  points: string[];
  lost_stones: number;
  won_stones: number;
}

export interface GameState {
  game_id: string;
  size: number;
  komi: number;
  grid: Cell[][];
  to_play: Color;
  user_color: Color;
  history: [Color, string][];
  captures: Record<Color, number>;
  ko_point: string | null;
  over: boolean;
  result: { text: string; score: number | null; ownership?: Ownership } | null;
  ownership?: Ownership | null;
  /** 방금 내 수·AI 수가 바꾼 예상 영역 (내 기준 칸 수 변화) + 주인이 바뀐 칸 */
  territory_change: { user: TerritoryDelta | null; ai: TerritoryDelta | null } | null;
  /** 최근 moves수 동안의 큰 실수(3집 이상) 유형 */
  weakness: { moves: number; types: Record<string, { count: number; avg_loss: number; label: string }> };
  review_url: string | null;
  last_ai: {
    move: string;
    loss: number;
    user_winrate: number;
    user_score_lead: number;
    explain: { move: string; summary: string; candidates: Candidate[] };
  } | null;
  last_eval: { move: string; best_move: string; score_loss: number; winrate_loss: number } | null;
  summary: {
    avg_score_loss: number | null;
    blunders: number;
    by_phase: Record<string, { moves: number; avg_score_loss: number }>;
  };
  rating: { rating: number; rd: number };
  estimate: {
    label: string;
    mean_label: string;
    confidence: number;
    moves: number;
    distribution: number[];
    ranks: string[];
    source?: string; // "ours"면 자체 학습 모델로 추정
  } | null;
  user_explain: { summary: string; best_pv: string[] } | null;
  estimate_evidence: { move: string; probs: number[] } | null;
  rating_change: { before: number; after: number } | null;
  message: string | null;
  visits: number;
  ai: {
    mode: Mode;
    human_available: boolean;
    profile: string | null;
    hybrid: boolean;
    target_loss: number;
    avg_loss: number;
    level: string | null;
  };
}

export interface Marker {
  vertex: string;
  label: string;
  color?: string;
}
