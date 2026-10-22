import { describe, expect, it } from "vitest";
import { buildMoments, chartLayout, noteAt, replayStones, toPlayAt, type DoneAnalysis, type LibraryGame } from "./library";

const turn = (t: number, wr: number) => ({ turn: t, winrate_black: wr, score_black: 0, best: "D4", best_pv: [] });

const analysis: DoneAnalysis = {
  state: "done",
  profile: "proyear_2016",
  turns: [turn(0, 0.5), turn(1, 0.6), turn(2, 0.2)],
  turning_points: [{ turn: 1, move_number: 2, text: "2수 결정적" }],
  surprises: [],
  my_mistakes: [{ turn: 1, move_number: 2, text: "2수 놓침" }],
};

const game: LibraryGame = {
  id: "f-x",
  size: 9,
  komi: 7,
  meta: {},
  moves: [["B", "D4"], ["W", "E5"]],
  notes: { "1": "첫 수" },
  analysis,
};

describe("replayStones", () => {
  it("따냄을 처리하고 돌 id는 수 번호", () => {
    // 흑 A1을 백 A2·B1로 따낸다
    const pos = replayStones(9, [["B", "A1"], ["W", "A2"], ["B", "pass"], ["W", "B1"]]);
    expect(pos).toHaveLength(5);
    expect(pos[1]).toEqual([{ id: 0, color: "B", row: 0, col: 0 }]);
    expect(pos[4].map((s) => s.id).sort()).toEqual([1, 3]);
  });
});

describe("toPlayAt / noteAt", () => {
  it("다음 둘 색과 메모", () => {
    expect(toPlayAt(game.moves, 0)).toBe("B");
    expect(toPlayAt(game.moves, 2)).toBe("B");
    expect(noteAt(game, 0)).toBe("다음 1수: 첫 수");
    expect(noteAt(game, 1)).toBe("1수: 첫 수");
    expect(noteAt(game, 2)).toBe("");
  });
});

describe("buildMoments", () => {
  it("내 실수가 먼저, 나머지는 수 순서", () => {
    const kinds = buildMoments(game, analysis).map((m) => [m.kind, m.turn]);
    expect(kinds).toEqual([["mine", 1], ["star", 0], ["dot", 1]]);
  });
});

describe("chartLayout", () => {
  it("좌표와 픽셀→수 변환", () => {
    const L = chartLayout(analysis, 336);
    expect(L.x(0)).toBe(30);
    expect(L.x(2)).toBe(330);
    expect(L.y(1)).toBe(8);
    expect(L.turnAtX(330)).toBe(2);
    expect(L.turnAtX(0)).toBe(0);
    expect(L.turning).toEqual([{ x: 330, y: L.y(0.2) }]);
  });
});
