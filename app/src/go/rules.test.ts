import { describe, expect, it } from "vitest";
import { Position } from "./rules";

const setup = (size: number, moves: string[]) => {
  const p = new Position(size);
  for (const m of moves) p.play(m);
  return p;
};

describe("Position", () => {
  it("한 점을 따내고 따낸 돌을 돌려준다", () => {
    const p = setup(9, ["D5", "E5", "F5", "A1", "E4", "A2"]);
    const removed = p.play("E6");
    expect(removed.map((s) => [s.row, s.col])).toEqual([[4, 4]]);
    expect(removed[0].id).toBe(1); // 두 번째 수로 놓인 백 E5
    expect(p.grid[4][4]).toBeNull();
    expect(p.captures.B).toBe(1);
  });

  it("자충수를 막는다", () => {
    const p = setup(9, ["D5", "A1", "F5", "A2", "E4", "A3", "E6"]);
    expect(p.check("E5")).toBe("suicide");
  });

  it("따내는 수는 자충수가 아니다", () => {
    const p = setup(9, ["A2", "A1", "B1"]);
    expect(p.grid[0][0]).toBeNull();
    expect(p.check("A1")).toBe("suicide"); // 이제는 따낼 게 없으니 자충수
  });

  it("패는 한 수 쉬어야 되따낼 수 있다", () => {
    const p = setup(9, ["D5", "E5", "E4", "F4", "E6", "F6", "J9", "G5", "F5"]);
    expect(p.check("E5")).toBe("ko");
    p.play("A1");
    p.play("A9");
    expect(p.check("E5")).toBeNull();
  });

  it("history 재생과 clone", () => {
    const p = Position.replay(9, [["B", "E5"], ["W", "pass"], ["B", "D4"]]);
    expect(p.toPlay).toBe("W");
    expect(p.stones().map((s) => s.id).sort()).toEqual([0, 2]);
    const q = p.clone();
    q.play("C3");
    expect(p.grid[2][2]).toBeNull();
  });
});
