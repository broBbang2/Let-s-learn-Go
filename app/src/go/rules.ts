// 바둑 규칙 (adaptive_go/board.py와 같은 규칙의 TS 버전)
//
// 서버가 정답(권위)이고, 이 파일은 화면 반응용이다.
// - 마우스를 올린 자리가 둘 수 있는 곳인지 즉시 판정 (자충수, 패)
// - 내가 둔 수로 따낸 돌을 서버 응답 전에 바로 들어내는 연출
// - 돌마다 "몇 번째 수로 놓였는지" id를 붙여, 화면이 어떤 돌이 새로 생기고 사라졌는지 알게 한다
//
// 좌표는 GTP 형식("D4"), row 0이 1행(아래). 단순 패만 처리(슈퍼코 미처리)도 board.py와 같다.

export type Color = "B" | "W";
export type Cell = "." | Color;
export type IllegalReason = "occupied" | "ko" | "suicide";

export const COLUMNS = "ABCDEFGHJKLMNOPQRST";

export interface Stone {
  id: number; // 이 돌을 놓은 수의 번호 (history 인덱스)
  color: Color;
  row: number;
  col: number;
}

export const opponent = (c: Color): Color => (c === "B" ? "W" : "B");
export const vertexOf = (row: number, col: number) => COLUMNS[col] + (row + 1);
export function pointOf(v: string): [number, number] {
  return [Number(v.slice(1)) - 1, COLUMNS.indexOf(v[0].toUpperCase())];
}

export class IllegalMove extends Error {
  constructor(public reason: IllegalReason, vertex: string) {
    super(`${vertex}: ${reason}`);
  }
}

export class Position {
  readonly grid: (Stone | null)[][];
  toPlay: Color = "B";
  koPoint: [number, number] | null = null;
  captures = { B: 0, W: 0 };
  moveCount = 0;

  constructor(readonly size: number) {
    this.grid = Array.from({ length: size }, () => Array<Stone | null>(size).fill(null));
  }

  /** 서버의 history([[색, 좌표], ...])를 처음부터 다시 둬서 같은 국면을 만든다. */
  static replay(size: number, history: [Color, string][]): Position {
    const p = new Position(size);
    for (const [color, v] of history) p.play(v, color);
    return p;
  }

  clone(): Position {
    const p = new Position(this.size);
    this.grid.forEach((row, r) => row.forEach((s, c) => (p.grid[r][c] = s)));
    p.toPlay = this.toPlay;
    p.koPoint = this.koPoint;
    p.captures = { ...this.captures };
    p.moveCount = this.moveCount;
    return p;
  }

  stones(): Stone[] {
    return this.grid.flat().filter((s): s is Stone => s !== null);
  }

  cells(): Cell[][] {
    return this.grid.map((row) => row.map((s) => (s ? s.color : ".")));
  }

  private neighbors(r: number, c: number): [number, number][] {
    const out: [number, number][] = [];
    if (r > 0) out.push([r - 1, c]);
    if (r < this.size - 1) out.push([r + 1, c]);
    if (c > 0) out.push([r, c - 1]);
    if (c < this.size - 1) out.push([r, c + 1]);
    return out;
  }

  /** 플러드필로 (r, c)가 속한 그룹과 활로 수를 구한다. grid는 시뮬레이션용 사본일 수 있다. */
  private group(grid: (Color | null)[][], r: number, c: number) {
    const color = grid[r][c];
    const key = (y: number, x: number) => y * this.size + x;
    const seen = new Set([key(r, c)]);
    const libs = new Set<number>();
    const members: [number, number][] = [[r, c]];
    for (let i = 0; i < members.length; i++) {
      for (const [y, x] of this.neighbors(...members[i])) {
        const s = grid[y][x];
        if (s === null) libs.add(key(y, x));
        else if (s === color && !seen.has(key(y, x))) {
          seen.add(key(y, x));
          members.push([y, x]);
        }
      }
    }
    return { members, liberties: libs };
  }

  /** 두면 따내는 돌. 둘 수 없으면 IllegalMove. 판은 바꾸지 않는다. */
  private simulate(r: number, c: number, color: Color) {
    const grid = this.grid.map((row) => row.map((s) => (s ? s.color : null)));
    grid[r][c] = color;
    const captured = new Map<number, [number, number]>();
    for (const [y, x] of this.neighbors(r, c)) {
      if (grid[y][x] === opponent(color)) {
        const g = this.group(grid, y, x);
        if (g.liberties.size === 0) for (const [gy, gx] of g.members) captured.set(gy * this.size + gx, [gy, gx]);
      }
    }
    for (const [y, x] of captured.values()) grid[y][x] = null;
    const mine = this.group(grid, r, c);
    return { captured: [...captured.values()], mine };
  }

  /** 둘 수 있으면 null, 없으면 이유. */
  check(vertex: string, color: Color = this.toPlay): IllegalReason | null {
    if (vertex.toLowerCase() === "pass") return null;
    const [r, c] = pointOf(vertex);
    if (this.grid[r][c]) return "occupied";
    if (color === this.toPlay && this.koPoint && this.koPoint[0] === r && this.koPoint[1] === c) return "ko";
    return this.simulate(r, c, color).mine.liberties.size === 0 ? "suicide" : null;
  }

  /** 착수. 따낸 돌 목록을 돌려준다. */
  play(vertex: string, color: Color = this.toPlay): Stone[] {
    const reason = this.check(vertex, color);
    if (reason) throw new IllegalMove(reason, vertex);
    const id = this.moveCount++;
    this.koPoint = null;
    this.toPlay = opponent(color);
    if (vertex.toLowerCase() === "pass") return [];

    const [r, c] = pointOf(vertex);
    const { captured, mine } = this.simulate(r, c, color);
    const removed = captured.map(([y, x]) => this.grid[y][x]!);
    for (const [y, x] of captured) this.grid[y][x] = null;
    this.grid[r][c] = { id, color, row: r, col: c };
    this.captures[color] += captured.length;
    // 단순 패: 한 점을 따냈고, 놓은 돌이 홀로 활로 1개(=따낸 자리)뿐이면 상대가 곧바로 되따낼 수 없다
    if (captured.length === 1 && mine.members.length === 1 && mine.liberties.size === 1) {
      const [ky, kx] = captured[0];
      if (mine.liberties.has(ky * this.size + kx)) this.koPoint = [ky, kx];
    }
    return removed;
  }
}

export const ILLEGAL_TEXT: Record<IllegalReason, string> = {
  occupied: "이미 돌이 있는 자리예요",
  ko: "패: 바로 되따낼 수 없어요. 다른 곳에 한 수 두고 오세요",
  suicide: "자충수: 두자마자 잡히는 자리라 둘 수 없어요",
};
