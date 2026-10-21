"""바둑 규칙 엔진.

좌표는 GTP 형식("D4", "pass")을 그대로 쓴다. KataGo와 변환 없이 통신하기 위해서다.
- 열은 A~T에서 I를 뺀다 (GTP 관례: I와 J 혼동 방지)
- 행은 아래가 1
"""

EMPTY, BLACK, WHITE = ".", "B", "W"
COLUMNS = "ABCDEFGHJKLMNOPQRST"


class IllegalMoveError(ValueError):
    pass


def opponent(color):
    return WHITE if color == BLACK else BLACK


class Board:
    def __init__(self, size=19, komi=7.5):
        if not 2 <= size <= len(COLUMNS):
            raise ValueError(f"지원하지 않는 판 크기: {size}")
        self.size = size
        self.komi = komi
        # grid[row][col], row 0 = GTP 1행(아래)
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.to_play = BLACK
        self.ko_point = None  # 다음 수에서 둘 수 없는 단순 패 자리
        self.history = []  # [(color, vertex), ...] GTP 수순
        self.captures = {BLACK: 0, WHITE: 0}  # 각 색이 따낸 돌 수
        self._consecutive_passes = 0

    # ---- 좌표 변환 ----

    def parse_vertex(self, vertex):
        """"D4" -> (row, col), "pass" -> None"""
        v = vertex.strip().upper()
        if v == "PASS":
            return None
        if len(v) < 2 or v[0] not in COLUMNS[: self.size]:
            raise IllegalMoveError(f"잘못된 좌표: {vertex}")
        try:
            row = int(v[1:]) - 1
        except ValueError:
            raise IllegalMoveError(f"잘못된 좌표: {vertex}") from None
        if not 0 <= row < self.size:
            raise IllegalMoveError(f"잘못된 좌표: {vertex}")
        return row, COLUMNS.index(v[0])

    def format_vertex(self, point):
        if point is None:
            return "pass"
        row, col = point
        return f"{COLUMNS[col]}{row + 1}"

    # ---- 그룹/활로 ----

    def _neighbors(self, point):
        row, col = point
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < self.size and 0 <= c < self.size:
                yield r, c

    def _group_and_liberties(self, grid, point):
        """플러드필로 point가 속한 그룹과 그 활로를 구한다."""
        color = grid[point[0]][point[1]]
        group, liberties = {point}, set()
        stack = [point]
        while stack:
            p = stack.pop()
            for n in self._neighbors(p):
                stone = grid[n[0]][n[1]]
                if stone == EMPTY:
                    liberties.add(n)
                elif stone == color and n not in group:
                    group.add(n)
                    stack.append(n)
        return group, liberties

    def _simulate(self, point, color):
        """point에 color를 놓은 결과 (grid, 따낸 돌 집합)를 반환. 자충수면 None."""
        grid = [row[:] for row in self.grid]
        grid[point[0]][point[1]] = color
        captured = set()
        for n in self._neighbors(point):
            if grid[n[0]][n[1]] == opponent(color):
                group, libs = self._group_and_liberties(grid, n)
                if not libs:
                    captured |= group
        for r, c in captured:
            grid[r][c] = EMPTY
        # 상대를 따내지 못했는데 내 그룹 활로가 0이면 자충수
        _, libs = self._group_and_liberties(grid, point)
        if not libs:
            return None
        return grid, captured

    # ---- 규칙 ----

    def is_legal(self, vertex, color=None):
        color = color or self.to_play
        try:
            point = self.parse_vertex(vertex)
        except IllegalMoveError:
            return False
        if point is None:
            return True
        if self.grid[point[0]][point[1]] != EMPTY:
            return False
        if color == self.to_play and point == self.ko_point:
            return False
        return self._simulate(point, color) is not None

    def legal_moves(self):
        moves = [
            self.format_vertex((r, c))
            for r in range(self.size)
            for c in range(self.size)
            if self.is_legal(self.format_vertex((r, c)))
        ]
        return moves + ["pass"]

    def play(self, vertex):
        color = self.to_play
        if not self.is_legal(vertex, color):
            raise IllegalMoveError(f"{color} {vertex}: 둘 수 없는 자리")
        point = self.parse_vertex(vertex)
        self.ko_point = None
        if point is None:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0
            grid, captured = self._simulate(point, color)
            self.grid = grid
            self.captures[color] += len(captured)
            # 단순 패: 한 점을 따냈고, 놓은 돌이 홀로 활로 1개(=따낸 자리)뿐이면
            # 상대가 곧바로 되따내는 것을 금지한다. (슈퍼코는 미처리)
            if len(captured) == 1:
                group, libs = self._group_and_liberties(grid, point)
                if len(group) == 1 and libs == captured:
                    self.ko_point = next(iter(captured))
        self.history.append((color, self.format_vertex(point)))
        self.to_play = opponent(color)

    def is_over(self):
        return self._consecutive_passes >= 2

    # ---- 계가 ----

    def area_score(self):
        """중국식 계가 (돌 + 집 - 덤). 양수면 흑 우세.

        사석 판정은 하지 않는다 — 판 위의 돌은 모두 산 돌로 센다.
        사석은 이후 KataGo ownership으로 판정할 예정.
        """
        score = {BLACK: 0, WHITE: 0}
        seen = set()
        for r in range(self.size):
            for c in range(self.size):
                stone = self.grid[r][c]
                if stone != EMPTY:
                    score[stone] += 1
                elif (r, c) not in seen:
                    region, borders = self._empty_region((r, c))
                    seen |= region
                    if len(borders) == 1:
                        score[borders.pop()] += len(region)
        return score[BLACK] - score[WHITE] - self.komi

    def _empty_region(self, point):
        region, borders = {point}, set()
        stack = [point]
        while stack:
            p = stack.pop()
            for n in self._neighbors(p):
                stone = self.grid[n[0]][n[1]]
                if stone == EMPTY and n not in region:
                    region.add(n)
                    stack.append(n)
                elif stone != EMPTY:
                    borders.add(stone)
        return region, borders

    def __str__(self):
        header = "   " + " ".join(COLUMNS[: self.size])
        lines = [header]
        for r in range(self.size - 1, -1, -1):
            lines.append(f"{r + 1:>2} " + " ".join(self.grid[r]) + f" {r + 1}")
        lines.append(header)
        return "\n".join(lines)


def format_score(score):
    if score == 0:
        return "0 (무승부)"
    return f"B+{score:g}" if score > 0 else f"W+{-score:g}"
