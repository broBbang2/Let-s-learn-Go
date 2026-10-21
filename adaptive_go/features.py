"""국면 → 신경망 입력 평면. 항상 "둘 차례" 기준으로 만든다(흑백을 뒤집어 같은 모델이 양쪽을 배움).

평면 (PLANES = 12, 각 size×size, 0/1):
  0 내 돌, 1 상대 돌, 2 빈 점
  3·4·5 활로가 1·2·3개 이상인 내 그룹의 돌
  6·7·8 활로가 1·2·3개 이상인 상대 그룹의 돌   ← 단수·잡기 같은 전술 신호. 신경망이 스스로 세기 어려워 직접 넣는다
  9 직전 수, 10 그 전 수                          ← 응수 위치 단서
  11 패 때문에 지금 둘 수 없는 점

수 인덱스: row * size + col (row 0이 1행), 패스는 size * size.
"""

import numpy as np

from .board import EMPTY, opponent

PLANES = 12


def move_index(vertex, size, board):
    if vertex.lower() == "pass":
        return size * size
    r, c = board.parse_vertex(vertex)
    return r * size + c


def index_vertex(index, size, board):
    if index == size * size:
        return "pass"
    return board.format_vertex(divmod(index, size))


def encode(board):
    n = board.size
    me, opp = board.to_play, opponent(board.to_play)
    x = np.zeros((PLANES, n, n), dtype=np.uint8)
    grid = board.grid
    seen = set()
    for r in range(n):
        for c in range(n):
            s = grid[r][c]
            if s == EMPTY:
                x[2, r, c] = 1
                continue
            x[0 if s == me else 1, r, c] = 1
            if (r, c) in seen:
                continue
            group, libs = board._group_and_liberties(grid, (r, c))
            seen |= group
            plane = (3 if s == me else 6) + min(len(libs), 3) - 1
            for gr, gc in group:
                x[plane, gr, gc] = 1
    played = [v for _, v in board.history[-2:]][::-1]  # 직전 수, 그 전 수
    for plane, v in zip((9, 10), played):
        if v != "pass":
            r, c = board.parse_vertex(v)
            x[plane, r, c] = 1
    if board.ko_point:
        x[11, board.ko_point[0], board.ko_point[1]] = 1
    return x


# ---- 8가지 대칭(회전·뒤집기): 학습 데이터를 8배로 불리는 데이터 증강 ----


def symmetry_planes(x, k):
    """x: (..., n, n). k = 0..7 (0~3 회전, 4~7 뒤집은 뒤 회전)."""
    if k >= 4:
        x = np.flip(x, axis=-1)
    return np.rot90(x, k % 4, axes=(-2, -1))


def symmetry_index(index, size, k):
    """수 인덱스를 같은 대칭으로 옮긴다 (패스는 그대로)."""
    if index == size * size:
        return index
    plane = np.zeros((size, size), dtype=np.uint8)
    plane[divmod(index, size)] = 1
    r, c = np.argwhere(symmetry_planes(plane, k))[0]
    return int(r) * size + int(c)


def symmetry_index_table(size):
    """[k][원래 인덱스] → 대칭 후 인덱스. 학습 중 매번 계산하지 않도록 미리 만든다."""
    return np.array([[symmetry_index(i, size, k) for i in range(size * size + 1)] for k in range(8)])
