import numpy as np

from adaptive_go.board import Board
from adaptive_go.features import (
    PLANES,
    encode,
    index_vertex,
    move_index,
    symmetry_index,
    symmetry_index_table,
    symmetry_planes,
)


def test_planes_from_side_to_move():
    b = Board(9)
    b.play("E5")  # 흑
    x = encode(b)  # 이제 백 차례 → 흑 돌이 "상대 돌"
    assert x.shape == (PLANES, 9, 9)
    assert x[1, 4, 4] == 1 and x[0].sum() == 0
    assert x[2].sum() == 80
    assert x[9, 4, 4] == 1  # 직전 수


def test_liberty_planes_and_atari():
    b = Board(9)
    for m in ["A2", "A1", "C1", "B1"]:  # 백 A1·B1 그룹이 활로 1개(… 흑 차례)
        b.play(m)
    x = encode(b)  # 흑 차례
    assert x[6, 0, 0] == 1 and x[6, 0, 1] == 1  # 상대(백) 활로 1
    assert x[4, 1, 0] == 1 or x[5, 1, 0] == 1  # 내(흑) A2: 활로 2 이상


def test_ko_plane():
    b = Board(9)
    for m in ["C4", "D4", "D3", "E3", "D5", "E5", "J9", "F4", "E4"]:
        b.play(m)
    assert encode(b)[11, 3, 3] == 1  # D4 되따냄 금지


def test_move_index_roundtrip():
    b = Board(19)
    assert move_index("A1", 19, b) == 0 and move_index("T19", 19, b) == 360 and move_index("pass", 19, b) == 361
    assert index_vertex(move_index("Q16", 19, b), 19, b) == "Q16"


def test_symmetry_index_matches_planes():
    table = symmetry_index_table(9)
    for k in range(8):
        plane = np.zeros((9, 9), dtype=np.uint8)
        plane[2, 7] = 1
        moved = symmetry_planes(plane, k)
        r, c = np.argwhere(moved)[0]
        assert table[k][2 * 9 + 7] == r * 9 + c == symmetry_index(2 * 9 + 7, 9, k)
        assert table[k][81] == 81  # 패스
    assert len({tuple(table[k]) for k in range(8)}) == 8  # 8가지 대칭이 서로 다름
