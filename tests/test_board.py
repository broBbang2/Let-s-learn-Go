import pytest

from adaptive_go.board import BLACK, EMPTY, WHITE, Board, IllegalMoveError, format_score


def play_all(board, moves):
    for m in moves:
        board.play(m)


def test_vertex_roundtrip_skips_i():
    b = Board(19)
    assert b.parse_vertex("J1") == (0, 8)
    assert b.format_vertex((18, 18)) == "T19"
    assert b.parse_vertex("pass") is None
    assert not b.is_legal("I5")


def test_alternating_turns_and_history():
    b = Board(9)
    play_all(b, ["D4", "E5"])
    assert b.to_play == BLACK
    assert b.history == [(BLACK, "D4"), (WHITE, "E5")]


def test_occupied_point_is_illegal():
    b = Board(9)
    b.play("D4")
    with pytest.raises(IllegalMoveError):
        b.play("D4")


def test_capture_single_stone():
    b = Board(9)
    # 흑이 백 E5를 둘러싸서 따냄
    play_all(b, ["E4", "E5", "D5", "A1", "F5", "A2", "E6"])
    assert b.grid[4][4] == EMPTY
    assert b.captures[BLACK] == 1


def test_capture_group_in_corner():
    b = Board(9)
    play_all(b, ["A2", "A1", "C1", "B1", "B2"])
    assert b.grid[0][0] == EMPTY and b.grid[0][1] == EMPTY
    assert b.captures[BLACK] == 2


def test_suicide_is_illegal():
    b = Board(9)
    play_all(b, ["A2", "J9", "B1"])
    assert not b.is_legal("A1")  # 백 A1은 활로 0, 아무것도 못 따냄


def test_capture_beats_suicide():
    b = Board(9)
    # 흑 A1은 놓는 순간 활로가 0이지만, 백 B1·A2를 따내므로 합법
    play_all(b, ["C1", "B1", "B2", "A2", "A3", "J9"])
    assert b.is_legal("A1")
    b.play("A1")
    assert b.grid[0][1] == EMPTY and b.grid[1][0] == EMPTY
    assert b.captures[BLACK] == 2


def test_simple_ko():
    b = Board(9)
    # 패 모양: 흑 D4 따냄 후 백이 즉시 되따내기 금지
    play_all(b, ["C4", "D4", "D3", "E3", "D5", "E5", "J9", "F4", "E4"])
    assert b.grid[3][3] == EMPTY  # 백 D4가 잡힘
    assert not b.is_legal("D4")  # 백의 즉시 되따냄 금지
    b.play("A1")  # 팻감
    b.play("A9")
    assert b.is_legal("D4")


def test_two_passes_end_game_and_area_score():
    b = Board(5, komi=0.5)
    # 흑이 C열 벽, 백이 D열 벽 → 흑 15점(A~C열), 백 10점(D~E열)
    for row in range(1, 6):
        b.play(f"C{row}")
        b.play(f"D{row}")
    b.play("pass")
    assert not b.is_over()
    b.play("pass")
    assert b.is_over()
    assert b.area_score() == 15 - 10 - 0.5
    assert format_score(b.area_score()) == "B+4.5"


def test_ascii_output():
    b = Board(5)
    b.play("A1")
    lines = str(b).splitlines()
    assert lines[0].split() == ["A", "B", "C", "D", "E"]
    assert lines[-2].startswith(" 1 B")
