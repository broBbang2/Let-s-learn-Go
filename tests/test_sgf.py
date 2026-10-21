import pytest

from adaptive_go.board import Board
from adaptive_go.sgf import SgfError, dump, gtp_to_sgf, parse, sgf_to_gtp

SAMPLE = r"""(;GM[1]FF[4]SZ[19]KM[6.5]PB[Black Player]PW[White \] Player]BR[9d]DT[2016-03-13]RE[W+R]
C[root comment]
;B[pd];W[dp]C[second move]
;B[pp](;W[dd];B[tt])(;W[qc]))"""


def test_coordinates_roundtrip():
    assert sgf_to_gtp("pd", 19) == "Q16"
    assert sgf_to_gtp("aa", 19) == "A19"
    assert sgf_to_gtp("ss", 19) == "T1"
    assert sgf_to_gtp("", 19) == "pass" and sgf_to_gtp("tt", 19) == "pass"
    assert gtp_to_sgf("Q16", 19) == "pd" and gtp_to_sgf("J10", 19) == "ij" and gtp_to_sgf("pass", 19) == ""


def test_parse_main_line_meta_and_comments():
    g = parse(SAMPLE)
    assert g["size"] == 19 and g["komi"] == 6.5
    assert g["meta"]["PW"] == "White ] Player"  # 이스케이프 처리
    assert g["meta"]["DT"] == "2016-03-13" and g["meta"]["RE"] == "W+R"
    # 본 수순은 첫 번째 변화도(W dd → B pass)를 따른다
    assert g["moves"] == [("B", "Q16"), ("W", "D4"), ("B", "Q4"), ("W", "D16"), ("B", "pass")]
    assert g["comments"] == {0: "root comment", 2: "second move"}
    assert not g["setup"]


def test_parse_detects_setup_stones():
    assert parse("(;SZ[9]AB[cc][gg];W[ee])")["setup"]


def test_dump_parse_roundtrip_and_replay():
    moves = [("B", "Q16"), ("W", "D4"), ("B", "pass"), ("W", "C3")]
    g = parse(dump(19, 7.5, moves, {"PB": "me", "PW": "AI v3(5k)"}))
    assert g["moves"] == moves and g["komi"] == 7.5 and g["meta"]["PW"] == "AI v3(5k)"
    b = Board(19)
    for _, v in g["moves"]:
        b.play(v)


def test_bad_sgf():
    with pytest.raises(SgfError):
        parse("not an sgf")
    with pytest.raises(SgfError):
        parse("(;SZ[19];B[pd]")
