import sys
import time
from pathlib import Path

import pytest

from adaptive_go.engine import KataGoEngine
from adaptive_go.library import Library, profile_for
from adaptive_go.records import Records

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]
SGF = "(;GM[1]SZ[9]KM[7]PB[Honinbo]PW[Gennan]DT[1846-07-21]RE[B+2]GN[Ear-reddening]C[famous];B[ee];W[dd]C[hmm];B[cc])"


@pytest.fixture
def lib(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "ear.sgf").write_text(SGF)
    (tmp_path / "lib" / "broken.sgf").write_text("not sgf")
    records = Records(tmp_path / "data")
    records.append_game(
        {"game_id": "20261021-120000-abc", "size": 9, "komi": 7.0, "user_color": "W",
         "moves": [["B", "E5"], ["W", "C3"]], "result": "W+3", "ai": {"mode": "human"}}
    )
    library = Library(tmp_path / "lib", records, lambda: KataGoEngine(FAKE))
    yield library
    library.close()


def wait_done(library, game_id):
    for _ in range(100):
        a = library.analysis(game_id)
        if a["state"] in ("done", "error"):
            return a
        time.sleep(0.05)
    raise AssertionError("분석이 끝나지 않음")


def test_profile_for_dates():
    assert profile_for("1846-07-21") == "proyear_1846"
    assert profile_for("2016-03-13") == "proyear_2016"
    assert profile_for("2026") == "proyear_2023"  # 지원 범위로 자름
    assert profile_for("") == "proyear_2023" and profile_for(None) == "proyear_2023"
    assert profile_for("1700") == "proyear_1800"


def test_list_famous_and_mine_skipping_broken(lib):
    items = {i["id"]: i for i in lib.list()}
    assert set(items) == {"f-ear", "m-20261021-120000-abc"}
    assert items["f-ear"]["title"] == "Ear-reddening" and items["f-ear"]["moves"] == 3
    assert items["m-20261021-120000-abc"]["black"] == "AI(human)" and items["m-20261021-120000-abc"]["white"] == "나"


def test_load_and_reject_path_traversal(lib):
    g = lib.load("f-ear")
    assert g["moves"][0] == ("B", "E5") and g["comments"] == {0: "famous", 2: "hmm"}
    assert lib.load("f-../secret") is None and lib.load("f-nope") is None and lib.load("x-1") is None
    assert lib.load("m-20261021-120000-abc")["moves"] == [("B", "E5"), ("W", "C3")]


def test_sgf_export(lib):
    assert lib.sgf("m-20261021-120000-abc").count(";") == 3


def test_analysis_runs_in_background_and_is_cached(lib):
    assert lib.analysis("f-ear") == {"state": "none"}
    lib.start("f-ear")
    a = wait_done(lib, "f-ear")
    assert a["state"] == "done" and a["profile"] == "proyear_1846"
    assert len(a["turns"]) == 4
    first = a["turns"][0]
    # 가짜 엔진: 흑 기준 승률 0.7, 3집 → 국면마다 같은 값이라 손실 0
    assert first["winrate_black"] == 0.7 and first["score_black"] == 3.0 and first["best"] == "C3"
    assert first["next"]["move"] == "E5" and first["next"]["loss"] == 0
    # 두 번째 수 W D6(SGF dd): 가짜 human 정책엔 없음 → 0
    assert a["turns"][1]["next"]["human_p"] == 0.0
    assert "next" not in a["turns"][3]
    assert (lib.cache_dir / "f-ear.json").exists()
    assert lib.start("f-ear")["state"] == "done"  # 캐시 재사용


def test_start_unknown_game(lib):
    assert lib.start("f-nope")["state"] == "error"


def test_notes_are_loaded(lib, tmp_path):
    (tmp_path / "lib" / "ear.notes.json").write_text('{"2": "famous move"}')
    assert lib.load("f-ear")["notes"] == {"2": "famous move"}
    assert lib.load("m-20261021-120000-abc")["notes"] == {}


def test_key_moments_skip_decided_positions():
    from adaptive_go.library import key_moments

    def turn(t, wr_black, color, loss, wr_loss, p):
        nxt = {"color": color, "move": "A1", "loss": loss, "winrate_loss": wr_loss, "human_p": p}
        return {"turn": t, "winrate_black": wr_black, "best": "B2", "next": nxt}

    turns = [turn(20, 0.5, "B", 0.0, 0.0, 0.001),   # 접전 중 드문 좋은 수 → 명수
             turn(21, 0.5, "W", 5.0, 0.3, 0.2),     # 접전 중 큰 실수 → 결정적 수
             turn(200, 0.99, "B", 0.0, 0.0, 0.0001),  # 흑이 이미 이긴 판 → 제외
             turn(201, 0.99, "W", 9.0, 0.009, 0.2)]   # 백이 이미 진 판(백 승률 1%) → 제외
    m = key_moments(turns, "proyear_2016")
    assert [x["move_number"] for x in m["surprises"]] == [21]
    assert [x["move_number"] for x in m["turning_points"]] == [22]
