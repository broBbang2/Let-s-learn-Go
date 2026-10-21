import sys
from pathlib import Path

import pytest

from adaptive_go.board import BLACK, WHITE, Board
from adaptive_go.engine import EngineError, KataGoEngine, ownership_score

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def engine():
    with KataGoEngine(FAKE, report_as=BLACK) as e:
        yield e


def test_moves_sorted_by_order_and_best(engine):
    a = engine.analyze(Board(9))
    assert [m.move for m in a.moves] == ["C3", "D4", "pass"]
    assert a.best.move == "C3"
    assert a.find("d4").score_lead == 1.5
    assert a.find("E5") is None


def test_black_to_move_keeps_black_perspective(engine):
    a = engine.analyze(Board(9))
    assert a.to_play == BLACK
    assert a.winrate == pytest.approx(0.7)
    assert a.score_lead == 3.0


def test_white_to_move_flips_to_side_to_move(engine):
    b = Board(9)
    b.play("E5")
    a = engine.analyze(b)
    assert a.to_play == WHITE
    assert a.winrate == pytest.approx(0.3)
    assert a.score_lead == -3.0
    # 백 기준으로도 손실 계산 방향이 같아야 한다: 최선수가 가장 큰 score_lead
    assert a.best.score_lead == -3.0


def test_sidetomove_config_is_not_flipped():
    b = Board(9)
    b.play("E5")
    with KataGoEngine(FAKE, report_as="SIDETOMOVE") as e:
        a = e.analyze(b)
    assert a.score_lead == 3.0


def test_ownership_mapping_top_left_is_last_row(engine):
    a = engine.analyze(Board(9), include_ownership=True)
    assert a.ownership_at("A9") == 1.0  # KataGo 배열 0번 = 맨 윗줄 왼쪽
    assert a.ownership_at("A1") == -1.0


def test_ownership_is_black_perspective_when_white_to_move(engine):
    b = Board(9)
    b.play("E5")
    a = engine.analyze(b, include_ownership=True)
    assert a.ownership_at("A9") == 1.0


def test_ownership_score(engine):
    b = Board(9, komi=7)
    a = engine.analyze(b, include_ownership=True)
    assert ownership_score(b, a) == 1 - 80 - 7


def test_engine_error_is_raised(engine):
    # Board는 19까지만 허용하므로 size를 직접 바꿔 에러 경로를 시험
    b = Board(19)
    b.size = 25
    with pytest.raises(EngineError, match="too big"):
        engine.analyze(b)


def test_dead_process_raises():
    e = KataGoEngine([sys.executable, "-c", "pass"])
    e._proc.wait()
    with pytest.raises(EngineError):
        e.analyze(Board(9))



def scripted(moves):
    it = iter(moves)
    return lambda _prompt: next(it)


def test_play_loop_retries_illegal_input_and_quits(engine):
    from adaptive_go.play import run

    b = Board(9, komi=7)
    out = []
    result = run(engine, b, BLACK, visits=10, read=scripted(["Z9", "E5", "quit"]), write=out.append)
    assert result is None
    assert b.history == [(BLACK, "E5"), (WHITE, "C3")]
    assert any("둘 수 없는" in line or "잘못된 좌표" in line for line in out)


def test_play_loop_scores_with_ownership_at_end(engine):
    from adaptive_go.play import run

    b = Board(9, komi=7)
    out = []
    # 흑 E5, 백 C3, 흑 pass, 백 pass(가짜 엔진) → 종국
    score = run(engine, b, BLACK, visits=10, read=scripted(["E5", "pass"]), write=out.append)
    assert b.is_over()
    assert score == 1 - 80 - 7
    assert "결과: W+86" in out[-1]


def test_human_policy_is_parsed_to_vertices(engine):
    a = engine.analyze(Board(9), human_profile="rank_5k")
    assert a.human_policy == {"D4": 0.6, "C3": 0.3, "A9": 0.1}


def test_no_human_policy_without_profile(engine):
    assert engine.analyze(Board(9)).human_policy is None


def test_analyze_game_returns_every_turn(engine):
    seen = []
    res = engine.analyze_game(9, 7.0, [("B", "E5"), ("W", "C3")], on_turn=lambda t, a: seen.append(t))
    assert sorted(res) == [0, 1, 2] and sorted(seen) == [0, 1, 2]
    assert res[0].to_play == "B" and res[1].to_play == "W"
    # 가짜 엔진 값은 흑 기준 3.0집 → 백 차례 국면은 백 기준 -3.0
    assert res[0].score_lead == 3.0 and res[1].score_lead == -3.0


def test_analyze_game_stops_at_illegal_move(engine):
    res = engine.analyze_game(9, 7.0, [("B", "E5"), ("W", "E5"), ("B", "C3")])
    assert sorted(res) == [0, 1]
