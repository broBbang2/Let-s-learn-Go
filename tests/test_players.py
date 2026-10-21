import random
import sys
from pathlib import Path

import pytest

from adaptive_go.arena import (
    bradley_terry,
    is_first_line,
    monotone_calibration,
    play_game,
    schedule,
    summarize,
    to_markdown,
)
from adaptive_go.board import Board
from adaptive_go.engine import KataGoEngine
from adaptive_go.players import (
    BestPlayer,
    HumanPlayer,
    HybridPlayer,
    TargetLossPlayer,
    parse_player,
    sample_human,
)

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def engine():
    with KataGoEngine(FAKE) as e:
        yield e


def test_sample_human_drops_tail_illegal_and_pass():
    b = Board(9)
    b.play("D4")
    policy = {"D4": 0.5, "C3": 0.3, "E5": 0.005, "pass": 0.2}
    picks = {sample_human(policy, b, random.Random(i)) for i in range(50)}
    assert picks == {"C3"}  # D4는 이미 돌이 있고, E5는 1% 미만, pass는 제외
    assert sample_human({}, b, random.Random(0)) == "pass"


def test_human_player_samples_from_human_policy(engine):
    # 가짜 human 정책: D4 0.6, C3 0.3, A9 0.1
    rng = random.Random(0)
    picks = [HumanPlayer("rank_5k").choose(engine, Board(9), rng) for _ in range(300)]
    assert set(picks) == {"D4", "C3", "A9"}
    assert picks.count("D4") > picks.count("C3") > picks.count("A9")


def test_hybrid_combines_human_prob_and_target_loss(engine):
    # 후보 C3(손실 0, 사람 0.3), D4(손실 1.5, 사람 0.6). 목표 1.5집이면 D4가 압도적
    rng = random.Random(0)
    picks = [HybridPlayer("rank_5k", 1.5, temperature=0.3).choose(engine, Board(9), rng) for _ in range(200)]
    assert picks.count("D4") > 190
    assert "A9" not in picks  # 엔진 후보가 아닌 수는 손실을 몰라 고르지 않는다


def test_players_pass_when_engine_best_is_pass(engine):
    b = Board(9)
    b.play("C3")  # 가짜 엔진: 이후 pass만 남음
    for player in [BestPlayer(), TargetLossPlayer(3), HumanPlayer("rank_5k"), HybridPlayer("rank_5k", 3)]:
        assert player.choose(engine, b, random.Random(0)) == "pass"


def test_parse_player_names():
    assert [parse_player(s).name for s in ["best", "v2:3", "v3:5k", "mix:1d:2.5"]] == [
        "최강", "v2(3집)", "v3(5k)", "혼합(1d,2.5집)",
    ]
    with pytest.raises(ValueError):
        parse_player("gnugo")


def test_play_game_records_moves_and_metrics(engine):
    g = play_game(engine, BestPlayer(), HumanPlayer("rank_5k"), random.Random(0))
    # 흑 C3 → 이후 모두 pass
    assert g["moves"][0] == {
        "player": "최강", "color": "B", "move": "C3", "loss": 0.0, "winrate_before": 0.7, "phase": "opening",
        "human_p": 0.3, "human_top3": True, "first_line": False,
    }
    assert [m["move"] for m in g["moves"]] == ["C3", "pass", "pass"]
    assert g["result"] == "W+86" and g["black_score"] == 0.0 and not g["capped"]


def test_first_line():
    assert is_first_line("A5", 9) and is_first_line("E9", 9) and not is_first_line("B2", 9)
    assert not is_first_line("pass", 9)


def test_schedule_alternates_colors_and_calibrates_self_play():
    assert schedule(["a", "b"], 2) == [("a", "b"), ("b", "a")]
    assert schedule(["a", "b"], 1, calibrate=True) == [("a", "a"), ("b", "b")]


def test_bradley_terry_orders_by_results():
    games = [{"black": "A", "white": "B", "black_score": 1.0}] * 3 + [{"black": "B", "white": "C", "black_score": 1.0}] * 3
    elo = bradley_terry(["A", "B", "C"], games)
    assert elo["A"] > elo["B"] > elo["C"]
    assert sum(elo.values()) / 3 == pytest.approx(1500, abs=1)


def test_summarize_and_markdown(engine):
    g = play_game(engine, BestPlayer(), HumanPlayer("rank_5k"), random.Random(0))
    rows = summarize([g])
    assert rows["최강"]["win_rate"] == 0.0 and rows["v3(5k)"]["win_rate"] == 1.0
    assert rows["최강"]["avg_loss"] == 0.0
    assert rows["v3(5k)"]["elo"] > rows["최강"]["elo"]
    assert "| v3(5k) | 1 | 100% |" in to_markdown(rows)


def test_resign_when_hopeless(engine):
    # 가짜 엔진: 흑 승률 0.7 → 백 차례엔 0.3. 기준을 0.5로 올리면 백이 바로 돌을 거둔다
    g = play_game(engine, BestPlayer(), BestPlayer(), random.Random(0), resign_winrate=0.5, resign_after=0)
    assert g["result"] == "B+R" and g["black_score"] == 1.0 and g["score"] is None
    assert [m["move"] for m in g["moves"]] == ["C3"]
    assert not g["capped"]


def test_monotone_calibration_enforces_rank_order():
    raw = {"20k": 2.15, "15k": 5.01, "10k": 2.07, "5k": 1.09, "1k": 1.51, "1d": 3.32, "3d": 0.55, "5d": 0.60}
    m = monotone_calibration(raw)
    order = ["20k", "15k", "10k", "5k", "1k", "1d", "3d", "5d"]
    assert all(m[a] >= m[b] for a, b in zip(order, order[1:]))
    assert m["20k"] == m["15k"] == pytest.approx((2.15 + 5.01) / 2, abs=1e-3)
    assert m["3d"] == m["5d"] == pytest.approx(0.575, abs=1e-3)
    assert monotone_calibration({"5k": 2.0, "1d": 1.0}) == {"5k": 2.0, "1d": 1.0}  # 이미 단조면 그대로
