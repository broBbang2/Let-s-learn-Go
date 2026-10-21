import sys
from pathlib import Path

import pytest

from adaptive_go.board import BLACK, Board
from adaptive_go.engine import KataGoEngine
from adaptive_go.records import Records
from adaptive_go.skill import LOSS_CAP, effective_loss, evaluate_move, phase_of, summarize

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def engine():
    with KataGoEngine(FAKE) as e:
        yield e


def test_loss_for_candidate_move(engine):
    # 가짜 엔진: 최선 C3 = 3.0집 / 0.7, D4 = 1.5집 / 0.65
    b = Board(9)
    ev = evaluate_move(engine, b, "d4")
    assert ev.move == "D4" and ev.best_move == "C3"
    assert ev.score_loss == 1.5
    assert ev.winrate_loss == pytest.approx(0.05)
    assert ev.in_candidates
    assert ev.move_number == 1 and ev.color == BLACK
    assert ev.candidates[0] == ["C3", 3.0, 0.7, 60]
    assert b.history == []  # 평가는 판을 바꾸지 않는다


def test_best_move_has_zero_loss(engine):
    ev = evaluate_move(engine, Board(9), "C3")
    assert ev.score_loss == 0 and ev.winrate_loss == 0


def test_move_outside_candidates_uses_position_after(engine):
    # E5는 후보수에 없음 → 둔 뒤 국면(백 차례)을 분석해 뒤집는다
    # 가짜 엔진의 그 국면 값은 백 기준 -3.0집 → 사용자(흑) 기준 3.0집 → 손실 0
    ev = evaluate_move(engine, Board(9), "E5")
    assert not ev.in_candidates
    assert ev.score_loss == 0


def test_phase_scales_with_board_size():
    assert phase_of(10, 9) == "opening"
    assert phase_of(30, 9) == "middle"
    assert phase_of(60, 9) == "endgame"
    assert phase_of(50, 19) == "opening"
    assert phase_of(150, 19) == "middle"


def test_summarize(engine):
    b = Board(9)
    evals = [evaluate_move(engine, b, "D4"), evaluate_move(engine, b, "pass")]
    s = summarize(evals)
    assert s["moves"] == 2
    assert s["avg_score_loss"] == pytest.approx((1.5 + 13.0) / 2)
    assert s["blunders"] == 1  # pass: 13집 손실
    assert s["by_phase"]["opening"]["moves"] == 2
    assert summarize([])["avg_score_loss"] is None


def test_effective_loss_counts_contested_positions_only():
    assert effective_loss(2.0, 0.5) == 2.0
    assert effective_loss(30.0, 0.5) == LOSS_CAP  # 대마 하나 죽는 수도 상한에서 자름
    assert effective_loss(30.0, 0.01) is None  # 이미 진 판
    assert effective_loss(30.0, 0.99) is None  # 이미 이긴 판


def test_recent_user_losses_skip_decided_positions(tmp_path):
    records = Records(tmp_path)
    for loss, wr in [(1.0, 0.5), (40.0, 0.02), (20.0, 0.6), (2.0, 0.4)]:
        records._append("moves.jsonl", {"score_loss": loss, "winrate_before": wr})
    assert records.recent_user_losses(10) == [1.0, LOSS_CAP, 2.0]
    assert records.recent_user_losses(2) == [LOSS_CAP, 2.0]
