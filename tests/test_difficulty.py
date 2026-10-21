import math
import random

import pytest

from adaptive_go.difficulty import (
    PRIOR_USER_LOSS,
    TARGET_USER_SCORE,
    DifficultyState,
    choose_move,
    level_label,
    level_of,
    recent_user_loss,
)
from adaptive_go.engine import Analysis, MoveInfo


def analysis(*moves):
    """(move, score_lead, visits) 목록으로 Analysis를 만든다. 첫 번째가 최선수."""
    infos = [MoveInfo(m, v, 0.5, s, i) for i, (m, s, v) in enumerate(moves)]
    return Analysis(to_play="B", moves=infos, winrate=0.5, score_lead=moves[0][1])


A = analysis(("E5", 2.0, 100), ("D4", 1.0, 30), ("C3", -1.0, 10), ("J9", -6.0, 1), ("pass", -20.0, 5))


def test_choose_move_tracks_target_loss():
    rng = random.Random(0)
    picks = [choose_move(A, 3.0, temperature=0.3, rng=rng)[0].move for _ in range(500)]
    # 손실 3집인 C3가 압도적으로 많이 뽑힌다
    assert picks.count("C3") > 400


def test_choose_move_excludes_pass_and_low_visit_moves():
    rng = random.Random(0)
    picks = {choose_move(A, 20.0, rng=rng)[0].move for _ in range(300)}
    assert "pass" not in picks and "J9" not in picks


def test_choose_move_returns_loss():
    move, loss = choose_move(A, 1.0, temperature=0.01, rng=random.Random(0))
    assert move.move == "D4" and loss == 1.0


def test_zero_target_and_best_pass_return_best():
    assert choose_move(A, 0)[0].move == "E5"
    end = analysis(("pass", 0.5, 50), ("A1", -1.0, 10))
    assert choose_move(end, 5.0)[0].move == "pass"


def test_recent_user_loss_shrinks_to_prior():
    assert recent_user_loss([]) == PRIOR_USER_LOSS
    assert recent_user_loss([1.0] * 20) == pytest.approx(2.0)  # 실제 20수 + 가상 20수
    # 최근 200수만 쓴다: (200 + 3.0 * 20) / 220
    assert recent_user_loss([9.0] * 1000 + [1.0] * 200) == pytest.approx(260 / 220)


def test_difficulty_converges_to_target_win_rate():
    # 사용자 승률이 목표와 같으면 계수가 기대값 기준으로 움직이지 않는다
    state = DifficultyState()
    win = state.after_game(1.0)
    loss = state.after_game(0.0)
    expected = TARGET_USER_SCORE * win.log_factor + (1 - TARGET_USER_SCORE) * loss.log_factor
    assert expected == pytest.approx(0.0)
    assert win.log_factor < 0 < loss.log_factor
    assert DifficultyState(log_factor=-0.5).target_loss(2.0) == pytest.approx(2.0 * math.exp(-0.5))


def test_difficulty_is_clamped():
    state = DifficultyState()
    for _ in range(100):
        state = state.after_game(0.0)
    assert state.log_factor == 3.0 and state.games == 100


def test_levels():
    assert level_of(0.1) == 0 and level_of(0.75) == 2 and level_of(20) == 6
    assert level_label(2) == "0.5~1집"
    assert level_label(6) == "8집 이상"


def test_rank_offset_moves_with_results():
    s = DifficultyState()
    assert s.after_game(1.0).rank_offset > 0  # 사용자가 이기면 AI 급수를 올린다
    assert s.after_game(0.0).rank_offset < 0
    for _ in range(100):
        s = s.after_game(0.0)
    assert s.rank_offset == -10.0
