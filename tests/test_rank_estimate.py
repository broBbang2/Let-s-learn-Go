import pytest

from adaptive_go.rank_estimate import RANKS, RankEstimate, nearest_rank, rank_label, stones


def policies(weak_p, strong_p, move="X"):
    """약한 급수일수록 move 확률이 weak_p, 강할수록 strong_p로 선형 변화."""
    n = len(RANKS)
    return {f"rank_{r}": {move: weak_p + (strong_p - weak_p) * i / (n - 1)} for i, r in enumerate(RANKS)}


def test_stones_scale():
    assert stones("1k") == 0 and stones("1d") == 1 and stones("5k") == -4 and stones("20k") == -19
    assert nearest_rank(-2.4) == "3k" and nearest_rank(100) == "9d"
    assert rank_label("5k") == "5급" and rank_label("3d") == "3단"


def test_uniform_start():
    e = RankEstimate()
    assert all(p == pytest.approx(1 / len(RANKS)) for p in e.probs())
    assert e.mean_stones() == pytest.approx(sum(stones(r) for r in RANKS) / len(RANKS))


def test_weak_moves_pull_estimate_down():
    e = RankEstimate()
    e.update(policies(0.3, 0.01), "X")  # 하수가 자주 두는 수
    one = e.mean_stones()
    for _ in range(29):
        e.update(policies(0.3, 0.01), "X")
    assert e.map_rank() == "20k"
    assert e.mean_stones() < one < RankEstimate().mean_stones()  # 수가 쌓일수록 더 약한 쪽으로
    assert e.confidence() > 0.8
    assert e.moves == 30


def test_strong_moves_pull_estimate_up():
    e = RankEstimate()
    for _ in range(10):
        e.update(policies(0.01, 0.3), "X")
    assert e.map_rank() == "9d"


def test_zero_probability_is_floored():
    e = RankEstimate()
    e.update(policies(0.0, 0.0), "X")  # 아무 급수도 안 둘 수 → 정보 없음, 터지지 않음
    assert all(p == pytest.approx(1 / len(RANKS)) for p in e.probs())


def test_next_game_prior_is_flatter():
    e = RankEstimate()
    for _ in range(10):
        e.update(policies(0.3, 0.01), "X")
    prior = e.next_game_prior()
    assert max(prior) < max(e.probs())
    assert RankEstimate.from_prior(prior).map_rank() == "20k"


def test_new_user_starts_from_population_prior():
    from adaptive_go.rank_estimate import POPULATION_PRIOR

    e = RankEstimate.from_prior(None)
    assert abs(sum(POPULATION_PRIOR) - 1) < 0.01
    assert nearest_rank(e.mean_stones()) == "7k"  # OGS 사용자 분포의 가운데
    assert e.probs()[-1] > 0  # 표본 없는 9단도 0은 아님
