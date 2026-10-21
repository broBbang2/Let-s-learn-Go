import pytest

from adaptive_go.rating import Rating, update


def test_glickman_paper_example():
    # Glickman (2013) 논문의 예제: 1400 승, 1550 패, 1700 패
    player = Rating(1500, 200, 0.06)
    results = [(Rating(1400, 30), 1), (Rating(1550, 100), 0), (Rating(1700, 300), 0)]
    new = update(player, results)
    assert new.rating == pytest.approx(1464.06, abs=0.01)
    assert new.rd == pytest.approx(151.52, abs=0.01)
    assert new.volatility == pytest.approx(0.05999, abs=1e-5)


def test_no_games_only_increases_rd():
    new = update(Rating(1500, 200, 0.06), [])
    assert new.rating == 1500
    assert new.rd == pytest.approx(200.27, abs=0.01)


def test_losing_to_much_stronger_opponent_costs_little():
    player = Rating()
    lose_strong = update(player, [(Rating(2800, 50), 0)])
    lose_equal = update(player, [(Rating(1500, 50), 0)])
    assert lose_strong.rating < player.rating
    assert player.rating - lose_strong.rating < player.rating - lose_equal.rating
    assert lose_strong.rd < player.rd
