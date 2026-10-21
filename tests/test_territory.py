from adaptive_go.territory import change, classify_mistake, expected, weakness


def grid_of(n, stones):
    g = [["."] * n for _ in range(n)]
    for (r, c), color in stones.items():
        g[r][c] = color
    return g


def test_expected_from_each_side():
    own = [[1.0, -1.0], [0.5, -0.25]]
    assert expected(own, "B") == (1.5, 1.25)
    assert expected(own, "W") == (1.25, 1.5)


def test_change_counts_gain_points_and_lost_stones():
    before = [[0.8, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, -0.9]]
    after = [[-0.8, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, -0.9]]
    # 흑 돌 두 개(A1, B1)가 백 땅으로 넘어가고, B2는 흑 땅이 됨
    grid = grid_of(3, {(0, 0): "B", (0, 1): "B"})
    d = change(before, after, "B", grid)
    assert d["lost_stones"] == 2 and d["won_stones"] == 0
    assert set(d["points"]) == {"A1", "B1", "B2"}
    assert d["mine"] == round((0.8 + 0.6) - (0.8 * 3), 1) and d["theirs"] == round((1.6 + 0.9) - 0.9, 1)
    # 백 쪽에서 보면 같은 변화가 반대로
    w = change(before, after, "W", grid)
    assert w["mine"] == d["theirs"] and w["won_stones"] == 2
    assert change(None, after, "B", grid) is None


def test_classify_mistake():
    life = {"lost_stones": 3}
    calm = {"lost_stones": 0}
    assert classify_mistake(1.0, "middle", life) is None  # 큰 실수 아님
    assert classify_mistake(None, "middle", life) is None  # 승부가 기운 국면
    assert classify_mistake(5.0, "opening", life) == "life"
    assert classify_mistake(5.0, "opening", calm) == "opening"
    assert classify_mistake(5.0, "endgame", None) == "endgame"


def test_weakness_summary():
    rows = [
        {"mistake_type": "life", "score_loss": 6.0},
        {"mistake_type": "life", "score_loss": 4.0},
        {"mistake_type": "endgame", "score_loss": 3.0},
        {"score_loss": 0.2},
    ]
    w = weakness(rows)
    assert list(w) == ["life", "endgame"]
    assert w["life"] == {"count": 2, "avg_loss": 5.0, "label": "사활·전투 (돌이 잡힘)"}
