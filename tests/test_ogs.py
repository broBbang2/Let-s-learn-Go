import gzip
import json

from adaptive_go.board import Board
from adaptive_go.ogs import games, ogs_to_stones, to_gtp, usable


def ogs_game(**over):
    g = {
        "game_id": 7, "width": 19, "height": 19, "handicap": 0, "ranked": True, "komi": 6.5,
        "initial_state": {"black": "", "white": ""}, "initial_player": "black", "rules": "japanese",
        "players": {"black": {"username": "a", "rank": 24.0}, "white": {"username": "b", "rank": 25.5}},
        "moves": [[15, 3, 100], [3, 15, 100], [-1, -1, 5]] + [[i % 19, 9 + i // 19, 1] for i in range(60)],
    }
    g.update(over)
    return g


def test_coordinates_and_rank_scale():
    assert to_gtp([15, 3]) == "Q16" and to_gtp([0, 18]) == "A1" and to_gtp([-1, -1]) == "pass"
    assert ogs_to_stones(29) == 0 and ogs_to_stones(30) == 1 and ogs_to_stones(25) == -4  # 1k, 1d, 5k


def test_filters():
    assert usable(ogs_game())
    assert not usable(ogs_game(width=9, height=9))
    assert not usable(ogs_game(handicap=2))
    assert not usable(ogs_game(ranked=False))
    assert not usable(ogs_game(moves=[[1, 1, 1]] * 10))
    assert not usable(ogs_game(initial_state={"black": "aabb", "white": ""}))
    assert not usable(ogs_game(players={"black": {"username": "katabot", "rank": 24}, "white": {"username": "b", "rank": 24}}))
    assert not usable(ogs_game(players={"black": {"username": "a", "rank": 20}, "white": {"username": "b", "rank": 25}}))


def test_games_reader(tmp_path):
    path = tmp_path / "s.json.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(ogs_game()) + "\n" + json.dumps(ogs_game(ranked=False)) + "\n")
    gs = list(games(path))
    assert len(gs) == 1
    g = gs[0]
    assert g["moves"][:3] == [("B", "Q16"), ("W", "D4"), ("B", "pass")]
    assert g["rank"] == {"B": -5.0, "W": -3.5}
    b = Board(19)
    for _, v in g["moves"][:3]:
        b.play(v)


def test_human_eval_metrics_and_sequences():
    import math

    from adaptive_go.human_eval import bucket_of, metrics, sequences
    from adaptive_go.rank_estimate import RANKS

    assert bucket_of(-18) == "20k대" and bucket_of(-4) == "5~1k" and bucket_of(2) == "1d+"
    # 정답 급수에서만 확률이 높은 합성 기록 → 수가 쌓이면 오차가 줄어야 한다
    def seq(true_rank, n):
        return {"true": {"20k": -19, "5k": -4, "3d": 3}[true_rank],
                "loglik": [[math.log(0.3 if r == true_rank else 0.05) for r in RANKS]] * n}
    m = metrics([seq("20k", 30), seq("5k", 30), seq("3d", 30)], checkpoints=[1, 30])
    assert m[30]["mae"] < m[1]["mae"] and m[30]["corr"] > 0.9

    # 정책 함수가 프로필 12개를 받고, 흑·백이 따로 기록되는지
    g = {"game_id": 1, "komi": 6.5, "rank": {"B": -5.0, "W": -3.0},
         "moves": [("B", "Q16"), ("W", "D4"), ("B", "pass"), ("W", "C3")]}
    calls = []
    def pol(board, profiles):
        calls.append(len(profiles))
        return {p: {"Q16": 0.5, "D4": 0.5, "C3": 0.5} for p in profiles}
    seqs = sequences(pol, [g], 10)
    assert calls == [12, 12, 12]  # 패스는 건너뜀
    assert [len(s["loglik"]) for s in seqs] == [1, 2] and [s["true"] for s in seqs] == [-5.0, -3.0]


def test_ogs_game_rows_are_per_color():
    from adaptive_go.ogs_data import game_rows

    g = {"game_id": 1, "komi": 6.5, "rank": {"B": -5.0, "W": -3.0},
         "moves": [("B", "Q16"), ("W", "D4"), ("B", "pass"), ("W", "C3"), ("B", "Q16")]}  # 마지막은 둘 수 없는 자리
    rows = game_rows(g, 10)
    assert [len(rows["B"]), len(rows["W"])] == [2, 2]  # 둘 수 없는 수에서 멈춤
    x, top_i, top_p, played, num = rows["W"][0]
    assert top_p == [1.0] and top_i == [played] and num == 1
