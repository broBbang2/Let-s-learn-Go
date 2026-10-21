import math

from adaptive_go.estimate_eval import evaluate, per_rank
from adaptive_go.rank_estimate import RANKS


def seq(true_rank, n, peak=0.3, other=0.1):
    """true_rank에서만 수 확률이 높은 합성 기록 n수."""
    return {"true_rank": true_rank, "loglik": [[math.log(peak if r == true_rank else other) for r in RANKS]] * n}


def test_evaluate_converges_with_more_moves():
    sequences = [seq("5k", 30), seq("3d", 30)]
    out = evaluate(sequences, weights=[0.5], checkpoints=[1, 30])[0.5]
    assert out[30]["mean_abs_err_stones"] < out[1]["mean_abs_err_stones"]
    assert out[30]["map_within_1"] == 1.0


def test_short_sequences_are_skipped_at_later_checkpoints():
    out = evaluate([seq("5k", 4)], weights=[0.5], checkpoints=[3, 5])[0.5]
    assert out[3]["n"] == 1 and 5 not in out


def test_per_rank_reports_nearest_grid_rank():
    assert per_rank([seq("5k", 30)], 0.5, 20)["5k"]["mean_estimate"] == "5k"
