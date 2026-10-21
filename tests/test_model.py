import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_go.board import Board
from adaptive_go.datagen import pack, play_game
from adaptive_go.engine import KataGoEngine
from adaptive_go.features import PLANES, encode
from adaptive_go.model import NetPolicy, RankPolicyNet, save
from adaptive_go.rank_estimate import RANKS
from adaptive_go.train import Batches, load_shards, run_epoch

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def shard_dir(tmp_path):
    with KataGoEngine(FAKE) as e:
        results = [(r, i, play_game(e, r, random.Random(i), size=9)) for i, r in enumerate(["20k", "5k", "3d"] * 4)]
    np.savez_compressed(tmp_path / "shard-000.npz", **pack(results, 9))
    return tmp_path


def test_forward_shape_and_illegal_mask():
    b = Board(9)
    b.play("E5")
    x = torch.from_numpy(encode(b)).float()[None]
    logits = RankPolicyNet(channels=8, blocks=1)(x, torch.tensor([0]))
    assert logits.shape == (1, 82)
    assert logits[0, 4 * 9 + 4] == float("-inf")  # E5엔 돌이 있음
    assert torch.isfinite(logits[0, 81])  # 패스는 항상 가능


def test_batches_augment_keeps_target_on_same_point(shard_dir):
    data = load_shards([shard_dir])
    rng = np.random.default_rng(0)
    batches = Batches(data, np.arange(len(data["rank"])), 8, True, rng)
    x, rank, target, played = next(iter(batches))
    assert torch.allclose(target.sum(1), torch.ones(len(x)))
    # 정답 분포는 빈 점에만 있어야 한다(대칭 변환이 판과 정답을 같이 돌렸는지)
    occupied = ((x[:, 0] + x[:, 1]) > 0).flatten(1)
    assert (target[:, :81][occupied] == 0).all()


def test_tiny_model_learns_teacher(shard_dir):
    data = load_shards([shard_dir])
    torch.manual_seed(0)
    model = RankPolicyNet(channels=16, blocks=1)
    rng = np.random.default_rng(0)
    batches = Batches(data, np.arange(len(data["rank"])), 32, False, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    before = run_epoch(model, batches, "cpu")["loss"]
    for _ in range(15):
        run_epoch(model, batches, "cpu", opt, sched)
    after = run_epoch(model, batches, "cpu")
    assert after["loss"] < before * 0.8
    assert after["top1_teacher"] > 0.5


def test_net_policy_has_human_policies_interface(tmp_path):
    path = tmp_path / "m.pt"
    save(RankPolicyNet(channels=8, blocks=1), path)
    policy = NetPolicy(path, device="cpu")
    b = Board(9)
    b.play("E5")
    out = policy.human_policies(b, ["rank_20k", "rank_9d"])
    assert set(out) == {"rank_20k", "rank_9d"}
    assert "E5" not in out["rank_20k"] and abs(sum(out["rank_9d"].values()) - 1) < 1e-3
    assert len(RANKS) == 12 and PLANES == 12


def test_eval_helpers(shard_dir, tmp_path):
    from adaptive_go.eval_model import loglik_sequences, val_games
    from adaptive_go.train import is_val

    data = load_shards([shard_dir])
    games = val_games(data, 1)
    assert games and all(is_val(g) for g in games)  # 검증용 판만
    ranks = [r for r, _ in games.values()]
    assert len(ranks) == len(set(ranks))  # 급수마다 1판
    path = tmp_path / "m.pt"
    save(RankPolicyNet(channels=8, blocks=1), path)
    seqs = loglik_sequences(NetPolicy(path, device="cpu").human_policies, games, 9, 10)
    assert {s["true_rank"] for s in seqs} == set(ranks)
    assert all(len(row) == 12 for s in seqs for row in s["loglik"])


def test_val_split_covers_every_rank():
    from adaptive_go.train import is_val

    seeds = np.array([g * 12 + i for g in range(80) for i in range(12)])  # datagen 번호 규칙
    val = is_val(seeds)
    assert 0.07 < val.mean() < 0.13
    assert all(val[i::12].sum() >= 3 for i in range(12))  # 모든 급수가 검증에 들어간다
