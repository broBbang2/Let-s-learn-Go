"""학습한 모델 평가: 검증용 판(학습에 안 쓴 판)에서 실력 추정을 teacher(KataGo human 모델)와 비교.

python -m adaptive_go.eval_model data/train/<시각> models/rank_policy.pt --games-per-rank 4 --moves 40

각 판의 수마다 급수 12개에 대한 log p(실제 수 | 급수)를 두 모델로 구하고,
estimate_eval.evaluate로 "몇 수 만에 급수를 맞히는가"를 같은 방식으로 잰다.
주의: 판 자체가 teacher로 생성됐으므로 teacher에게 유리한 조건(홈 경기)이다.
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import torch

from . import arena, progress
from .board import Board
from .engine import KataGoEngine
from .estimate_eval import evaluate
from .features import encode, index_vertex
from .model import NetPolicy
from .rank_estimate import PROB_FLOOR, RANKS
from .train import is_val, load_shards


def val_games(data, games_per_rank):
    """검증 판들: {게임 번호: (급수, 수 인덱스 목록)}. 급수마다 앞에서 games_per_rank판."""
    by_game = defaultdict(list)
    for i in np.nonzero(is_val(data["game"]))[0]:
        by_game[int(data["game"][i])].append((int(data["move_no"][i]), int(data["played"][i]), int(data["rank"][i])))
    picked, count = {}, defaultdict(int)
    for g in sorted(by_game):
        rows = sorted(by_game[g])
        rank = RANKS[rows[0][2]]
        if count[rank] < games_per_rank:
            picked[g] = (rank, [mv for _, mv, _ in rows])
            count[rank] += 1
    return picked


def loglik_sequences(policy_fn, games, size, moves):
    """policy_fn(board) → {프로필: 분포}. 판마다 흑·백 따로 [급수별 log p] 목록."""
    profiles = [f"rank_{r}" for r in RANKS]
    sequences = []
    for rank, played in games.values():
        board = Board(size)
        per_color = {"B": [], "W": []}
        for idx in played[:moves]:
            move = index_vertex(idx, size, board)
            if move != "pass":
                pol = policy_fn(board, profiles)
                per_color[board.to_play].append([math.log(max(pol[p].get(move, 0.0), PROB_FLOOR)) for p in profiles])
            board.play(move)
        sequences += [{"true_rank": rank, "loglik": seq} for seq in per_color.values()]
    return sequences


def main(argv=None):
    p = argparse.ArgumentParser(description="학습한 모델의 실력 추정 평가")
    p.add_argument("data", nargs="+")
    p.add_argument("--model", default=str(arena.ROOT / "models/rank_policy.pt"))
    p.add_argument("--games-per-rank", type=int, default=4)
    p.add_argument("--moves", type=int, default=60, help="판마다 앞에서 몇 수까지 (추정은 초반 수십 수가 관심사)")
    args = p.parse_args(argv)
    progress.tee_stdout("eval_model")

    data = load_shards(args.data)
    size = data["size"]
    games = val_games(data, args.games_per_rank)
    print(f"검증 판 {len(games)}개, 판마다 {args.moves}수", flush=True)

    ours = NetPolicy(args.model)
    seq_ours = loglik_sequences(ours.human_policies, games, size, args.moves)
    with KataGoEngine.from_paths(
        "katago", str(arena.DEFAULT_MODEL), str(arena.DEFAULT_CONFIG), human_model=str(arena.HUMAN_MODEL)
    ) as engine:
        seq_teacher = loglik_sequences(engine.human_policies, games, size, args.moves)

    checkpoints = [3, 5, 10, 20, 30]
    result = {
        "ours": evaluate(seq_ours, weights=[0.5], checkpoints=checkpoints)[0.5],
        "teacher": evaluate(seq_teacher, weights=[0.5], checkpoints=checkpoints)[0.5],
        "model": args.model,
        "games": len(games),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
