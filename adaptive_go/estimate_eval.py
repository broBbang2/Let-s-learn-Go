"""실력 추정 검증: 급수를 아는 v3 봇을 "사용자"로 두고, 몇 수 만에 맞히는지 잰다.

python -m adaptive_go.estimate_eval 15k 5k 1k 3d --games 3 --workers 3

주의: 봇은 추정에 쓰는 바로 그 human 모델에서 수를 뽑으므로, 실제 사람보다 맞히기 쉽다(낙관적 상한).
실제 사용자 데이터가 쌓이면 그걸로 다시 검증해야 한다.
"""

import argparse
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import arena, progress
from .board import BLACK, Board
from .players import HumanPlayer
from .rank_estimate import PROB_FLOOR, RANKS, RankEstimate, nearest_rank, stones

CHECKPOINTS = [3, 5, 10, 15, 20, 30]
WEIGHTS = [0.3, 0.5, 1.0]


def play_and_record(engine, rank, rng, size=9, komi=7.0):
    """같은 급수끼리 한 판. 양쪽 각각의 수마다 급수별 log p(수 | 급수)를 기록."""
    board = Board(size, komi)
    player = HumanPlayer(f"rank_{rank}")
    loglik = {BLACK: [], "W": []}
    max_moves = int(size * size * 1.5)
    while not board.is_over() and len(board.history) < max_moves:
        # 돌 거두기는 하지 않는다: 같은 급수끼리면 25수 무렵 승부가 기울어 기권하는데,
        # 그러면 한쪽당 13수 정도라 "몇 수면 맞히는가"를 15수 이후로 볼 수 없었다
        move, _ = player.decide(engine, board, rng)
        if not board.is_legal(move):
            move = "pass"
        if move != "pass":
            pol = engine.human_policies(board, [f"rank_{r}" for r in RANKS])
            loglik[board.to_play].append([math.log(max(pol[f"rank_{r}"].get(move, 0.0), PROB_FLOOR)) for r in RANKS])
        board.play(move)
    return [{"true_rank": rank, "loglik": seq} for seq in loglik.values()]


def _run(job):
    rank, seed = job
    return play_and_record(arena._engine, rank, random.Random(seed))


def evaluate(sequences, weights=WEIGHTS, checkpoints=CHECKPOINTS):
    """가중치·수 개수별로: 평균 추정의 절대 오차(점), 가장 그럴듯한 급수가 격자 ±1칸 안인 비율."""
    out = {}
    for w in weights:
        rows = {}
        for n in checkpoints:
            errs, hits = [], []
            for s in sequences:
                if len(s["loglik"]) < n:
                    continue
                e = RankEstimate([w * sum(col) for col in zip(*s["loglik"][:n])])
                errs.append(abs(e.mean_stones() - stones(s["true_rank"])))
                hits.append(abs(RANKS.index(e.map_rank()) - RANKS.index(s["true_rank"])) <= 1)
            if errs:
                rows[n] = {
                    "n": len(errs),
                    "mean_abs_err_stones": round(sum(errs) / len(errs), 2),
                    "map_within_1": round(sum(hits) / len(hits), 2),
                }
        out[w] = rows
    return out


def per_rank(sequences, weight, n):
    """급수별로 n수 뒤 평균 추정 (어느 쪽으로 치우치는지 보기 위함)."""
    out = {}
    for rank in sorted({s["true_rank"] for s in sequences}, key=stones):
        ests = [
            RankEstimate([weight * sum(col) for col in zip(*s["loglik"][:n])]).mean_stones()
            for s in sequences
            if s["true_rank"] == rank and len(s["loglik"]) >= n
        ]
        if ests:
            mean = sum(ests) / len(ests)
            out[rank] = {"games": len(ests), "mean_estimate": nearest_rank(mean), "mean_stones": round(mean, 2)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="실력 추정 검증")
    p.add_argument("ranks", nargs="+")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=str(arena.ROOT / "data/arena"))
    args = p.parse_args(argv)
    progress.tee_stdout("estimate")
    started = time.time()

    jobs = [(r, 1000 + i) for i, r in enumerate(r for r in args.ranks for _ in range(args.games))]
    out_dir = Path(args.out) / ("estimate-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)}판 → {out_dir}", flush=True)
    init = (str(arena.DEFAULT_MODEL), str(arena.DEFAULT_CONFIG), str(arena.HUMAN_MODEL))
    sequences = []
    with ProcessPoolExecutor(args.workers, initializer=arena._init_worker, initargs=init) as pool:
        for i, seqs in enumerate(pool.map(_run, jobs), 1):
            sequences += seqs
            print(f"[{i}/{len(jobs)}] {seqs[0]['true_rank']}: 수 {[len(s['loglik']) for s in seqs]}"
                  f"   {progress.status(i, len(jobs), started)}", flush=True)
    (out_dir / "sequences.json").write_text(json.dumps(sequences))
    result = {
        "by_weight": evaluate(sequences),
        "per_rank_w0.5_n10": per_rank(sequences, 0.5, 10),
        "per_rank_w0.5_n30": per_rank(sequences, 0.5, 30),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
