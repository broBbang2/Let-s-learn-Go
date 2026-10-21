"""학습 데이터 생성: KataGo human 모델로 급수별 대국을 두며 (국면, 급수, 정답 분포)를 모은다.

python -m adaptive_go.datagen --games-per-rank 60 --workers 3

- 수 선택: 급수 프로필의 사람 정책에서 확률 비례로 뽑는다 (탐색 없이 maxVisits=1 → 빠름)
- 정답(teacher): 그 국면의 사람 정책 상위 TOP_K개 확률. 실제 둔 한 수보다 정보가 많아
  작은 모델이 더 빨리 배운다(지식 증류). 실제 둔 수도 같이 저장해 정확도 지표로 쓴다
- 저장: 워커마다 npz 조각. 평면은 비트로 압축(국면당 12×361비트 ≈ 0.5KB)

검증용은 게임 단위로 나눈다(같은 판의 국면이 학습·검증에 섞이면 점수가 부풀려짐).
"""

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import arena, progress
from .board import Board, IllegalMoveError
from .features import PLANES, encode, move_index
from .rank_estimate import RANKS

TOP_K = 16
# 꼬리의 아주 드문 수는 뽑지 않는다 (players.HUMAN_MIN_PROB보다 낮게 잡아 다양성은 남김)
SAMPLE_MIN_PROB = 0.005
MAX_MOVES = 260


def play_game(engine, rank, rng, size=19, komi=7.5):
    """한 판. 국면마다 (평면, 정답 인덱스·확률, 실제 수) 목록을 돌려준다."""
    board = Board(size, komi)
    rows = []
    profile = f"rank_{rank}"
    while not board.is_over() and len(board.history) < MAX_MOVES:
        policy = engine.analyze(board, max_visits=1, human_profile=profile).human_policy
        legal = {m: p for m, p in policy.items() if board.is_legal(m)}
        if not legal:
            break
        top = sorted(legal, key=legal.get, reverse=True)[:TOP_K]
        pool = [m for m in legal if legal[m] >= SAMPLE_MIN_PROB] or top[:1]
        move = rng.choices(pool, [legal[m] for m in pool])[0]
        rows.append(
            (
                encode(board),
                [move_index(m, size, board) for m in top],
                [legal[m] for m in top],
                move_index(move, size, board),
                len(board.history),
            )
        )
        try:
            board.play(move)
        except IllegalMoveError:  # 엔진·board.py 패 규칙 차이 대비
            break
    return rows


def _run(job):
    rank, seed, size = job
    rows = play_game(arena._engine, rank, random.Random(seed), size)
    return rank, seed, rows


def pack(results, size):
    """[(rank, seed, rows)] → npz용 배열들."""
    planes, idx, prob, played, rank_i, game, move_no = [], [], [], [], [], [], []
    for rank, seed, rows in results:
        for x, top_i, top_p, mv, num in rows:
            planes.append(np.packbits(x.reshape(-1)))
            idx.append(top_i + [-1] * (TOP_K - len(top_i)))
            prob.append(top_p + [0.0] * (TOP_K - len(top_p)))
            played.append(mv)
            rank_i.append(RANKS.index(rank))
            game.append(seed)
            move_no.append(num)
    return {
        "planes": np.array(planes, dtype=np.uint8),
        "top_idx": np.array(idx, dtype=np.int16),
        "top_p": np.array(prob, dtype=np.float16),
        "played": np.array(played, dtype=np.int16),
        "rank": np.array(rank_i, dtype=np.int8),
        "game": np.array(game, dtype=np.int32),
        "move_no": np.array(move_no, dtype=np.int16),
        "size": np.array(size),
        "planes_n": np.array(PLANES),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="학습 데이터 생성")
    p.add_argument("--games-per-rank", type=int, default=60)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--size", type=int, default=19)
    p.add_argument("--shard-games", type=int, default=60, help="이 판 수마다 조각 파일로 저장")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(arena.ROOT / "data/train"))
    args = p.parse_args(argv)
    progress.tee_stdout("datagen")

    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    # 급수를 섞어서 배치해 조각마다 급수가 고르게 들어가게
    jobs = [(r, args.seed + g * len(RANKS) + i, args.size) for g in range(args.games_per_rank) for i, r in enumerate(RANKS)]
    print(f"{len(jobs)}판 → {out}", flush=True)
    init = (str(arena.DEFAULT_MODEL), str(arena.DEFAULT_CONFIG), str(arena.HUMAN_MODEL))
    buf, shard, positions, t0 = [], 0, 0, time.time()
    with ProcessPoolExecutor(args.workers, initializer=arena._init_worker, initargs=init) as pool:
        for i, res in enumerate(pool.map(_run, jobs), 1):
            buf.append(res)
            positions += len(res[2])
            print(f"[{i}/{len(jobs)}] {res[0]} 국면 {positions}개, {time.time() - t0:.0f}초"
                  f"   {progress.status(i, len(jobs), t0)}", flush=True)
            if len(buf) >= args.shard_games or i == len(jobs):
                np.savez_compressed(out / f"shard-{shard:03d}.npz", **pack(buf, args.size))
                print(f"  조각 저장: shard-{shard:03d}.npz ({len(buf)}판)", flush=True)
                shard, buf = shard + 1, []


if __name__ == "__main__":
    main()
