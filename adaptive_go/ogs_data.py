"""OGS 실제 사람 기보 → 학습 데이터(npz 조각). datagen과 같은 형식이라 train.py가 그대로 읽는다.

python -m adaptive_go.ogs_data --max-moves 120

- 검증용 판(train.is_val)은 뺀다 → human_eval 결과가 학습 데이터에 오염되지 않음
- 정답: 사람이 실제로 둔 수 하나 (teacher 분포가 없으니 top_idx 첫 칸에 그 수, 확률 1)
- 급수: 둔 사람의 OGS 급수에서 가장 가까운 격자 급수(rank_estimate.RANKS). 흑·백 급수가 달라서
  한 판의 흑 수와 백 수를 별개 "판"으로 묶어 pack에 넘긴다 (게임 번호는 같다)
- 판 끝부분은 끝내기·집 정리라 급수 신호가 약하고 양이 많아, 앞 max_moves수만 쓴다
"""

import argparse
import time
from pathlib import Path

import numpy as np

from . import progress
from .board import Board, IllegalMoveError
from .datagen import pack
from .features import encode, move_index
from .ogs import SAMPLE, games
from .rank_estimate import nearest_rank
from .train import is_val

ROOT = Path(__file__).resolve().parent.parent


def game_rows(g, max_moves):
    """한 판 → {"B": rows, "W": rows}. rows는 datagen.play_game과 같은 모양."""
    board = Board(19, g["komi"])
    rows = {"B": [], "W": []}
    for color, move in g["moves"][:max_moves]:
        if color != board.to_play:
            break
        try:
            idx = move_index(move, 19, board)
            x = encode(board)
            num = len(board.history)
            board.play(move)
        except IllegalMoveError:
            break
        rows[color].append((x, [idx], [1.0], idx, num))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="OGS 사람 기보 → 학습 데이터")
    p.add_argument("--max-moves", type=int, default=120)
    p.add_argument("--shard-games", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="판 수 제한 (0이면 전부)")
    p.add_argument("--out", default=str(ROOT / "data/train"))
    args = p.parse_args(argv)
    progress.tee_stdout("ogs_data")

    out = Path(args.out) / ("ogs-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"OGS 기보 변환 → {out}", flush=True)
    buf, shard, n_games, n_pos, t0 = [], 0, 0, 0, time.time()
    for g in games(SAMPLE):
        if is_val(g["game_id"]):
            continue
        rows = game_rows(g, args.max_moves)
        for color in ("B", "W"):
            if rows[color]:
                buf.append((nearest_rank(g["rank"][color]), g["game_id"], rows[color]))
                n_pos += len(rows[color])
        n_games += 1
        if n_games % args.shard_games == 0:
            np.savez_compressed(out / f"shard-{shard:03d}.npz", **pack(buf, 19))
            shard, buf = shard + 1, []
            print(f"[{n_games}] 국면 {n_pos}개, {time.time() - t0:.0f}초", flush=True)
        if args.limit and n_games >= args.limit:
            break
    if buf:
        np.savez_compressed(out / f"shard-{shard:03d}.npz", **pack(buf, 19))
    print(f"끝: 판 {n_games}개, 국면 {n_pos}개, {time.time() - t0:.0f}초 → {out}", flush=True)


if __name__ == "__main__":
    main()
