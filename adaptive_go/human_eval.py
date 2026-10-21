"""실제 사람(OGS) 기보로 실력 추정 검증.

python -m adaptive_go.human_eval --per-bucket 30 --moves 60 --model models/rank_policy.pt

- 검증 판: train.is_val(game_id)인 판만 (학습 데이터 변환에서도 같은 함수로 빼므로 겹치지 않음)
- 급수대 5개(20k대·10k대·5k대·1k대·단)에서 판을 고르게 뽑는다 (OGS는 10k~1k에 몰려 있음)
- 한 판의 흑·백은 각자 자기 OGS 급수를 정답으로 하는 별개의 기록
- 지표 (점 척도, 1급 = 1점): 평균 절대 오차, 치우침(추정 − 실제의 평균), 상관계수, 급수대별 평균 추정
  OGS 급수 체계가 KataGo human 모델의 급수 체계와 다를 수 있어, 오차만이 아니라 치우침·상관을 같이 본다
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import arena, progress
from .board import Board, IllegalMoveError
from .ogs import SAMPLE, games
from .rank_estimate import PROB_FLOOR, RANKS, RankEstimate
from .train import is_val

BUCKETS = [(-30, -15, "20k대"), (-15, -10, "15~11k"), (-10, -5, "10~6k"), (-5, 0, "5~1k"), (0, 20, "1d+")]
CHECKPOINTS = [5, 10, 20, 30]
WEIGHT = 0.5
PROFILES = [f"rank_{r}" for r in RANKS]


def bucket_of(stones):
    return next(name for lo, hi, name in BUCKETS if lo <= stones < hi)


def pick_validation(per_bucket, path=SAMPLE):
    """검증용 판을 급수대별로 per_bucket판씩 (두 사람 평균 급수 기준)."""
    picked, count = [], defaultdict(int)
    for g in games(path):
        if not is_val(g["game_id"]):
            continue
        b = bucket_of((g["rank"]["B"] + g["rank"]["W"]) / 2)
        if count[b] < per_bucket:
            picked.append(g)
            count[b] += 1
        if len(count) == len(BUCKETS) and all(v >= per_bucket for v in count.values()):
            break
    return picked


def sequences(policy_fn, gs, moves):
    """판마다 흑·백 각각 [급수별 log p] 목록. board.py 규칙으로 못 두는 수가 나오면 그 판은 거기까지."""
    out = []
    for g in gs:
        board = Board(19, g["komi"])
        per = {"B": [], "W": []}
        for color, move in g["moves"][:moves]:
            if color != board.to_play:  # 연속 착수 같은 이상한 기록은 거기서 멈춘다
                break
            if move != "pass":
                pol = policy_fn(board, PROFILES)
                per[color].append([math.log(max(pol[p].get(move, 0.0), PROB_FLOOR)) for p in PROFILES])
            try:
                board.play(move)
            except IllegalMoveError:
                break
        for color in ("B", "W"):
            out.append({"game_id": g["game_id"], "true": g["rank"][color], "loglik": per[color]})
    return out


def metrics(seqs, weight=WEIGHT, checkpoints=CHECKPOINTS):
    rows = {}
    for n in checkpoints:
        est, true = [], []
        for s in seqs:
            if len(s["loglik"]) < n:
                continue
            e = RankEstimate([weight * sum(col) for col in zip(*s["loglik"][:n])])
            est.append(e.mean_stones())
            true.append(s["true"])
        if len(est) < 3:
            continue
        est, true = np.array(est), np.array(true)
        err = est - true
        by_bucket = defaultdict(list)
        for e_, t_ in zip(est, true):
            by_bucket[bucket_of(t_)].append(e_)
        rows[n] = {
            "n": len(est),
            "mae": round(float(np.abs(err).mean()), 2),
            "bias": round(float(err.mean()), 2),
            "corr": round(float(np.corrcoef(est, true)[0, 1]), 3),
            "within_3": round(float((np.abs(err) <= 3).mean()), 2),
            "mean_by_bucket": {b: round(float(np.mean(v)), 1) for b, v in by_bucket.items()},
        }
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="실제 사람 기보로 실력 추정 검증")
    p.add_argument("--per-bucket", type=int, default=30)
    p.add_argument("--moves", type=int, default=60)
    p.add_argument("--model", action="append", default=[], help="비교할 자체 모델 (여러 개 가능)")
    p.add_argument("--no-teacher", action="store_true", help="KataGo human 모델 비교 생략")
    p.add_argument("--out", default=str(arena.ROOT / "data/arena"))
    args = p.parse_args(argv)
    progress.tee_stdout("human_eval")

    gs = pick_validation(args.per_bucket)
    truth_by_bucket = defaultdict(list)
    for g in gs:
        for c in ("B", "W"):
            truth_by_bucket[bucket_of(g["rank"][c])].append(g["rank"][c])
    print(f"검증 판 {len(gs)}개 (사람 {2 * len(gs)}명), 판마다 앞 {args.moves}수", flush=True)
    print("급수대별 실제 평균:", {b: round(float(np.mean(v)), 1) for b, v in truth_by_bucket.items()}, flush=True)

    result = {"games": len(gs), "moves": args.moves, "models": {}}
    sources = [(Path(m).stem, m) for m in args.model]
    if not args.no_teacher:
        sources.append(("katago_human", None))
    out_dir = Path(args.out) / ("human-eval-" + time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, path) in enumerate(sources, 1):
        t = time.time()
        if path:
            from .model import NetPolicy

            seqs = sequences(NetPolicy(path).human_policies, gs, args.moves)
        else:
            from .engine import KataGoEngine

            with KataGoEngine.from_paths(
                "katago", str(arena.DEFAULT_MODEL), str(arena.DEFAULT_CONFIG), human_model=str(arena.HUMAN_MODEL)
            ) as engine:
                seqs = sequences(engine.human_policies, gs, args.moves)
        (out_dir / f"{name}.sequences.json").write_text(json.dumps(seqs))
        result["models"][name] = metrics(seqs)
        print(f"[{i}/{len(sources)}] {name}: {json.dumps(result['models'][name], ensure_ascii=False)} ({time.time() - t:.0f}초)", flush=True)
    result["truth_by_bucket"] = {b: round(float(np.mean(v)), 1) for b, v in truth_by_bucket.items()}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"저장: {out_dir}/result.json")


if __name__ == "__main__":
    main()
