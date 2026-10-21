"""AI끼리 리그전.

python -m adaptive_go.arena best v2:3 v3:5k mix:5k:3 --games 2 --workers 2
python -m adaptive_go.arena --calibrate 20k 10k 5k 1d 5d     # v3 급수별 수당 손실 측정

측정 (모든 플레이어에게 같은 평가 질의를 따로 던져 공정하게 잰다):
- 승률, Bradley-Terry 레이팅(상대적 Elo 척도)
- 수당 평균 손실(접전 국면만, skill.effective_loss), 큰 실수(3집 이상) 비율 — 엔진 최선수 대비
- 사람다움: 기준 급수(기본 10급) human 모델이 그 수에 준 확률 평균, 사람 top-3 안에 든 비율,
  포석·중반 1선 착수율 (사람이 거의 안 두는 수 = v2의 약점)
"""

import argparse
import itertools
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import progress
from .board import BLACK, WHITE, Board, format_score
from .engine import KataGoEngine, ownership_score
from .players import parse_player
from .play import DEFAULT_CONFIG, DEFAULT_MODEL
from .skill import BLUNDER_LOSS, EVAL_VISITS, effective_loss, evaluate_move

ROOT = Path(__file__).resolve().parent.parent
HUMAN_MODEL = ROOT / "models/b18c384nbt-humanv0.bin.gz"
REF_PROFILE = "rank_10k"
# 둘 차례 승률이 이 미만이면 돌을 거둔다. 사람처럼 끝내고, 진 판을 끝까지 두며 생기는
# 의미 없는 손실과 시간 낭비를 막는다. 초반 승률은 출렁이니 판 넓이의 30% 수 이후에만
RESIGN_WINRATE = 0.02
RESIGN_AFTER = 0.3

RANK_ORDER = [f"{n}k" for n in range(30, 0, -1)] + [f"{n}d" for n in range(1, 10)]  # 약 → 강


def is_first_line(move, size):
    if move == "pass":
        return False
    col = "ABCDEFGHJKLMNOPQRST".index(move[0])
    row = int(move[1:]) - 1
    return col in (0, size - 1) or row in (0, size - 1)


def play_game(
    engine,
    black,
    white,
    rng,
    size=9,
    komi=7.0,
    ref_profile=REF_PROFILE,
    eval_visits=EVAL_VISITS,
    resign_winrate=RESIGN_WINRATE,
    resign_after=None,
):
    """한 판을 두고 기록을 돌려준다. black/white는 players의 Player."""
    board = Board(size, komi)
    # 9x9 기준 121수. 약한 프로필끼리 두면 집을 메우며 안 끝날 수 있어 상한을 둔다
    max_moves = int(size * size * 1.5)
    resign_after = int(size * size * RESIGN_AFTER) if resign_after is None else resign_after
    moves = []
    resigned = None
    while not board.is_over() and len(board.history) < max_moves:
        player = black if board.to_play == BLACK else white
        before = engine.analyze(board, max_visits=eval_visits, human_profile=ref_profile)
        if len(board.history) >= resign_after and before.winrate < resign_winrate:
            resigned = board.to_play
            break
        move = player.choose(engine, board, rng)
        if not board.is_legal(move):
            move = "pass"  # 엔진과 board.py의 패 규칙 차이 등 예외 상황 방어
        ev = evaluate_move(engine, board, move, before=before, visits=eval_visits)
        hp = before.human_policy or {}
        top3 = sorted(hp, key=hp.get, reverse=True)[:3]
        moves.append(
            {
                "player": player.name,
                "color": board.to_play,
                "move": ev.move,
                "loss": ev.score_loss,
                "winrate_before": ev.winrate_before,
                "phase": ev.phase,
                "human_p": round(hp.get(ev.move, 0.0), 4),
                "human_top3": ev.move in top3,
                "first_line": is_first_line(ev.move, size),
            }
        )
        board.play(move)
    if resigned:
        score, result = None, f"{WHITE if resigned == BLACK else BLACK}+R"
        black_score = 0.0 if resigned == BLACK else 1.0
    else:
        final = engine.analyze(board, max_visits=eval_visits, include_ownership=True)
        score = ownership_score(board, final)
        result, black_score = format_score(score), 0.5 if score == 0 else float(score > 0)
    return {
        "black": black.name,
        "white": white.name,
        "score": score,
        "result": result,
        "black_score": black_score,
        "capped": not board.is_over() and not resigned,
        "moves": moves,
    }


# ---- 병렬 실행: 프로세스마다 KataGo를 하나씩 띄운다 ----

_engine = None


def _init_worker(model, config, human_model):
    global _engine
    _engine = KataGoEngine.from_paths("katago", model, config, human_model=human_model)


def _run(job):
    black_spec, white_spec, seed, size, komi = job
    t = time.time()
    record = play_game(_engine, parse_player(black_spec), parse_player(white_spec), random.Random(seed), size, komi)
    record["seconds"] = round(time.time() - t, 1)
    record["seed"] = seed
    return record


def schedule(specs, games_per_pair, calibrate=False):
    """리그전: 모든 쌍이 흑백을 바꿔 가며 games_per_pair판. 보정 모드: 각자 자기 자신과."""
    pairs = [(s, s) for s in specs] if calibrate else list(itertools.combinations(specs, 2))
    jobs = []
    for a, b in pairs:
        for g in range(games_per_pair):
            black, white = (a, b) if g % 2 == 0 else (b, a)
            jobs.append((black, white))
    return jobs


# ---- 집계 ----


def bradley_terry(names, games, iters=500):
    """Bradley-Terry 강도를 MM 알고리즘으로 추정해 Elo 척도(평균 1500)로 돌려준다.

    전승·전패 플레이어는 강도가 무한대로 발산하므로, 모두에게 "평균 상대와 1무"를 가상으로 더한다.
    """
    strength = {n: 1.0 for n in names}
    for _ in range(iters):
        new = {}
        for n in names:
            wins, denom = 0.5, 1 / (strength[n] + 1)  # 가상의 1무 (강도 1인 상대)
            for g in games:
                if n not in (g["black"], g["white"]) or g["black"] == g["white"]:
                    continue
                opp = g["white"] if g["black"] == n else g["black"]
                wins += g["black_score"] if g["black"] == n else 1 - g["black_score"]
                denom += 1 / (strength[n] + strength[opp])
            new[n] = wins / denom
        strength = new
    elos = {n: 400 * math.log10(s) for n, s in strength.items()}
    mean = sum(elos.values()) / len(elos)
    return {n: round(e - mean + 1500) for n, e in elos.items()}


def summarize(games):
    names = sorted({g["black"] for g in games} | {g["white"] for g in games})
    rows = {}
    for n in names:
        mv = [m for g in games for m in g["moves"] if m["player"] == n]
        played = [g for g in games if n in (g["black"], g["white"]) and g["black"] != g["white"]]
        score = sum(g["black_score"] if g["black"] == n else 1 - g["black_score"] for g in played)
        early = [m for m in mv if m["phase"] != "endgame"]
        eff = [x for m in mv if (x := effective_loss(m["loss"], m["winrate_before"])) is not None]
        rows[n] = {
            "games": len(played),
            "win_rate": round(score / len(played), 3) if played else None,
            "moves": len(mv),
            "avg_loss": round(sum(eff) / len(eff), 3) if eff else None,
            "raw_avg_loss": round(sum(m["loss"] for m in mv) / len(mv), 3) if mv else None,
            "blunder_rate": round(sum(m["loss"] >= BLUNDER_LOSS for m in mv) / len(mv), 3) if mv else None,
            "human_p": round(sum(m["human_p"] for m in mv) / len(mv), 4) if mv else None,
            "human_top3": round(sum(m["human_top3"] for m in mv) / len(mv), 3) if mv else None,
            "first_line_early": round(sum(m["first_line"] for m in early) / len(early), 3) if early else None,
        }
    if any(g["black"] != g["white"] for g in games):
        for n, elo in bradley_terry(names, games).items():
            rows[n]["elo"] = elo
    return rows


def monotone_calibration(table):
    """급수가 높을수록 손실이 줄어야 한다는 순서를 강제한다 (isotonic regression, PAVA).

    판 수가 적으면 측정값이 순서를 거스르는데(예: 1d가 1k보다 손실이 큼), 그대로 쓰면
    목표 손실 → 급수 선택이 뒤죽박죽이 된다. 순서를 어기는 이웃끼리 평균으로 묶어 편다.
    """
    ranks = [r for r in RANK_ORDER if r in table]
    blocks = []  # [합, 개수, 급수들]
    for r in ranks:
        blocks.append([table[r], 1, [r]])
        # 약 → 강 순서에서 손실은 줄어야(비증가) 한다. 뒤 블록 평균이 더 크면 합친다
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n, rs = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][2] += rs
    return {r: round(s / n, 3) for s, n, rs in blocks for r in rs}


def to_markdown(rows):
    header = "| 플레이어 | 판 | 승률 | Elo | 수당 손실(접전) | 큰 실수 | 사람 확률 | 사람 top3 | 1선(포석·중반) |"
    lines = [header, "|" + "---|" * 9]
    # Elo 높은 순, Elo가 없으면(보정 모드) 손실 낮은 순
    for n, r in sorted(rows.items(), key=lambda kv: (-kv[1].get("elo", 0), kv[1]["avg_loss"] or 0)):
        fmt = lambda v, p="{:.0%}": "-" if v is None else p.format(v)
        lines.append(
            f"| {n} | {r['games']} | {fmt(r['win_rate'])} | {r.get('elo', '-')} | {fmt(r['avg_loss'], '{:.2f}')} "
            f"| {fmt(r['blunder_rate'])} | {fmt(r['human_p'], '{:.3f}')} | {fmt(r['human_top3'])} "
            f"| {fmt(r['first_line_early'])} |"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="AI끼리 리그전")
    p.add_argument("players", nargs="+", help='best, v2:<집>, v3:<급수>, mix:<급수>:<집>  (보정 모드에선 급수만)')
    p.add_argument("--games", type=int, default=2, help="쌍마다 판 수 (흑백 교대)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.0)
    p.add_argument("--calibrate", action="store_true", help="v3 급수별 자기 대국으로 수당 손실 측정")
    p.add_argument("--out", default=str(ROOT / "data/arena"))
    args = p.parse_args(argv)
    progress.tee_stdout("arena")
    started = time.time()

    specs = [f"v3:{r}" for r in args.players] if args.calibrate else args.players
    jobs = schedule(specs, args.games, args.calibrate)
    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)}판, 워커 {args.workers}개 → {run_dir}", flush=True)

    games = []
    init = (str(DEFAULT_MODEL), str(DEFAULT_CONFIG), str(HUMAN_MODEL))
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=init) as pool:
        futures = [pool.submit(_run, (b, w, seed, args.size, args.komi)) for seed, (b, w) in enumerate(jobs)]
        for f in futures:
            g = f.result()
            games.append(g)
            with open(run_dir / "games.jsonl", "a", encoding="utf-8") as out:
                out.write(json.dumps(g, ensure_ascii=False) + "\n")
            flag = " (수 상한)" if g["capped"] else ""
            print(f"[{len(games)}/{len(jobs)}] {g['black']} vs {g['white']}: {g['result']}{flag} "
                  f"{len(g['moves'])}수 {g['seconds']}초   {progress.status(len(games), len(jobs), started)}",
                  flush=True)

    rows = summarize(games)
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    md = to_markdown(rows)
    (run_dir / "summary.md").write_text(md + "\n")
    print("\n" + md)

    if args.calibrate:
        # v3 급수 → 수당 평균 손실 표. 웹 대국의 v3 맞춤 모드가 목표 손실에 맞는 급수를 고를 때 쓴다
        raw = {name.removeprefix("v3(").removesuffix(")"): r["avg_loss"] for name, r in rows.items()}
        table = monotone_calibration(raw)
        (run_dir / "calibration_raw.json").write_text(json.dumps(raw, indent=2))
        path = Path(args.out) / "human_calibration.json"
        path.write_text(json.dumps(table, indent=2))
        print(f"\n측정값: {raw}\n단조 보정: {table}\n보정표 저장: {path}")


if __name__ == "__main__":
    main()
