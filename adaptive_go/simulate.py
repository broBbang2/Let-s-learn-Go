"""가상 사용자로 적응형 대국 검증 (A안).

python -m adaptive_go.simulate --games 15 --workers 3

사람이 직접 두지 않아도 웹 대국과 똑같은 Game 로직(실력 추정 → AI 급수 조정 → 판마다 난이도 보정)이
제대로 도는지 본다. 사용자 역할은 OGS 사람 기보로 학습한 자체 모델(mixed2)이 정해진 급수로 수를 뽑아 맡는다.
AI는 KataGo human 모델이라 "같은 모델끼리 두는" 문제를 줄인다 (그래도 실력 추정에 자체 모델이 30/70으로 섞여
있어 실제 사람보다는 쉬운 조건이다).

시나리오 (각자 별도 기록 폴더 → 판 사이에 배운 난이도·실력 추정이 이어진다):
- fixed-15k / 10k / 5k / 1k: 실력이 고정된 사용자 → 승률이 목표(45%) 근처로 가는가, 추정이 실제 급수로 가는가
- growing: 판마다 15k → 5k로 강해지는 사용자 → 추정과 AI 급수가 성장을 따라가는가 ("같이 발전하는 AI"의 핵심)
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import arena, progress
from .difficulty import TARGET_USER_SCORE
from .engine import KataGoEngine
from .players import sample_human
from .rank_estimate import mix_policies, nearest_rank, stones
from .records import Records
from .server import HUMAN, RANK_MODEL, Game

VISITS = 100  # AI 탐색량. 실험 속도를 위해 웹 기본(200)보다 낮춘다
RESIGN_WINRATE = 0.03
RESIGN_AFTER = 100
MAX_MOVES = 320


def rank_at(scenario, i, n):
    """시나리오의 i번째 판(0부터)에서 사용자 실력(점 척도)."""
    if scenario == "growing":
        return -14 + (10 * i / max(1, n - 1))  # 15k(-14) → 5k(-4)
    return stones(scenario.removeprefix("fixed-"))


def play_one(game, user_policy, user_stones, color, rng):
    """한 판. 사용자 수는 user_policy(국면, [프로필])에서 뽑는다."""
    profile = f"rank_{nearest_rank(user_stones)}"
    s = game.new_game(size=19, komi=7.5, color=color, visits=VISITS, mode=HUMAN)
    while not s["over"]:
        n = len(s["history"])
        ai = s["last_ai"]
        last_ai_move = s["history"][-1][1] if s["history"] and s["history"][-1][0] != color else None
        if n >= RESIGN_AFTER and ai and ai["user_winrate"] < RESIGN_WINRATE:
            s = game.resign()
            break
        if n >= MAX_MOVES or (n >= 150 and last_ai_move == "pass"):
            move = "pass"  # 끝내기가 끝났다고 보고 AI 패스에 맞춰 패스
        else:
            policy = user_policy(game.board, [profile])[profile]
            move = sample_human(policy, game.board, rng)
        s = game.user_move(move)
        if s.get("message") and "둘 수 없는" in s["message"]:
            s = game.user_move("pass")
    return s


def run_scenario(job):
    scenario, games, out_dir, seed = job
    from .model import NetPolicy  # torch는 워커 안에서만

    rng = random.Random(seed)
    records = Records(Path(out_dir) / scenario)
    ours = NetPolicy(str(RANK_MODEL))
    with KataGoEngine.from_paths(
        "katago", str(arena.DEFAULT_MODEL), str(arena.DEFAULT_CONFIG), human_model=str(arena.HUMAN_MODEL)
    ) as engine:
        # 웹 서버 기본과 같은 실력 추정(KataGo 30% + 자체 70%)
        rank_policy = lambda b, p: mix_policies(engine.human_policies(b, p), ours.human_policies(b, p))  # noqa: E731
        game = Game(
            engine, records, rng=rng, human_available=True, rank_policy=rank_policy, rank_source="mix", live_territory=False
        )
        rows = []
        for i in range(games):
            true = rank_at(scenario, i, games)
            color = "B" if i % 2 == 0 else "W"
            t = time.time()
            s = play_one(game, ours.human_policies, true, color, rng)
            row = {
                "scenario": scenario,
                "game": i,
                "true_stones": round(true, 2),
                "user_color": color,
                "result": s["result"]["text"],
                "user_score": records.read("games.jsonl")[-1]["user_score"],
                "moves": len(s["history"]),
                "estimate_stones": s["estimate"]["mean_stones"],
                "ai_profile": s["ai"]["profile"],
                "rank_offset": round(game.current_difficulty.rank_offset, 2),
                "ai_avg_loss": s["ai"]["avg_loss"],
                "user_avg_loss": s["summary"]["avg_score_loss"],
                "seconds": round(time.time() - t),
            }
            rows.append(row)
            with open(Path(out_dir) / f"{scenario}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(rows):
    rows = sorted(rows, key=lambda r: r["game"])
    n = len(rows)
    half = n // 2
    win = lambda rs: round(sum(r["user_score"] for r in rs) / len(rs), 2) if rs else None  # noqa: E731
    last = rows[-1]
    return {
        "games": n,
        "win_rate_first_half": win(rows[:half]),
        "win_rate_second_half": win(rows[half:]),
        "win_rate_last5": win(rows[-5:]),
        "target": TARGET_USER_SCORE,
        "final_true": last["true_stones"],
        "final_estimate": last["estimate_stones"],
        "estimate_error_last5": round(sum(abs(r["estimate_stones"] - r["true_stones"]) for r in rows[-5:]) / min(5, n), 2),
        "ai_profiles": [r["ai_profile"] for r in rows],
        "rank_offsets": [r["rank_offset"] for r in rows],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="가상 사용자로 적응형 대국 검증")
    p.add_argument("--games", type=int, default=15)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--scenarios", nargs="+", default=["fixed-15k", "fixed-10k", "fixed-5k", "fixed-1k", "growing"])
    p.add_argument("--out", default=str(arena.ROOT / "data/sim"))
    args = p.parse_args(argv)
    progress.tee_stdout("simulate")

    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(sc, args.games, str(out), 1000 + i) for i, sc in enumerate(args.scenarios)]
    print(f"시나리오 {len(jobs)}개 × {args.games}판 → {out}", flush=True)
    t0 = time.time()
    summary = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, rows in enumerate(pool.map(run_scenario, jobs), 1):
            sc = rows[0]["scenario"]
            summary[sc] = summarize(rows)
            print(f"[{i}/{len(jobs)}] {sc}: {json.dumps(summary[sc], ensure_ascii=False)}   "
                  f"{progress.status(i, len(jobs), t0)}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"저장: {out}/summary.json")


if __name__ == "__main__":
    main()
