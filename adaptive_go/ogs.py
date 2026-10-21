"""OGS(online-go.com) 실제 사람 기보 읽기.

출처: https://za3k.com/ogs/ 의 sample-100k.json.gz (OGS 허락 하에 재배포, OGS 측이 연구 용도 사용을 허용,
모음 자체는 퍼블릭 도메인). 한 줄 = 한 판의 OGS API 원본 JSON. data/ogs/에 압축된 채로 두고 줄 단위로 읽는다.

쓰는 판: 19줄, 호선(치석·배치 돌 없음), 50수 이상, 랭크 대국, 봇 제외, 두 사람 급수 차 3 이하.
OGS 급수: players.<색>.rank 실수. 29 = 1급, 30 = 1단 → rank_estimate의 점 척도(1k=0, 1d=1)로는 rank - 29.
OGS 급수 체계는 KataGo human 모델의 급수 체계와 완전히 같지 않다(서버마다 기준이 다름) → 평가에서 치우침도 잰다.
착수: [x, y, 시간], y=0이 맨 윗줄. 패스는 x < 0.
"""

import gzip
import json
from pathlib import Path

from .board import COLUMNS

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data/ogs/sample-100k.json.gz"
MIN_MOVES = 50
MAX_RANK_GAP = 3.0


def ogs_to_stones(rank):
    """OGS rank 실수 → 점 척도 (1k = 0, 1d = 1, 5k = -4)."""
    return rank - 29.0


def to_gtp(move, size=19):
    x, y = move[0], move[1]
    if x < 0 or y < 0:
        return "pass"
    return f"{COLUMNS[x]}{size - y}"


def usable(g):
    """검증·학습에 쓸 판인지. 이유 없이 거르면 나중에 헷갈리니 조건을 한곳에 모은다."""
    if g.get("width") != 19 or g.get("height") != 19:
        return False
    if len(g.get("moves") or []) < MIN_MOVES or g.get("handicap"):
        return False
    init = g.get("initial_state") or {}
    if any(init.values()):  # 배치 돌
        return False
    if not g.get("ranked"):
        return False
    p = g.get("players") or {}
    b, w = p.get("black") or {}, p.get("white") or {}
    if "bot" in (b.get("username", "") + w.get("username", "")).lower():
        return False
    if b.get("rank") is None or w.get("rank") is None:
        return False
    return abs(b["rank"] - w["rank"]) <= MAX_RANK_GAP


def games(path=SAMPLE):
    """쓸 수 있는 판을 차례로: {game_id, komi, moves: [(색, gtp)], rank: {"B": 점, "W": 점}}"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            g = json.loads(line)
            if not usable(g):
                continue
            first = g.get("initial_player") or "black"
            colors = ("B", "W") if first == "black" else ("W", "B")
            yield {
                "game_id": int(g["game_id"]),
                "komi": float(g.get("komi") or 0),
                "rules": g.get("rules"),
                "moves": [(colors[i % 2], to_gtp(m)) for i, m in enumerate(g["moves"])],
                "rank": {
                    "B": ogs_to_stones(g["players"]["black"]["rank"]),
                    "W": ogs_to_stones(g["players"]["white"]["rank"]),
                },
            }
