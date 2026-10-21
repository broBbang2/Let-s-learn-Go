"""형세(ownership) 계산: 예상 영역, 수 한 번에 따른 영역 변화, 실수 유형 분류.

ownership은 흑 기준 [row][col] (+1 흑 땅 ~ −1 백 땅). 여기 함수들은 모두 "color 쪽 기준"으로 바꿔서 센다.
"""

from .board import COLUMNS

# 칸 하나가 이만큼 넘게 바뀌면 "이 수로 주인이 바뀐 칸"으로 판에 강조한다
CHANGED = 0.35
# 돌이 "내 쪽 → 상대 쪽"으로 넘어갔다고 볼 기준 (잡힐 위기 표시와 같은 정도)
FLIP = 0.3
# 실수 유형은 이 이상 손해 본 수만 나눈다 (skill.BLUNDER_LOSS와 같은 "큰 실수")
MISTAKE_LOSS = 3.0
# 이만큼 넘는 내 돌이 상대 땅으로 넘어간 큰 실수는 "사활·전투"
LOST_STONES = 2

MISTAKE_TYPES = {
    "life": "사활·전투 (돌이 잡힘)",
    "opening": "포석·방향",
    "middle": "중반 판단",
    "endgame": "끝내기",
}


def expected(own, color):
    """(내 예상 영역, 상대 예상 영역). 칸마다 내 땅일 정도(0~1)를 더한다."""
    sign = 1 if color == "B" else -1
    mine = theirs = 0.0
    for row in own:
        for o in row:
            v = o * sign
            if v > 0:
                mine += v
            else:
                theirs -= v
    return mine, theirs


def change(before, after, color, grid):
    """before → after 국면 사이 color 쪽 기준 영역 변화. grid는 after 국면의 돌 배치.

    before는 "그 수 대신 패스했다면"(수의 효과) 또는 "최선수를 뒀다면"(실수의 대가) 국면을 넣는다.
    둘 다 after와 둘 차례가 같아서, 차례의 이점이 섞이지 않는다.
    """
    if before is None or after is None:
        return None
    sign = 1 if color == "B" else -1
    m0, t0 = expected(before, color)
    m1, t1 = expected(after, color)
    n = len(after)
    points, lost, won = [], 0, 0
    for r in range(n):
        for c in range(n):
            v0, v1 = before[r][c] * sign, after[r][c] * sign
            if abs(v1 - v0) > CHANGED:
                points.append(f"{COLUMNS[c]}{r + 1}")
            stone = grid[r][c]
            if stone == color and v0 > FLIP and v1 < -FLIP:
                lost += 1
            elif stone not in (".", color) and v0 < -FLIP and v1 > FLIP:
                won += 1
    return {
        "mine": round(m1 - m0, 1),
        "theirs": round(t1 - t0, 1),
        "points": points,
        "lost_stones": lost,
        "won_stones": won,
    }


def classify_mistake(loss, phase, delta):
    """큰 실수(loss ≥ MISTAKE_LOSS)의 유형. 큰 실수가 아니거나 loss가 None(승부가 기운 국면)이면 None."""
    if loss is None or loss < MISTAKE_LOSS:
        return None
    if delta and delta["lost_stones"] >= LOST_STONES:
        return "life"
    return phase if phase in MISTAKE_TYPES else "middle"


def weakness(rows):
    """moves.jsonl 행들 → 유형별 {count, avg_loss, label}. 많이 나온 순."""
    agg = {}
    for r in rows:
        t = r.get("mistake_type")
        if not t:
            continue
        a = agg.setdefault(t, {"count": 0, "loss": 0.0})
        a["count"] += 1
        a["loss"] += r["score_loss"]
    out = {
        t: {"count": a["count"], "avg_loss": round(a["loss"] / a["count"], 1), "label": MISTAKE_TYPES[t]}
        for t, a in agg.items()
    }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["count"]))
