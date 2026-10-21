"""테스트용 가짜 KataGo Analysis Engine.

실제 KataGo처럼 stdin JSON 한 줄 → stdout JSON 한 줄로 답한다.
값은 항상 흑 기준(reportAnalysisWinratesAs = BLACK)으로 고정:
- 현재 국면: 흑 승률 0.7, 흑 3집 우세
- 후보수: C3(최선), D4(1.5집 손해), pass — 단, C3나 D4가 이미 놓였으면 pass만 남음
- ownership: 맨 윗줄 왼쪽 점만 흑 1.0, 나머지 백 -1.0
- humanSLProfile을 주면 humanPolicy: D4 0.6, C3 0.3, A9(맨 윗줄 왼쪽) 0.1, 나머지 0, pass 자리 -1
"""

import json
import sys

for line in sys.stdin:
    q = json.loads(line)
    # 실제 엔진처럼 질의와 무관한 경고 줄을 섞는다
    print(json.dumps({"warning": "fake warning"}), flush=True)
    if q["boardXSize"] > 19:
        print(json.dumps({"id": q["id"], "error": "board too big"}), flush=True)
        continue
    n = q["boardXSize"] * q["boardYSize"]
    resp = {
        "id": q["id"],
        "isDuringSearch": False,
        "turnNumber": len(q["moves"]),
        "moveInfos": [
            {"move": "D4", "visits": 30, "winrate": 0.65, "scoreLead": 1.5, "order": 1, "pv": ["D4"]},
            {"move": "C3", "visits": 60, "winrate": 0.7, "scoreLead": 3.0, "order": 0, "pv": ["C3", "D4"]},
            {"move": "pass", "visits": 1, "winrate": 0.2, "scoreLead": -10.0, "order": 2, "pv": ["pass"]},
        ],
        "rootInfo": {"winrate": 0.7, "scoreLead": 3.0, "visits": 91},
    }
    if any(v in ("C3", "D4") for _, v in q["moves"]):
        resp["moveInfos"] = [{"move": "pass", "visits": 1, "winrate": 0.7, "scoreLead": 3.0, "order": 0}]
    if q.get("overrideSettings", {}).get("humanSLProfile"):
        size = q["boardXSize"]
        policy = [0.0] * (n + 1)
        policy[(size - 4) * size + 3] = 0.6  # D4
        policy[(size - 3) * size + 2] = 0.3  # C3
        policy[0] = 0.1  # A9
        policy[n] = -1  # pass
        resp["humanPolicy"] = policy
    if q.get("includeOwnership"):
        resp["ownership"] = [1.0] + [-1.0] * (n - 1)
    turns = q.get("analyzeTurns")
    if turns:
        for t in reversed(turns):  # 실제 엔진처럼 끝난 순서대로(순서 보장 없음)
            print(json.dumps({**resp, "turnNumber": t}), flush=True)
    else:
        print(json.dumps(resp), flush=True)
