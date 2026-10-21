"""기보 보관함: 명국(library/*.sgf)과 내 대국(data/games.jsonl)을 목록·재생·분석한다.

판 전체 분석은 오래 걸려서(19줄 250수 기준 1~2분) 백그라운드 스레드에서 돌리고,
결과는 data/library_analysis/<id>.json에 캐시한다. 대국 중인 엔진을 막지 않도록
분석 전용 KataGo 프로세스를 따로 띄운다(engine_factory, 처음 분석할 때 생성).

분석에서 뽑는 근거:
- 국면별 흑 승률·집 차이 (그래프), KataGo 최선수와 예상 진행
- 실제 수의 손실(집·승률)과 "그 시대 프로가 이 수를 둘 확률"(human 모델 proyear_<연도> 프로필)
- 결정적 수: 승률을 가장 크게 잃은 수 / 예상 밖의 명수: 프로도 거의 안 둘 수인데 손실이 거의 없는 수
"""

import json
import re
import threading
from pathlib import Path

from .sgf import SgfError, dump, parse
from .skill import CONTESTED_WINRATE

ANALYSIS_VISITS = 100
ANALYSIS_VERSION = 1  # 분석 형식이 바뀌면 올려서 캐시를 무효화
# 결정적 수: 승률 손실 상위 N개 (이 이상 잃은 수만)
TURNING_POINTS = 5
TURNING_MIN_WINRATE_LOSS = 0.08
# 예상 밖의 명수: 그 시대 프로 확률이 이 미만인데 손실이 이 이하
SURPRISE_MAX_HUMAN_P = 0.05
SURPRISE_MAX_LOSS = 0.5
SURPRISES = 3
# 내 대국 복기: 내 수 중 승률을 가장 많이 잃은 수 (결정적 수보다 기준을 낮춰 배울 장면을 넉넉히)
MY_MISTAKES = 3
# human 모델 proyear 프로필이 지원하는 연도 범위
PROYEAR_RANGE = (1800, 2023)


def profile_for(date):
    """기보 날짜 → "그 시대 프로" 프로필. 날짜가 없으면 현대 프로."""
    m = re.search(r"\d{4}", date or "")
    year = int(m.group()) if m else PROYEAR_RANGE[1]
    return f"proyear_{max(PROYEAR_RANGE[0], min(PROYEAR_RANGE[1], year))}"


class Library:
    def __init__(self, library_dir, records, engine_factory):
        self.dir = Path(library_dir)
        self.records = records
        self.engine_factory = engine_factory
        self._engine = None
        self.cache_dir = records.dir / "library_analysis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.jobs = {}  # id → {"state", "done", "total", "error"}

    # ---- 목록·불러오기 ----

    def list(self):
        items = []
        for path in sorted(self.dir.glob("*.sgf")):
            try:
                g = parse(path.read_text(encoding="utf-8", errors="replace"))
            except SgfError:
                continue
            m = g["meta"]
            items.append(
                {
                    "id": f"f-{path.stem}",
                    "kind": "famous",
                    "title": m.get("GN") or (f"{m['EV']} 제{m['RO']}국" if "EV" in m and "RO" in m else m.get("EV")) or path.stem,
                    "black": m.get("PB", "흑"),
                    "white": m.get("PW", "백"),
                    "date": m.get("DT", ""),
                    "result": m.get("RE", ""),
                    "event": m.get("EV", ""),
                    "size": g["size"],
                    "moves": len(g["moves"]),
                }
            )
        for row in reversed(self.records.read("games.jsonl")):
            you, ai = "나", f"AI({row['ai'].get('mode', '')})"
            black, white = (you, ai) if row["user_color"] == "B" else (ai, you)
            items.append(
                {
                    "id": f"m-{row['game_id']}",
                    "kind": "mine",
                    "title": f"내 대국 {row['game_id'][:13]}",
                    "black": black,
                    "white": white,
                    "date": row["game_id"][:8],
                    "result": row["result"],
                    "event": "",
                    "size": row["size"],
                    "moves": len(row["moves"]),
                }
            )
        return items

    def load(self, game_id):
        """{id, size, komi, meta, moves, comments}. 없는 id면 None."""
        if game_id.startswith("f-"):
            stem = game_id[2:]
            # 경로 조작 방지: 목록에 있는 파일 이름만 허용
            if not re.fullmatch(r"[\w.\-]+", stem) or not (self.dir / f"{stem}.sgf").is_file():
                return None
            g = parse((self.dir / f"{stem}.sgf").read_text(encoding="utf-8", errors="replace"))
            # 널리 알려진 명장면 메모 (<이름>.notes.json, {"수 번호": "설명"}). 기보 원본은 건드리지 않는다
            notes_path = self.dir / f"{stem}.notes.json"
            g["notes"] = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
            if g["setup"]:
                g["warning"] = "치석(배치 돌)이 있는 기보는 아직 정확히 재생하지 못합니다."
            return {"id": game_id, **g}
        if game_id.startswith("m-"):
            for row in self.records.read("games.jsonl"):
                if row["game_id"] == game_id[2:]:
                    return {
                        "id": game_id,
                        "size": row["size"],
                        "komi": row["komi"],
                        "meta": {"RE": row["result"], "DT": row["game_id"][:8]},
                        "moves": [tuple(m) for m in row["moves"]],
                        "comments": {},
                        "notes": {},
                        "setup": False,
                        "user_color": row["user_color"],
                    }
        return None

    def sgf(self, game_id):
        g = self.load(game_id)
        return dump(g["size"], g["komi"], g["moves"], g["meta"]) if g else None

    # ---- 분석 ----

    def analysis(self, game_id):
        """{state: none|running|done|error, done, total, (done이면) turns, turning_points, surprises}"""
        cache = self.cache_dir / f"{game_id}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
            if data.get("version") == ANALYSIS_VERSION:
                # 주요 장면은 국면 데이터에서 매번 다시 뽑는다 → 기준을 바꿔도 재분석이 필요 없다
                user_color = None
                if game_id.startswith("m-"):
                    g = self.load(game_id)
                    user_color = g and g["user_color"]
                return {"state": "done", **data, **key_moments(data["turns"], data["profile"], user_color)}
        with self.lock:
            job = self.jobs.get(game_id)
            return dict(job) if job else {"state": "none"}

    def start(self, game_id):
        g = self.load(game_id)
        if g is None:
            return {"state": "error", "error": "없는 기보"}
        current = self.analysis(game_id)
        if current["state"] in ("done", "running"):
            return current
        with self.lock:
            if any(j["state"] == "running" for j in self.jobs.values()):
                return {"state": "busy", "error": "다른 기보를 분석하는 중입니다"}
            self.jobs[game_id] = {"state": "running", "done": 0, "total": len(g["moves"]) + 1}
        threading.Thread(target=self._run, args=(game_id, g), daemon=True).start()
        return self.analysis(game_id)

    def _run(self, game_id, g):
        try:
            if self._engine is None:
                self._engine = self.engine_factory()
            profile = profile_for(g["meta"].get("DT"))

            def progress(t, _):
                with self.lock:
                    self.jobs[game_id]["done"] += 1

            results = self._engine.analyze_game(
                g["size"], g["komi"], g["moves"], ANALYSIS_VISITS, human_profile=profile, on_turn=progress
            )
            data = summarize_analysis(g, results, profile)
            (self.cache_dir / f"{game_id}.json").write_text(json.dumps(data, ensure_ascii=False))
            with self.lock:
                self.jobs.pop(game_id, None)
        except Exception as e:  # 분석 스레드가 조용히 죽지 않게 상태로 남긴다
            with self.lock:
                self.jobs[game_id] = {"state": "error", "error": str(e)}

    def close(self):
        if self._engine:
            self._engine.close()


def summarize_analysis(g, results, profile):
    """국면별 분석 → 화면용 수치와 근거."""
    n = max(results) + 1
    turns = []
    for t in range(n):
        a = results[t]
        to_black = 1 if a.to_play == "B" else -1
        turn = {
            "turn": t,
            "winrate_black": round(a.winrate if to_black == 1 else 1 - a.winrate, 4),
            "score_black": round(to_black * a.score_lead, 2),
            "best": a.best.move,
            "best_pv": a.best.pv[:8],
        }
        # 이 국면에서 실제로 둔 수(t+1번째 수)의 평가
        if t + 1 < n:
            color, move = g["moves"][t]
            after = results[t + 1]
            human = a.human_policy or {}
            ranked = sorted(human, key=human.get, reverse=True)
            engine_order = next((m.order for m in a.moves if m.move == move), None)
            turn["next"] = {
                "color": color,
                "move": move,
                # 둔 쪽 기준: 두기 전 값 - 둔 뒤 값(상대 기준이라 뒤집음)
                "loss": round(max(0.0, a.score_lead + after.score_lead), 2),
                "winrate_loss": round(max(0.0, a.winrate - (1 - after.winrate)), 4),
                "human_p": round(human.get(move, 0.0), 4),
                "human_rank": ranked.index(move) + 1 if move in ranked else None,
                "engine_rank": engine_order + 1 if engine_order is not None else None,
            }
        turns.append(turn)

    return {
        "version": ANALYSIS_VERSION,
        "visits": ANALYSIS_VISITS,
        "profile": profile,
        "total": n,
        "done": n,
        "turns": turns,
        **key_moments(turns, profile),
    }


def key_moments(turns, profile, user_color=None):
    """결정적 수와 예상 밖의 명수. 승부가 이미 기운 국면의 수는 뺀다(skill과 같은 접전 기준).

    user_color(내 대국)를 주면 "내가 놓친 장면"(내 수 중 승률을 가장 많이 잃은 수)도 따로 뽑는다.

    끝난 판에선 어떤 수를 둬도 손실이 거의 없어서 "드물지만 손해 없는 수"가 쏟아지고,
    큰 승률 변화도 의미가 없기 때문이다.
    """
    lo, hi = CONTESTED_WINRATE

    def contested(t):
        wr = t["winrate_black"] if t["next"]["color"] == "B" else 1 - t["winrate_black"]
        return lo <= wr <= hi

    played = [t for t in turns if "next" in t and t["next"]["move"] != "pass" and contested(t)]
    turning = sorted(
        (t for t in played if t["next"]["winrate_loss"] >= TURNING_MIN_WINRATE_LOSS),
        key=lambda t: -t["next"]["winrate_loss"],
    )[:TURNING_POINTS]
    # 초반 10수는 정석·포석이라 "예상 밖"의 의미가 약해 뺀다
    surprising = sorted(
        (
            t
            for t in played
            if t["turn"] >= 10
            and t["next"]["human_p"] < SURPRISE_MAX_HUMAN_P
            and t["next"]["loss"] <= SURPRISE_MAX_LOSS
        ),
        key=lambda t: t["next"]["human_p"],
    )[:SURPRISES]

    def moment(t, kind):
        nx = t["next"]
        num = t["turn"] + 1
        who = "흑" if nx["color"] == "B" else "백"
        era = profile.removeprefix("proyear_")
        if kind == "turning":
            text = (
                f"{num}수 {who} {nx['move']}: 승률을 {nx['winrate_loss']:.0%} 잃었습니다 ({nx['loss']:.1f}집). "
                f"KataGo는 {t['best']}를 더 좋게 봤습니다."
            )
        else:
            text = (
                f"{num}수 {who} {nx['move']}: {era}년 프로 모델이 이 수를 둘 확률은 {nx['human_p']:.1%}로 아주 드물지만, "
                f"KataGo 기준 손실은 {nx['loss']:.1f}집뿐인 예상 밖의 좋은 수입니다."
            )
        return {"turn": t["turn"], "move_number": num, "text": text}

    out = {
        "turning_points": sorted((moment(t, "turning") for t in turning), key=lambda m: m["turn"]),
        "surprises": sorted((moment(t, "surprise") for t in surprising), key=lambda m: m["turn"]),
    }
    if user_color:
        mine = sorted(
            (t for t in played if t["next"]["color"] == user_color and t["next"]["winrate_loss"] > 0.02),
            key=lambda t: -t["next"]["winrate_loss"],
        )[:MY_MISTAKES]
        out["my_mistakes"] = sorted((moment(t, "turning") for t in mine), key=lambda m: m["turn"])
    return out
