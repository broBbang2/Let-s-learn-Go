"""KataGo Analysis Engine 연결.

`katago analysis`는 stdin으로 JSON 질의를 한 줄씩 받고 stdout으로 JSON 응답을 한 줄씩 준다.
여기서는 그 프로세스를 subprocess로 띄우고, 응답을 다루기 쉬운 dataclass로 바꾼다.

관점(perspective) 규칙:
- KataGo는 config의 reportAnalysisWinratesAs(BLACK / WHITE / SIDETOMOVE) 기준으로 값을 준다.
- 이 모듈은 winrate·scoreLead를 항상 "둘 차례인 쪽" 기준으로 바꿔서 돌려준다.
  → 사용자 착수 손실 = best.score_lead - 사용자 수.score_lead 를 색 구분 없이 계산하기 위해서
- ownership은 항상 흑 기준(+1 흑 집, -1 백 집)으로 바꾼다. 사석·계가 판정은 색이 고정된 쪽이 편하다.
"""

import itertools
import json
import subprocess
from dataclasses import dataclass, field

from .board import BLACK, COLUMNS, Board, IllegalMoveError


class EngineError(RuntimeError):
    pass


@dataclass
class MoveInfo:
    move: str  # GTP 좌표
    visits: int
    winrate: float  # 둘 차례 기준
    score_lead: float  # 둘 차례 기준, 집
    order: int  # 0이 엔진 최선수
    pv: list = field(default_factory=list)


@dataclass
class Analysis:
    to_play: str
    moves: list  # MoveInfo, order 순
    winrate: float  # 현재 국면, 둘 차례 기준
    score_lead: float
    ownership: list = None  # 흑 기준, KataGo 순서(맨 윗줄 왼쪽부터)
    board_size: int = 19
    # human 모델이 예측한 "이 급수의 사람이 여기 둘 확률" {GTP 좌표: 확률}. 둘 수 없는 자리는 빠짐
    human_policy: dict = None

    @property
    def best(self):
        return self.moves[0]

    def find(self, move):
        """후보수 중 move를 찾는다. 탐색이 안 된 수면 None."""
        move = move.upper()
        return next((m for m in self.moves if m.move.upper() == move), None)

    def ownership_at(self, vertex):
        """GTP 좌표의 ownership (흑 기준). KataGo는 y=0이 맨 윗줄이라 행을 뒤집는다."""
        col = COLUMNS.index(vertex[0].upper())
        row = int(vertex[1:])
        return self.ownership[(self.board_size - row) * self.board_size + col]


class KataGoEngine:
    def __init__(self, command, report_as=BLACK, stderr=subprocess.DEVNULL):
        """command: 실행할 명령 리스트. 보통 from_paths()로 만든다.

        report_as: config의 reportAnalysisWinratesAs 값 ("B", "W", "SIDETOMOVE").
        stderr: KataGo 로그. 파이프로 받으면 읽지 않을 때 버퍼가 차서 멈추므로 기본은 버린다.
        """
        self.report_as = report_as
        self._ids = itertools.count()
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )

    @classmethod
    def from_paths(cls, katago, model, config, human_model=None, **kwargs):
        command = [katago, "analysis", "-model", model, "-config", config]
        if human_model:
            command += ["-human-model", human_model]
        return cls(command, **kwargs)

    def analyze(self, board, max_visits=None, include_ownership=False, override_settings=None, human_profile=None):
        """override_settings: 이 질의에만 적용할 KataGo 탐색 설정 (예: {"wideRootNoise": 1.0})
        human_profile: "rank_5k" 같은 급수 프로필. 주면 human_policy를 같이 받는다 (-human-model 필요).
        humanSLProfile만 지정하면 탐색 자체는 바뀌지 않고, 사람 정책 출력만 추가된다.
        """
        query = self._query(board, max_visits, include_ownership, override_settings, human_profile)
        return self._parse(self._request_many([query])[0], board)

    def human_policies(self, board, profiles):
        """같은 국면을 여러 급수 프로필로 한꺼번에 묻는다. {프로필: human_policy}

        탐색 없이(maxVisits=1) 신경망 정책만 필요해서 가볍고, 질의를 한 번에 보내면
        KataGo가 병렬로 처리해 급수 12개도 하나 묻는 시간과 비슷하게 끝난다.
        """
        queries = [self._query(board, 1, False, None, p) for p in profiles]
        responses = self._request_many(queries)
        return {p: self._parse(r, board).human_policy for p, r in zip(profiles, responses)}

    def analyze_game(self, size, komi, moves, max_visits=100, human_profile=None, on_turn=None):
        """기보 전체를 한 번에 분석한다. {수 번호 t: t수 둔 뒤 국면의 Analysis}

        KataGo의 analyzeTurns로 질의 하나에 모든 국면을 맡기면 엔진이 병렬로 처리하고
        끝나는 대로 국면별 응답을 보낸다. on_turn(t, analysis)로 진행 상황을 알 수 있다.
        moves: [(color, gtp)]. board.py 규칙으로 둘 수 없는 수가 나오면 그 앞까지만 분석.
        """
        board = Board(size, komi)
        # 국면별로 _parse에 필요한 건 둘 차례와 판 크기뿐이라 가벼운 빈 판으로 기억해 둔다
        views = []

        def snapshot():
            view = Board(size, komi)
            view.to_play = board.to_play
            views.append(view)

        snapshot()
        for _, vertex in moves:
            try:
                board.play(vertex)
            except IllegalMoveError:
                break
            snapshot()
        query = self._query(board, max_visits, False, None, human_profile)
        query["analyzeTurns"] = list(range(len(views)))
        results = {}
        for data in self._collect_turns(query):
            t = data["turnNumber"]
            results[t] = self._parse(data, views[t])
            if on_turn:
                on_turn(t, results[t])
        return results

    def _collect_turns(self, query):
        """analyzeTurns 질의 하나를 보내고 국면별 응답을 도착 순서대로 내보낸다."""
        if self._proc.poll() is not None:
            raise EngineError(f"KataGo 프로세스가 종료됨 (code {self._proc.returncode})")
        self._proc.stdin.write(json.dumps(query) + "\n")
        self._proc.stdin.flush()
        remaining = set(query["analyzeTurns"])
        while remaining:
            line = self._proc.stdout.readline()
            if not line:
                raise EngineError("KataGo 응답 없이 출력이 끝남")
            data = json.loads(line)
            if data.get("id") != query["id"] or data.get("isDuringSearch"):
                continue
            if "error" in data:
                raise EngineError(data["error"])
            remaining.discard(data["turnNumber"])
            yield data

    def _query(self, board, max_visits, include_ownership, override_settings, human_profile):
        query = {
            "id": str(next(self._ids)),
            "moves": [[color, vertex] for color, vertex in board.history],
            # board.py가 중국식 계가 + 단순 패라서 규칙을 맞춘다
            "rules": "chinese",
            "komi": board.komi,
            "boardXSize": board.size,
            "boardYSize": board.size,
            "includeOwnership": include_ownership,
        }
        if max_visits is not None:
            query["maxVisits"] = max_visits
        if human_profile:
            override_settings = {**(override_settings or {}), "humanSLProfile": human_profile}
            query["includePolicy"] = True
        if override_settings:
            query["overrideSettings"] = override_settings
        return query

    def _request_many(self, queries):
        """질의들을 모두 보낸 뒤, 응답을 id로 모아 보낸 순서대로 돌려준다."""
        if self._proc.poll() is not None:
            raise EngineError(f"KataGo 프로세스가 종료됨 (code {self._proc.returncode})")
        for q in queries:
            self._proc.stdin.write(json.dumps(q) + "\n")
        self._proc.stdin.flush()
        pending = {q["id"] for q in queries}
        got = {}
        # 응답은 끝난 순서대로 오고, 경고 메시지나 다른 id 응답이 섞일 수 있다
        while pending:
            line = self._proc.stdout.readline()
            if not line:
                raise EngineError("KataGo 응답 없이 출력이 끝남 (모델/설정 경로를 확인)")
            data = json.loads(line)
            if data.get("id") not in pending:
                continue
            if "error" in data:
                raise EngineError(data["error"])
            if data.get("isDuringSearch"):
                continue
            got[data["id"]] = data
            pending.discard(data["id"])
        return [got[q["id"]] for q in queries]

    def _request(self, query):
        return self._request_many([query])[0]

    def _sign(self, to_play):
        """KataGo 보고 기준 값을 둘 차례 기준으로 바꾸는 부호."""
        if self.report_as == "SIDETOMOVE" or self.report_as == to_play:
            return 1
        return -1

    def _parse(self, data, board):
        to_play = board.to_play
        sign = self._sign(to_play)

        def winrate(w):
            return w if sign == 1 else 1 - w

        moves = sorted(
            (
                MoveInfo(
                    move=m["move"],
                    visits=m["visits"],
                    winrate=winrate(m["winrate"]),
                    score_lead=sign * m["scoreLead"],
                    order=m["order"],
                    pv=m.get("pv", []),
                )
                for m in data["moveInfos"]
            ),
            key=lambda m: m.order,
        )

        ownership = data.get("ownership")
        if ownership is not None:
            # 둘 차례 기준으로 바꾼 뒤, 흑 기준으로 한 번 더 바꾼다
            to_black = sign * (1 if to_play == BLACK else -1)
            ownership = [to_black * v for v in ownership]

        human_policy = None
        if "humanPolicy" in data:
            # 배열 순서는 ownership과 같고 마지막 칸이 pass. 둘 수 없는 자리는 -1
            n = board.size
            human_policy = {}
            for i, p in enumerate(data["humanPolicy"]):
                if p > 0:
                    human_policy["pass" if i == n * n else board.format_vertex((n - 1 - i // n, i % n))] = p

        root = data["rootInfo"]
        return Analysis(
            to_play=to_play,
            moves=moves,
            winrate=winrate(root["winrate"]),
            score_lead=sign * root["scoreLead"],
            ownership=ownership,
            board_size=board.size,
            human_policy=human_policy,
        )

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def ownership_score(board, analysis, threshold=0.5):
    """ownership으로 사석을 반영한 중국식 계가. 양수면 흑 우세.

    board.area_score()는 판 위 돌을 모두 산 돌로 세지만, 여기서는 각 점을
    KataGo가 판단한 주인에게 준다. 죽은 돌은 상대 집으로, 애매한 점(|v| < threshold)은 공배.
    """
    total = 0
    for r in range(board.size):
        for c in range(board.size):
            v = analysis.ownership_at(board.format_vertex((r, c)))
            if v >= threshold:
                total += 1
            elif v <= -threshold:
                total -= 1
    return total - board.komi

