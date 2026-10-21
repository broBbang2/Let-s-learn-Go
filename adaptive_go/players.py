"""AI 착수 전략들. 웹 대국(server)과 AI끼리 대국(arena)이 같이 쓴다.

모두 decide(engine, board, rng) -> (GTP 수, 그 수를 고를 때 쓴 Analysis)를 돌려준다.
분석 결과를 같이 주는 건 호출하는 쪽이 추가 질의 없이 그 수의 손실을 계산하게 하기 위해서다.
choose()는 수만 필요할 때 쓰는 단축형.
- BestPlayer        최강: KataGo 최선수
- TargetLossPlayer  v2: 목표 손실 근처 후보수를 확률적으로 (손실 "크기"를 맞춤)
- HumanPlayer       v3: human 모델이 예측한 "이 급수 사람의 수" 분포에서 뽑기 (실수 "모양"을 맞춤)
- HybridPlayer      v3 + v2: 사람 확률 × 목표 손실 가중치 (사람다운 수 중에서 강도를 맞춤)

공통 규칙: 엔진 최선수가 pass면 pass. 판이 끝난 국면에서 집을 메우며 대국을 끌지 않게 한다.
"""

import math
from dataclasses import dataclass

from .difficulty import MIN_VISITS, WIDE_ROOT_NOISE, choose_move

DEFAULT_VISITS = 200
# 사람 분포에서 이 확률 미만인 수는 버린다. 꼬리의 수백 개 수를 합치면 무시 못 할 확률이 되어
# 가끔 터무니없는 수가 나오기 때문 (사람도 실제로는 그런 수를 거의 안 둔다)
HUMAN_MIN_PROB = 0.01


def sample_human(policy, board, rng, min_prob=HUMAN_MIN_PROB):
    """human_policy에서 pass를 빼고 확률 비례로 하나 뽑는다."""
    items = [(m, p) for m, p in policy.items() if m != "pass" and p >= min_prob and board.is_legal(m)]
    if not items:
        items = [(m, p) for m, p in policy.items() if board.is_legal(m)]
    if not items:
        return "pass"
    moves, weights = zip(*items)
    return rng.choices(moves, weights)[0]


class Player:
    def choose(self, engine, board, rng):
        return self.decide(engine, board, rng)[0]


@dataclass
class BestPlayer(Player):
    visits: int = DEFAULT_VISITS

    @property
    def name(self):
        return "최강"

    def decide(self, engine, board, rng):
        a = engine.analyze(board, max_visits=self.visits)
        return a.best.move, a


@dataclass
class TargetLossPlayer(Player):
    target_loss: float
    visits: int = DEFAULT_VISITS

    @property
    def name(self):
        return f"v2({self.target_loss:g}집)"

    def decide(self, engine, board, rng):
        a = engine.analyze(board, max_visits=self.visits, override_settings={"wideRootNoise": WIDE_ROOT_NOISE})
        return choose_move(a, self.target_loss, rng=rng)[0].move, a


@dataclass
class HumanPlayer(Player):
    profile: str  # "rank_5k", "rank_1d" ...
    visits: int = DEFAULT_VISITS  # pass 판단용 탐색

    @property
    def name(self):
        return f"v3({self.profile.removeprefix('rank_')})"

    def decide(self, engine, board, rng):
        a = engine.analyze(board, max_visits=self.visits, human_profile=self.profile)
        if a.best.move == "pass":
            return "pass", a
        return sample_human(a.human_policy, board, rng), a


@dataclass
class HybridPlayer(Player):
    profile: str
    target_loss: float
    temperature: float = 1.0
    visits: int = DEFAULT_VISITS

    @property
    def name(self):
        return f"혼합({self.profile.removeprefix('rank_')},{self.target_loss:g}집)"

    def decide(self, engine, board, rng):
        a = engine.analyze(
            board,
            max_visits=self.visits,
            override_settings={"wideRootNoise": WIDE_ROOT_NOISE},
            human_profile=self.profile,
        )
        best = a.best
        if best.move == "pass":
            return "pass", a
        pool = [m for m in a.moves if m.move != "pass" and m.visits >= MIN_VISITS]
        weights = [
            a.human_policy.get(m.move, 0.0)
            * math.exp(-abs(max(0.0, best.score_lead - m.score_lead) - self.target_loss) / self.temperature)
            for m in pool
        ]
        if sum(weights) <= 0:
            return sample_human(a.human_policy, board, rng), a
        return rng.choices(pool, weights)[0].move, a


def parse_player(spec):
    """CLI용: "best", "v2:3", "v3:5k", "mix:5k:3" → Player"""
    kind, *args = spec.split(":")
    if kind == "best":
        return BestPlayer()
    if kind == "v2":
        return TargetLossPlayer(float(args[0]))
    if kind == "v3":
        return HumanPlayer(f"rank_{args[0]}")
    if kind == "mix":
        return HybridPlayer(f"rank_{args[0]}", float(args[1]))
    raise ValueError(f"알 수 없는 플레이어: {spec}")
