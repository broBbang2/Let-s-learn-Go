"""대국 중 실시간 실력(급수) 추정 — 베이즈 갱신.

human 모델은 "급수 r의 사람이 이 국면에서 각 자리에 둘 확률" p(수 | r)을 준다.
사용자가 수를 둘 때마다 급수 후보 전체에 대해
  log P(r | 지금까지의 수) += w · log p(실제 둔 수 | r)
로 갱신하면, 몇 수 만에 "이런 수를 두는 사람은 몇 급일 가능성이 높다"가 좁혀진다.
(체스 Maia 연구의 급수 추정과 같은 아이디어. 별도 학습이 필요 없다)

w(LIKELIHOOD_WEIGHT) < 1로 낮추는 이유: 같은 판 안의 수들은 서로 독립이 아니고(한 수가 다음 수를 강제),
human 모델도 완벽하지 않아서 그대로 곱하면 너무 빨리 한 급수로 확신해 버린다.
"""

import math
from dataclasses import dataclass, field

# 추정할 급수 격자. 급수 사이 간격이 약할수록 넓은 건 실력 차이가 하수 쪽에서 더 크기 때문
RANKS = ["20k", "15k", "10k", "7k", "5k", "3k", "1k", "1d", "3d", "5d", "7d", "9d"]
# 실제 사람(OGS 300명) 검증에서 사전분포를 OGS 분포로 둘 때 1.0이 0.5보다 조금 나았다(10수 오차 4.45→4.24)
LIKELIHOOD_WEIGHT = 1.0
# human 정책이 0을 준 수(둘 수 없는 자리 외에 신경망이 사실상 0으로 본 수)도 사람은 둘 수 있으니 바닥값을 둔다
PROB_FLOOR = 1e-4
# 처음 보는 사용자의 출발점: OGS 랭크 대국 사용자의 급수 분포(검증 판 제외 33,264명을 가장 가까운 격자로).
# 균등하게 시작하면 추정이 격자 가운데(약 3급)로 끌려가 하수를 크게 강하게 봤다(치우침 +2.2~3.4급) →
# 이 분포로 시작하니 치우침이 +0.2~0.4급으로 줄었다. 우리 사용자가 쌓이면 그 분포로 바꿔야 한다
POPULATION_PRIOR = [0.070, 0.178, 0.238, 0.161, 0.120, 0.115, 0.055, 0.040, 0.020, 0.004, 0.0, 0.0]
PRIOR_FLOOR = 1e-3  # 표본이 없는 고단(7d·9d)도 아예 불가능하진 않게

# 지난 판 추정을 다음 판 출발점으로 쓸 때 분포를 이만큼 평평하게 (지수 < 1) → 실력 변화에 다시 적응할 여지
CARRY_OVER = 0.5


def stones(rank):
    """급수를 '점' 척도로: 1k = 0, 1d = 1, 5k = -4, 20k = -19. 평균·차이를 계산하기 위한 숫자 척도."""
    n = int(rank[:-1])
    return 1 - n if rank.endswith("k") else n


def nearest_rank(value):
    """점 척도 값에 가장 가까운 격자 급수."""
    return min(RANKS, key=lambda r: abs(stones(r) - value))


def rank_label(rank):
    return rank[:-1] + ("급" if rank.endswith("k") else "단")


@dataclass
class RankEstimate:
    log_probs: list = field(default_factory=lambda: [0.0] * len(RANKS))  # 정규화 안 된 로그 사후확률
    moves: int = 0  # 추정에 쓴 수 개수 (이번 판)

    @classmethod
    def from_prior(cls, probs=None):
        """probs가 없으면(처음 보는 사용자) POPULATION_PRIOR에서 출발."""
        if not probs:
            probs = [max(p, PRIOR_FLOOR) for p in POPULATION_PRIOR]
        return cls([math.log(max(p, 1e-12)) for p in probs])

    def update(self, policies, move, weight=LIKELIHOOD_WEIGHT):
        """policies: {"rank_5k": human_policy, ...}. move는 사용자가 실제로 둔 수."""
        for i, r in enumerate(RANKS):
            p = policies[f"rank_{r}"].get(move, 0.0)
            self.log_probs[i] += weight * math.log(max(p, PROB_FLOOR))
        self.moves += 1

    def probs(self):
        m = max(self.log_probs)
        ws = [math.exp(x - m) for x in self.log_probs]
        total = sum(ws)
        return [w / total for w in ws]

    def mean_stones(self):
        return sum(p * stones(r) for p, r in zip(self.probs(), RANKS))

    def map_rank(self):
        probs = self.probs()
        return RANKS[probs.index(max(probs))]

    def confidence(self):
        """가장 그럴듯한 급수와 그 양옆 격자를 합친 확률. 0.25(모름) ~ 1(확신)."""
        probs = self.probs()
        i = probs.index(max(probs))
        return sum(probs[max(0, i - 1) : i + 2])

    def next_game_prior(self):
        """다음 판 출발점: 분포를 CARRY_OVER 지수로 평평하게 한 확률."""
        flat = [p**CARRY_OVER for p in self.probs()]
        total = sum(flat)
        return [p / total for p in flat]

    def summary(self):
        return {
            "rank": self.map_rank(),
            "label": rank_label(self.map_rank()),
            "mean_stones": round(self.mean_stones(), 2),
            "mean_label": rank_label(nearest_rank(self.mean_stones())),
            "confidence": round(self.confidence(), 3),
            "moves": self.moves,
            "distribution": [round(p, 4) for p in self.probs()],
        }


# 실력 추정에서 KataGo human 모델과 자체 모델을 섞는 비율 (KataGo 쪽).
# 실제 사람 300명 검증: KataGo만 30수 오차 3.64, 자체만 3.52, 30% 혼합 3.28(상관 0.81, ±3급 58%)로 가장 좋았다
MIX_KATAGO = 0.3


def mix_policies(katago, ours, alpha=MIX_KATAGO):
    """두 공급원의 {프로필: {수: 확률}}을 확률 평균으로 섞는다 (로그 평균이 아니라: 한쪽이라도 그럴듯하면 인정)."""
    out = {}
    for profile, k in katago.items():
        o = ours[profile]
        out[profile] = {m: alpha * k.get(m, 0.0) + (1 - alpha) * o.get(m, 0.0) for m in set(k) | set(o)}
    return out

