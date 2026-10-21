"""난이도 조절 v2: 목표 손실 기반 수 선택.

아이디어: AI가 매 수 "사용자와 비슷한 만큼" 손해를 보면 한 판 전체로는 비슷한 승부가 된다.
  목표 손실 = 사용자 최근 평균 손실 × 보정 계수
보정 계수는 판이 끝날 때마다 사용자 승률이 TARGET_USER_SCORE로 수렴하도록 조정한다.
(평균 손실이 같아도 큰 실수가 섞인 사람과 고른 사람은 승률이 달라서, 결과로 한 번 더 맞춘다)
"""

import math
import random
from bisect import bisect_right
from dataclasses import asdict, dataclass

from .rating import Rating, update

# 사용자 승률 목표. 50%보다 살짝 낮게 잡아 "이길 듯 말 듯" 도전감을 준다
TARGET_USER_SCORE = 0.45
# 보정 계수(로그 척도) 학습률. 한 판 지면 계수가 exp(0.3 * 0.45) ≈ 1.14배 → AI가 14% 더 실수
LEARNING_RATE = 0.3
LOG_FACTOR_LIMIT = 3.0
# 사람형(v3) 모드: AI 급수 = 추정한 사용자 급수 + rank_offset(점 단위). 판마다 결과로 조정
# 한 판 지면 0.45 × 2 = 0.9점 약하게, 이기면 1.1점 강하게
RANK_OFFSET_LR = 2.0
RANK_OFFSET_LIMIT = 10.0
# 기록이 적을 때 가정하는 사용자 수당 손실(집)과 그 가정의 무게(가상의 수 개수)
PRIOR_USER_LOSS = 3.0
PRIOR_WEIGHT = 20
RECENT_MOVES = 200
# 맞춤 모드 질의에만 쓰는 KataGo 설정. 첫 수 후보를 넓게 탐색해 "적당히 나쁜 수"의 값도 얻는다
# (9x9 실측: 200 visits에서 3회 이상 탐색된 후보 9개 → 19개, 최대 손실 5집 → 12집)
WIDE_ROOT_NOISE = 1.0
# 탐색이 이보다 적은 후보는 값이 부정확해서 고르지 않는다
MIN_VISITS = 2


def choose_move(analysis, target_loss, temperature=1.0, rng=random):
    """목표 손실 근처의 후보수를 확률적으로 고른다. (MoveInfo, 손실) 반환.

    가중치 exp(-|손실 - 목표| / temperature): 목표에서 멀수록 덜 뽑힌다.
    temperature가 작을수록 목표에 정확히 맞추고, 크면 더 다양하게 둔다.
    """
    best = analysis.best
    # 엔진이 패스를 최선으로 보면 대국을 끝내야 할 때 → 일부러 집을 메우며 끌지 않는다
    if best.move == "pass" or target_loss <= 0:
        return best, 0.0
    # 대국 중 패스는 사람이 보기엔 실수가 아니라 이상한 행동이라 후보에서 뺀다
    pool = [m for m in analysis.moves if m.move != "pass" and m.visits >= MIN_VISITS] or [best]
    losses = [max(0.0, best.score_lead - m.score_lead) for m in pool]
    weights = [math.exp(-abs(loss - target_loss) / temperature) for loss in losses]
    i = rng.choices(range(len(pool)), weights)[0]
    return pool[i], losses[i]


def recent_user_loss(losses):
    """최근 손실들의 평균. 기록이 적으면 PRIOR_USER_LOSS 쪽으로 당긴다 (첫 판부터 극단값 방지)."""
    losses = losses[-RECENT_MOVES:]
    return (sum(losses) + PRIOR_USER_LOSS * PRIOR_WEIGHT) / (len(losses) + PRIOR_WEIGHT)


@dataclass
class DifficultyState:
    log_factor: float = 0.0
    games: int = 0
    rank_offset: float = 0.0

    def target_loss(self, user_loss):
        return user_loss * math.exp(self.log_factor)

    def after_game(self, user_score):
        """사용자가 이기면(1) 계수를 줄여 AI를 강하게, 지면(0) 키워 약하게."""
        log_factor = self.log_factor - LEARNING_RATE * (user_score - TARGET_USER_SCORE)
        log_factor = max(-LOG_FACTOR_LIMIT, min(LOG_FACTOR_LIMIT, log_factor))
        rank_offset = self.rank_offset + RANK_OFFSET_LR * (user_score - TARGET_USER_SCORE)
        rank_offset = max(-RANK_OFFSET_LIMIT, min(RANK_OFFSET_LIMIT, rank_offset))
        return DifficultyState(log_factor, self.games + 1, rank_offset)

    def to_dict(self):
        return asdict(self)


# ---- AI 레벨 레이팅 ----
# AI 강도는 판마다 달라지므로, 그 판에서 AI가 실제로 낸 수당 평균 손실 구간을 "레벨"로 보고
# 레벨마다 Glicko-2 레이팅을 둔다. 사용자 레이팅과 서로 갱신하면서 보정된다.
LEVEL_EDGES = [0.25, 0.5, 1, 2, 4, 8]  # 집/수
# 레벨별 초기 가정치. RD를 크게 잡아 실제 결과로 빨리 움직이게 한다
LEVEL_PRIORS = [2600, 2300, 2000, 1700, 1400, 1100, 800]
LEVEL_PRIOR_RD = 300.0


def level_of(avg_ai_loss):
    return bisect_right(LEVEL_EDGES, avg_ai_loss)


def level_label(level):
    lo = 0 if level == 0 else LEVEL_EDGES[level - 1]
    hi = LEVEL_EDGES[level] if level < len(LEVEL_EDGES) else None
    return f"{lo}~{hi}집" if hi is not None else f"{lo}집 이상"


def default_level_ratings():
    return {str(i): asdict(Rating(r, LEVEL_PRIOR_RD)) for i, r in enumerate(LEVEL_PRIORS)}


def update_level_rating(level_rating, user_rating, user_score):
    """AI 레벨 쪽 레이팅 갱신. AI 입장 점수 = 1 - 사용자 점수."""
    return update(level_rating, [(user_rating, 1 - user_score)])
