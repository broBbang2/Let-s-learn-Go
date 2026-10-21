"""사용자 착수 평가: 엔진 최선수 대비 손실.

손실은 항상 "사용자 기준"이라 흑백 구분 없이 같은 식을 쓴다 (engine.py가 둘 차례 기준으로 맞춰 줌).
  score_loss   = 최선수 scoreLead - 사용자 수 scoreLead   (집)
  winrate_loss = 최선수 winrate  - 사용자 수 winrate     (0~1)
집 손실은 국면 전체에서 고르게 의미가 있고, 승률 손실은 승부가 기운 뒤엔 0에 가까워진다.
둘 다 남겨서 나중에 모델 특징으로 고를 수 있게 한다.
"""

from dataclasses import asdict, dataclass, field

from .board import Board

# 실력 측정은 AI 난이도와 무관하게 같은 탐색량으로 해야 기록끼리 비교할 수 있다
EVAL_VISITS = 200
# 이 이상 손해 본 수를 "큰 실수"로 센다
BLUNDER_LOSS = 3.0
# 실력 지표용 손실은 "승부가 아직 안 갈린 국면"의 수만 센다.
# 대마가 죽어 승부가 기운 뒤엔 매 수 수십 집씩 손실로 잡혀 평균을 지배하기 때문
# (arena 실측: 급수와 무관하게 손실 중앙값은 ~0이고, 평균은 이런 몇 수가 좌우했다)
CONTESTED_WINRATE = (0.05, 0.95)
# 접전 중 한 수 손실도 이 값에서 자른다. 대마 하나 죽는 수가 몇 판 치 평균을 뒤집지 않게
LOSS_CAP = 15.0
# 기록에 남길 후보수 개수 (학습 시 정답 분포·순위 특징으로 쓰기 위해)
TOP_CANDIDATES = 5


@dataclass
class MoveEval:
    move_number: int  # 1부터, 이 수가 몇 번째 수인지
    color: str
    move: str
    best_move: str
    score_loss: float  # 음수는 0으로 자름 (탐색 노이즈)
    winrate_loss: float
    score_before: float  # 두기 전 국면, 사용자 기준 집 차이
    winrate_before: float
    phase: str
    in_candidates: bool  # 엔진 후보수 안에 있었는지 (없으면 착수 후 국면을 따로 분석)
    candidates: list  # [[move, score_lead, winrate, visits], ...] 상위 TOP_CANDIDATES개
    best_pv: list = field(default_factory=list)  # 최선수 이후 엔진 예상 진행 (근거 표시용)

    def to_dict(self):
        return asdict(self)


def effective_loss(score_loss, winrate_before):
    """실력 지표용 손실. 접전이 아닌 국면의 수면 None (집계에서 뺀다)."""
    lo, hi = CONTESTED_WINRATE
    if not lo <= winrate_before <= hi:
        return None
    return min(score_loss, LOSS_CAP)


def phase_of(move_number, size):
    """수 번호로 대략적인 국면 단계를 나눈다. 판 넓이 대비 비율이라 9줄·19줄 모두 쓸 수 있다.

    9줄: 포석 ~12수, 중반 ~40수 / 19줄: 포석 ~54수, 중반 ~180수.
    정밀한 구분은 아니고, 단계별 손실 경향을 보기 위한 1차 기준.
    """
    ratio = move_number / (size * size)
    if ratio <= 0.15:
        return "opening"
    if ratio <= 0.5:
        return "middle"
    return "endgame"


def evaluate_move(engine, board, move, before=None, visits=EVAL_VISITS):
    """board에 move를 두기 *전에* 호출한다. board는 바꾸지 않는다.

    before: 이미 분석해 둔 현재 국면 결과가 있으면 재사용.
    """
    before = before or engine.analyze(board, max_visits=visits)
    best = before.best
    played = before.find(move)

    if played is not None:
        played_score, played_winrate = played.score_lead, played.winrate
    else:
        # 후보수에 없는 수: 둔 뒤 국면을 분석해 상대 기준 값을 뒤집는다
        after_board = _copy_and_play(board, move)
        after = engine.analyze(after_board, max_visits=visits)
        played_score, played_winrate = -after.score_lead, 1 - after.winrate

    move_number = len(board.history) + 1
    return MoveEval(
        move_number=move_number,
        color=board.to_play,
        move=move.upper() if move.lower() != "pass" else "pass",
        best_move=best.move,
        score_loss=round(max(0.0, best.score_lead - played_score), 3),
        winrate_loss=round(max(0.0, best.winrate - played_winrate), 4),
        score_before=round(before.score_lead, 3),
        winrate_before=round(before.winrate, 4),
        phase=phase_of(move_number, board.size),
        in_candidates=played is not None,
        candidates=[
            [m.move, round(m.score_lead, 3), round(m.winrate, 4), m.visits] for m in before.moves[:TOP_CANDIDATES]
        ],
        best_pv=best.pv[:8],
    )


def _copy_and_play(board, move):
    copy = Board(board.size, board.komi)
    for _, vertex in board.history:
        copy.play(vertex)
    copy.play(move)
    return copy


def summarize(evals):
    """한 판(또는 진행 중인 판)의 손실 요약."""
    if not evals:
        return {
            "moves": 0,
            "avg_score_loss": None,
            "avg_effective_loss": None,
            "avg_winrate_loss": None,
            "blunders": 0,
            "by_phase": {},
        }

    def avg(xs):
        return round(sum(xs) / len(xs), 3)

    effective = [x for e in evals if (x := effective_loss(e.score_loss, e.winrate_before)) is not None]

    by_phase = {}
    for phase in ("opening", "middle", "endgame"):
        losses = [e.score_loss for e in evals if e.phase == phase]
        if losses:
            by_phase[phase] = {"moves": len(losses), "avg_score_loss": avg(losses)}
    return {
        "moves": len(evals),
        "avg_score_loss": avg([e.score_loss for e in evals]),
        "avg_effective_loss": avg(effective) if effective else None,
        "avg_winrate_loss": avg([e.winrate_loss for e in evals]),
        "blunders": sum(e.score_loss >= BLUNDER_LOSS for e in evals),
        "by_phase": by_phase,
    }
