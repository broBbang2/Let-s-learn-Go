"""AI 수의 근거를 사람이 읽을 수 있게 정리한다.

"왜 이 수를 뒀나"를 두 가지로 보여준다.
- summary: 어떤 방식(최강/손실형/사람형/혼합)으로, 어떤 수치를 보고 골랐는지 한 문장
- candidates: 비교한 후보수 표 (엔진 손실, 승률, 사람 선택률, 탐색 수, 예상 진행)
"""

from .difficulty import MIN_VISITS
from .players import BestPlayer, HumanPlayer, HybridPlayer, TargetLossPlayer
from .rank_estimate import rank_label

TOP = 5


def _profile_label(profile):
    return rank_label(profile.removeprefix("rank_"))


def candidate_rows(analysis, chosen, top=TOP):
    """엔진 순서 상위 후보 + 실제로 고른 수(상위에 없으면 덧붙임)."""
    best = analysis.best
    human = analysis.human_policy or {}
    pool = [m for m in analysis.moves if m.visits >= MIN_VISITS or m.move == chosen][:top]
    if chosen not in [m.move for m in pool]:
        extra = analysis.find(chosen)
        if extra:
            pool.append(extra)
    return [
        {
            "move": m.move,
            "loss": round(max(0.0, best.score_lead - m.score_lead), 2),
            "winrate": round(m.winrate, 3),
            "human_p": round(human[m.move], 3) if m.move in human else None,
            "visits": m.visits,
            "pv": m.pv[:8],
            "chosen": m.move == chosen,
        }
        for m in pool
    ]


def explain_ai_move(player, move, analysis, loss):
    """player: players의 Player, analysis: 고를 때 쓴 분석, loss: 실제 손실(집, evaluate_move 기준)."""
    best = analysis.best
    human = analysis.human_policy or {}
    if move == "pass":
        summary = "엔진이 더 둘 곳이 없다고 판단해(최선수가 패스) 패스했습니다."
    elif isinstance(player, BestPlayer):
        summary = f"엔진 최선수입니다. {best.visits}회 탐색한 후보 중 승률이 가장 높았습니다 ({best.winrate:.0%})."
    elif isinstance(player, TargetLossPlayer):
        summary = (
            f"목표 손실 {player.target_loss:.1f}집에 맞춘 수입니다. 최선수 {best.move} 대비 {loss:.1f}집 손해인 후보를 "
            "목표에 가까울수록 높은 확률로 골랐습니다."
        )
    elif isinstance(player, HybridPlayer):
        p = human.get(move, 0.0)
        summary = (
            f"{_profile_label(player.profile)} 사람이 둘 확률({p:.0%})과 목표 손실 {player.target_loss:.1f}집에 가까운 정도를 "
            f"곱한 가중치로 골랐습니다. 최선수 {best.move} 대비 {loss:.1f}집입니다."
        )
    elif isinstance(player, HumanPlayer):
        p = human.get(move, 0.0)
        ranked = sorted(human, key=human.get, reverse=True)
        order = f"{ranked.index(move) + 1}위" if move in ranked else "순위 밖"
        summary = (
            f"{_profile_label(player.profile)} 사람 모델이 이 국면에서 이 수를 둘 확률은 {p:.0%}({order})입니다. "
            f"그 확률 분포에서 뽑았습니다. 엔진 기준으로는 최선수 {best.move} 대비 {loss:.1f}집입니다."
        )
    else:
        summary = ""
    return {"move": move, "summary": summary, "candidates": candidate_rows(analysis, move)}


def explain_user_move(move_eval, best_pv):
    """사용자 수 평가 근거: 최선수와 그 예상 진행, 상위 후보."""
    e = move_eval
    if e.move == e.best_move:
        summary = "엔진 최선수와 같은 수입니다."
    else:
        summary = f"엔진 최선수는 {e.best_move}였고, 둔 수는 그보다 {e.score_loss:.1f}집 손해로 평가됩니다."
    return {
        "summary": summary,
        "best_pv": best_pv[:8],
        "candidates": [{"move": m, "score": s, "winrate": w, "visits": v} for m, s, w, v in e.candidates],
    }
