import sys
from pathlib import Path

import pytest

from adaptive_go.board import BLACK, WHITE
from adaptive_go.engine import KataGoEngine
from adaptive_go.records import Records
from adaptive_go.difficulty import DifficultyState
from adaptive_go.server import ADAPTIVE, BEST, HUMAN, Game, needs_hybrid

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def records(tmp_path):
    return Records(tmp_path)


class MaxRng:
    """가중치가 가장 큰 후보를 항상 고르는 결정적 rng (테스트용)."""

    def choices(self, population, weights):
        return [population[weights.index(max(weights))]]


@pytest.fixture
def engine():
    with KataGoEngine(FAKE) as e:
        yield e


@pytest.fixture
def game(engine, records):
    # 기존 테스트는 최강 모드 기준 (AI가 항상 C3)
    g = Game(engine, records, rng=MaxRng())
    g.new_game(size=9, komi=7.0, mode=BEST)
    return g


def test_user_move_gets_ai_reply(game):
    s = game.user_move("E5")
    assert s["history"] == [(BLACK, "E5"), (WHITE, "C3")]
    # 가짜 엔진: 흑 기준 승률 0.7 → 백(AI) 기준 0.3 → 사용자(흑) 기준 0.7
    assert s["last_ai"]["user_winrate"] == pytest.approx(0.7)
    assert s["last_ai"]["user_score_lead"] == 3.0
    assert s["message"] is None


def test_illegal_move_keeps_turn(game):
    s = game.user_move("Z9")
    assert s["history"] == []
    assert s["message"]


def test_ai_moves_first_when_user_is_white(game):
    s = game.new_game(size=9, komi=7.0, color=WHITE, mode=BEST)
    assert s["history"] == [(BLACK, "C3")]
    assert s["to_play"] == WHITE


def test_game_ends_with_ownership_result(game):
    game.user_move("E5")  # AI C3
    s = game.user_move("pass")  # AI pass(가짜 엔진) → 종국
    assert s["over"]
    assert s["result"]["text"] == "W+86"
    assert s["result"]["ownership"][8][0] == 1.0  # A9


def test_resign(game):
    s = game.resign()
    assert s["over"] and s["result"]["text"] == "W+R (기권)"
    assert game.user_move("E5")["message"] == "이미 끝난 대국"


def test_user_move_is_evaluated_and_logged(game, records):
    s = game.user_move("D4", think_ms=1234)
    assert s["last_eval"] == {"move": "D4", "best_move": "C3", "score_loss": 1.5, "winrate_loss": 0.05}
    rows = records.read("moves.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["moves"] == []  # 두기 직전 수순
    assert row["move"] == "D4" and row["score_loss"] == 1.5
    assert row["think_ms"] == 1234
    assert row["game_id"] == s["game_id"]
    assert row["ai"]["visits"] == 200


def test_illegal_move_is_not_logged(game, records):
    game.user_move("Z9")
    assert records.read("moves.jsonl") == []


def test_finished_game_updates_rating_and_persists(game, records):
    game.user_move("E5")
    s = game.user_move("pass")  # 종국, 사용자(흑) 패배
    assert s["rating_change"]["after"] < s["rating_change"]["before"]
    games = records.read("games.jsonl")
    assert len(games) == 1 and games[0]["user_score"] == 0.0
    assert games[0]["summary"]["moves"] == 2
    # 서버를 다시 띄워도 레이팅이 이어진다
    assert records.load_rating().rating == pytest.approx(s["rating_change"]["after"], abs=0.1)


def test_resign_records_once(game, records):
    game.resign()
    game.resign()
    assert len(records.read("games.jsonl")) == 1


def test_adaptive_ai_picks_move_near_target_loss(engine, records):
    g = Game(engine, records, rng=MaxRng())
    # 기록이 없으면 목표 손실 = 사전값 3.0집 → 후보 C3(0집), D4(1.5집) 중 D4가 더 가깝다
    s = g.new_game(size=9, komi=7.0, color=WHITE, mode=ADAPTIVE)
    assert s["ai"]["target_loss"] == 3.0
    assert s["history"] == [(BLACK, "D4")]
    assert s["last_ai"]["loss"] == 1.5


def test_adaptive_game_end_updates_difficulty_and_level(engine, records):
    g = Game(engine, records, rng=MaxRng())
    g.new_game(size=9, komi=7.0, color=WHITE, mode=ADAPTIVE)  # AI D4 (1.5집)
    s = g.user_move("pass")  # AI pass(가짜 엔진) → 종국, W+86 → 사용자(백) 승리
    assert s["over"] and s["result"]["text"] == "W+86"
    # 사용자가 이겼으니 보정 계수가 줄어 다음 판 AI가 강해진다
    diff = records.load_json("difficulty.json")[ADAPTIVE]
    assert diff["games"] == 1 and diff["log_factor"] < 0
    game_row = records.read("games.jsonl")[0]
    assert game_row["ai"]["avg_loss"] == 0.75  # (1.5 + 0) / 2
    assert game_row["ai"]["level"] == 2  # 0.5~1집 구간
    levels = records.load_json("ai_levels.json")
    assert levels["2"]["rating"] < 2000  # AI 레벨이 졌으니 레벨 레이팅 하락
    # 다음 판 목표 손실은 줄어든다 (사용자 손실 기록 1개 + 사전값)
    nxt = g.new_game(size=9, komi=7.0, color=WHITE, mode=ADAPTIVE)
    assert nxt["ai"]["target_loss"] < 3.0


def test_human_mode_plays_human_policy_move(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True, calibration={"20k": 1.8, "5k": 1.0})
    s = g.new_game(size=9, komi=7.0, color=WHITE)  # human 모델이 있으면 기본이 사람형 맞춤
    # 첫 판: 급수 추정이 균등 → 평균 약 -2.4점 → 가장 가까운 격자 3k
    assert s["ai"]["mode"] == HUMAN and s["ai"]["profile"] == "3k" and not s["ai"]["hybrid"]
    # 가짜 human 정책 D4 0.6 / C3 0.3 / A9 0.1 → 확률 비례 추출 (MaxRng는 최댓값 D4)
    assert s["history"] == [(BLACK, "D4")]
    assert s["last_ai"]["loss"] == 1.5


def test_human_mode_falls_back_without_model(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=False)
    assert g.new_game(size=9, komi=7.0, mode=HUMAN)["ai"]["mode"] == ADAPTIVE


def test_difficulty_kept_per_mode_and_old_format_migrates(engine, records):
    records.save_json("difficulty.json", {"log_factor": 0.5, "games": 3})  # 이전 형식
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    assert g.difficulties[ADAPTIVE].log_factor == 0.5
    assert g.difficulties[HUMAN].log_factor == 0.0


def test_hybrid_when_user_below_weakest_profile(engine, records):
    table = {"20k": 1.8, "5k": 1.0}
    assert needs_hybrid(3.0, table) and not needs_hybrid(1.5, table)
    records.save_json("difficulty.json", {HUMAN: DifficultyState(rank_offset=-30).to_dict()})
    g = Game(engine, records, rng=MaxRng(), human_available=True, calibration=table)
    s = g.new_game(size=9, komi=7.0, color=WHITE)  # 추정 + 보정이 20k보다 약함, 목표 3.0집 > 20k 1.8집 → 20k 혼합
    assert s["ai"]["profile"] == "20k" and s["ai"]["hybrid"]
    # 혼합: 후보 C3(손실 0, 사람 0.3) / D4(손실 1.5, 사람 0.6) 중 D4
    assert s["history"] == [(BLACK, "D4")]


def test_user_move_updates_rank_estimate(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    s = g.user_move("E5")
    assert s["estimate"]["moves"] == 1
    assert records.load_json("rank_estimate.json")["current"]["moves"] == 1
    assert records.read("moves.jsonl")[0]["hint_used"] is False


def test_hint_suggests_good_human_moves_and_excludes_move_from_estimate(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    s = g.hint()
    # 후보 중 손실 1집 이하이면서 탐색 2회 이상인 수: C3만 (D4는 1.5집 손해, pass는 탐색 1회)
    assert [h["move"] for h in s["hints"]["moves"]] == ["C3"]
    assert s["hints"]["moves"][0]["pv"] == ["C3", "D4"]
    assert s["hint_used"]
    s = g.user_move("C3")
    assert s["estimate"]["moves"] == 0  # 힌트 본 수는 실력 추정에서 제외
    assert s["hints"] is None and not s["hint_used"]
    assert records.read("moves.jsonl")[0]["hint_used"] is True
    assert records.recent_user_losses(10) == []  # 손실 통계에서도 제외


def test_hint_only_on_users_turn(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    g.resign()
    assert g.hint()["hints"] is None


def test_ai_move_has_explanation(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    s = g.new_game(size=9, komi=7.0, color=WHITE)  # 사람형 3k → D4
    ex = s["last_ai"]["explain"]
    assert ex["move"] == "D4"
    assert "3급 사람 모델이 이 국면에서 이 수를 둘 확률은 60%(1위)" in ex["summary"]
    assert "최선수 C3 대비 1.5집" in ex["summary"]
    rows = {r["move"]: r for r in ex["candidates"]}
    assert rows["D4"]["chosen"] and rows["D4"]["human_p"] == 0.6 and rows["D4"]["loss"] == 1.5
    assert rows["C3"]["pv"] == ["C3", "D4"] and not rows["C3"]["chosen"]


def test_best_mode_explanation(game):
    s = game.user_move("E5")
    assert s["last_ai"]["explain"]["summary"].startswith("엔진 최선수입니다. 60회 탐색")


def test_user_move_explanation_and_estimate_evidence(engine, records):
    g = Game(engine, records, rng=MaxRng(), human_available=True)
    s = g.user_move("D4")
    assert s["user_explain"]["summary"] == "엔진 최선수는 C3였고, 둔 수는 그보다 1.5집 손해로 평가됩니다."
    assert s["user_explain"]["best_pv"] == ["C3", "D4"]
    assert s["estimate_evidence"]["move"] == "D4" and s["estimate_evidence"]["probs"][0] == 0.6
