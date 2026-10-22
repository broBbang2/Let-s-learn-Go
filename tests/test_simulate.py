import random
import sys
from pathlib import Path

import pytest

from adaptive_go.engine import KataGoEngine
from adaptive_go.records import Records
from adaptive_go.server import BEST, Game
from adaptive_go.simulate import play_one, rank_at, summarize

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


def test_rank_schedule():
    assert rank_at("fixed-5k", 3, 15) == -4
    assert rank_at("growing", 0, 11) == -14 and rank_at("growing", 10, 11) == -4  # 15k → 5k


def test_play_one_finishes_a_game(tmp_path):
    with KataGoEngine(FAKE) as e:
        g = Game(e, Records(tmp_path), rng=random.Random(0), human_available=True, live_territory=False)
        # 사용자 정책: 항상 D4 (가짜 엔진: D4 뒤엔 AI가 패스 → 150수 전이라도 판은 이어짐)
        policy = lambda board, profiles: {p: {"D4": 0.9, "pass": 0.1} for p in profiles}
        s = play_one(g, policy, -4, "B", random.Random(0))
        assert s["over"]


def test_summarize():
    rows = [{"game": i, "user_score": float(i % 2), "true_stones": -4, "estimate_stones": -5, "ai_profile": "rank_5k", "rank_offset": 0.0}
            for i in range(10)]
    s = summarize(rows)
    assert s["games"] == 10 and s["win_rate_last5"] == 0.6 and s["estimate_error_last5"] == 1
