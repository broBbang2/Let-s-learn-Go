import random
import sys
from pathlib import Path

import numpy as np
import pytest

from adaptive_go.datagen import TOP_K, pack, play_game
from adaptive_go.engine import KataGoEngine
from adaptive_go.features import PLANES

FAKE = [sys.executable, str(Path(__file__).with_name("fake_katago.py"))]


@pytest.fixture
def engine():
    with KataGoEngine(FAKE) as e:
        yield e


def test_play_game_records_teacher_distribution(engine):
    rows = play_game(engine, "5k", random.Random(0), size=9)
    x, top_i, top_p, played, num = rows[0]
    assert x.shape == (PLANES, 9, 9) and num == 0
    # 가짜 human 정책: D4 0.6, C3 0.3, A9 0.1 (확률 순)
    assert top_i == [3 * 9 + 3, 2 * 9 + 2, 8 * 9 + 0] and top_p == [0.6, 0.3, 0.1]
    assert played in top_i


def test_pack_shapes_and_padding(engine):
    rows = play_game(engine, "5k", random.Random(0), size=9)
    d = pack([("5k", 7, rows)], 9)
    n = len(rows)
    assert d["planes"].shape == (n, (PLANES * 81 + 7) // 8)
    assert d["top_idx"].shape == (n, TOP_K) and d["top_idx"][0, 3] == -1  # 빈 칸은 -1
    assert d["rank"][0] == 4 and d["game"][0] == 7  # RANKS.index("5k") == 4
    x = np.unpackbits(d["planes"][0])[: PLANES * 81].reshape(PLANES, 9, 9)
    assert (x == rows[0][0]).all()
