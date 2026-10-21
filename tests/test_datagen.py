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


def test_resume_skips_saved_games(tmp_path, monkeypatch):
    import adaptive_go.datagen as dg

    # 엔진 없이: 워커 풀 대신 즉석 실행하는 가짜로 바꿔 "어떤 판을 생성하려 했는지"만 본다
    asked = []

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, jobs):
            for rank, seed, size in jobs:
                asked.append(seed)
                yield rank, seed, []

    monkeypatch.setattr(dg, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(dg.progress, "tee_stdout", lambda kind: None)
    out = tmp_path / "run"
    out.mkdir()
    (out / "args.json").write_text('{"games_per_rank": 2, "seed": 100, "size": 9, "shard_games": 60}')
    # 이미 저장된 조각: 게임 100, 101
    rows = [("20k", 100, [(np.zeros((12, 9, 9), np.uint8), [0], [1.0], 0, 0)]), ("15k", 101, [(np.zeros((12, 9, 9), np.uint8), [0], [1.0], 0, 0)])]
    np.savez_compressed(out / "shard-000.npz", **pack(rows, 9))
    dg.main(["--resume", str(out)])
    assert 100 not in asked and 101 not in asked and len(asked) == 2 * 12 - 2
    assert (out / "shard-001.npz").exists()  # 조각 번호는 이어서
