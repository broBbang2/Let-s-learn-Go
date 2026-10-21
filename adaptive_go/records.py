"""대국 기록 저장. 나중에 학습 데이터로 바로 쓰도록 한 줄 = 한 샘플(JSONL)로 남긴다.

data/
  moves.jsonl  사용자 착수 한 수 = 한 줄. 국면 복원용 수순(moves)과 엔진 평가 포함
               → 급수 예측·사람 수 예측 모델의 학습 샘플
  games.jsonl  끝난 대국 한 판 = 한 줄. 결과, 손실 요약, 레이팅 변화
  player.json  현재 Glicko-2 레이팅
  difficulty.json  난이도 보정 계수
  ai_levels.json   AI 레벨별 레이팅
"""

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .rating import Rating
from .skill import effective_loss

SCHEMA_VERSION = 1  # 기록 형식이 바뀌면 올린다 (학습 코드가 버전별로 읽을 수 있게)


def new_game_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class Records:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def append_move(self, game_id, board, move_eval, think_ms, rating, ai_settings, extra=None):
        """move_eval을 board에 두기 직전 상태로 호출한다 (board.history = 이 수 이전 수순)."""
        self._append(
            "moves.jsonl",
            {
                "schema": SCHEMA_VERSION,
                "game_id": game_id,
                "ts": time.time(),
                "size": board.size,
                "komi": board.komi,
                "rules": "chinese",
                "moves": [[c, v] for c, v in board.history],
                **move_eval.to_dict(),
                "think_ms": think_ms,
                "player_rating": round(rating.rating, 1),
                "player_rd": round(rating.rd, 1),
                "ai": ai_settings,
                **(extra or {}),
            },
        )

    def append_game(self, record):
        self._append("games.jsonl", {"schema": SCHEMA_VERSION, "ts": time.time(), **record})

    def load_rating(self):
        data = self.load_json("player.json")
        return Rating(**data) if data else Rating()

    def save_rating(self, rating):
        self.save_json("player.json", asdict(rating))

    def recent_user_losses(self, n):
        """최근 접전 국면 사용자 착수 n개의 실력 지표용 손실 (오래된 것부터). 힌트를 본 수는 뺀다."""
        rows = [r for r in self.read("moves.jsonl") if not r.get("hint_used")]
        losses = [effective_loss(r["score_loss"], r["winrate_before"]) for r in rows]
        return [x for x in losses if x is not None][-n:]

    def load_json(self, name, default=None):
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else default

    def save_json(self, name, data):
        (self.dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def read(self, name):
        path = self.dir / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def _append(self, name, row):
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
