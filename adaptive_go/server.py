"""localhost 웹 대국 서버.

python -m adaptive_go.server  →  http://localhost:8000

외부 의존성 없이 표준 라이브러리 http.server만 쓴다. 프론트는 web/index.html 한 장.
대국은 서버 메모리에 한 판만 둔다 (혼자 쓰는 로컬 서버라 세션 관리는 아직 불필요).
"""

import argparse
import json
import random
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import difficulty
from .board import BLACK, Board, format_score, opponent
from .difficulty import DifficultyState, level_label, level_of
from .explain import explain_ai_move, explain_user_move
from .library import Library
from .engine import KataGoEngine, ownership_score
from .play import DEFAULT_CONFIG, DEFAULT_MODEL
from .players import BestPlayer, HumanPlayer, HybridPlayer, TargetLossPlayer
from .rank_estimate import RANKS, RankEstimate, nearest_rank, rank_label
from .rating import Rating, update
from .records import Records, new_game_id
from .skill import evaluate_move, summarize

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
APP_DIST = ROOT / "app" / "dist"  # React 대국 화면 빌드 결과 (cd app && npm run build)
HUMAN_MODEL = ROOT / "models/b18c384nbt-humanv0.bin.gz"
LIBRARY_DIR = ROOT / "library"


# 최강 모드(KataGo 최선수만 둠)의 레이팅 가정치. 맞춤 모드는 difficulty의 레벨별 레이팅을 쓴다.
# 갱신하지 않는 고정점이라 전체 레이팅 척도의 기준(앵커) 역할도 한다.
AI_RATING = Rating(rating=2800, rd=50, volatility=0.06)
ADAPTIVE, HUMAN, BEST = "adaptive", "human", "best"  # v2 손실 맞춤, v3 사람형 맞춤, 최강

# v3 급수별 수당 평균 손실. arena --calibrate 결과(data/arena/human_calibration.json)가 있으면 그걸 쓴다
DEFAULT_HUMAN_CALIBRATION = {"20k": 6.0, "15k": 4.0, "10k": 2.5, "5k": 1.5, "1k": 1.0, "1d": 0.8, "5d": 0.5, "9d": 0.3}


def needs_hybrid(target_loss, calibration):
    """목표 손실이 가장 약한 급수보다도 크면 v3만으로는 그만큼 약해질 수 없다.

    arena 실측: v3 급수별 접전 손실은 약 0.4~1.9집으로 범위가 좁아, 수당 2집 넘게 잃는
    초보에겐 20급 프로필도 강하다. 이때는 가장 약한 급수의 사람 확률 × 목표 손실(혼합)로 둔다.
    """
    return target_loss > max(calibration.values())


# 대국 중 형세(누구 땅인지) 질의 탐색량. 매 수 보여주므로 빠르게, 그래도 초반에 흔들리지 않을 만큼
LIVE_OWNERSHIP_VISITS = 64

STATIC_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".json": "application/json",
}


class Game:
    """한 판의 상태와 엔진 호출을 묶는다. HTTP와 분리해 테스트하기 쉽게 했다."""

    def __init__(self, engine, records, rng=None, human_available=False, calibration=None, rank_policy=None):
        self.engine = engine
        # 실력 추정에 쓸 "급수별 수 확률" 공급원. 기본은 KataGo human 모델, 직접 학습한 모델로 바꿀 수 있다
        self.rank_policy = rank_policy or engine.human_policies
        self.rank_source = "ours" if rank_policy else "katago"
        self.records = records
        self.rng = rng or random.Random()
        self.human_available = human_available
        self.calibration = calibration or DEFAULT_HUMAN_CALIBRATION
        self.rating = records.load_rating()
        # 모드마다 "목표 손실 → 실제 강도" 관계가 달라서 보정 계수를 따로 둔다
        saved = records.load_json("difficulty.json", {})
        if "log_factor" in saved:  # 이전 형식(v2 하나만 있던 때)
            saved = {ADAPTIVE: saved}
        self.difficulties = {m: DifficultyState(**saved.get(m, {})) for m in (ADAPTIVE, HUMAN)}
        self.level_ratings = records.load_json("ai_levels.json") or difficulty.default_level_ratings()
        self.rank_prior = (records.load_json("rank_estimate.json") or {}).get("prior")
        # 엔진 프로세스는 한 번에 한 질의만 처리하므로 요청들을 직렬화한다
        self.lock = threading.Lock()
        self.new_game()

    def new_game(self, size=19, komi=7.5, color=BLACK, visits=200, mode=None):
        mode = mode or (HUMAN if self.human_available else ADAPTIVE)
        with self.lock:
            # 도중에 새 판을 시작하면 이전 판은 games.jsonl에 남기지 않는다 (moves.jsonl에는 남음)
            self.game_id = new_game_id()
            self.board = Board(size, komi)
            self.user_color = color
            self.visits = visits
            if mode == HUMAN and not self.human_available:
                mode = ADAPTIVE  # human 모델 파일이 없으면 v2로 대신한다
            self.mode = mode
            # 목표 손실은 판 시작 때 정하고 한 판 동안 고정 (판 중간에 바뀌면 결과로 보정하기 어려움)
            self.user_loss = difficulty.recent_user_loss(self.records.recent_user_losses(difficulty.RECENT_MOVES))
            self.target_loss = self.current_difficulty.target_loss(self.user_loss) if mode != BEST else 0.0
            # 실력 추정은 모드와 상관없이 매 판 지난 추정에서 출발해 수마다 갱신
            self.estimate = RankEstimate.from_prior(self.rank_prior)
            self.profile, self.hybrid = None, False
            self._update_profile()
            self.user_explain = None
            self.estimate_evidence = None
            self.ai_losses = []
            self.evals = []
            self.last_ai = None
            self.result = None
            self.rating_change = None
            self.message = None
            self.ownership = None
            if self.board.to_play != color:
                self._ai_move()
                self._update_live_ownership()
            return self.state()

    def user_move(self, move, think_ms=None):
        with self.lock:
            self.message = None
            if self.board.is_over() or self.result is not None:
                self.message = "이미 끝난 대국"
            elif self.board.to_play != self.user_color:
                self.message = "AI 차례"
            elif not self.board.is_legal(move):
                self.message = f"{self.user_color} {move}: 둘 수 없는 자리"
            else:
                # 두기 전에 평가해야 "그 국면에서 최선수 대비" 손실이 나온다
                ev = evaluate_move(self.engine, self.board, move)
                self._update_estimate(move)
                extra = {"rank_estimate": self.estimate.summary()["mean_stones"]}
                self.records.append_move(
                    self.game_id, self.board, ev, think_ms, self.rating, self._ai_settings(), extra
                )
                self.board.play(move)
                self.evals.append(ev)
                self.user_explain = explain_user_move(ev, ev.best_pv)
                self._update_profile()  # 방금 수로 추정이 바뀌었으니 AI 급수도 다시 맞춘다
                if not self.board.is_over():
                    self._ai_move()
                self._update_live_ownership()
            self._finish_if_over()
            return self.state()

    def resign(self):
        with self.lock:
            if self.result is None:
                self.result = {"text": f"{opponent(self.user_color)}+R (기권)", "score": None}
                self._record_game(user_score=0.0)
            return self.state()

    @property
    def current_difficulty(self):
        return self.difficulties.get(self.mode, DifficultyState())

    def _ai_settings(self):
        return {
            "engine": "katago",
            "mode": self.mode,
            "visits": self.visits,
            "target_loss": round(self.target_loss, 3),
            "log_factor": round(self.current_difficulty.log_factor, 4),
            "profile": self.profile,
            "hybrid": self.hybrid,
        }

    def _update_estimate(self, move):
        """사용자 수로 급수 추정 갱신. pass는 실력 정보가 아니라서 건너뛴다."""
        if not self.human_available or move.lower() == "pass":
            return
        policies = self.rank_policy(self.board, [f"rank_{r}" for r in RANKS])
        self.estimate.update(policies, move.upper())
        # 근거: 급수별로 "이 수를 둘 확률". 확률이 높은 급수 쪽으로 추정이 움직인다
        self.estimate_evidence = {
            "move": move.upper(),
            "probs": [round(policies[f"rank_{r}"].get(move.upper(), 0.0), 4) for r in RANKS],
        }
        self.rank_prior = self.estimate.next_game_prior()
        self.records.save_json(
            "rank_estimate.json", {"prior": self.rank_prior, "current": self.estimate.summary()}
        )

    def _update_profile(self):
        """사람형 모드: AI 급수 = 추정 사용자 급수 + 판 결과로 배운 보정(rank_offset)."""
        if self.mode != HUMAN:
            return
        rank = nearest_rank(self.estimate.mean_stones() + self.current_difficulty.rank_offset)
        self.profile = f"rank_{rank}"
        # 가장 약한 급수로도 부족하면(초보) 사람 확률 × 목표 손실로 더 약하게
        self.hybrid = rank == RANKS[0] and needs_hybrid(self.target_loss, self.calibration)

    def _player(self):
        if self.mode == HUMAN and self.hybrid:
            return HybridPlayer(self.profile, self.target_loss, visits=self.visits)
        if self.mode == HUMAN:
            return HumanPlayer(self.profile, self.visits)
        if self.mode == ADAPTIVE:
            return TargetLossPlayer(self.target_loss, self.visits)
        return BestPlayer(self.visits)

    def _ai_avg_loss(self):
        return sum(self.ai_losses) / len(self.ai_losses) if self.ai_losses else 0.0

    def _ai_move(self):
        player = self._player()
        move, analysis = player.decide(self.engine, self.board, self.rng)
        # 고를 때 쓴 분석을 재사용해 손실 계산 (후보에 없던 수만 추가 질의)
        ev = evaluate_move(self.engine, self.board, move, before=analysis, visits=self.visits)
        self.board.play(move)
        self.ai_losses.append(ev.score_loss)
        # AI 기준 값을 사용자 기준으로 뒤집어 보여준다 (AI가 둔 뒤 국면 = 두기 전 값 - 손실)
        self.last_ai = {
            "move": ev.move,
            "loss": ev.score_loss,
            "user_winrate": 1 - (ev.winrate_before - ev.winrate_loss),
            "user_score_lead": -(ev.score_before - ev.score_loss),
            "explain": explain_ai_move(player, ev.move, analysis, ev.score_loss),
        }

    def _ownership_grid(self, analysis):
        """ownership을 grid와 같은 [row][col] 배열로 (흑 +1 ~ 백 -1)."""
        n = self.board.size
        return [[analysis.ownership_at(self.board.format_vertex((r, c))) for c in range(n)] for r in range(n)]

    def _update_live_ownership(self):
        """AI 응수까지 끝난 국면의 형세. 웹이 판 위에 누구 땅이 될지 은은하게 칠한다."""
        if self.board.is_over():
            return
        a = self.engine.analyze(self.board, max_visits=LIVE_OWNERSHIP_VISITS, include_ownership=True)
        self.ownership = self._ownership_grid(a)

    def _finish_if_over(self):
        if self.board.is_over() and self.result is None:
            final = self.engine.analyze(self.board, max_visits=self.visits, include_ownership=True)
            score = ownership_score(self.board, final)
            self.result = {
                "text": format_score(score),
                "score": score,
                "ownership": self._ownership_grid(final),
            }
            if score == 0:
                user_score = 0.5
            else:
                user_score = 1.0 if (score > 0) == (self.user_color == BLACK) else 0.0
            self._record_game(user_score)

    def _record_game(self, user_score):
        before = self.rating
        if self.mode != BEST:
            # 이번 판 AI가 실제로 낸 평균 손실로 레벨을 정하고, 사용자와 서로 레이팅을 갱신
            level = level_of(self._ai_avg_loss())
            ai_rating = Rating(**self.level_ratings[str(level)])
            self.level_ratings[str(level)] = asdict(difficulty.update_level_rating(ai_rating, before, user_score))
            self.records.save_json("ai_levels.json", self.level_ratings)
            self.difficulties[self.mode] = self.current_difficulty.after_game(user_score)
            self.records.save_json("difficulty.json", {m: d.to_dict() for m, d in self.difficulties.items()})
        else:
            level, ai_rating = None, AI_RATING
        self.rating = update(before, [(ai_rating, user_score)])
        self.rating_change = {"before": round(before.rating, 1), "after": round(self.rating.rating, 1)}
        self.records.save_rating(self.rating)
        self.records.append_game(
            {
                "game_id": self.game_id,
                "size": self.board.size,
                "komi": self.board.komi,
                "user_color": self.user_color,
                "moves": [[c, v] for c, v in self.board.history],
                "result": self.result["text"],
                "user_score": user_score,
                "summary": summarize(self.evals),
                "rating_before": asdict(before),
                "rating_after": asdict(self.rating),
                "ai": {**self._ai_settings(), "avg_loss": round(self._ai_avg_loss(), 3), "level": level},
                "ai_rating": asdict(ai_rating),
            }
        )

    def state(self):
        b = self.board
        last = self.evals[-1] if self.evals else None
        return {
            "game_id": self.game_id,
            "size": b.size,
            "komi": b.komi,
            "grid": b.grid,  # grid[row][col], row 0 = 1행(아래)
            "to_play": b.to_play,
            "user_color": self.user_color,
            "history": b.history,
            "captures": b.captures,
            "ko_point": b.format_vertex(b.ko_point) if b.ko_point else None,
            "over": b.is_over() or self.result is not None,
            "result": self.result,
            "ownership": self.ownership,  # 대국 중 형세 (종국 후엔 result.ownership)
            "last_ai": self.last_ai,
            "last_eval": last and {
                "move": last.move,
                "best_move": last.best_move,
                "score_loss": last.score_loss,
                "winrate_loss": last.winrate_loss,
            },
            "summary": summarize(self.evals),
            "rating": {"rating": round(self.rating.rating, 1), "rd": round(self.rating.rd, 1)},
            "estimate": {**self.estimate.summary(), "ranks": [rank_label(r) for r in RANKS], "source": self.rank_source}
            if self.human_available
            else None,
            "user_explain": self.user_explain,
            "estimate_evidence": self.estimate_evidence,
            "rating_change": self.rating_change,
            "message": self.message,
            "visits": self.visits,
            "ai": {
                "mode": self.mode,
                "human_available": self.human_available,
                "profile": self.profile and self.profile.removeprefix("rank_"),
                "hybrid": self.hybrid,
                "target_loss": round(self.target_loss, 2),
                "avg_loss": round(self._ai_avg_loss(), 2),
                "level": level_label(level_of(self._ai_avg_loss())) if self.ai_losses else None,
            },
        }


def make_handler(game, library=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path.startswith("/api/library") and library:
                return self._library_get(path)
            if path == "/library":
                return self._file(WEB_DIR / "library.html", "text/html; charset=utf-8")
            if self.path == "/api/state":
                with game.lock:
                    return self._json(game.state())
            if path in ("/", "/index.html"):
                # React 빌드가 있으면 그걸, 없으면 예전 한 장짜리 화면
                if (APP_DIST / "index.html").exists():
                    return self._file(APP_DIST / "index.html", "text/html; charset=utf-8")
                return self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if path == "/classic":
                return self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if path.startswith("/assets/"):
                return self._static(APP_DIST, path)
            if self.path == "/board.js":
                return self._file(WEB_DIR / "board.js", "text/javascript; charset=utf-8")
            self.send_error(404)

        def _library_get(self, path):
            parts = path.strip("/").split("/")  # api, library, <id>, (analysis|sgf)
            if len(parts) == 2:
                return self._json(library.list())
            game_id = parts[2]
            if len(parts) == 3:
                g = library.load(game_id)
                return self._json({**g, "analysis": library.analysis(game_id)}) if g else self.send_error(404)
            if parts[3] == "analysis":
                return self._json(library.analysis(game_id))
            if parts[3] == "sgf":
                text = library.sgf(game_id)
                if text is None:
                    return self.send_error(404)
                raw = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-go-sgf; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{game_id}.sgf"')
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return None
            return self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            parts = self.path.split("?")[0].strip("/").split("/")
            if library and len(parts) == 4 and parts[:2] == ["api", "library"] and parts[3] == "analyze":
                return self._json(library.start(parts[2]))
            if self.path == "/api/new":
                return self._json(
                    game.new_game(
                        size=int(body.get("size", 19)),
                        komi=float(body.get("komi", 7.5)),
                        color=body.get("color", BLACK),
                        visits=int(body.get("visits", 200)),
                        mode=body.get("mode"),
                    )
                )
            if self.path == "/api/move":
                think_ms = body.get("think_ms")
                return self._json(game.user_move(str(body.get("move", "")), think_ms=think_ms))
            if self.path == "/api/resign":
                return self._json(game.resign())
            self.send_error(404)

        def _json(self, data):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _static(self, root, path):
            target = (root / path.lstrip("/")).resolve()
            # ../ 로 빌드 폴더 밖을 읽지 못하게
            if root.resolve() not in target.parents or not target.is_file():
                return self.send_error(404)
            return self._file(target, STATIC_TYPES.get(target.suffix, "application/octet-stream"))

        def _file(self, path, content_type):
            raw = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            pass  # 매 요청 로그는 생략

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="localhost 웹 대국 서버")
    p.add_argument("--katago", default="katago")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="대국 기록·레이팅 저장 위치")
    p.add_argument("--library-dir", default=str(LIBRARY_DIR), help="명국 SGF 폴더")
    p.add_argument("--rank-model", help="실력 추정에 직접 학습한 모델 사용 (예: models/rank_policy.pt)")
    args = p.parse_args(argv)

    human_model = str(HUMAN_MODEL) if HUMAN_MODEL.exists() else None
    if not human_model:
        print(f"human 모델 없음({HUMAN_MODEL}) → 사람형 맞춤(v3) 대신 손실 맞춤(v2)으로 동작")
    records = Records(args.data_dir)
    calibration = records.load_json("arena/human_calibration.json")
    with KataGoEngine.from_paths(args.katago, args.model, args.config, human_model=human_model) as engine:
        print("KataGo 초기화 중... (첫 실행은 20초 정도 걸림)")
        engine.analyze(Board(9), max_visits=1)  # 미리 데워서 첫 수 대기를 없앤다
        rank_policy = None
        if args.rank_model:
            from .model import NetPolicy  # torch는 이 옵션을 쓸 때만 불러온다

            rank_policy = NetPolicy(args.rank_model).human_policies
            print(f"실력 추정: 직접 학습한 모델 {args.rank_model}")
        game = Game(engine, records, human_available=bool(human_model), calibration=calibration, rank_policy=rank_policy)
        # 기보 분석은 대국 엔진을 막지 않도록 KataGo를 하나 더 띄운다 (처음 분석할 때)
        library = Library(
            args.library_dir,
            records,
            lambda: KataGoEngine.from_paths(args.katago, args.model, args.config, human_model=human_model),
        )
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game, library))
        print(f"http://localhost:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            library.close()


if __name__ == "__main__":
    main()
