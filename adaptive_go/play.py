"""터미널 텍스트 대국. 2단계: 엔진은 아직 풀파워(최선수)로 둔다.

python -m adaptive_go.play --size 9 --color B
"""

import argparse
from pathlib import Path

from .board import BLACK, Board, IllegalMoveError, format_score
from .engine import KataGoEngine, ownership_score

# Homebrew katago가 모델과 예제 설정을 같이 설치한다
KATAGO_SHARE = Path("/opt/homebrew/share/katago")
DEFAULT_MODEL = KATAGO_SHARE / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
DEFAULT_CONFIG = KATAGO_SHARE / "configs/analysis_example.cfg"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="KataGo와 텍스트 대국")
    p.add_argument("--katago", default="katago")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.0)
    p.add_argument("--color", choices=["B", "W"], default=BLACK, help="사용자 색")
    p.add_argument("--visits", type=int, default=200)
    return p.parse_args(argv)


def run(engine, board, user_color, visits, read=input, write=print):
    """대국 루프. 입출력을 주입받아 테스트에서 가짜 입력으로 돌릴 수 있게 한다."""
    while not board.is_over():
        write(f"\n{board}\n")
        if board.to_play == user_color:
            move = read(f"{user_color} 차례 (예: D4, pass, quit) > ").strip()
            if move.lower() == "quit":
                write("대국 중단")
                return None
            try:
                board.play(move)
            except IllegalMoveError as e:
                write(str(e))
        else:
            analysis = engine.analyze(board, max_visits=visits)
            move = analysis.best.move
            board.play(move)
            write(f"AI: {move}  (AI 기준 승률 {analysis.best.winrate:.0%}, {analysis.best.score_lead:+.1f}집)")

    # 종국: board.area_score()는 사석을 못 가리므로 KataGo ownership으로 계가
    write(f"\n{board}\n")
    final = engine.analyze(board, max_visits=visits, include_ownership=True)
    score = ownership_score(board, final)
    write(f"결과: {format_score(score)} (사석 반영)  / 판 위 돌 그대로: {format_score(board.area_score())}")
    return score


def main(argv=None):
    args = parse_args(argv)
    board = Board(args.size, args.komi)
    with KataGoEngine.from_paths(args.katago, args.model, args.config) as engine:
        run(engine, board, args.color, args.visits)


if __name__ == "__main__":
    main()
