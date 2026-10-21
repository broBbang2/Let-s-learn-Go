# 적응형 바둑 AI — 작업 인계 노트

> claude.ai 대화에서 시작한 작업을 VS Code(Claude Code)에서 이어가기 위한 문서.
> Claude Code에게: 이 파일을 먼저 읽고 "현재 상태"와 "다음 할 일"부터 이어서 진행해줘.

## 목표

사용자와 바둑을 두면서 **사용자의 실력에 맞춰 같이 발전하는 바둑 AI**를 만들고, 최종적으로 웹 게임으로 배포한다.

## 핵심 설계 결정

- 강한 AI를 처음부터 학습시키지 않는다. 두뇌는 오픈소스 엔진 **KataGo**를 쓰고, 직접 설계하는 부분은 "얼마나·어떻게 맞춰 둘지"와 "사용자가 얼마나 늘었는지"를 판단하는 층이다.
- 시스템은 3개 층으로 나눈다.
  1. **엔진 층**: KataGo Analysis Engine(JSON)으로 매 수의 후보수, 승률, 예상 집 차이를 받는다.
  2. **사용자 모델 층**: 사용자 실력 추정.
     - 대국 결과 기반: Glicko-2 레이팅 (신규 유저 불확실성 처리)
     - 수 단위 기반: 사용자 착수의 **엔진 최선수 대비 평균 손실(집)** — 더 정밀하고 한 판 안에서도 갱신 가능
  3. **난이도 조절 층** (알고리즘의 핵심):
     - v1: 탐색량(visits) 제한 — 단순하지만 약해지는 방식이 부자연스러움
     - v2: **목표 손실 기반 수 선택** — 사용자 실력에 맞는 목표 손실 근처의 후보수를 확률적으로 고름
     - v3: KataGo의 **human 모델**(급수 조건부로 사람이 둘 법한 수 예측, 체스 Maia와 같은 아이디어) — 사람 같은 실수
     - 목표 함수: 장기적으로 사용자 승률이 **40~50%**에 머물도록 AI 레벨을 상향/하향

```python
def choose_move(candidates, target_loss, temperature):
    # candidates: [(move, score_loss), ...]  엔진 최선수 대비 손실
    weights = [exp(-abs(loss - target_loss) / temperature) for _, loss in candidates]
    return random.choices(candidates, weights)[0].move
```

## 전체 단계

1. ✅ **바둑 규칙 엔진** (Python) — 보드, 따냄, 패, 자충수, 계가
2. ✅ **KataGo 연결** — 내 보드 ↔ 엔진 통신, 일단 풀파워로 두게 하기
3. ✅ **사용자 실력 측정** — 수당 손실 계산, 레이팅 저장
4. ✅ **난이도 조절** — v2 목표 손실 + v3 human 모델 + AI 리그전 — 목표 손실 방식 → human 모델 방식으로 개선
5. ⬜ **웹 게임화** (프론트엔드)
6. ⬜ (선택, 학습용) **9x9 미니 AlphaZero를 PyTorch로 직접 학습** — MCTS + 정책/가치망 이해용. ML 로드맵 Phase 4~5(CNN, ResNet) 이후에 붙이면 좋음

## 현재 상태 (4단계 완료)

> 원래 claude.ai에서 만든 `board.py`는 로컬에 없어서, 이 문서 명세대로 다시 작성했다.

파일:
- `adaptive_go/board.py` — 규칙 엔진
- `adaptive_go/engine.py` — KataGo Analysis Engine 연결
- `adaptive_go/play.py` — 터미널 텍스트 대국 (엔진은 풀파워)
- `adaptive_go/server.py` — localhost 웹 대국 서버 (표준 라이브러리 http.server, 대국 한 판을 메모리에 보관)
- `web/index.html` — 웹 바둑판 (SVG, 클릭 착수, 승률 막대, 기보, 종국 시 ownership으로 집·사석 표시, 레이팅·매 수 손실 패널)
- `adaptive_go/skill.py` — 매 수 손실(집·승률), 국면 단계(포석/중반/끝내기), 한 판 요약
- `adaptive_go/rating.py` — Glicko-2 (Glickman 2013 논문 예제로 검증)
- `adaptive_go/records.py` — 학습용 JSONL 기록
- `adaptive_go/difficulty.py` — 목표 손실 기반 수 선택, 승률 45% 보정, AI 레벨별 레이팅
- `adaptive_go/players.py` — AI 전략 공통 인터페이스: 최강 / v2 목표 손실 / v3 사람형 / 혼합
- `adaptive_go/arena.py` — AI끼리 리그전 + v3 급수 보정 (`python -m adaptive_go.arena --help`)
- `adaptive_go/rank_estimate.py` — 대국 중 실시간 급수 추정 (human 정책 베이즈 갱신)
- `adaptive_go/estimate_eval.py` — 급수 아는 봇으로 추정 정확도 검증
- `models/b18c384nbt-humanv0.bin.gz` — KataGo human 모델 (v1.15.0 릴리스, sha256 637746e4…)
- `tests/test_board.py`, `test_engine.py`, `test_server.py`, `test_skill.py`, `test_rating.py`, `test_difficulty.py`, `test_players.py`, `test_rank_estimate.py`, `test_estimate_eval.py` — pytest 81개, 전부 통과
- `tests/fake_katago.py` — KataGo 없이 JSON 프로토콜을 흉내 내는 가짜 엔진 (테스트용)

`board.py` 요약:
- 좌표는 **GTP 형식**(`"D4"`, `"pass"`, 열에 `I` 없음, 아래가 1행) — KataGo와 그대로 통신하기 위함
- `Board(size, komi)`, `play(vertex)`, `is_legal()`, `legal_moves()`, `is_over()`(연속 패스 2번), `area_score()`(중국식 계가), `history`(GTP 수순), `print(board)`로 ASCII 출력
- 따냄은 플러드필로 그룹·활로 계산, 자충수 금지, **단순 패**만 처리(슈퍼코 미처리)
- 사석 판정은 안 함 → `engine.ownership_score()`가 KataGo ownership으로 대신 계가

`engine.py` 요약:
- `KataGoEngine.from_paths(katago, model, config)` → `analyze(board, max_visits, include_ownership)` → `Analysis`
- `Analysis.moves`는 `MoveInfo(move, visits, winrate, score_lead, order, pv)` 리스트, `best`, `find(move)`, `ownership_at(vertex)`
- **관점 통일**: winrate·score_lead는 항상 "둘 차례" 기준으로 변환 (config의 `reportAnalysisWinratesAs = BLACK`을 `report_as` 인자로 받아 뒤집음). ownership은 항상 흑 기준
- 규칙은 `chinese`로 질의 (board.py의 중국식 계가와 맞춤)

KataGo 환경 (Homebrew `katago` 1.18.2):
- 모델: `/opt/homebrew/share/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` (그 외 g170 모델 2개)
- 설정: `/opt/homebrew/share/katago/configs/analysis_example.cfg`
- human 모델 예제 설정(`gtp_human5k_example.cfg` 등)도 같이 설치됨 → 4단계 v3에서 사용. human 모델 파일(`b18c384nbt-humanv0`)은 별도 다운로드 필요
- 첫 질의는 약 20초 걸림(엔진 초기화)

실행:
```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest -q
.venv/bin/python -m adaptive_go.server        # → http://localhost:8000 (대국은 여기서)
.venv/bin/python -m adaptive_go.play --size 9 --color B --visits 200   # 터미널 대국
```

API: `GET /api/state`, `POST /api/new {size, komi, color, visits}`, `POST /api/move {move}`(사용자 수 + AI 응수까지 한 번에), `POST /api/resign`

### 3단계 구현 메모

- 손실 = 최선수 값 − 사용자 수 값 (집·승률 둘 다, 사용자 기준). 음수(탐색 노이즈)는 0으로 자름
- 사용자 수가 후보수에 없으면 착수 후 국면을 분석해 뒤집은 값을 쓴다 (`in_candidates=false`로 기록)
- 실력 측정 탐색량은 AI 난이도와 별개로 `EVAL_VISITS=200` 고정 → 기록끼리 비교 가능
- 국면 단계는 수 번호 / 판 넓이 비율 (≤15% 포석, ≤50% 중반) — 1차 기준
- 레이팅: 한 판 = 한 레이팅 기간. 상대 AI 레이팅은 **가정치 2800±50** (`server.AI_RATING`) → 4단계에서 보정
- 도중에 새 판을 시작하면 `games.jsonl`엔 안 남고 `moves.jsonl`엔 남음

데이터 (`data/`, `--data-dir`로 변경 가능):
- `moves.jsonl` — 사용자 착수 1수 = 1줄: `moves`(직전까지 수순, 국면 복원용), `move`, `best_move`, `score_loss`, `winrate_loss`, `score_before`, `phase`, `candidates`(상위 5개 `[수, 집, 승률, visits]`), `think_ms`, `player_rating`, `ai` 설정, `schema` 버전
- `games.jsonl` — 끝난 판 1줄: 결과, `user_score`, 손실 요약, 레이팅 전후
- `player.json` — 현재 Glicko-2 레이팅

### 4단계 v2 구현 메모

- **목표 손실** = 사용자 최근 200수 평균 손실(사전값 3집 × 가상 20수로 축소 추정) × exp(log_factor). 판 시작 때 정하고 한 판 동안 고정
- **수 선택** `choose_move`: 가중치 exp(−|손실 − 목표| / T), T=1집. pass와 탐색 2회 미만 후보는 제외. 엔진 최선이 pass면 그대로 pass (집 메우며 판 끌기 방지)
- 맞춤 모드 질의만 `overrideSettings: {"wideRootNoise": 1.0}` — 9x9 실측으로 3회 이상 탐색 후보 9→19개, 최대 손실 5→12집
- **승률 보정**: 판마다 log_factor −= 0.3 × (사용자 점수 − 0.45), ±3 클램프 → 기대값상 사용자 승률 45%에서 평형
- **AI 레벨 레이팅**: 그 판 AI 실제 평균 손실 구간(0.25/0.5/1/2/4/8집)별 Glicko-2. 사전값 2600~800, RD 300. 사용자·레벨이 서로 갱신. 최강 모드는 2800 고정(척도 기준점)
- 저장: `data/difficulty.json`, `data/ai_levels.json`. `games.jsonl`의 `ai`에 `target_loss`, `avg_loss`, `level`
- 실측: 목표 3.0집 → 6수 평균 3.05집으로 잘 따라감
- **한계 (v3가 필요한 이유)**: 목표 손실이 크면(6집 이상) 1선·구석 같은 사람이 안 두는 수를 둔다. 손실 "크기"만 맞추고 실수 "모양"은 사람답지 않음

### 4단계 v3 구현 메모

- 엔진: `-human-model`로 같이 로드. 질의에 `overrideSettings.humanSLProfile`(예: `rank_5k`) + `includePolicy`를 주면 응답에 `humanPolicy`가 붙음 → `Analysis.human_policy` {좌표: 확률}. 프로필만 지정하면 탐색은 그대로이고 사람 정책 출력만 추가됨
- v3 `HumanPlayer`: 엔진 최선이 pass면 pass, 아니면 사람 정책에서 확률 1% 이상 수만 남겨 확률 비례 추출
- 혼합 `HybridPlayer`: 엔진 후보 중 사람 확률 × exp(−|손실 − 목표| / T)
- 웹 맞춤 모드: 목표 손실(v2와 같은 방식) → 보정표에서 수당 손실이 가장 가까운 급수 선택. 목표가 가장 약한 급수 손실보다 크면 그 급수로 **혼합** (`server.needs_hybrid`). 보정표는 `data/arena/human_calibration.json`(arena `--calibrate` 결과), 없으면 `server.DEFAULT_HUMAN_CALIBRATION` 가정치
- 보정 계수(`difficulty.json`)는 모드별로 따로 (`{"adaptive": …, "human": …}`). 이전 형식은 자동 변환
- 웹 AI 선택: 사람형(v3, 기본) / 손실형(v2) / 최강. human 모델 파일이 없으면 사람형 비활성

### AI 리그전 (`arena.py`)

- 측정을 공정하게: 모든 플레이어의 매 수에 같은 평가 질의(200 visits + 기준 프로필 `rank_10k` 사람 정책)를 따로 던짐
- 지표: 승률, Bradley-Terry Elo(평균 1500), 수당 손실, 큰 실수율, 사람 확률(기준 10급이 그 수에 준 확률), 사람 top-3 비율, 포석·중반 1선 착수율
- 9x9 한 판 약 90초. `--workers`로 KataGo 프로세스 여러 개 병렬
- 결과: `data/arena/<시각>/games.jsonl`, `summary.md`, `summary.json`

### 리그전 결과 (2026-10-21, 9x9, M3 Max Metal)

**측정 기준 수정 이력**: 첫 실행에서 손실 평균이 급수 순서를 뒤집음. 원인은 승부가 기운 뒤(대마 사망 후) 매 수 수십 집씩 잡히는 손실 — 중앙값은 전 급수 ~0. 그래서
- 실력 지표 손실 = 두기 전 승률 5~95% 국면의 수만, 한 수 15집 상한 (`skill.effective_loss`). 사용자 목표 손실도 같은 기준
- 리그전에 돌 거두기 (판 넓이 30% 이후 승률 2% 미만) → 79판 중 77판 기권 종료, 수 상한 0판, 판당 약 59초

리그전 (각 12판, 쌍마다 흑백 교대 2판, `data/arena/20261021-111432`):

| 플레이어 | 승률 | Elo | 접전 손실 | 큰 실수 | 사람 확률 | 사람 top3 |
|---|---|---|---|---|---|---|
| 최강 | 100% | 2101 | 0.00 | 0% | 0.315 | 73% |
| v3(1d) | 75% | 1742 | 0.41 | 10% | 0.279 | 70% |
| v3(5k) | 58% | 1587 | 0.78 | 20% | 0.352 | 78% |
| 혼합(10k,2집) | 50% | 1514 | 1.16 | 17% | 0.359 | 75% |
| v2(1집) | 50% | 1514 | 1.21 | 11% | 0.134 | 30% |
| v3(15k) | 17% | 1163 | 1.75 | 36% | 0.287 | 72% |
| v2(3집) | 0% | 879 | 2.98 | 62% | 0.029 | 9% |

해석:
- **사람다움**: v3·혼합은 사람 top3 70~78%, v2는 9~30%. v2의 "사람이 안 두는 수" 문제가 수치로 확인됨
- v3는 급수 순서대로 강하고 손실도 순서대로 (리그전 기준). 같은 승률이면 v3가 v2보다 손실이 작음 (v3(5k) 0.78집으로 58%, v2(1집) 1.21집으로 50%) → v3의 실수는 승부에 덜 치명적인 곳에서 나옴
- **v3 손실 범위가 좁음** (접전 0.4~1.9집) → 초보에겐 20급도 강함 → 혼합 폴백 추가
- 12판씩이라 Elo 오차가 ±100~150 정도로 큼
- **자기 대국 보정은 불안정** (`20261021-110323`, 급수당 4판): 9d 1.03 > 1d 0.71 등 순서 섞임. 같은 급수끼리 두면 상대 수준 따라 국면 난도가 달라서로 추정. 9x9 사람 기보가 적어 고단 프로필 정확도가 낮을 가능성도 있음. 단조 보정 후 `data/arena/human_calibration.json` = 20k 1.86 ~ 9d 0.68
- 개선안: 보정을 자기 대국 대신 **고정 상대(예: v3(5k))와의 대국**으로, 판 수를 늘려서


### 대국 중 실력 추정 + 힌트 (5단계 전 기능 추가)

- **실시간 급수 추정** (`rank_estimate.py`): 급수 격자 12개(20k…9d)에 대해 사용자 수마다 `log P(급수) += 0.5 · log p_human(그 수 | 급수)`. human 정책 12개는 `engine.human_policies`로 한 번에 질의(maxVisits=1, 약 0.09초). 정책 0인 수는 1e-4 바닥값. pass·힌트 본 수는 갱신 안 함
- 판 사이: 사후분포를 지수 0.5로 평평하게 해서 다음 판 사전분포로 (`data/rank_estimate.json`)
- **AI 급수 실시간 조정**: 사람형 모드에서 매 수 `AI 급수 = 추정 평균(점) + rank_offset`. rank_offset은 판 결과로 학습 (`difficulty.json`의 `human.rank_offset`, 지면 −0.9점·이기면 +1.1점). 가장 약한 급수로도 부족하면 혼합
- **힌트** (`POST /api/hint`, 단축키 H): 추정 급수 +3점 프로필로 분석 → 최선 대비 1집 이하 & 탐색 2회 이상 후보를 사람 확률 순으로 최대 3개. 판에 A/B/C, 패널에 손실·승률·사람 선택률·예상 진행(마우스 올리면 판에 번호 돌)
- 힌트 본 수는 `moves.jsonl`에 `hint_used: true` → 실력 추정·목표 손실 통계에서 제외
- **검증** (`estimate_eval`, 9x9, v3 봇을 사용자로, 16판 32기록, 돌 거두기 없이): w=0.5에서

  | 둔 수 | 3 | 5 | 10 | 20 | 30 |
  |---|---|---|---|---|---|
  | 가장 그럴듯한 급수가 ±1칸 안 | 31% | 56% | 75% | 89% | 87% |
  | 평균 추정 오차(점) | 4.0 | 3.7 | 2.7 | 1.9 | 1.8 |

  30수 뒤 급수별 평균: 5k→5k, 1k→1k, 3d→3d, 15k→10k(하수를 강하게 보는 치우침). 가중치 0.3/0.5/1.0 차이는 작음.
  **봇은 추정에 쓰는 모델에서 수를 뽑으므로 실제 사람보다 쉬운 조건(낙관적 상한)**. 첫 실행은 같은 급수끼리 25수 무렵 기권해 한쪽당 13수뿐이라 돌 거두기를 끄고 재실행

## 다음 할 일

1. 보정을 고정 상대 대국으로 다시 (급수당 10판 이상) → 급수별 손실표 안정화
2. 실제 사용자 대국 데이터 쌓기 → 보정 계수·레벨 레이팅이 실제로 수렴하는지 확인
3. 5단계(웹 게임화): 대국 목록·복기 화면, 수별 손실 그래프, 세션/다중 사용자

머신러닝 방향 (데이터가 쌓인 뒤):
- 급수 조건부 수 예측 모델(Maia 방식, 9x9 ResNet) → 같은 모델로 급수 추정
- 내 기보로 파인튜닝 → "나를 닮은 상대"
- 실수 유형 분류 + 지식 추적으로 약점 분석, 밴딧으로 난이도 조절

## 선행 지식 체크리스트

- 바둑 규칙, 계가, 급·단 체계
- 탐색: 미니맥스 → MCTS(UCT) → PUCT(AlphaZero)
- 신경망/CNN, ResNet (바둑판 = 19×19 이미지, AlphaGo Zero = ResNet + 정책 헤드 + 가치 헤드)
- 강화학습 기초: 정책/가치, 셀프플레이 (6단계에서 깊게)
- 레이팅: Elo, Glicko-2
- 게임 디자인: 동적 난이도 조절(DDA)

## 작업 방식 메모

- 만식은 프론트엔드 출신으로 ML을 공부 중 (Python → PyTorch → 비전 로드맵 진행 중)
- 코드에 한국어 주석으로 "왜"를 남기고, 각 단계는 테스트와 함께 진행
- 포트폴리오 관점: "바둑 AI"보다 "사용자 실력을 수 단위로 추정하고 사람 같은 실수를 하는 적응형 상대를 설계"가 강점
