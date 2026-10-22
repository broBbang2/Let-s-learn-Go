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
6. 🔶 **직접 학습한 모델** — 급수 조건부 수 예측(Maia-2 방식, 지식 증류) 진행 중. 9x9 미니 AlphaZero는 이후 — MCTS + 정책/가치망 이해용. ML 로드맵 Phase 4~5(CNN, ResNet) 이후에 붙이면 좋음

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
- `adaptive_go/explain.py` — AI 수·내 수·실력 추정의 근거 정리
- `adaptive_go/sgf.py` — SGF 읽기(본 수순)·쓰기
- `adaptive_go/library.py` — 기보 보관함: 명국·내 대국 목록, 판 전체 분석(백그라운드, 캐시)
- `web/board.js` — 바둑판 SVG 공용 렌더러, `web/library.html` — 기보 보관함 화면 (`/library`)
- `adaptive_go/features.py` — 국면 → 신경망 입력 12평면, 8대칭 데이터 증강
- `adaptive_go/datagen.py` — KataGo human 모델로 급수별 19줄 대국 생성 → 학습 데이터(npz)
- `adaptive_go/model.py` — 급수 조건부 수 예측 ResNet + `NetPolicy`(human_policies와 같은 인터페이스)
- `adaptive_go/train.py` — PyTorch 학습(mps), `adaptive_go/eval_model.py` — teacher와 실력 추정 비교
- `library/` — 명국 SGF (+ `<이름>.notes.json` 명장면 메모, 출처는 `library/SOURCES.md`)
- `models/b18c384nbt-humanv0.bin.gz` — KataGo human 모델 (v1.15.0 릴리스, sha256 637746e4…)
- `tests/test_board.py`, `test_engine.py`, `test_server.py`, `test_skill.py`, `test_rating.py`, `test_difficulty.py`, `test_players.py`, `test_rank_estimate.py`, `test_estimate_eval.py`, `test_sgf.py`, `test_library.py`, `test_features.py`, `test_datagen.py`, `test_model.py` — pytest 133개, 전부 통과
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


### 대국 중 실력 추정 + 형세 표시 (5단계 전 기능 추가)

- **힌트는 제거함** (사용자 결정, 2026-10-21): `/api/hint`·자동 힌트·힌트 카드 모두 삭제. 예전 기록의 `hint_used`/`hint_followed` 수는 손실 통계에서 계속 제외
- **형세(영역) 표시가 대신함**: 매 수(AI 응수 후) `LIVE_OWNERSHIP_VISITS=64`로 ownership 질의 → `state.ownership`. React 판(`app/src/components/GoBoard.tsx`)이 빈 칸을 칸 크기 그대로 흑/백으로 칠함(확실할수록 진하게, |값|0.15부터, 최대 불투명도 0.5), 상대 땅 위에 선 돌(|값|>0.6)은 작은 네모로 "잡힐 위기" 표시. 상단 "형세 보기"로 끄고 켬(기본 켜짐, localStorage)
- **형세 카드** (`TerritoryCard`): 내 땅 | 미정 | 상대 땅 막대와 "약 N칸". 칸마다 내 땅일 정도(0~1)를 더한 **예상 영역** — 확실한 칸(>0.5)만 세면 초반엔 361칸 중 342칸이 미정이라 의미가 없었음. 잡힐 위기 돌 개수도 표시

- **실시간 급수 추정** (`rank_estimate.py`): 급수 격자 12개(20k…9d)에 대해 사용자 수마다 `log P(급수) += 0.5 · log p_human(그 수 | 급수)`. human 정책 12개는 `engine.human_policies`로 한 번에 질의(maxVisits=1, 약 0.09초). 정책 0인 수는 1e-4 바닥값. pass는 갱신 안 함
- 판 사이: 사후분포를 지수 0.5로 평평하게 해서 다음 판 사전분포로 (`data/rank_estimate.json`)
- **AI 급수 실시간 조정**: 사람형 모드에서 매 수 `AI 급수 = 추정 평균(점) + rank_offset`. rank_offset은 판 결과로 학습 (`difficulty.json`의 `human.rank_offset`, 지면 −0.9점·이기면 +1.1점). 가장 약한 급수로도 부족하면 혼합
- **검증** (`estimate_eval`, 9x9, v3 봇을 사용자로, 16판 32기록, 돌 거두기 없이): w=0.5에서

  | 둔 수 | 3 | 5 | 10 | 20 | 30 |
  |---|---|---|---|---|---|
  | 가장 그럴듯한 급수가 ±1칸 안 | 31% | 56% | 75% | 89% | 87% |
  | 평균 추정 오차(점) | 4.0 | 3.7 | 2.7 | 1.9 | 1.8 |

  30수 뒤 급수별 평균: 5k→5k, 1k→1k, 3d→3d, 15k→10k(하수를 강하게 보는 치우침). 가중치 0.3/0.5/1.0 차이는 작음.
  **봇은 추정에 쓰는 모델에서 수를 뽑으므로 실제 사람보다 쉬운 조건(낙관적 상한)**. 첫 실행은 같은 급수끼리 25수 무렵 기권해 한쪽당 13수뿐이라 돌 거두기를 끄고 재실행

### 19줄 기본 + 수의 근거 + 기보 보관함

- 웹 대국 기본: **19x19, 덤 7.5** (human 모델이 19줄 사람 기보로 학습되어 추정·사람형 AI가 이쪽에서 더 자연스러움)
- **수 해설 카드** (`explain.py`): AI 수마다 방식별 근거 문장(예: "3급 사람 모델이 이 수를 둘 확률 23%(2위)… 최선수 C15 대비 0.1집") + 후보 표(손실·AI 승률·사람 선택률·탐색 수, 마우스 올리면 판에 위치). 내 수는 최선수와 예상 진행, 실력 추정은 급수별 "이 수를 둘 확률" 막대
- **기보 보관함** (`/library`): 명국과 끝난 내 대국 목록 → 재생(◀▶, 슬라이더, ←→ 키) → "분석 시작"으로 판 전체 KataGo 분석
  - 엔진 `analyze_game`: `analyzeTurns`로 질의 하나에 전 국면, 진행률 콜백. 대국 엔진과 별도 KataGo 프로세스. 100 visits, 19줄 180수 약 2분
  - 국면별: 흑 승률·집, KataGo 최선수(판에 "AI")와 예상 진행. 실제 수: 손실, 엔진 후보 순위, **그 시대 프로 확률**(human 모델 `proyear_<기보 연도>`, 1800~2023 지원 — 같은 국면에서 1846·1933년 프로는 소목, 2023년은 화점 선호를 실측 확인)
  - 결정적 수(승률 8% 이상 손실 상위 5), 예상 밖의 명수(프로 확률 5% 미만 & 손실 0.5집 이하, 10수 이후, 상위 3), 명장면 메모(`notes.json`, 사람이 쓴 설명). 둘 다 **접전 국면(둘 쪽 승률 5~95%)만** — 끝난 판의 수가 뽑히던 문제 수정. 캐시된 국면 데이터에서 읽을 때마다 다시 계산하므로 기준을 바꿔도 재분석 불필요 (2국에선 이 기준으로 37수 P10이 자동으로 뽑힘)
  - 흑 승률 그래프: 단일 계열 2px 선, 50% 기준선, 십자선 툴팁, 클릭 이동. 색은 dataviz 검증 통과(#1a7f5a / #c2410c), 표식은 모양(원·마름모·별)으로도 구분
  - 결과 캐시 `data/library_analysis/<id>.json`, 내 대국은 SGF로 내보내기 (`/api/library/<id>/sgf`)
- **이세돌 vs 알파고 5국** (`library/2016-alphago-lee-sedol-1..5.sgf`, 사용자 승인 후 CWI 기보 아카이브에서 받음): 수 수 일치, 4국 78수 = 백 L11, 2국 37수 = 흑 P10 확인
  - 4국 KataGo 분석: 78수 L11은 2016년 프로 확률 0.68%(22위), KataGo 손실 0집·후보 7위(최선 J11), 직전 흑 승률 97%. 승부를 가른 건 알파고 쪽 89수 R9(−26%)·93수 S9(−25%)

### React 대국 화면 (`app/`, 2026-10-21)

- `web/index.html`(한 장짜리)을 **Vite + React + TypeScript + Motion**으로 옮김. 기능은 그대로(패널·해설; 힌트·단축키 H는 이후 제거되고 형세 표시로 대체), 예전 화면은 `/classic`
- 서버: `app/dist/index.html`이 있으면 `/`에서 React 화면, `/assets/*` 정적 서빙(폴더 밖 경로 차단). 기보 보관함(`/library`)은 아직 예전 HTML
- **실시간 형세**: AI 응수 뒤 `LIVE_OWNERSHIP_VISITS=32`로 ownership을 한 번 더 질의 → `state.ownership`. 빈 자리에 흑/백 반투명 칠, 바뀔 땐 방금 둔 자리에서 거리 × 28ms씩 늦게 전환(번지는 효과). 상단 "형세 보기"로 끔
- **애니메이션** (`app/src/components/GoBoard.tsx`): 착수 스프링 드롭 + 파문, 따낸 돌은 흔들림 → 떠오르며 사라짐(따낸 자리에서 가까운 순 35ms 시차), 마지막 수 표시 지연 등장, 호버 미리보기, 둘 수 없는 자리는 빨간 점선 + 클릭 시 흔들림과 이유 문구. `prefers-reduced-motion`이면 줄임
- **TS 규칙 엔진** (`app/src/go/rules.ts`): board.py와 같은 규칙. 서버 history를 재생해 돌마다 id(놓인 수 번호)를 붙이고, 직전 렌더와 비교해 새 돌·따낸 돌을 안다. 내 수는 서버 응답 전에 바로 두고 따낸 돌도 바로 들어냄. 재생 결과가 서버 판과 다르면 서버 판을 씀
- **feedback 모듈** (`app/src/feedback.ts`): `place / capture(n) / illegal` 한곳에 모음. 지금은 Web Audio 합성음 + `navigator.vibrate`(안드로이드만). 앱으로 옮길 때 여기에 햅틱 추가
- 실행: `cd app && npm install && npm run dev` → http://localhost:5173 (API는 8000으로 프록시) / 배포용 `npm run build` → Python 서버가 바로 서빙. 테스트 `npm test`(규칙 5개), 서버 pytest 108개 통과

### 직접 학습한 모델: 급수 조건부 수 예측 (Maia-2 방식, 지식 증류)

- **목적**: 국면 + 급수 → "그 급수 사람이 둘 수"의 분포. 베이즈 실력 추정에 KataGo human 모델 대신 꽂을 수 있게 (`NetPolicy.human_policies`, 서버 `--rank-model models/rank_policy.pt`)
- **데이터** (`datagen.py`, `data/train/20261021-163648`): KataGo human 모델로 19줄 대국을 급수 12개 × 80판 = 960판, 24만 8천 국면 (46분, 압축 17MB). 수는 사람 정책에서 추출(0.5% 미만 제외, maxVisits=1), 정답은 그 국면 사람 정책 상위 16개(증류). 판당 260수 상한
- **입력** (`features.py`): 둘 차례 기준 12평면 — 내/상대/빈, 내·상대 그룹 활로 1/2/3+, 직전 두 수, 패 금지점 + 상수 1 평면. 8대칭 증강
- **모델** (`model.py`): ResNet 96채널 × 8블록(135만 파라미터, 5.4MB). 급수 임베딩을 블록마다 채널 편향으로 주입. 둘 수 없는 점 −inf 마스크
- **학습** (`train.py`, mps): AdamW + OneCycle 2e-3, 배치 256, 12에폭 약 37분. 검증 = 게임 번호 MD5 해시 10%
  - 검증 지표(12에폭): teacher 1순위 일치 **62.8%**, 실제 둔 수 1순위 **42.1%**, 손실 2.054 (아직 감소 중 → 더 키우거나 더 돌릴 여지)
  - 64×6(46만)은 58.9% / 40.6%
- **검증 분할 버그 2번 수정**: "게임 번호 끝자리 0" → 번호 규칙(라운드×12+급수) 때문에 짝수 급수만 검증에 들어감. Knuth 곱셈 해시도 12 간격 입력에서 급수별로 0~20판으로 치우침 → MD5로 해결(급수별 6~11판)
- **실력 추정 비교** (`eval_model.py`, 급수당 4판 48판 96기록, w=0.5, 가장 그럴듯한 급수가 ±1칸 안 / 평균 오차 점):

  | 둔 수 | 3 | 5 | 10 | 20 | 30 |
  |---|---|---|---|---|---|
  | 우리 모델 (5.4MB) | 41% / 5.8 | 56% / 4.7 | 66% / 3.3 | 73% / 2.3 | 75% / 2.1 |
  | KataGo human (99MB) | 48% / 5.5 | 72% / 4.3 | 78% / 2.6 | 89% / 1.7 | 94% / 1.5 |

  크기 1/18로 초반(3~10수)은 teacher의 85% 수준, 30수에선 격차가 벌어짐. **판이 teacher로 생성돼 teacher에 유리한 조건**. 기본 서버는 여전히 KataGo human 모델로 추정

- 진행 확인 도구 `adaptive_go/progress.py` (`python -m adaptive_go.progress -f`): 작업 로그를 `logs/`에 같이 남김 (사용자가 추가)

### 실제 사람 검증 · 사람 기보 학습 · 형세 학습 도구 (2026-10-21~22)

**사람 기보** (`data/ogs/sample-100k.json.gz`, 사용자 승인 후 받음): za3k.com/ogs의 OGS 무작위 10만 판 샘플(79MB, sha256 c9b274fa…). OGS 허락 하에 재배포, OGS 측이 연구 용도 허용, 모음은 퍼블릭 도메인. `ogs.py`가 줄 단위로 읽음
- 쓰는 판: 19줄·호선·50수+·랭크·봇 제외·급수 차 ≤3 → 18,554판. 급수는 18k~1d에 몰림(3d+ 약 300판)
- OGS rank: 29 = 1k, 30 = 1d → 점 척도 `rank − 29`. OGS 체계 ≠ KataGo human 체계 → 오차와 함께 치우침·상관을 봄

**1. 실제 사람으로 실력 추정 검증** (`human_eval.py`, 검증 판 = `train.is_val`, 급수대 5개 × 30판 = 300명, 앞 60수):
- 생성 데이터만으로 학습한 모델(`models/rank_policy_distill_v1.pt`), 30수(한 사람 기준) 뒤: 상관 0.73, 평균 오차 4.5점, **치우침 +3.4점**(사람을 실제보다 강하게 봄), ±3점 안 46%
- 급수대별 평균 추정: 20k대 −12.5(실제 −18.3), 15~11k −7.2(−12.8), 10~6k −3.6(−7.9), 5~1k −0.6(−2.7), 1d+ 1.1(1.6) → 순서는 맞지만 하수 쪽을 크게 끌어올림
- KataGo human 모델(teacher), 30수 뒤: 상관 0.78, 평균 오차 3.7점, **치우침 +2.4점**, ±3점 안 52% (`data/arena/human-eval-20261021-233325/result.json`). 급수대별: 20k대 −12.6, 15~11k −8.6, 10~6k −4.9, 5~1k −1.9, 1d+ 0.4 → teacher도 OGS 하수를 강하게 봄. 우리 생성-데이터 모델은 teacher보다 약간 뒤짐

**2. 사람 기보로 학습** (`ogs_data.py`): 검증 판을 뺀 16,632판의 앞 120수 → 사람이 실제 둔 수를 정답으로 190만 국면(64MB). 급수는 둔 사람의 OGS 급수에서 가장 가까운 격자. 생성 데이터(teacher 분포)와 섞어 학습 → `models/rank_policy_mixed.pt` (96×8, 2에폭, 사람 190만 + 생성 25만 국면. 맥북이 밤새 잠들어 1에폭이 8시간 걸림)
- 실제 사람 검증, 30수 뒤 (`data/arena/human-eval-20261022-103822/result.json`):

  | | 상관 | 평균 오차 | 치우침 | ±3점 안 |
  |---|---|---|---|---|
  | KataGo human (teacher) | **0.78** | **3.7** | +2.4 | **52%** |
  | 생성 데이터만 (distill_v1) | 0.73 | 4.5 | +3.4 | 46% |
  | 사람 + 생성 (mixed) | 0.73 | 4.0 | **+2.2** | 48% |

  사람 데이터로 오차·치우침은 줄었지만 teacher를 넘진 못함. 초반(5~10수)은 오히려 mixed가 가장 나쁨(치우침 +4.0~4.8)
- 세 모델 모두 하수를 가운데로 끌어올림(실제 20k대 → 추정 −12~−14) → 모델보다 추정 방식(12칸 격자를 균등에서 시작해 평균, 격자 최저 20k) 쪽 문제로 봄
- **선형 보정은 효과 없음**: 판 단위로 반씩 나눠 "추정 → OGS 급수" 직선을 맞추고 나머지 반에 적용 → 오차가 그대로이거나 늘어남(teacher 30수 3.38→3.49, mixed 3.60→4.16). 남은 오차는 일정한 치우침보다 개인별 편차(30수로 한 사람을 정하기엔 부족 + OGS 급수 자체의 불확실성)가 큼
- **추정 방식 개선 (적용함)**: 저장된 검증 기록으로 출발 분포(균등 vs OGS 사용자 분포) × w(0.5/1.0) × 답 고르기(평균/최빈) 8가지 비교
  - OGS 분포로 시작하면 치우침이 크게 줄어듦: mixed +2.2 → **+0.2~0.4**, 10수 오차 5.6 → 4.7. teacher도 10수 4.6 → 4.2
  - 최선: **OGS 분포 + w=1.0 + 사후 평균** → teacher 10수 4.24 / 30수 3.63(±3점 53%), mixed 4.66 / 3.81. 최빈 급수는 나아지지 않음
  - `rank_estimate.POPULATION_PRIOR`(검증 판 제외 33,264명), `LIKELIHOOD_WEIGHT = 1.0`. 처음 보는 사용자는 7k에서 출발(전엔 3k). 저장된 사용자 추정(`rank_estimate.json`)이 있으면 그걸 씀. `human_eval.metrics`도 같은 규칙
  - 주의: OGS 사용자 분포라 우리 사용자와 다를 수 있음 → 사용자가 쌓이면 교체. 같은 검증 판으로 설정을 골라 실제 효과는 약간 작을 수 있음
  - 격자를 25k까지 넓히는 건 불가: KataGo human 모델 프로필이 rank_20k까지
- 생성 데이터 추가분 2,400판(`data/train/20261021-232836`, 62만 국면) 생성 완료 → mixed에서 출발해 전체 데이터(280만 국면)로 2에폭 더 학습 중 → `models/rank_policy_mixed2.pt` (`--init`, lr 7e-4, `caffeinate`로 잠자기 방지)
- 생성 데이터 추가분 2,400판(seed 10000~) 생성 중 → `data/train/20261021-232836`

**3. 형세 기반 학습 도구** (`territory.py`, 서버·React):
- **수의 효과**: 내 수 = "둔 뒤" − "그 자리에서 패스했다면", AI 수 = "AI가 둔 뒤" − "AI가 패스했다면" (둘 다 같은 쪽 차례라 차례 이점이 상쇄). 처음엔 "두기 전"과 비교했다가 모든 수가 상대에게 유리하게 나와(상대 다음 수의 가치가 섞임) 바꿈. 형세 카드에 "내 수 D4: 내 땅 +12.8 · 상대 땅 −3.8", 주인이 바뀐 칸(|Δ|>0.35)은 판에 파란 테두리로 잠깐 강조
- **실수 유형**: 접전 국면의 3집+ 실수만. "최선수를 뒀다면"과 비교해 내 돌 2개 이상이 상대 땅으로 넘어갔으면 사활·전투, 아니면 포석·방향 / 중반 판단 / 끝내기. `moves.jsonl`에 `territory_change`·`mistake_type`, 최근 500수 집계를 "자주 하는 실수" 카드로
- **대국 후 복기**: 판이 끝나면 `on_finished` → 기보 보관함 분석 자동 시작, 상태에 `review_url`("복기 보기 → 내가 놓친 장면"). 내 대국 분석에 `my_mistakes`(내 수 중 승률 2% 넘게 잃은 상위 3)
- 한 차례 추가 비용: 형세 질의 2~3번(64 visits) ≈ 0.2~0.3초

### 끊긴 작업 이어 돌리기 (2026-10-22)

컴퓨터를 끄거나 작업이 끊겨도 처음부터 다시 할 필요 없게 만들어 둠.

```bash
# 진행 상황 보기 (실행 중인 작업, 남은 시간)
.venv/bin/python -m adaptive_go.progress

# 1) 학습 데이터 생성 이어 하기: 그 폴더의 조각에 이미 있는 판은 건너뜀 (설정은 폴더의 args.json을 따름)
.venv/bin/python -m adaptive_go.datagen --resume data/train/20261021-232836

# 2) 사람 + 생성 데이터 섞어 학습 이어 하기: models/rank_policy_mixed.ckpt에서 끊긴 배치부터
.venv/bin/python -m adaptive_go.train --resume --out models/rank_policy_mixed.pt

# 3) 학습이 끝나면 실제 사람으로 검증 (teacher 결과는 이미 있으니 생략 가능)
.venv/bin/python -m adaptive_go.human_eval --per-bucket 30 --moves 60 --model models/rank_policy_mixed.pt --no-teacher
```

- 데이터 생성: 60판마다 조각 저장 → 끊기면 최대 59판만 다시 함. 판 번호는 `seed + 라운드×12 + 급수`라 같은 설정이면 같은 판
- 학습: 500배치마다 `<out>.ckpt`(모델·옵티마이저·학습률 일정·섞은 순서·난수 상태) 저장, 임시 파일에 쓴 뒤 바꿔치기라 저장 중 끊겨도 안전. 마지막 배치 직후 끊겨도 그 에폭의 검증·기록부터 이어감
- 사람 기보(`data/ogs/`)와 생성 데이터(`data/train/`)는 git에 안 올림. 사람 기보는 `curl -fLo data/ogs/sample-100k.json.gz https://za3k.com/ogs/sample-100k.json.gz` (sha256 c9b274fa…)

## 다음 할 일

1. 자체 모델 개선: 데이터 늘리기(급수당 200판+), 더 큰 모델·더 긴 학습, 급수 구분을 직접 겨냥한 보조 손실(같은 국면에서 급수별 분포 차이)
2. 실제 사람 기보(급수 붙은 공개 기보)로 검증 → teacher 홈 경기 문제 해소
3. 보정을 고정 상대 대국으로 다시 (급수당 10판 이상) → 급수별 손실표 안정화
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
