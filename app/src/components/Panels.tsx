// 대국 화면 오른쪽 패널들 (예전 web/index.html의 카드를 그대로 옮김)
import { motion } from "motion/react";
import { useState, type FormEvent } from "react";
import type { Color } from "../go/rules";
import type { GameState, Marker, Mode } from "../types";

const PHASE_NAMES: Record<string, string> = { opening: "포석", middle: "중반", endgame: "끝내기" };
const MODE_NAMES: Record<Mode, string> = { human: "사람형 맞춤 (v3)", adaptive: "손실형 맞춤 (v2)", best: "최강" };

const colorName = (c: Color) => (c === "B" ? "흑" : "백");
// 1% 미만은 유효숫자 2자리로 (0%로 뭉개지지 않게)
const pct = (p: number | null | undefined) =>
  p == null ? "-" : p > 0 && p < 0.01 ? `${(p * 100).toPrecision(2)}%` : `${(p * 100).toFixed(0)}%`;

type HoverMarkers = (m: Marker[] | null) => void;

export function StatusCard({ state, busy, message }: { state: GameState; busy: boolean; message: string | null }) {
  const ai = state.last_ai;
  let status: React.ReactNode;
  if (state.over) status = state.result ? `종국 · ${state.result.text}` : "종국";
  else if (busy) status = <span className="thinking">AI가 생각 중</span>;
  else if (state.to_play === state.user_color) status = `내 차례 (${colorName(state.user_color)})`;
  else status = "AI 차례";

  return (
    <section className="card">
      <div className="status">{status}</div>
      <div className="message">{message}</div>
      {ai && (
        <div className="eval">
          <div className={`eval-bar me-${state.user_color}`}>
            <div style={{ width: `${(ai.user_winrate * 100).toFixed(1)}%` }} />
          </div>
          <div className="eval-text">
            <span>내 승률 {(ai.user_winrate * 100).toFixed(0)}%</span>
            <span>
              {ai.user_score_lead >= 0
                ? `내가 ${ai.user_score_lead.toFixed(1)}집 앞섬`
                : `내가 ${(-ai.user_score_lead).toFixed(1)}집 뒤짐`}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}

function RankBars({ values, ranks, highlightTop }: { values: number[]; ranks: string[]; highlightTop?: boolean }) {
  const max = Math.max(...values, 1e-9);
  return (
    <>
      <div className={highlightTop ? "est-bars" : "ev-bars"}>
        {values.map((p, i) => (
          <div
            key={i}
            className={highlightTop && p === max ? "top" : undefined}
            style={{ height: `${Math.max(4, (p / max) * 100)}%` }}
            title={`${ranks[i]} ${pct(p)}`}
          />
        ))}
      </div>
      <div className="est-axis">
        <span>20급</span>
        <span>1급</span>
        <span>9단</span>
      </div>
    </>
  );
}

export function SkillCard({ state }: { state: GameState }) {
  const est = state.estimate;
  const ch = state.rating_change;
  const d = ch ? ch.after - ch.before : 0;
  const ev = state.last_eval;
  const sum = state.summary;
  return (
    <section className="card">
      <h2>내 실력</h2>
      <div className="rating">
        <b>{Math.round(state.rating.rating)}</b>
        <span className="rd">± {Math.round(state.rating.rd)}</span>
        {ch && <span className={`delta ${d >= 0 ? "up" : "down"}`}>{`${d >= 0 ? "+" : ""}${d.toFixed(1)}`}</span>}
      </div>
      {est && (
        <div>
          <div className="kv">
            <span>추정 급수</span>
            <b>
              {est.moves
                ? `${est.label} (확신 ${Math.round(est.confidence * 100)}%, ${est.moves}수 기준)`
                : `${est.mean_label} (이전 기록 기준)`}
              {est.source === "ours" ? " · 자체 모델" : ""}
            </b>
          </div>
          <RankBars values={est.distribution} ranks={est.ranks} highlightTop />
        </div>
      )}
      <div className="kv">
        <span>직전 수</span>
        <b className={ev ? (ev.score_loss < 1 ? "loss-good" : ev.score_loss >= 3 ? "loss-bad" : "") : ""}>
          {ev
            ? `${ev.move} · ${ev.score_loss.toFixed(1)}집 손실 (${ev.move === ev.best_move ? "최선수" : `최선 ${ev.best_move}`})`
            : "-"}
        </b>
      </div>
      <div className="kv">
        <span>이번 판 평균 손실</span>
        <b>{sum.avg_score_loss == null ? "-" : `${sum.avg_score_loss.toFixed(2)}집`}</b>
      </div>
      <div className="kv">
        <span>큰 실수 (3집 이상)</span>
        <b>{sum.blunders}</b>
      </div>
      {/* 종국 후에만 국면별 손실 (대국 중엔 정보가 너무 많음) */}
      {state.over &&
        Object.entries(sum.by_phase).map(([k, v]) => (
          <div className="kv" key={k}>
            <span>
              {PHASE_NAMES[k]} ({v.moves}수)
            </span>
            <b>{v.avg_score_loss.toFixed(2)}집</b>
          </div>
        ))}
    </section>
  );
}

export function AiCard({ state }: { state: GameState }) {
  const ai = state.ai;
  const rank = ai.profile ? ai.profile.replace("k", "급").replace("d", "단") : "-";
  return (
    <section className="card">
      <h2>AI 상대</h2>
      <div className="kv">
        <span>모드</span>
        <b>{MODE_NAMES[ai.mode]}</b>
      </div>
      {ai.profile && (
        <div className="kv">
          <span>상대 급수</span>
          <b>{ai.hybrid ? `${rank} + 손실 맞춤` : rank}</b>
        </div>
      )}
      {ai.mode !== "best" && (
        <div className="kv">
          <span>목표 손실</span>
          <b>수당 {ai.target_loss.toFixed(2)}집</b>
        </div>
      )}
      <div className="kv">
        <span>실제 평균 손실</span>
        <b>{ai.level ? `수당 ${ai.avg_loss.toFixed(2)}집` : "-"}</b>
      </div>
    </section>
  );
}

/** 숫자가 바뀌면 살짝 튀어 오른다 (따낸 돌 카운터) */
function Bump({ value }: { value: number }) {
  return (
    <motion.b
      key={value}
      style={{ display: "inline-block" }}
      initial={value ? { scale: 1.6, color: "var(--accent)" } : false}
      animate={{ scale: 1, color: "var(--ink)" }}
      transition={{ type: "spring", stiffness: 500, damping: 18, delay: 0.25 }}
    >
      {value}
    </motion.b>
  );
}

export function StatsCard({ state, captures }: { state: GameState; captures: Record<Color, number> }) {
  return (
    <section className="card">
      <div className="stats">
        <div>
          흑 따낸 돌 <Bump value={captures.B} />
        </div>
        <div>
          백 따낸 돌 <Bump value={captures.W} />
        </div>
        <div>
          덤 <b>{state.komi}</b>
        </div>
        <div>
          수 <b>{state.history.length}</b>
        </div>
      </div>
    </section>
  );
}

export function ActionsCard(p: { locked: boolean; canResign: boolean; onPass: () => void; onResign: () => void }) {
  return (
    <section className="card">
      <div className="buttons">
        <button onClick={p.onPass} disabled={p.locked}>
          패스
        </button>
        <button className="danger" onClick={p.onResign} disabled={!p.canResign}>
          기권
        </button>
      </div>
    </section>
  );
}

/** 형세: KataGo가 지금 국면에서 본 "누구 땅이 될지"를 칸 수로 센다 (판 위 색칠의 범례 겸 요약). */
export function TerritoryCard({ state }: { state: GameState }) {
  const own = state.ownership;
  if (!own || state.over) return null;
  const sign = state.user_color === "B" ? 1 : -1; // ownership은 흑 기준(+1)이라 내 기준으로 뒤집는다
  // 예상 영역: 칸마다 "내 땅일 정도"(0~1)를 더한다. 확실한 칸만 세면 초반엔 거의 다 "미정"이라 의미가 없다
  let mine = 0;
  let theirs = 0;
  let myDanger = 0;
  let theirDanger = 0;
  own.forEach((row, r) =>
    row.forEach((o, c) => {
      const v = o * sign;
      const cell = state.grid[r][c];
      if (v > 0) mine += v;
      else theirs -= v;
      if (cell === state.user_color && v < -0.6) myDanger++;
      else if (cell !== "." && cell !== state.user_color && v > 0.6) theirDanger++;
    }),
  );
  const total = state.size * state.size;
  mine = Math.round(mine);
  theirs = Math.round(theirs);
  const open = Math.max(0, total - mine - theirs);
  const pct = (n: number) => `${(n / total) * 100}%`;
  const meName = colorName(state.user_color);
  const themName = colorName(state.user_color === "B" ? "W" : "B");
  return (
    <section className="card">
      <h2>형세 · 누구 땅인가</h2>
      <div className="land-bar" title={`가운데 빈 부분 = 아직 주인이 정해지지 않은 ${open}칸`} aria-label={`내 땅 ${mine}칸, 미정 ${open}칸, 상대 땅 ${theirs}칸`}>
        <div className={`land me-${state.user_color}`} style={{ width: pct(mine) }} />
        <div className="land open" style={{ width: pct(open) }} />
        <div className={`land them-${state.user_color === "B" ? "W" : "B"}`} style={{ width: pct(theirs) }} />
      </div>
      <div className="land-legend">
        <span><i className={`sw ${state.user_color}`} />내 땅({meName}) 약 <b>{mine}</b>칸</span>
        <span>상대 땅({themName}) 약 <b>{theirs}</b>칸<i className={`sw ${state.user_color === "B" ? "W" : "B"}`} style={{ marginLeft: 4, marginRight: 0 }} /></span>
      </div>
      {(myDanger > 0 || theirDanger > 0) && (
        <div className="kv">
          <span>잡힐 위기</span>
          <b>내 돌 {myDanger}개 · 상대 돌 {theirDanger}개</b>
        </div>
      )}
      <div className="note">KataGo가 지금 국면에서 본 예상 영역입니다. 판의 진한 칸일수록 그 땅이 확실하고, 돌 위의 작은 네모는 그 돌이 잡힐 위기라는 뜻입니다. 칸 수는 돌을 포함하며 덤은 빠져 있습니다.</div>
    </section>
  );
}

export function ExplainCard({ state, setHoverMarkers }: { state: GameState; setHoverMarkers: HoverMarkers }) {
  const ai = state.last_ai?.explain;
  const me = state.user_explain;
  const evd = state.estimate_evidence;
  if (!ai && !me && !evd) return null;
  const hoverAt = (vertex: string, label: string) => ({
    onMouseEnter: () => setHoverMarkers([{ vertex, label, color: "#b5651d" }]),
    onMouseLeave: () => setHoverMarkers(null),
  });

  let why = "";
  if (evd && state.estimate) {
    const ranks = state.estimate.ranks;
    const maxP = Math.max(...evd.probs);
    const top = ranks[evd.probs.indexOf(maxP)];
    why =
      maxP < 0.01
        ? `어느 급수에서도 드문 수(최대 ${pct(maxP)})지만, 상대적으로 ${top} 쪽 확률이 가장 높아 추정이 그쪽으로 움직였습니다.`
        : `${top} 쪽 확률이 가장 높아(${pct(maxP)}) 추정이 그쪽으로 움직였습니다.`;
  }
  const hasHuman = ai?.candidates.some((c) => c.human_p != null);

  return (
    <section className="card explain">
      <h2>수 해설</h2>
      {ai && (
        <div>
          <h3>AI의 수 · {ai.move}</h3>
          <p>{ai.summary}</p>
          {ai.candidates.length > 0 && (
            <table className="cands">
              <thead>
                <tr>
                  <th>후보</th>
                  <th>손실</th>
                  <th>AI 승률</th>
                  {hasHuman && <th>사람</th>}
                  <th>탐색</th>
                </tr>
              </thead>
              <tbody>
                {ai.candidates.map((c) => (
                  <tr
                    key={c.move}
                    className={c.chosen ? "chosen" : undefined}
                    title={c.pv.length ? `예상 진행: ${c.pv.join(" → ")}` : ""}
                    {...hoverAt(c.move, c.move === ai.move ? "✓" : "?")}
                  >
                    <td>
                      {c.move}
                      {c.chosen ? " ✓" : ""}
                    </td>
                    <td>{c.loss.toFixed(1)}집</td>
                    <td>{pct(c.winrate)}</td>
                    {hasHuman && <td>{pct(c.human_p)}</td>}
                    <td>{c.visits}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
      {me && state.last_eval && (
        <div>
          <h3>내 수 · {state.last_eval.move}</h3>
          <p>{me.summary}</p>
          {me.best_pv.length > 0 && (
            <div className="pv" {...hoverAt(me.best_pv[0], "최선")}>
              최선 예상 진행: {me.best_pv.join(" → ")}
            </div>
          )}
        </div>
      )}
      {evd && state.estimate && (
        <div>
          <h3>실력 추정 근거 · {evd.move}</h3>
          <p>급수별로 이 수를 둘 확률입니다. {why}</p>
          <RankBars values={evd.probs} ranks={state.estimate.ranks} />
        </div>
      )}
    </section>
  );
}

function Seg<T extends string>(p: { name: string; value: T; options: { value: T; label: string; title?: string; disabled?: boolean }[]; onChange: (v: T) => void; disabled: boolean }) {
  return (
    <span className="seg">
      {p.options.map((o) => (
        <label key={o.value} title={o.title} className={p.value === o.value ? "on" : undefined}>
          <input
            type="radio"
            name={p.name}
            value={o.value}
            checked={p.value === o.value}
            disabled={p.disabled || o.disabled}
            onChange={() => p.onChange(o.value)}
          />
          {o.label}
        </label>
      ))}
    </span>
  );
}

export function NewGameCard({ disabled, humanAvailable, onStart }: { disabled: boolean; humanAvailable: boolean; onStart: (o: { size: number; color: Color; mode: Mode }) => void }) {
  const [size, setSize] = useState("19");
  const [color, setColor] = useState<Color>("B");
  const [mode, setMode] = useState<Mode>("human");
  // human 모델이 없으면 사람형을 고를 수 없게
  const effectiveMode = !humanAvailable && mode === "human" ? "adaptive" : mode;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    onStart({ size: Number(size), color, mode: effectiveMode });
  };
  return (
    <section className="card">
      <h2>새 대국</h2>
      <form className="new-game" onSubmit={submit}>
        <div className="field">
          <label>판 크기</label>
          <Seg name="size" value={size} onChange={setSize} disabled={disabled} options={["9", "13", "19"].map((v) => ({ value: v, label: v }))} />
        </div>
        <div className="field">
          <label>내 돌</label>
          <Seg name="color" value={color} onChange={setColor} disabled={disabled} options={[{ value: "B", label: "흑" }, { value: "W", label: "백" }]} />
        </div>
        <div className="field">
          <label>AI</label>
          <Seg
            name="mode"
            value={effectiveMode}
            onChange={setMode}
            disabled={disabled}
            options={[
              { value: "human", label: "사람형", title: "사람 기보로 학습한 모델이 내 수준의 사람처럼 둠 (v3)", disabled: !humanAvailable },
              { value: "adaptive", label: "손실형", title: "목표 손실만큼 일부러 손해 보는 수를 둠 (v2)" },
              { value: "best", label: "최강" },
            ]}
          />
        </div>
        <button className="primary" type="submit" disabled={disabled}>
          시작
        </button>
      </form>
    </section>
  );
}

export function MovesCard({ history }: { history: [Color, string][] }) {
  const rows = [];
  for (let i = 0; i < history.length; i += 2) {
    rows.push(
      <div className="move-row" key={i}>
        <span className="n">{i / 2 + 1}.</span>
        <span>{history[i]?.[1] ?? ""}</span>
        <span>{history[i + 1]?.[1] ?? ""}</span>
      </div>,
    );
  }
  return (
    <section className="card">
      <h2>기보</h2>
      <div
        className="moves"
        ref={(el) => {
          if (el) el.scrollTop = el.scrollHeight;
        }}
      >
        {history.length ? rows : <span className="empty">아직 둔 수가 없습니다</span>}
      </div>
    </section>
  );
}
