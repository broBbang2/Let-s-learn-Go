// 기보 보관함 (예전 web/library.html을 옮김). 목록 ↔ 감상은 주소 #<기보 id>로 오간다.
import { MotionConfig } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState, type MouseEvent } from "react";
import { GoBoard } from "../components/GoBoard";
import { pct } from "../core/http";
import {
  buildMoments,
  chartLayout,
  engineText,
  eraOf,
  gameHeader,
  humanText,
  itemMeta,
  leadText,
  libraryApi,
  noteAt,
  toPlayAt,
  type DoneAnalysis,
  type LibraryGame,
  type LibraryItem,
} from "../core/library";
import { useLibraryGame } from "../core/useLibraryGame";
import type { Marker } from "../types";

function useHashId() {
  const read = () => decodeURIComponent(window.location.hash.slice(1)) || null;
  const [id, setId] = useState(read);
  useEffect(() => {
    const on = () => setId(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return id;
}

export function LibraryApp() {
  const id = useHashId();
  return (
    <MotionConfig reducedMotion="user">
      <nav className="top wide">
        <a href="/">대국</a>
        <a href="/library" className="on">
          기보 보관함
        </a>
      </nav>
      <main className="library">{id ? <Viewer id={id} /> : <GameList />}</main>
    </MotionConfig>
  );
}

// ---------- 목록 ----------

function GameList() {
  const [items, setItems] = useState<LibraryItem[] | null>(null);
  useEffect(() => {
    libraryApi.list().then(setItems, () => setItems([]));
  }, []);

  const section = (kind: LibraryItem["kind"], empty: string) => {
    const list = (items ?? []).filter((g) => g.kind === kind);
    return (
      <div className="games">
        {items && !list.length && <div className="empty">{empty}</div>}
        {list.map((g) => (
          <button key={g.id} className="game-card" onClick={() => (window.location.hash = g.id)}>
            <div className="title">{g.title}</div>
            <div className="players">
              흑 {g.black} · 백 {g.white}
            </div>
            <div className="meta">{itemMeta(g)}</div>
          </button>
        ))}
      </div>
    );
  };

  return (
    <section>
      <h1>기보 보관함</h1>
      <h2 className="section">명국</h2>
      {section("famous", "library 폴더에 SGF 기보를 넣으면 여기에 나타납니다.")}
      <h2 className="section">내 대국</h2>
      {section("mine", "끝까지 둔 대국이 아직 없습니다.")}
    </section>
  );
}

// ---------- 감상 ----------

function Viewer({ id }: { id: string }) {
  const { game, error, turn, total, positions, analysis, goTo, analyze } = useLibraryGame(id);
  const [hoverPv, setHoverPv] = useState<string[] | null>(null);

  // 한 수 앞으로 갈 때만 착수·따냄 연출, 되돌리거나 건너뛸 때는 바로 바꾼다
  const prevTurn = useRef(turn);
  const instant = turn !== prevTurn.current + 1;
  useEffect(() => {
    prevTurn.current = turn;
  });

  useEffect(() => setHoverPv(null), [turn]);
  useEffect(() => {
    if (error) window.location.hash = "";
  }, [error]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === "ArrowLeft") goTo(turn - 1);
      else if (e.key === "ArrowRight") goTo(turn + 1);
      else if (e.key === "Home") goTo(0);
      else if (e.key === "End") goTo(total);
      else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [goTo, turn, total]);

  if (!game) return <div className="loading">불러오는 중</div>;

  const done = analysis.state === "done" ? analysis : null;
  const header = gameHeader(game);
  const last = turn > 0 ? game.moves[turn - 1] : null;
  const a = done?.turns[turn];
  const markers: Marker[] = a && a.best !== "pass" && turn < total ? [{ vertex: a.best, label: "AI", color: "var(--series)" }] : [];
  const note = noteAt(game, turn);
  const comment = game.comments?.[turn];

  return (
    <section className="viewer">
      <div className="board-wrap">
        <GoBoard
          size={game.size}
          gameId={game.id}
          stones={positions[turn]}
          lastMove={last && last[1] !== "pass" ? last[1] : null}
          ownership={null}
          territory="off"
          markers={markers}
          preview={hoverPv ? { moves: hoverPv, firstColor: toPlayAt(game.moves, turn) } : null}
          clickColor={null}
          check={() => null}
          onPlay={() => {}}
          onIllegal={() => {}}
          instant={instant}
        />
      </div>
      <aside>
        <div className="card">
          <div className="game-title">{header.title}</div>
          <div className="sub">{header.players}</div>
          <div className="sub">{header.meta}</div>
          {game.warning && <div className="sub warn">{game.warning}</div>}
        </div>

        <div className="card">
          <div className="nav-row">
            <button onClick={() => goTo(0)} disabled={turn === 0} title="처음 (Home)">⏮</button>
            <button onClick={() => goTo(turn - 1)} disabled={turn === 0} title="이전 (←)">◀</button>
            <button onClick={() => goTo(turn + 1)} disabled={turn === total} title="다음 (→)">▶</button>
            <button onClick={() => goTo(total)} disabled={turn === total} title="끝 (End)">⏭</button>
            <span className="move-no">
              {last ? `${turn} / ${total} · ${last[0] === "B" ? "흑" : "백"} ${last[1]}` : `0 / ${total}`}
            </span>
          </div>
          <input type="range" min={0} max={total} value={turn} onChange={(e) => goTo(Number(e.target.value))} />
          {note && <div className="note-box">{note}</div>}
          {comment && <div className="comment">{comment}</div>}
        </div>

        <div className="card">
          <h3>KataGo 분석</h3>
          {(analysis.state === "none" || analysis.state === "error" || analysis.state === "busy") && (
            <div>
              <div className="sub" style={{ marginBottom: 8 }}>
                모든 국면을 분석해 승률 그래프, 결정적 수, 수마다 근거를 보여줍니다. 19줄 기준 1~2분 걸립니다.
              </div>
              <button className="primary" onClick={() => void analyze()}>
                {analysis.state === "none" ? "분석 시작" : `다시 시도 (${analysis.error})`}
              </button>
            </div>
          )}
          {analysis.state === "running" && (
            <div>
              <div className="sub">
                국면 {analysis.done} / {analysis.total} 분석 중
              </div>
              <div className="progress">
                <div style={{ width: `${(analysis.done / analysis.total) * 100}%` }} />
              </div>
            </div>
          )}
          {done && (
            <div>
              <WinrateChart analysis={done} turn={turn} onPick={goTo} />
              <div className="key">
                <span><i className="dot" />결정적 수</span>
                <span><i className="diamond" />예상 밖의 명수(자동)</span>
              </div>
              <Here analysis={done} turn={turn} setHoverPv={setHoverPv} />
            </div>
          )}
        </div>

        {done && <Moments game={game} analysis={done} onPick={goTo} />}

        <div className="card">
          <a className="sub" href={libraryApi.sgfUrl(game.id)}>
            SGF로 받기
          </a>
        </div>
      </aside>
    </section>
  );
}

function Here({ analysis, turn, setHoverPv }: { analysis: DoneAnalysis; turn: number; setHoverPv: (pv: string[] | null) => void }) {
  const a = analysis.turns[turn];
  if (!a) return null;
  const n = a.next;
  return (
    <div>
      <div className="kv">
        <span>이 국면</span>
        <b>
          흑 승률 {pct(a.winrate_black)} · {leadText(a.score_black)}
        </b>
      </div>
      {n && (
        <>
          <div className="reason">
            <b>
              다음 수 {turn + 1}. {n.color === "B" ? "흑" : "백"} {n.move}
            </b>
            <div className="kv">
              <span>엔진 평가</span>
              <b>{engineText(n)}</b>
            </div>
            <div className="kv">
              <span>{eraOf(analysis)}년 프로라면</span>
              <b>이 수를 둘 확률 {humanText(n)}</b>
            </div>
          </div>
          {a.best !== n.move && a.best !== "pass" ? (
            <div className="pv" onMouseEnter={() => setHoverPv(a.best_pv)} onMouseLeave={() => setHoverPv(null)}>
              KataGo 최선 {a.best} (판에 AI 표시) · 예상 진행 {a.best_pv.join(" → ")}
            </div>
          ) : (
            <div className="pv">실제 수가 KataGo 최선수와 같습니다.</div>
          )}
        </>
      )}
    </div>
  );
}

function Moments({ game, analysis, onPick }: { game: LibraryGame; analysis: DoneAnalysis; onPick: (t: number) => void }) {
  const list = buildMoments(game, analysis);
  if (!list.length) return null;
  return (
    <div className="card">
      <h3>주요 장면</h3>
      <div className="moments">
        {list.map((m, i) => (
          // 그 수를 두기 직전 국면으로 가서 AI 최선수와 비교해 볼 수 있게
          <button key={i} className="moment" onClick={() => onPick(m.turn)}>
            <i className={m.kind} />
            <span>{m.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// 흑 승률 그래프: 한 계열 선(2px) + 50% 기준선 + 현재 위치 + 주요 장면 표식. 마우스로 십자선·툴팁, 클릭하면 이동
function WinrateChart({ analysis, turn, onPick }: { analysis: DoneAnalysis; turn: number; onPick: (t: number) => void }) {
  const ref = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(340);
  const [hover, setHover] = useState<number | null>(null);

  useLayoutEffect(() => {
    const svg = ref.current;
    if (!svg) return;
    const ro = new ResizeObserver(() => setWidth(svg.clientWidth || 340));
    ro.observe(svg);
    return () => ro.disconnect();
  }, []);

  const L = chartLayout(analysis, width);
  const { padL, padR, padT, padB, width: W, height: H } = L;
  const turnAt = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return L.turnAtX(((e.clientX - rect.left) / rect.width) * W);
  };
  const hoverTurn = hover != null ? analysis.turns[hover] : null;

  return (
    <div className="chart-wrap">
      <svg
        ref={ref}
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        aria-label="흑 승률 그래프"
        onMouseMove={(e) => setHover(turnAt(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => onPick(turnAt(e))}
      >
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1={padL} x2={W - padR} y1={L.y(p)} y2={L.y(p)} stroke="var(--line)" strokeWidth={1} strokeDasharray={p === 0.5 ? "3 3" : undefined} />
            <text x={padL - 4} y={L.y(p)} fontSize={10} fill="var(--muted)" textAnchor="end" dominantBaseline="central">
              {p * 100}%
            </text>
          </g>
        ))}
        <text x={W - padR} y={H - 4} fontSize={10} fill="var(--muted)" textAnchor="end">
          {L.n}수
        </text>
        <text x={padL} y={H - 4} fontSize={10} fill="var(--muted)">
          흑 승률
        </text>
        <path d={L.path} fill="none" stroke="var(--series)" strokeWidth={2} strokeLinejoin="round" />
        <line x1={L.x(turn)} x2={L.x(turn)} y1={padT} y2={H - padB} stroke="var(--ink)" strokeWidth={1} opacity={0.5} />
        {L.turning.map((m, i) => (
          <circle key={`t${i}`} cx={m.x} cy={m.y} r={4} fill="var(--turning)" stroke="var(--panel)" strokeWidth={2} />
        ))}
        {L.surprises.map((m, i) => (
          <rect key={`s${i}`} x={m.x - 4} y={m.y - 4} width={8} height={8} fill="var(--series)" stroke="var(--panel)" strokeWidth={2} transform={`rotate(45 ${m.x} ${m.y})`} />
        ))}
        {hover != null && <line x1={L.x(hover)} x2={L.x(hover)} y1={padT} y2={H - padB} stroke="var(--muted)" strokeWidth={1} />}
      </svg>
      {hover != null && hoverTurn && (
        <div className="tooltip" style={{ left: `${(L.x(hover) / W) * 100}%`, top: `${(L.y(hoverTurn.winrate_black) / H) * 100}%` }}>
          {hover}수 · 흑 승률 {pct(hoverTurn.winrate_black)}
        </div>
      )}
    </div>
  );
}
