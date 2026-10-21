// 바둑판 SVG + 애니메이션.
//
// 연출 3가지:
// 1. 착수: 돌이 살짝 위에서 떨어지듯 커졌다가 안착(스프링) + 퍼지는 파문
// 2. 따냄: 따인 돌이 잠깐 흔들린 뒤 위로 떠오르며 사라짐. 여러 점이면 따낸 자리에서 가까운 순으로 시차
// 3. 형세: 누구 땅이 될지 빈 자리에 은은하게 칠하고, 바뀔 때는 방금 둔 자리에서부터 번지듯 전환
//
// 어떤 돌이 새로 생기고 사라졌는지는 돌 id(놓인 수 번호)로 직전 렌더와 비교해서 안다.
import { motion, type TargetAndTransition } from "motion/react";
import { useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { COLUMNS, pointOf, vertexOf, type Color, type IllegalReason, type Stone } from "../go/rules";
import type { Marker, Ownership } from "../types";

export type TerritoryMode = "live" | "final" | "off";

interface Props {
  size: number;
  gameId: string;
  stones: Stone[];
  lastMove: string | null;
  ownership: Ownership | null;
  territory: TerritoryMode;
  markers: Marker[];
  preview: { moves: string[]; firstColor: Color } | null;
  /** 방금 수로 주인이 바뀐 칸: 테두리로 잠깐 강조 */
  highlights?: string[];
  /** 지금 클릭으로 둘 수 있는 색. null이면 클릭 불가 */
  clickColor: Color | null;
  check: (vertex: string) => IllegalReason | null;
  onPlay: (vertex: string) => void;
  onIllegal: (reason: IllegalReason) => void;
  /** 직전 렌더 대비 새로 놓인 돌, 따여서 사라진 돌 (같은 판 안에서만) */
  onStonesChange?: (added: Stone[], removed: Stone[]) => void;
}

interface Ghost {
  key: string;
  stone: Stone;
  order: number;
}

// SVG 요소를 자기 중심 기준으로 확대·이동시키기 위한 설정
const ORIGIN = { transformBox: "fill-box", transformOrigin: "center" } as const;

function starPoints(size: number): [number, number][] {
  if (size < 9) return [];
  const e = size >= 13 ? 3 : 2;
  const pts = [e, size - 1 - e];
  const out: [number, number][] = [];
  for (const r of pts) for (const c of pts) out.push([r, c]);
  if (size % 2 === 1) {
    const m = (size - 1) / 2;
    out.push([m, m]);
    if (size >= 19) out.push([e, m], [size - 1 - e, m], [m, e], [m, size - 1 - e]);
  }
  return out;
}

export function GoBoard(props: Props) {
  const { size: n, gameId, stones } = props;
  const cell = 100 / (n + 1);
  const x = (c: number) => cell * (c + 1);
  const y = (r: number) => cell * (n - r);
  const sr = cell * 0.47;
  const keyOf = (s: Stone) => `${gameId}-${s.id}`;

  // ---------- 돌 변화 감지 → 따낸 돌 유령 ----------
  const prev = useRef<{ gameId: string; stones: Map<string, Stone> } | null>(null);
  const firstRender = useRef(true);
  // 화면을 처음 열었을 때 이미 있던 돌은 등장 연출 없이 그린다 (판 전체가 한꺼번에 튀지 않게)
  const initialKeys = useRef<Set<string> | null>(null);
  initialKeys.current ??= new Set(stones.map(keyOf));
  const [ghosts, setGhosts] = useState<Ghost[]>([]);
  const onStonesChange = props.onStonesChange;

  useLayoutEffect(() => {
    const next = new Map(stones.map((s) => [keyOf(s), s]));
    const before = prev.current;
    prev.current = { gameId, stones: next };
    firstRender.current = false;
    if (!before || before.gameId !== gameId) {
      setGhosts([]);
      return;
    }
    const added = stones.filter((s) => !before.stones.has(keyOf(s)));
    const removed = [...before.stones.entries()].filter(([k]) => !next.has(k)).map(([, s]) => s);
    if (!added.length && !removed.length) return;
    onStonesChange?.(added, removed);
    if (removed.length) {
      // 따낸 수(가장 최근에 놓인 돌)에서 가까운 돌부터 튕겨 나가게
      const src = added.reduce<Stone | null>((a, s) => (!a || s.id > a.id ? s : a), null);
      const dist = (s: Stone) => (src ? Math.abs(s.row - src.row) + Math.abs(s.col - src.col) : 0);
      const sorted = [...removed].sort((a, b) => dist(a) - dist(b));
      setGhosts((g) => [
        ...g,
        ...sorted.map((stone, order) => ({ key: `ghost-${gameId}-${stone.id}-${Date.now()}`, stone, order })),
      ]);
    }
    // keyOf는 gameId에만 의존
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stones, gameId, onStonesChange]);

  // ---------- 마우스·터치 ----------
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [shake, setShake] = useState<{ vertex: string; n: number } | null>(null);
  const occupied = useMemo(() => new Set(stones.map((s) => vertexOf(s.row, s.col))), [stones]);

  function vertexAt(e: PointerEvent<SVGSVGElement>): string | null {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    const c = Math.round(p.x / cell - 1);
    const r = Math.round(n - p.y / cell);
    return r >= 0 && r < n && c >= 0 && c < n ? vertexOf(r, c) : null;
  }

  const canClick = !!props.clickColor;
  const hoverReason = hover && canClick && !occupied.has(hover) ? props.check(hover) : null;

  function click(e: PointerEvent<SVGSVGElement>) {
    const v = vertexAt(e);
    if (!v || !canClick || occupied.has(v)) return;
    const reason = props.check(v);
    if (reason) {
      setShake((s) => ({ vertex: v, n: (s?.n ?? 0) + 1 }));
      props.onIllegal(reason);
      return;
    }
    setHover(null);
    props.onPlay(v);
  }

  // ---------- 판 배경 (크기가 바뀔 때만 다시 계산) ----------
  const background = useMemo(() => {
    const lineW = n >= 19 ? 0.12 : 0.18;
    const fs = cell * (n >= 13 ? 0.34 : 0.26);
    const off = cell * 0.25;
    const label = { fontSize: fs, fill: "var(--wood-line)", textAnchor: "middle", dominantBaseline: "central", opacity: 0.7 } as const;
    const idx = Array.from({ length: n }, (_, i) => i);
    return (
      <g>
        <rect x={0} y={0} width={100} height={100} fill="var(--wood)" />
        {idx.map((i) => (
          <g key={i}>
            <line x1={x(0)} y1={y(i)} x2={x(n - 1)} y2={y(i)} stroke="var(--wood-line)" strokeWidth={lineW} />
            <line x1={x(i)} y1={y(0)} x2={x(i)} y2={y(n - 1)} stroke="var(--wood-line)" strokeWidth={lineW} />
            <text {...label} x={x(i)} y={off}>{COLUMNS[i]}</text>
            <text {...label} x={x(i)} y={100 - off}>{COLUMNS[i]}</text>
            <text {...label} x={off} y={y(i)}>{i + 1}</text>
            <text {...label} x={100 - off} y={y(i)}>{i + 1}</text>
          </g>
        ))}
        {starPoints(n).map(([r, c]) => (
          <circle key={`${r}-${c}`} cx={x(c)} cy={y(r)} r={cell * 0.09} fill="var(--wood-line)" />
        ))}
      </g>
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n]);

  // ---------- 형세 ----------
  const own = props.territory === "off" ? null : props.ownership;
  const final = props.territory === "final";
  const lastPt = props.lastMove && props.lastMove !== "pass" ? pointOf(props.lastMove) : null;
  const stoneAt = useMemo(() => {
    const m = new Map<string, Stone>();
    for (const s of stones) m.set(`${s.row},${s.col}`, s);
    return m;
  }, [stones]);

  // 돌이 선 자리를 상대 땅으로 보면 그 돌은 죽은(종국) / 잡힐 위기인(대국 중) 돌
  const onEnemyLand = (s: Stone, th: number) =>
    own != null && ((s.color === "B" && own[s.row][s.col] < -th) || (s.color === "W" && own[s.row][s.col] > th));
  const isDead = (s: Stone) => final && onEnemyLand(s, 0.5);
  const inDanger = (s: Stone) => !final && onEnemyLand(s, 0.6);

  const territory = Array.from({ length: n * n }, (_, i) => {
    const r = Math.floor(i / n);
    const c = i % n;
    const s = stoneAt.get(`${r},${c}`);
    const o = own ? own[r][c] : 0;
    let alpha = 0;
    if (own) {
      if (final) alpha = Math.abs(o) > 0.5 && (!s || isDead(s)) ? 0.9 : 0;
      // 대국 중: 빈 점을 칸 전체 크기로 칠해 땅이 "지역"으로 보이게. 확실할수록 진하게
      else if (!s) alpha = Math.min(1, Math.max(0, (Math.abs(o) - 0.15) / 0.6)) * 0.5;
    }
    const w = cell * (final ? 0.3 : 0.98);
    // 방금 둔 자리에서 멀수록 늦게 바뀌어 번지는 것처럼 보인다
    const delay = lastPt ? Math.min(450, (Math.abs(r - lastPt[0]) + Math.abs(c - lastPt[1])) * 28) : 0;
    const common = {
      x: x(c) - w / 2,
      y: y(r) - w / 2,
      width: w,
      height: w,
      rx: final ? 0 : w * 0.12,
      className: "territory",
      style: { transitionDelay: `${delay}ms` },
    };
    return (
      <g key={i}>
        <rect {...common} fill="#111" fillOpacity={o > 0 ? alpha : 0} stroke={final ? "#555" : "none"} strokeWidth={0.05} strokeOpacity={o > 0 ? alpha : 0} />
        <rect {...common} fill="#fff" fillOpacity={o < 0 ? alpha : 0} stroke={final ? "#555" : "none"} strokeWidth={0.05} strokeOpacity={o < 0 ? alpha : 0} />
      </g>
    );
  });

  const stoneFill = (c: Color) => (c === "B" ? "url(#gb)" : "url(#gw)");
  const stoneBody = (c: Color) => (
    <>
      <circle cx={cell * 0.03} cy={cell * 0.05} r={sr} fill="rgba(0,0,0,.25)" />
      <circle r={sr} fill={stoneFill(c)} stroke={c === "W" ? "#b9b2a3" : "none"} strokeWidth={0.08} />
    </>
  );

  const lastStone = lastPt ? stoneAt.get(`${lastPt[0]},${lastPt[1]}`) : undefined;
  const hoverPt = hover && canClick && !occupied.has(hover) ? pointOf(hover) : null;

  return (
    <svg
      ref={svgRef}
      className={`board${canClick ? "" : " locked"}`}
      viewBox="0 0 100 100"
      onPointerMove={(e) => setHover(vertexAt(e))}
      onPointerLeave={() => setHover(null)}
      onClick={click}
      role="img"
      aria-label={`${n}줄 바둑판`}
    >
      <defs>
        <radialGradient id="gb" cx="35%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#5a5650" />
          <stop offset="100%" stopColor="#111" />
        </radialGradient>
        <radialGradient id="gw" cx="35%" cy="30%" r="75%">
          <stop offset="0%" stopColor="#fff" />
          <stop offset="100%" stopColor="#d9d4c8" />
        </radialGradient>
      </defs>
      {background}

      {!final && <g pointerEvents="none">{territory}</g>}

      {/* 돌 */}
      <g pointerEvents="none">
        {stones.map((s) => {
          const animate = !initialKeys.current!.has(keyOf(s));
          const enter: TargetAndTransition | false = animate ? { scale: 1.28, y: -cell * 0.22, opacity: 0.5 } : false;
          return (
            <g key={keyOf(s)} transform={`translate(${x(s.col)} ${y(s.row)})`}>
              {animate && (
                <motion.circle
                  r={sr}
                  fill="none"
                  stroke={s.color === "B" ? "#111" : "#fff"}
                  strokeWidth={cell * 0.05}
                  style={ORIGIN}
                  initial={{ scale: 1, opacity: 0.5 }}
                  animate={{ scale: 1.9, opacity: 0 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              )}
              <motion.g
                style={ORIGIN}
                initial={enter}
                animate={{ scale: 1, y: 0, opacity: isDead(s) ? 0.45 : 1 }}
                transition={{ type: "spring", stiffness: 900, damping: 30, mass: 0.6 }}
              >
                {stoneBody(s.color)}
              </motion.g>
            </g>
          );
        })}
      </g>

      {/* 따낸 돌: 흔들림 → 떠오르며 사라짐 */}
      <g pointerEvents="none">
        {ghosts.map((g) => {
          const j = cell * 0.08;
          return (
            <g key={g.key} transform={`translate(${x(g.stone.col)} ${y(g.stone.row)})`}>
              <motion.g
                style={ORIGIN}
                initial={{ x: 0, y: 0, scale: 1, opacity: 1 }}
                animate={{
                  x: [0, -j, j, -j / 2, 0, 0],
                  y: [0, 0, 0, 0, 0, -cell * 0.9],
                  scale: [1, 1, 1, 1, 1.06, 0.5],
                  opacity: [1, 1, 1, 1, 1, 0],
                }}
                transition={{ duration: 0.62, times: [0, 0.08, 0.16, 0.24, 0.34, 1], delay: 0.06 + g.order * 0.035, ease: "easeOut" }}
                onAnimationComplete={() => setGhosts((gs) => gs.filter((o) => o.key !== g.key))}
              >
                {stoneBody(g.stone.color)}
              </motion.g>
            </g>
          );
        })}
      </g>

      {final && <g pointerEvents="none">{territory}</g>}

      {/* 방금 수로 주인이 바뀐 칸: 테두리를 그렸다가 천천히 흐려진다 (판이 바뀔 때마다 다시) */}
      {!final && props.highlights && props.highlights.length > 0 && (
        <g pointerEvents="none" key={`hl-${props.lastMove}-${props.highlights.length}`}>
          {props.highlights.map((v) => {
            const [r, c] = pointOf(v);
            const w = cell * 0.9;
            return (
              <motion.rect
                key={v}
                x={x(c) - w / 2}
                y={y(r) - w / 2}
                width={w}
                height={w}
                rx={w * 0.12}
                fill="none"
                stroke="var(--changed)"
                strokeWidth={cell * 0.07}
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 1, 0.35] }}
                transition={{ duration: 2.2, times: [0, 0.12, 0.6, 1] }}
              />
            );
          })}
        </g>
      )}

      {/* 대국 중 잡힐 위기의 돌: 돌 위에 그 자리 주인 색 표시 (종국의 사석 표시와 같은 모양) */}
      {!final && own && (
        <g pointerEvents="none">
          {stones.filter(inDanger).map((s) => {
            const w = cell * 0.3;
            return (
              <rect
                key={`danger-${keyOf(s)}`}
                className="territory"
                x={x(s.col) - w / 2}
                y={y(s.row) - w / 2}
                width={w}
                height={w}
                fill={s.color === "B" ? "#fff" : "#111"}
                stroke="#555"
                strokeWidth={0.05}
              />
            );
          })}
        </g>
      )}

      {/* 마지막 수 표시: 돌이 안착한 뒤 나타난다 */}
      {lastStone && (
        <motion.circle
          key={`last-${keyOf(lastStone)}`}
          cx={x(lastStone.col)}
          cy={y(lastStone.row)}
          r={sr * 0.38}
          fill="none"
          stroke={lastStone.color === "B" ? "#fff" : "#111"}
          strokeWidth={cell * 0.06}
          pointerEvents="none"
          style={ORIGIN}
          initial={firstRender.current ? false : { scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.14, duration: 0.18 }}
        />
      )}

      {/* 호버 미리보기: 둘 수 있으면 반투명 돌, 없으면 빨간 점선 */}
      {hoverPt && props.clickColor && (
        <g transform={`translate(${x(hoverPt[1])} ${y(hoverPt[0])})`} pointerEvents="none">
          {hoverReason ? (
            <circle r={sr * 0.9} fill="none" stroke="var(--danger)" strokeWidth={cell * 0.06} strokeDasharray={`${cell * 0.12} ${cell * 0.1}`} opacity={0.8} />
          ) : (
            <circle r={sr} fill={props.clickColor === "B" ? "#111" : "#fff"} opacity={0.42} />
          )}
        </g>
      )}

      {/* 둘 수 없는 자리를 눌렀을 때: 고개 젓듯 흔들기 */}
      {shake && (
        <motion.circle
          key={`shake-${shake.n}`}
          cx={x(pointOf(shake.vertex)[1])}
          cy={y(pointOf(shake.vertex)[0])}
          r={sr * 0.9}
          fill="color-mix(in srgb, var(--danger) 18%, transparent)"
          stroke="var(--danger)"
          strokeWidth={cell * 0.07}
          pointerEvents="none"
          initial={{ x: 0, opacity: 1 }}
          animate={{ x: [0, -cell * 0.18, cell * 0.18, -cell * 0.12, cell * 0.08, 0], opacity: [1, 1, 1, 1, 1, 0] }}
          transition={{ duration: 0.5, times: [0, 0.12, 0.28, 0.44, 0.6, 1] }}
        />
      )}

      {/* 힌트 A/B/C, 후보수 표시 */}
      {props.markers
        .filter((m) => m.vertex !== "pass")
        .map((m) => {
          const [r, c] = pointOf(m.vertex);
          return (
            <motion.g
              key={`m-${m.vertex}-${m.label}`}
              transform={`translate(${x(c)} ${y(r)})`}
              pointerEvents="none"
            >
              <motion.g style={ORIGIN} initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 600, damping: 22 }}>
                <circle r={sr * 0.62} fill={m.color || "var(--accent)"} opacity={0.88} />
                <text fontSize={cell * (m.label.length > 1 ? 0.26 : 0.36)} fill="#fff" textAnchor="middle" dominantBaseline="central" fontWeight={600}>
                  {m.label}
                </text>
              </motion.g>
            </motion.g>
          );
        })}

      {/* 예상 진행: 번호 붙은 반투명 돌이 차례로 나타난다 */}
      {props.preview &&
        props.preview.moves.map((v, i) => {
          if (v === "pass") return null;
          const [r, c] = pointOf(v);
          if (stoneAt.has(`${r},${c}`)) return null;
          const color = (i % 2 === 0) === (props.preview!.firstColor === "B") ? "B" : "W";
          return (
            <motion.g
              key={`pv-${i}-${v}`}
              transform={`translate(${x(c)} ${y(r)})`}
              pointerEvents="none"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.06, duration: 0.15 }}
            >
              <circle r={sr} fill={color === "B" ? "#111" : "#fff"} opacity={0.55} stroke="#555" strokeWidth={0.06} />
              <text fontSize={cell * 0.34} fill={color === "B" ? "#fff" : "#111"} textAnchor="middle" dominantBaseline="central">
                {i + 1}
              </text>
            </motion.g>
          );
        })}
    </svg>
  );
}
