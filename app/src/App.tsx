import { MotionConfig } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { GoBoard, type TerritoryMode } from "./components/GoBoard";
import { ActionButtons, AiCard, ExplainCard, MovesCard, NewGameCard, SettingsCard, SkillCard, StatsCard, StatusCard, TerritoryCard, WeaknessCard } from "./components/Panels";
import { feedback, setSoundEnabled, soundEnabled } from "./feedback";
import { ILLEGAL_TEXT, Position, type IllegalReason, type Stone } from "./go/rules";
import type { GameState, Marker, Mode } from "./types";

const TERRITORY_KEY = "adaptive-go:territory";
const TAB_KEY = "adaptive-go:tab";

// 오른쪽 패널(모바일은 하단 시트) 탭: 대국 중에 보는 것 / 쌓인 실력 / 대국 준비
const TABS = [
  { id: "now", label: "지금" },
  { id: "skill", label: "내 실력" },
  { id: "game", label: "대국" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function readTerritoryPref(): boolean {
  try {
    return localStorage.getItem(TERRITORY_KEY) !== "off";
  } catch {
    return true;
  }
}

function readTabPref(): Tab {
  try {
    const t = localStorage.getItem(TAB_KEY);
    return TABS.some((x) => x.id === t) ? (t as Tab) : "now";
  } catch {
    return "now";
  }
}

/** 서버 history를 다시 둬서 돌마다 id가 붙은 국면을 만든다. 서버 판과 다르면 서버 판을 믿는다. */
function positionOf(s: GameState): Position {
  try {
    const p = Position.replay(s.size, s.history);
    if (JSON.stringify(p.cells()) === JSON.stringify(s.grid)) return p;
    console.warn("규칙 엔진 재생 결과가 서버 판과 달라 서버 판을 그대로 씁니다");
  } catch (e) {
    console.warn("history 재생 실패", e);
  }
  const p = new Position(s.size);
  s.grid.forEach((row, r) =>
    row.forEach((cell, c) => {
      if (cell !== ".") p.grid[r][c] = { id: -(r * s.size + c) - 1, color: cell, row: r, col: c };
    }),
  );
  p.toPlay = s.to_play;
  p.captures = { ...s.captures };
  p.moveCount = s.history.length;
  return p;
}

export function App() {
  const [state, setState] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<string | null>(null); // AI 응답을 기다리는 동안 미리 그려 둘 내 수
  const [hoverMarkers, setHoverMarkers] = useState<Marker[] | null>(null); // 해설에서 마우스를 올린 후보수
  const [localMsg, setLocalMsg] = useState<string | null>(null);
  const [territoryOn, setTerritoryOn] = useState(readTerritoryPref);
  const [sound, setSound] = useState(soundEnabled);
  const [tab, setTabState] = useState<Tab>(readTabPref);
  const [sheetOpen, setSheetOpen] = useState(false); // 모바일 하단 시트 펼침
  const setTab = (t: Tab) => {
    setTabState(t);
    setSheetOpen(true);
    try {
      localStorage.setItem(TAB_KEY, t);
    } catch {
      /* 저장 못 해도 지금은 적용 */
    }
  };
  const turnStart = useRef<number | null>(null); // 내 차례 시작 시각 → 생각한 시간(think_ms)

  useEffect(() => {
    api("/api/state").then((s) => {
      setState(s);
      turnStart.current = performance.now();
    });
  }, []);

  const position = useMemo(() => (state ? positionOf(state) : null), [state]);
  // 내 수는 서버 응답 전에 바로 반영 (따낸 돌도 바로 들어냄)
  const shown = useMemo(() => {
    if (!position || !state || !pending || pending === "pass") return position;
    const p = position.clone();
    try {
      p.play(pending, state.user_color);
    } catch {
      return position;
    }
    return p;
  }, [position, pending, state]);
  const stones = useMemo(() => shown?.stones() ?? [], [shown]);

  const locked = busy || !state || state.over || state.to_play !== state.user_color;

  const request = useCallback(async (path: string, body: unknown, optimistic: string | null = null) => {
    setBusy(true);
    setPending(optimistic);
    setLocalMsg(null);
    try {
      setState(await api(path, body));
    } catch (e) {
      setLocalMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setPending(null);
      turnStart.current = performance.now();
    }
  }, []);

  const play = useCallback(
    (move: string) => {
      if (locked) return;
      const think_ms = turnStart.current == null ? null : Math.round(performance.now() - turnStart.current);
      void request("/api/move", { move, think_ms }, move === "pass" ? null : move);
    },
    [locked, request],
  );

  const onStonesChange = useCallback((added: Stone[], removed: Stone[]) => {
    if (added.length) feedback.place();
    if (removed.length) feedback.capture(removed.length);
  }, []);

  const onIllegal = useCallback((reason: IllegalReason) => {
    setLocalMsg(ILLEGAL_TEXT[reason]);
    feedback.illegal();
  }, []);

  const toggleTerritory = () => {
    const on = !territoryOn;
    setTerritoryOn(on);
    try {
      localStorage.setItem(TERRITORY_KEY, on ? "on" : "off");
    } catch {
      /* 저장 못 해도 지금은 적용 */
    }
  };
  const toggleSound = () => {
    setSoundEnabled(!sound);
    setSound(!sound);
    if (!sound) feedback.place(); // 켤 때 한 번 들려준다
  };

  if (!state || !shown) {
    return <div className="loading">불러오는 중</div>;
  }

  const lastMove = pending ?? [...state.history].reverse().find(([, v]) => v !== "pass")?.[1] ?? null;
  const finalOwn = state.result?.ownership ?? null;
  const territory: TerritoryMode = finalOwn ? "final" : territoryOn && state.ownership ? "live" : "off";

  return (
    <MotionConfig reducedMotion="user">
      <nav className="top">
        <a href="/" className="on">
          대국
        </a>
        <a href="/library">기보 보관함</a>
      </nav>
      <main>
        <div className="board-wrap">
          <GoBoard
            size={state.size}
            gameId={state.game_id}
            stones={stones}
            lastMove={lastMove}
            ownership={finalOwn ?? state.ownership ?? null}
            territory={territory}
            markers={hoverMarkers ?? []}
            preview={null}
            highlights={
              territoryOn && !busy && state.territory_change
                ? [...new Set([...(state.territory_change.user?.points ?? []), ...(state.territory_change.ai?.points ?? [])])]
                : []
            }
            clickColor={locked ? null : state.user_color}
            check={(v) => shown.check(v, state.user_color)}
            onPlay={play}
            onIllegal={onIllegal}
            onStonesChange={onStonesChange}
          />
        </div>

        <aside className={`panel${sheetOpen ? " open" : ""}`}>
          <button className="sheet-handle" onClick={() => setSheetOpen(!sheetOpen)} aria-expanded={sheetOpen} aria-label={sheetOpen ? "패널 접기" : "패널 펼치기"}>
            <span />
          </button>
          <StatusCard state={state} busy={busy} message={localMsg ?? state.message}>
            <ActionButtons
              locked={locked}
              canResign={!busy && !state.over}
              onPass={() => play("pass")}
              onResign={() => void request("/api/resign", {})}
            />
          </StatusCard>
          <div className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "on" : undefined} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
          <div className="tab-body" role="tabpanel">
            {tab === "now" && (
              <>
                {territoryOn && <TerritoryCard state={state} />}
                <ExplainCard state={state} setHoverMarkers={setHoverMarkers} />
                {!state.last_ai && !state.user_explain && <div className="empty tab-empty">수를 두면 여기에 AI의 수와 내 수 해설이 나옵니다.</div>}
                <StatsCard state={state} captures={shown.captures} />
              </>
            )}
            {tab === "skill" && (
              <>
                <SkillCard state={state} />
                <WeaknessCard state={state} />
                <AiCard state={state} />
              </>
            )}
            {tab === "game" && (
              <>
                <NewGameCard
                  disabled={busy}
                  humanAvailable={state.ai.human_available}
                  onStart={({ size, color, mode }: { size: number; color: "B" | "W"; mode: Mode }) =>
                    void request("/api/new", { size, color, mode, visits: 200, komi: size === 19 ? 7.5 : 7 })
                  }
                />
                <SettingsCard territoryOn={territoryOn} onTerritory={toggleTerritory} sound={sound} onSound={toggleSound} />
                <MovesCard history={state.history} />
              </>
            )}
          </div>
        </aside>
      </main>
    </MotionConfig>
  );
}
