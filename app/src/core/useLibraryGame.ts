// 기보 하나를 불러와 재생·분석하는 상태 훅. 화면(웹 SVG / 앱 react-native-svg)과 무관하게 같이 쓴다.
import { useCallback, useEffect, useMemo, useState } from "react";
import { libraryApi, replayStones, type Analysis, type LibraryGame } from "./library";

export function useLibraryGame(id: string | null) {
  const [game, setGame] = useState<LibraryGame | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [turn, setTurn] = useState(0);
  const [analysis, setAnalysis] = useState<Analysis>({ state: "none" });

  useEffect(() => {
    setGame(null);
    setError(null);
    setTurn(0);
    setAnalysis({ state: "none" });
    if (!id) return;
    let alive = true;
    libraryApi.load(id).then(
      (g) => {
        if (!alive) return;
        setGame(g);
        setAnalysis(g.analysis);
      },
      (e) => alive && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      alive = false;
    };
  }, [id]);

  // 분석 중이면 1초마다 진행률을 물어본다
  useEffect(() => {
    if (!game || analysis.state !== "running") return;
    const timer = setTimeout(() => {
      libraryApi.analysis(game.id).then(setAnalysis, () => {});
    }, 1000);
    return () => clearTimeout(timer);
  }, [game, analysis]);

  const positions = useMemo(() => (game ? replayStones(game.size, game.moves) : []), [game]);
  const total = game?.moves.length ?? 0;

  const goTo = useCallback((t: number) => setTurn(Math.max(0, Math.min(total, t))), [total]);

  const analyze = useCallback(async () => {
    if (game) setAnalysis(await libraryApi.analyze(game.id));
  }, [game]);

  return { game, error, turn, total, positions, analysis, goTo, analyze };
}
