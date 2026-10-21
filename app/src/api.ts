import type { GameState } from "./types";

export async function api(path: string, body?: unknown): Promise<GameState> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
  return res.json();
}
