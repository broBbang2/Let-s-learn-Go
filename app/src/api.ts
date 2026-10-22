import { request } from "./core/http";
import type { GameState } from "./types";

export function api(path: string, body?: unknown): Promise<GameState> {
  return request<GameState>(path, body);
}
