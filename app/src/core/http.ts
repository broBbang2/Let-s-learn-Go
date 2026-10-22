// 서버 호출 공용 함수. DOM에 의존하지 않아 React Native에서도 그대로 쓴다.
//
// 웹: 같은 서버에서 페이지를 받으므로 상대 경로("/api/...")로 충분하다.
// 앱: 서버 주소가 따로 있으므로 시작할 때 setApiBase("http://192.168.0.10:8000")처럼 한 번 지정한다.

let base = "";

export function setApiBase(url: string) {
  base = url.replace(/\/$/, "");
}

export function apiUrl(path: string) {
  return base + path;
}

export async function request<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
  return res.json() as Promise<T>;
}

// 1% 미만은 유효숫자 2자리로 (0%로 뭉개지지 않게)
export const pct = (p: number | null | undefined) =>
  p == null ? "-" : p > 0 && p < 0.01 ? `${(p * 100).toPrecision(2)}%` : `${(p * 100).toFixed(0)}%`;
