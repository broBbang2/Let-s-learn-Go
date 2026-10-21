// 바둑판 SVG 그리기. 대국 화면(index.html)과 명국 감상(library.html)이 같이 쓴다.
// 좌표는 GTP 형식("D4"), grid[row][col]에서 row 0이 1행(아래).

const COLUMNS = "ABCDEFGHJKLMNOPQRST";
const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs, parent) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function vertexOf(row, col) { return COLUMNS[col] + (row + 1); }

function pointOf(v) { return [+v.slice(1) - 1, COLUMNS.indexOf(v[0])]; }

function starPoints(size) {
  if (size < 9) return [];
  const e = size >= 13 ? 3 : 2;
  const pts = [e, size - 1 - e];
  const out = [];
  for (const r of pts) for (const c of pts) out.push([r, c]);
  if (size % 2 === 1) {
    const m = (size - 1) / 2;
    out.push([m, m]);
    if (size >= 19) out.push([e, m], [size - 1 - e, m], [m, e], [m, size - 1 - e]);
  }
  return out;
}

/**
 * opts:
 *   size, grid                판 크기와 돌 배치
 *   lastMove                  마지막 수 표시할 좌표
 *   ownership                 종국 후 집·사석 표시용 [row][col] (흑 +, 백 -)
 *   markers                   [{vertex, label, color}] 원형 표시 (힌트 A/B/C, 후보수 등)
 *   preview                   {moves, firstColor} 번호 붙은 반투명 돌 (예상 진행)
 *   clickColor, onClick       빈 자리 클릭 착수 (clickColor가 없으면 클릭 불가)
 */
function drawGoBoard(svg, opts) {
  const n = opts.size;
  const cell = 100 / (n + 1);
  // 판 좌표: 0~100, 위쪽이 n행 (화면 위 = 큰 행 번호)
  const x = (c) => cell * (c + 1);
  const y = (r) => cell * (n - r);
  const at = (v) => { const [r, c] = pointOf(v); return [x(c), y(r)]; };
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.innerHTML = "";
  svg.classList.toggle("locked", !opts.clickColor);

  svgEl("rect", { x: 0, y: 0, width: 100, height: 100, fill: "var(--wood)" }, svg);
  const lineW = n >= 19 ? 0.12 : 0.18;
  for (let i = 0; i < n; i++) {
    svgEl("line", { x1: x(0), y1: y(i), x2: x(n - 1), y2: y(i), stroke: "var(--wood-line)", "stroke-width": lineW }, svg);
    svgEl("line", { x1: x(i), y1: y(0), x2: x(i), y2: y(n - 1), stroke: "var(--wood-line)", "stroke-width": lineW }, svg);
  }
  for (const [r, c] of starPoints(n)) {
    svgEl("circle", { cx: x(c), cy: y(r), r: cell * 0.09, fill: "var(--wood-line)" }, svg);
  }

  // 좌표 라벨: 가장자리 돌(반지름 0.47칸)과 겹치지 않도록 판 바깥쪽 0.25칸 위치에 둔다
  const fs = cell * (n >= 13 ? 0.34 : 0.26); // 19줄은 칸이 작아 글자를 키운다
  const off = cell * 0.25;
  for (let i = 0; i < n; i++) {
    const style = { "font-size": fs, fill: "var(--wood-line)", "text-anchor": "middle", "dominant-baseline": "central", opacity: 0.7 };
    svgEl("text", { ...style, x: x(i), y: off }, svg).textContent = COLUMNS[i];
    svgEl("text", { ...style, x: x(i), y: 100 - off }, svg).textContent = COLUMNS[i];
    svgEl("text", { ...style, x: off, y: y(i) }, svg).textContent = i + 1;
    svgEl("text", { ...style, x: 100 - off, y: y(i) }, svg).textContent = i + 1;
  }

  const defs = svgEl("defs", {}, svg);
  const gb = svgEl("radialGradient", { id: "gb", cx: "35%", cy: "30%", r: "70%" }, defs);
  svgEl("stop", { offset: "0%", "stop-color": "#5a5650" }, gb);
  svgEl("stop", { offset: "100%", "stop-color": "#111" }, gb);
  const gw = svgEl("radialGradient", { id: "gw", cx: "35%", cy: "30%", r: "75%" }, defs);
  svgEl("stop", { offset: "0%", "stop-color": "#fff" }, gw);
  svgEl("stop", { offset: "100%", "stop-color": "#d9d4c8" }, gw);

  const grid = opts.grid;
  const sr = cell * 0.47;
  const own = opts.ownership;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const s = grid[r][c];
      const v = vertexOf(r, c);
      if (s !== ".") {
        const dead = own && ((s === "B" && own[r][c] < -0.5) || (s === "W" && own[r][c] > 0.5));
        svgEl("circle", { cx: x(c) + cell * 0.03, cy: y(r) + cell * 0.05, r: sr, fill: "rgba(0,0,0,.25)" }, svg);
        svgEl("circle", {
          cx: x(c), cy: y(r), r: sr,
          fill: s === "B" ? "url(#gb)" : "url(#gw)",
          stroke: s === "W" ? "#b9b2a3" : "none", "stroke-width": 0.08,
          opacity: dead ? 0.45 : 1,
        }, svg);
        if (v === opts.lastMove) {
          svgEl("circle", {
            cx: x(c), cy: y(r), r: sr * 0.38, fill: "none",
            stroke: s === "B" ? "#fff" : "#111", "stroke-width": cell * 0.06,
          }, svg);
        }
      }
      // 종국 후 집 표시
      if (own && Math.abs(own[r][c]) > 0.5 && (s === "." || own[r][c] * (s === "B" ? 1 : -1) < 0)) {
        const w = cell * 0.3;
        svgEl("rect", {
          x: x(c) - w / 2, y: y(r) - w / 2, width: w, height: w,
          fill: own[r][c] > 0 ? "#111" : "#fff", stroke: "#555", "stroke-width": 0.05,
        }, svg);
      }
      if (s === "." && opts.clickColor) {
        const hit = svgEl("rect", {
          class: "hit", x: x(c) - cell / 2, y: y(r) - cell / 2, width: cell, height: cell, fill: "transparent",
        }, svg);
        const ghost = svgEl("circle", {
          cx: x(c), cy: y(r), r: sr, fill: opts.clickColor === "B" ? "#111" : "#fff",
          opacity: 0, "pointer-events": "none",
        }, svg);
        hit.addEventListener("mouseenter", () => ghost.setAttribute("opacity", 0.4));
        hit.addEventListener("mouseleave", () => ghost.setAttribute("opacity", 0));
        hit.addEventListener("click", () => opts.onClick(v));
      }
    }
  }

  for (const m of opts.markers || []) {
    if (m.vertex === "pass") continue;
    const [cx, cy] = at(m.vertex);
    svgEl("circle", { cx, cy, r: sr * 0.62, fill: m.color || "var(--accent)", opacity: 0.88, "pointer-events": "none" }, svg);
    svgEl("text", {
      x: cx, y: cy, "font-size": cell * (m.label.length > 1 ? 0.26 : 0.36), fill: "#fff", "text-anchor": "middle",
      "dominant-baseline": "central", "font-weight": 600, "pointer-events": "none",
    }, svg).textContent = m.label;
  }

  // 예상 진행: 번호 붙은 반투명 돌 (firstColor부터 번갈아). 이미 돌이 있는 자리는 건너뛴다
  if (opts.preview) {
    opts.preview.moves.forEach((v, i) => {
      if (v === "pass") return;
      const [r, c] = pointOf(v);
      if (grid[r][c] !== ".") return;
      const color = (i % 2 === 0) === (opts.preview.firstColor === "B") ? "B" : "W";
      const [cx, cy] = at(v);
      svgEl("circle", {
        cx, cy, r: sr, fill: color === "B" ? "#111" : "#fff", opacity: 0.55,
        stroke: "#555", "stroke-width": 0.06, "pointer-events": "none",
      }, svg);
      svgEl("text", {
        x: cx, y: cy, "font-size": cell * 0.34, fill: color === "B" ? "#fff" : "#111",
        "text-anchor": "middle", "dominant-baseline": "central", "pointer-events": "none",
      }, svg).textContent = i + 1;
    });
  }
}
