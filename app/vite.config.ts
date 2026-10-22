import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// 개발 중: npm run dev → http://localhost:5173
// API는 Python 서버(localhost:8000)로 넘긴다
const backend = "http://127.0.0.1:8000";

// 서버와 같은 주소(/library)로 기보 보관함 페이지를 열 수 있게
const libraryRoute: Plugin = {
  name: "library-route",
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      const r = req as { url?: string }; // @types/node 없이
      if (r.url === "/library" || r.url?.startsWith("/library?")) r.url = r.url.replace("/library", "/library.html");
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), libraryRoute],
  server: {
    proxy: {
      "/api": backend,
      "/board.js": backend,
      "/classic": backend,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: { input: { main: "index.html", library: "library.html" } },
  },
});
