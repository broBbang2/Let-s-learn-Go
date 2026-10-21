import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 개발 중: npm run dev → http://localhost:5173
// API·기보 보관함은 Python 서버(localhost:8000)로 넘긴다
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": backend,
      "/library": backend,
      "/board.js": backend,
      "/classic": backend,
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
