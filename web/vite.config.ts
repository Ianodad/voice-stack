import { defineConfig } from "vite";

// Dev workflow (optional; production is `npm run build` + `uv run voice-stack web`,
// which serves web/dist at http://localhost:7860):
//
//   1. Start the backend with:  VOICE_STACK_DEV=1 uv run voice-stack web
//   2. Start Vite:              cd web && npm run dev   -> http://localhost:5173
//
// The server rejects any request whose Origin is not its own Host. VOICE_STACK_DEV=1
// makes it also accept Origin http://localhost:5173. The proxy must therefore keep
// the browser's Origin header (we do NOT rewrite it) while Host is rewritten to the
// target (changeOrigin: true sets Host=127.0.0.1:7860, which the Host guard allows).
export default defineConfig({
  server: {
    host: "localhost",
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:7860", changeOrigin: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
