import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard calls the API on same-origin paths (/score, /demo/inject, ...).
// In development the API runs separately on :8000, so proxy those paths rather
// than hardcoding a host — that way the built bundle works unchanged when the
// dashboard is served from the same origin as the API.
const API_PATHS = [
  "/score",
  "/batch-score",
  "/transactions",
  "/demo",
  "/health",
];

// The one port this dashboard is served on, in dev and in preview alike.
const PORT = 5173;

export default defineConfig({
  plugins: [react()],
  server: {
    port: PORT,
    // Without this, `port` is only a preference: a busy 5173 sends vite walking
    // up to 5174, 5175, ... and the URL changes under you between runs. With it
    // vite refuses to start instead, which is the honest failure — the port is
    // occupied, and silently serving somewhere else only hides that.
    strictPort: true,
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [
        path,
        { target: "http://127.0.0.1:8000", changeOrigin: true },
      ]),
    ),
  },
  // `npm run preview` serves the built bundle and otherwise picks 4173. Pinning
  // it to the same port means the dev and production-build URLs are one URL.
  preview: {
    port: PORT,
    strictPort: true,
  },
});
