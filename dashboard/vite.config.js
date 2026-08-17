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

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      API_PATHS.map((path) => [
        path,
        { target: "http://127.0.0.1:8000", changeOrigin: true },
      ]),
    ),
  },
});
