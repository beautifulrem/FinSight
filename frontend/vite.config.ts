/// <reference types="vitest/config" />
import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build is written into the Python package so FastAPI can serve it without Node:
// `/` renders dist/index.html and `/static/app/*` serves the hashed assets.
const outDir = fileURLToPath(new URL("../query_intelligence/web/dist", import.meta.url));
const backend = process.env.FINSIGHT_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  base: "/static/app/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    outDir,
    emptyOutDir: true,
    target: "es2022",
    // react-dom alone is ~200 kB minified; the whole app entry is ~165 kB gzipped.
    chunkSizeWarningLimit: 560,
    reportCompressedSize: true,
  },
  server: {
    port: 5173,
    // `pnpm dev` proxies API calls to a locally running uvicorn.
    proxy: Object.fromEntries(["/chat", "/agent", "/health"].map((path) => [path, { target: backend, changeOrigin: true }])),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
