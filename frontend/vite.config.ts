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
    reportCompressedSize: true,
    // Vite's default 500 kB chunk warning applies. Libraries that change rarely get their own
    // long-cached chunks (Vite 8 / Rolldown `codeSplitting.groups`, which replaces `manualChunks`);
    // the price chart and the settings dialog are loaded on demand with `React.lazy`.
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: "react", test: /[\\/]node_modules[\\/](?:\.pnpm[\\/])?(?:react|react-dom|scheduler)[@\\/]/, priority: 30 },
            { name: "radix", test: /[\\/]node_modules[\\/](?:\.pnpm[\\/])?(?:@radix-ui|radix-ui|@floating-ui)[+@\\/]/, priority: 20 },
            { name: "motion", test: /[\\/]node_modules[\\/](?:\.pnpm[\\/])?(?:motion|framer-motion|motion-dom|motion-utils)[@\\/]/, priority: 20 },
          ],
        },
      },
    },
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
    // (round 13, I17) Wall-clock budgets, not performance checks: on a loaded host (load 35-80, several suites at
    // once) every jsdom test runs 5-25x slower and the first user-event test of a file passed 5 s. A hang still fails.
    testTimeout: 30_000,
    css: false,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
