import { defineConfig } from "@playwright/test";

/**
 * Browser regression tests (round 12): real Chrome against the offline FastAPI server, which serves the committed
 * build (query_intelligence/web/dist). Run `pnpm run build` first so the tests see the current UI.
 *
 *   FINSIGHT_PYTHON=../.venv/bin/python pnpm run e2e
 *
 * The server is started here with every live source off and traces off, so answers come from the offline snapshot
 * and are the same on every machine. Both offline layers are pinned on (the extension and the corrections layer, the
 * shipped default; round-13 review M17), so a variable inherited from the shell cannot switch the tests to the seed. `FINSIGHT_E2E_PORT` moves it off 8905. `FINSIGHT_E2E_START_TIMEOUT_MS` sets how
 * long to wait for it to come up (default 10 min locally, 4 min in CI).
 */
const port = Number(process.env.FINSIGHT_E2E_PORT ?? 8905);
// (round 13, I18) Start-up is the NLU model load: about a minute on an idle laptop and 55 s in CI, but more than the
// old 4 min on a loaded host (round-9 review, load ~35), where the run failed before any test started. CI keeps 4 min
// so a server that really hangs fails fast; locally the default is 10 min, and the variable overrides both.
const startTimeout = Number(process.env.FINSIGHT_E2E_START_TIMEOUT_MS || (process.env.CI ? 240_000 : 600_000));
if (!Number.isFinite(startTimeout) || startTimeout <= 0) {
  throw new Error(`FINSIGHT_E2E_START_TIMEOUT_MS must be a positive number of milliseconds, got ${process.env.FINSIGHT_E2E_START_TIMEOUT_MS}`);
}
const python = process.env.FINSIGHT_PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: "chrome",
    headless: true,
    locale: "zh-CN",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: `${python} -m uvicorn query_intelligence.api.app:create_app --factory --host 127.0.0.1 --port ${port}`,
    cwd: "..",
    url: `http://127.0.0.1:${port}/health`,
    timeout: startTimeout,
    reuseExistingServer: !process.env.CI,
    gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
    stdout: "ignore",
    stderr: "pipe",
    env: {
      QI_USE_LIVE_MARKET: "0",
      QI_USE_LIVE_MACRO: "0",
      QI_USE_LIVE_NEWS: "0",
      QI_USE_LIVE_ANNOUNCEMENT: "0",
      QI_AGENT_TRACE_DIR: "off",
      // what the product serves by default: v1 seed + 2026-09-30 extension + FY2025 corrections layer
      QI_OFFLINE_SNAPSHOT_EXT: "true",
      QI_OFFLINE_SNAPSHOT_FIX: "true",
      // no LLM: the deterministic path answers, so the expected numbers hold on every machine
      DEEPSEEK_API_KEY: "",
    },
  },
});
