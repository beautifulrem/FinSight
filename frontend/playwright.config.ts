import { defineConfig } from "@playwright/test";

/**
 * Browser regression tests (round 12): real Chrome against the offline FastAPI server, which serves the committed
 * build (query_intelligence/web/dist). Run `pnpm run build` first so the tests see the current UI.
 *
 *   FINSIGHT_PYTHON=../.venv/bin/python pnpm run e2e
 *
 * The server is started here with every live source off and traces off, so answers come from the offline snapshot
 * and are the same on every machine. `FINSIGHT_E2E_PORT` moves it off 8905.
 */
const port = Number(process.env.FINSIGHT_E2E_PORT ?? 8905);
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
    // loading the NLU models takes about a minute on a laptop
    timeout: 240_000,
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
      // no LLM: the deterministic path answers, so the expected numbers hold on every machine
      DEEPSEEK_API_KEY: "",
    },
  },
});
