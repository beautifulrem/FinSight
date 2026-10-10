import { expect, test, type Page } from "@playwright/test";

/**
 * Round-12 browser regressions against the offline server (see playwright.config.ts):
 * - a three-turn gap session in Chinese computes the ROE gap and cites both companies' fundamentals;
 * - a fact-check with a stated industry average shows the relation row and the "stated in the claim" row.
 * Every test also fails on a console error.
 */

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}

async function openFresh(page: Page) {
  // a new browser context has empty storage: Chinese UI, chat view, new session id
  await page.goto("/");
  // a reload before the first question must not ask the server for a session it never saw (a 404 in the console)
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
}

async function ask(page: Page, question: string, turn: number) {
  const input = page.locator("#query-input");
  await input.fill(question);
  await input.press("Enter");
  const card = page.locator(".turn").nth(turn - 1).locator(".answer-card").last();
  await expect(card.locator(".answer-text")).toBeVisible({ timeout: 60_000 });
  return card;
}

test("three-turn gap session (zh) shows the computed gap with two citations", async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await openFresh(page);

  const first = await ask(page, "五粮液的ROE是多少", 1);
  await expect(first.locator(".answer-text")).toContainText("ROE（加权） 6.89%");
  const second = await ask(page, "贵州茅台呢", 2);
  await expect(second.locator(".answer-text")).toContainText("ROE（加权） 32.53%");
  const third = await ask(page, "两者差几个点", 3);
  const answer = third.locator(".answer-text");
  await expect(answer).toContainText("两者相差 25.64 个百分点（贵州茅台更高）");

  // The citations that close the gap sentence: everything between "25.64 个百分点" and the next "。".
  const cited = await answer.evaluate((root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let inGap = false;
    const ids: string[] = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent ?? "";
        if (!inGap && text.includes("25.64 个百分点")) inGap = true;
        else if (inGap && text.includes("。")) break;
      } else if (inGap && (node as Element).classList.contains("citation-chip")) {
        ids.push((node as Element).getAttribute("data-evidence-id") ?? "");
      }
    }
    return ids;
  });
  expect(cited).toEqual(["fundamental_000858.SZ", "fundamental_600519.SH"]);

  // The run inspector names the computed comparison in words, not as a raw route-reason chip (H14).
  await page.getByRole("tab", { name: "过程" }).click();
  const inspector = page.getByRole("complementary");
  await expect(inspector.getByText("计算ROE差值：五粮液 对比 贵州茅台")).toBeVisible();
  await expect(inspector).not.toContainText("Frame difference");

  // A reload restores the three turns under a banner that counts them, and shows them (H14).
  await page.reload();
  const history = page.locator(".session-history");
  await expect(history).toContainText("已恢复本会话的 3 轮历史");
  await expect(history).toHaveAttribute("open", "");
  await expect(history.getByText("两者差几个点")).toBeVisible();

  expect(errors).toEqual([]);
});

test("fact-check with a stated industry average shows the relation and the stated-average row", async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await openFresh(page);
  await page.getByRole("tab", { name: "核查" }).click();
  await page.locator("#claim-input").fill("茅台市盈率比白酒行业平均的30倍低");
  await page.locator("#claim-submit").click();

  const report = page.locator(".claim-report");
  await expect(report).toBeVisible({ timeout: 60_000 });
  const relation = report.locator('.claim-check[data-kind="relation"]');
  await expect(relation).toHaveAttribute("data-status", "supported");
  await expect(relation.locator(".claim-claimed")).toContainText("贵州茅台 < 白酒行业");
  await expect(relation.locator(".claim-actual")).toHaveText("21.44 倍");
  await expect(relation.locator(".claim-reference")).toHaveText("白酒行业 27.3 倍");

  const stated = report.locator('.claim-check[data-kind="stated_reference"]');
  await expect(stated).toHaveAttribute("data-status", "contradicted");
  await expect(stated.locator(".claim-stated")).toHaveText("说法给出的数值");
  await expect(stated.locator(".claim-target-name")).toContainText("白酒行业平均");
  await expect(stated.locator(".claim-claimed")).toHaveText("30 倍");
  await expect(stated.locator(".claim-actual")).toHaveText("27.3 倍");

  // (round 13) the top bar stays put: only the fact-check panel scrolls, never the page
  const header = page.locator(".app-header");
  await page.mouse.move(400, 500);
  await page.mouse.wheel(0, 4000);
  await page.keyboard.press("End");
  const layout = await page.evaluate(() => ({
    pageOverflow: document.scrollingElement!.scrollHeight - window.innerHeight,
    pageScroll: window.scrollY,
    panelScroll: document.querySelector(".check-main")!.scrollTop,
  }));
  expect(layout.pageOverflow).toBeLessThanOrEqual(0);
  expect(layout.pageScroll).toBe(0);
  expect(layout.panelScroll).toBeGreaterThan(0);
  expect((await header.boundingBox())?.y).toBe(0);
  await expect(header).toBeInViewport();

  expect(errors).toEqual([]);
});
