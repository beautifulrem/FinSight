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
  await expect(first.locator(".answer-text")).toContainText("ROE 29.4%");
  const second = await ask(page, "贵州茅台呢", 2);
  await expect(second.locator(".answer-text")).toContainText("ROE 33%");
  const third = await ask(page, "两者差几个点", 3);
  const answer = third.locator(".answer-text");
  await expect(answer).toContainText("两者相差 3.6 个百分点（贵州茅台更高）");

  // The citations that close the gap sentence: everything between "3.6 个百分点" and the next "。".
  const cited = await answer.evaluate((root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    let inGap = false;
    const ids: string[] = [];
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent ?? "";
        if (!inGap && text.includes("3.6 个百分点")) inGap = true;
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
  await expect(relation.locator(".claim-actual")).toHaveText("24.6 倍");
  await expect(relation.locator(".claim-reference")).toHaveText("白酒行业 27.3 倍");

  const stated = report.locator('.claim-check[data-kind="stated_reference"]');
  await expect(stated).toHaveAttribute("data-status", "contradicted");
  await expect(stated.locator(".claim-stated")).toHaveText("说法给出的数值");
  await expect(stated.locator(".claim-target-name")).toContainText("白酒行业平均");
  await expect(stated.locator(".claim-claimed")).toHaveText("30 倍");
  await expect(stated.locator(".claim-actual")).toHaveText("27.3 倍");

  expect(errors).toEqual([]);
});
