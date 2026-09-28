import { sourceNameLabel } from "./i18n";

describe("sourceNameLabel", () => {
  it("localises source ids and generic labels in English and keeps Chinese labels in Chinese (B27)", () => {
    const snapshot = { source: "offline_snapshot", source_label: "离线快照" };
    expect(sourceNameLabel("zh", snapshot)).toBe("离线快照");
    expect(sourceNameLabel("en", snapshot)).toBe("Offline snapshot");
    expect(sourceNameLabel("en", { source: "sina.kline", source_label: "新浪财经行情" })).toBe("Sina Finance quotes");
    expect(sourceNameLabel("en", null, "seed")).toBe("Offline seed data");
    expect(sourceNameLabel("en", null, "每日经济新闻")).toBe("每日经济新闻"); // a publisher's name stays as published
    expect(sourceNameLabel("en", null, null)).toBeUndefined();
  });
});
