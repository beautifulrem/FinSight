import { distinctKeyPoints } from "./view";

describe("distinctKeyPoints", () => {
  it("hides key points that only repeat the answer", () => {
    const answer = "收盘价 1409.5 [price_1]。PE 24.6 [fundamental_1]。";
    expect(distinctKeyPoints(answer, ["收盘价 1409.5 [price_1]。", "PE 24.6 [fundamental_1]。"])).toEqual([]);
  });

  it("keeps key points that add information", () => {
    expect(distinctKeyPoints("走势偏弱。", ["RSI 10.9"])).toEqual(["RSI 10.9"]);
  });
});
