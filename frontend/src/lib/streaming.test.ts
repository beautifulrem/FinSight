import { answerEdited, streamingText } from "./streaming";

describe("streaming helpers", () => {
  it("hides complete and half-received citation markers while streaming", () => {
    expect(streamingText("茅台收于 1413 元 [price_600519.SH]，PE 24.6 [fundam")).toBe("茅台收于 1413 元，PE 24.6 ");
  });

  it("detects when the verified answer differs from the streamed draft", () => {
    expect(answerEdited("PE 24.6", "PE 24.6 [fundamental_600519.SH]")).toBe(false);
    expect(answerEdited("PE 24.6", "PE 24.6，条件性判断")).toBe(true);
    expect(answerEdited(undefined, "anything")).toBe(false);
  });
});
