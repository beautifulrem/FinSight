import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { I18nContext, makeTranslate } from "@/lib/i18n";

import { SettingsDialog } from "./SettingsDialog";

function renderDialog(props: Partial<React.ComponentProps<typeof SettingsDialog>> = {}) {
  const onApiKey = vi.fn();
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify({ session_id: "s1", turns: [], pending_clarification: null }), { status: 200 }),
  );
  render(
    <I18nContext.Provider value={{ lang: "en", t: makeTranslate("en") }}>
      <SettingsDialog
        open
        onOpenChange={() => {}}
        apiKey="k1"
        rememberKey={false}
        onApiKey={onApiKey}
        sessionId="s1"
        lang="en"
        onLang={() => {}}
        theme="system"
        onTheme={() => {}}
        {...props}
      />
    </I18nContext.Provider>,
  );
  return onApiKey;
}

describe("SettingsDialog API key (C19)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("remembering the key is an explicit, unticked-by-default opt-in with the trade-off stated", async () => {
    const onApiKey = renderDialog();
    const remember = screen.getByRole("checkbox", { name: /remember the api key on this device/i });

    expect(remember).not.toBeChecked();
    expect(screen.getByText(/any script running on this page can read it/i)).toBeInTheDocument();
    expect(screen.getByText(/kept for this tab only by default/i)).toBeInTheDocument();

    await userEvent.click(remember);
    expect(onApiKey).toHaveBeenLastCalledWith("k1", true);
    await userEvent.click(remember);
    expect(onApiKey).toHaveBeenLastCalledWith("k1", false);
  });

  it("shows the stored choice and saves edits with it", async () => {
    const onApiKey = renderDialog({ rememberKey: true });

    expect(screen.getByRole("checkbox", { name: /remember/i })).toBeChecked();
    await userEvent.clear(screen.getByLabelText("API key"));
    await userEvent.type(screen.getByLabelText("API key"), " k2 ");
    await userEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(onApiKey).toHaveBeenLastCalledWith("k2", true);
  });
});
