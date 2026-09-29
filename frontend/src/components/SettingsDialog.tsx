import { Dialog, ToggleGroup } from "radix-ui";
import { Eye, EyeOff, RefreshCw, X } from "lucide-react";
import { useEffect, useState } from "react";

import { fetchSession } from "@/lib/api";
import { cn } from "@/lib/cn";
import { useI18n, type Lang } from "@/lib/i18n";
import type { SessionInfo } from "@/lib/types";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

export type ThemePref = "light" | "dark" | "system";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  apiKey: string;
  /** Whether the key is kept in localStorage across restarts (opt-in); otherwise sessionStorage only. */
  rememberKey: boolean;
  onApiKey: (key: string, remember: boolean) => void;
  sessionId: string;
  lang: Lang;
  onLang: (lang: Lang) => void;
  theme: ThemePref;
  onTheme: (theme: ThemePref) => void;
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(next) => next && onChange(next as T)}
      aria-label={label}
      className="inline-flex rounded-lg bg-surface-2 p-0.5"
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.value}
          value={option.value}
          className="h-7 rounded-md px-3 text-[13px] text-muted data-[state=on]:bg-surface data-[state=on]:text-ink data-[state=on]:shadow-sm"
        >
          {option.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

export function SettingsDialog(props: Props) {
  const { t } = useI18n();
  return (
    <Dialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-[2px]" />
        <Dialog.Content
          className="settings-dialog fixed top-1/2 left-1/2 z-50 flex max-h-[88dvh] w-[min(34rem,calc(100vw-1.5rem))] -translate-x-1/2 -translate-y-1/2 flex-col rounded-2xl border border-line bg-surface shadow-2xl outline-none"
          aria-describedby={undefined}
        >
          <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
            <Dialog.Title className="text-[16px] font-semibold">{t("settings.title")}</Dialog.Title>
            <Dialog.Close asChild>
              <Button size="icon-sm" aria-label={t("inspector.close")}>
                <X />
              </Button>
            </Dialog.Close>
          </div>
          {/* Mounted only while open, so the draft key and session memory are fresh each time. */}
          <SettingsBody {...props} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function SettingsBody({ apiKey, rememberKey, onApiKey, sessionId, lang, onLang, theme, onTheme }: Props) {
  const { t } = useI18n();
  const [draft, setDraft] = useState(apiKey);
  const [remember, setRemember] = useState(rememberKey);
  const [reveal, setReveal] = useState(false);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchSession(sessionId, { apiKey, signal: controller.signal })
      .then((info) => {
        setSession(info);
        setError(null);
      })
      .catch((exc: unknown) => {
        if (!controller.signal.aborted) setError(exc instanceof Error ? exc.message : String(exc));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [sessionId, apiKey, reloads]);

  const reload = () => {
    setLoading(true);
    setReloads((n) => n + 1);
  };

  return (
    <>
      <div className="scrollbar-thin space-y-6 overflow-y-auto px-5 py-5">
        <section className="space-y-2">
          <label htmlFor="api-key-input" className="text-[13.5px] font-medium">
            {t("settings.apiKey")}
          </label>
          <div className="flex gap-2">
            <input
              id="api-key-input"
              type={reveal ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={() => onApiKey(draft.trim(), remember)}
              className="h-9 min-w-0 flex-1 rounded-lg border border-line bg-bg px-3 font-mono text-[13px] outline-none focus:border-cobalt"
            />
            <Button variant="outline" size="icon" aria-label={reveal ? t("settings.hide") : t("settings.show")} onClick={() => setReveal(!reveal)}>
              {reveal ? <EyeOff /> : <Eye />}
            </Button>
          </div>
          <p className="text-[12.5px] text-muted">{t("settings.apiKeyHint")}</p>
          <label className="flex items-start gap-2 text-[13px]">
            <input
              id="api-key-remember"
              type="checkbox"
              checked={remember}
              onChange={(event) => {
                setRemember(event.target.checked);
                onApiKey(draft.trim(), event.target.checked);
              }}
              className="mt-0.5 size-4 accent-cobalt"
            />
            <span>
              {t("settings.rememberKey")}
              <span className="block text-[12px] text-faint">{t("settings.rememberKeyHint")}</span>
            </span>
          </label>
        </section>

        <section className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-[13.5px] font-medium">{t("settings.language")}</span>
          <Segmented<Lang>
            label={t("settings.language")}
            value={lang}
            onChange={onLang}
            options={[
              { value: "zh", label: "中文" },
              { value: "en", label: "English" },
            ]}
          />
        </section>
        <section className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-[13.5px] font-medium">{t("settings.theme")}</span>
          <Segmented<ThemePref>
            label={t("settings.theme")}
            value={theme}
            onChange={onTheme}
            options={[
              { value: "light", label: t("settings.light") },
              { value: "dark", label: t("settings.dark") },
              { value: "system", label: t("settings.system") },
            ]}
          />
        </section>

        <section className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-[13.5px] font-medium">{t("settings.memory")}</h3>
            <Button size="sm" onClick={reload} disabled={loading}>
              <RefreshCw className={cn(loading && "motion-safe:animate-spin")} />
              {t("settings.refresh")}
            </Button>
          </div>
          <p className="text-[12.5px] text-muted">{t("settings.memoryHint")}</p>
          <p className="text-[12px] text-faint">
            {t("settings.session")}: <code id="session-id" className="font-mono break-all text-muted">{sessionId}</code>
          </p>
          {error ? (
            <p className="text-[12.5px] text-up">{t("settings.memoryError", { e: error })}</p>
          ) : session && session.turns.length ? (
            <ol className="session-memory space-y-1.5">
              {session.turns.map((turn, i) => (
                <li key={i} className="rounded-lg border border-line bg-bg px-3 py-2 text-[12.5px]">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-faint">#{i + 1}</span>
                    <span className="min-w-0 flex-1 truncate font-medium">{turn.query}</span>
                    {turn.route && <Badge tone="cobalt">{turn.route}</Badge>}
                  </div>
                  {(turn.entities ?? []).filter((entity) => entity.symbol).length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {(turn.entities ?? [])
                        .filter((entity) => entity.symbol)
                        .map((entity) => (
                          <Badge key={`${entity.name}-${entity.symbol}`}>
                            {entity.name} {entity.symbol}
                          </Badge>
                        ))}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-[12.5px] text-faint">{loading ? "…" : t("settings.memoryEmpty")}</p>
          )}
        </section>
      </div>
      <div className="flex justify-end border-t border-line px-5 py-3">
        <Dialog.Close asChild>
          <Button variant="primary" onClick={() => onApiKey(draft.trim(), remember)}>
            {t("settings.done")}
          </Button>
        </Dialog.Close>
      </div>
    </>
  );
}

export default SettingsDialog;
