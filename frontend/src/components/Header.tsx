import { Languages, Moon, PanelRight, Plus, Settings, Sun } from "lucide-react";

import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";

import { Button } from "./ui/button";
import { Tooltip } from "./ui/tooltip";

export type AppStatus = "ready" | "running" | "waiting" | "error" | "offline";

interface Props {
  status: AppStatus;
  dark: boolean;
  onToggleTheme: () => void;
  onToggleLang: () => void;
  onNewSession: () => void;
  onSettings: () => void;
  onInspector: () => void;
  canInspect: boolean;
}

const STATUS_KEY = {
  ready: "app.ready",
  running: "app.running",
  waiting: "app.waiting",
  error: "app.error",
  offline: "app.offline",
} as const;

export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="8" className="fill-cobalt" />
      <path d="M9 22V10h10M9 16h7" className="stroke-cobalt-ink" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="22.5" cy="21.5" r="3.2" className="fill-gilt" />
    </svg>
  );
}

export function Header({ status, dark, onToggleTheme, onToggleLang, onNewSession, onSettings, onInspector, canInspect }: Props) {
  const { lang, t } = useI18n();
  return (
    <header className="app-header flex h-14 shrink-0 items-center gap-2 border-b border-line bg-surface/80 px-3 backdrop-blur-md sm:px-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <Logo className="size-7 shrink-0" />
        <div className="min-w-0 leading-tight">
          <div className="text-[15px] font-semibold tracking-tight">FinSight</div>
          <div className="hidden truncate text-[11.5px] text-muted sm:block">{t("app.tagline")}</div>
        </div>
      </div>
      <div
        id="status-pill"
        role="status"
        data-status={status}
        className={cn(
          "ml-1 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[12px] whitespace-nowrap sm:ml-3",
          status === "error" || status === "offline" ? "border-up/30 text-up" : status === "waiting" ? "border-gilt/40 text-gilt" : "border-line text-muted",
        )}
      >
        <span
          aria-hidden
          className={cn(
            "size-1.5 rounded-full",
            status === "ready" && "bg-down",
            status === "running" && "animate-pulse bg-cobalt",
            status === "waiting" && "bg-gilt",
            (status === "error" || status === "offline") && "bg-up",
          )}
        />
        {t(STATUS_KEY[status])}
      </div>
      <nav className="ml-auto flex items-center gap-0.5" aria-label="toolbar">
        <Tooltip content={t("header.inspector")}>
          <Button size="icon" className="lg:hidden" onClick={onInspector} disabled={!canInspect} aria-label={t("header.inspector")}>
            <PanelRight />
          </Button>
        </Tooltip>
        <Tooltip content={t("header.newSession")}>
          <Button id="new-session" size="md" className="max-sm:size-9 max-sm:px-0" onClick={onNewSession} aria-label={t("header.newSession")}>
            <Plus />
            <span className="max-sm:hidden">{t("header.newSession")}</span>
          </Button>
        </Tooltip>
        <Tooltip content={t("header.lang")}>
          <Button id="lang-toggle" size="icon" onClick={onToggleLang} aria-label={t("header.lang")}>
            <Languages />
            <span className="sr-only">{lang}</span>
          </Button>
        </Tooltip>
        <Tooltip content={t("header.theme")}>
          <Button id="theme-toggle" size="icon" onClick={onToggleTheme} aria-label={t("header.theme")} aria-pressed={dark}>
            {dark ? <Sun /> : <Moon />}
          </Button>
        </Tooltip>
        <Tooltip content={t("header.settings")}>
          <Button id="settings-toggle" size="icon" onClick={onSettings} aria-label={t("header.settings")}>
            <Settings />
          </Button>
        </Tooltip>
      </nav>
    </header>
  );
}
