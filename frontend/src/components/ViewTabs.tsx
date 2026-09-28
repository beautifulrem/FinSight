import { Tabs } from "radix-ui";
import { MessagesSquare, SearchCheck } from "lucide-react";

import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";

export type AppView = "chat" | "check";

const VIEWS: { value: AppView; icon: typeof MessagesSquare; label: "view.chat" | "view.check" }[] = [
  { value: "chat", icon: MessagesSquare, label: "view.chat" },
  { value: "check", icon: SearchCheck, label: "view.check" },
];

/** Ask / Fact-check switch in the header; the panels are `Tabs.Content` in App (arrow keys move between tabs). */
export function ViewTabs({ className }: { className?: string }) {
  const { t } = useI18n();
  return (
    <Tabs.List aria-label={t("view.label")} className={cn("view-tabs inline-flex rounded-lg bg-surface-2 p-0.5", className)}>
      {VIEWS.map(({ value, icon: Icon, label }) => (
        <Tabs.Trigger
          key={value}
          value={value}
          data-view={value}
          className={cn(
            "inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md px-3 text-[13px] font-medium whitespace-nowrap text-muted transition-colors",
            "hover:text-ink data-[state=active]:bg-surface data-[state=active]:text-cobalt data-[state=active]:shadow-sm",
          )}
        >
          <Icon className="size-3.5" aria-hidden />
          {t(label)}
        </Tabs.Trigger>
      ))}
    </Tabs.List>
  );
}
