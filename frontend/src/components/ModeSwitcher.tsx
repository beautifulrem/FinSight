import { ToggleGroup } from "radix-ui";
import { Bot, Layers, Sparkles, Workflow } from "lucide-react";

import { cn } from "@/lib/cn";
import { useI18n, type MessageKey } from "@/lib/i18n";
import type { UiMode } from "@/lib/types";

import { Tooltip } from "./ui/tooltip";

const MODES: { value: UiMode; icon: typeof Bot }[] = [
  { value: "auto", icon: Sparkles },
  { value: "agent", icon: Bot },
  { value: "workflow", icon: Workflow },
  { value: "classic", icon: Layers },
];

export function ModeSwitcher({ value, onChange, disabled }: { value: UiMode; onChange: (mode: UiMode) => void; disabled?: boolean }) {
  const { t } = useI18n();
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(next) => next && onChange(next as UiMode)}
      aria-label={t("mode.label")}
      disabled={disabled}
      className="mode-switcher inline-flex rounded-lg bg-surface-2 p-0.5"
    >
      {MODES.map(({ value: mode, icon: Icon }) => (
        <Tooltip key={mode} content={t(`mode.${mode}.hint` as MessageKey)} side="top">
          <ToggleGroup.Item
            value={mode}
            data-mode={mode}
            aria-label={t(`mode.${mode}` as MessageKey)}
            className={cn(
              "inline-flex h-7 items-center gap-1 rounded-md px-2 text-[12.5px] font-medium whitespace-nowrap text-muted transition-colors",
              "hover:text-ink data-[state=on]:bg-surface data-[state=on]:text-cobalt data-[state=on]:shadow-sm",
            )}
          >
            <Icon className="size-3.5" aria-hidden />
            {/* Narrow screens: icons only, plus the label of the active mode. */}
            <span className={mode === value ? undefined : "max-sm:sr-only"}>{t(`mode.${mode}` as MessageKey)}</span>
          </ToggleGroup.Item>
        </Tooltip>
      ))}
    </ToggleGroup.Root>
  );
}
