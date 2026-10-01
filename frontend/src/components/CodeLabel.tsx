import type { ComponentProps } from "react";

import { cn } from "@/lib/cn";
import { humanizeCode, localizeNames, type CodeKind } from "@/lib/codes";
import { useI18n } from "@/lib/i18n";

import { Badge } from "./ui/badge";
import { Tooltip } from "./ui/tooltip";

function RawCode({ code }: { code: string }) {
  const { t } = useI18n();
  return (
    <span>
      <span className="opacity-70">{t("code.raw")}: </span>
      <code className="font-mono break-all">{code}</code>
    </span>
  );
}

/** A humanised code as a badge; the raw code is in the tooltip (hover or keyboard focus). */
export function CodeBadge({
  code,
  kind,
  names,
  className,
  ...props
}: { code: string; kind?: CodeKind; names?: ReadonlyMap<string, string> } & Omit<ComponentProps<typeof Badge>, "children">) {
  const { lang } = useI18n();
  return (
    <Tooltip content={<RawCode code={code} />}>
      {/* Route reasons are sentences ("Computed the ROE gap: Wuliangye vs Kweichow Moutai"): they wrap instead of
          being cut off, since the tooltip shows the raw code, not the label. */}
      <Badge
        tabIndex={0}
        data-code={code}
        className={cn("code-label max-w-full", kind === "reason" ? "whitespace-normal break-words" : "truncate", className)}
        {...props}
      >
        {localizeNames(lang, humanizeCode(lang, code, kind), names)}
      </Badge>
    </Tooltip>
  );
}

/** Inline humanised text (e.g. inside a limitations list) with the raw code in a tooltip. */
export function CodeText({ code, label, className }: { code: string; label?: string; className?: string }) {
  const { lang } = useI18n();
  return (
    <Tooltip content={<RawCode code={code} />}>
      <span
        tabIndex={0}
        data-code={code}
        className={cn("code-label underline decoration-dotted decoration-from-font underline-offset-2", className)}
      >
        {label ?? humanizeCode(lang, code)}
      </span>
    </Tooltip>
  );
}
