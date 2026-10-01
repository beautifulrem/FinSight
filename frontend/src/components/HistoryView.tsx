import { ChevronDown, History } from "lucide-react";
import { useState } from "react";

import { stripCitations } from "@/lib/citations";
import { useI18n } from "@/lib/i18n";
import type { SessionTurn } from "@/lib/types";

/**
 * Earlier turns of a restored session: shown compactly, without their evidence trail. Open by default, so the banner
 * "Restored 5 earlier turns" is never shown above an empty page (round 12, H14); the reader can fold them away.
 */
export function HistoryView({ turns }: { turns: SessionTurn[] }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(true);
  return (
    <details
      className="session-history group rounded-xl border border-dashed border-line px-4 py-2.5"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 text-[13px] text-muted">
        <History className="size-3.5" aria-hidden />
        <span>{turns.length === 1 ? t("answer.restoredOne") : t("answer.restored", { n: turns.length })}</span>
        <span className="ml-auto inline-flex items-center gap-1 text-[12px] text-faint">
          {open ? t("answer.historyHide") : t("answer.historyShow")}
          <ChevronDown className="size-3.5 transition-transform group-open:rotate-180 motion-reduce:transition-none" aria-hidden />
        </span>
      </summary>
      <ol className="mt-2 space-y-3 border-t border-line pt-3">
        {turns.map((turn, i) => (
          <li key={i} className="space-y-1 text-[13.5px]">
            <p className="font-medium text-ink">{turn.query}</p>
            {turn.answer && <p className="line-clamp-3 text-muted">{stripCitations(turn.answer)}</p>}
          </li>
        ))}
      </ol>
    </details>
  );
}
