import { History } from "lucide-react";

import { stripCitations } from "@/lib/citations";
import { useI18n } from "@/lib/i18n";
import type { SessionTurn } from "@/lib/types";

/** Earlier turns of a restored session: shown compactly, without their evidence trail. */
export function HistoryView({ turns }: { turns: SessionTurn[] }) {
  const { t } = useI18n();
  return (
    <details className="session-history group rounded-xl border border-dashed border-line px-4 py-2.5">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-[13px] text-muted">
        <History className="size-3.5" aria-hidden />
        {t("answer.restored", { n: turns.length })}
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
