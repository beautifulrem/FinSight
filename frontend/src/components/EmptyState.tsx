import { m as motion } from "motion/react";
import { FileSearch, Route, ShieldCheck, Wrench } from "lucide-react";

import { useI18n, type MessageKey } from "@/lib/i18n";

const STAGES: { key: MessageKey; icon: typeof Route }[] = [
  { key: "pipeline.nlu", icon: Route },
  { key: "pipeline.tools", icon: Wrench },
  { key: "pipeline.verify", icon: FileSearch },
  { key: "pipeline.compliance", icon: ShieldCheck },
];

const EXAMPLES: MessageKey[] = ["empty.q1", "empty.q2", "empty.q3", "empty.q4"];

/** `active`: the chat is the visible view, so its title is the page's one h1 (the hidden fact-check view's is not). */
export function EmptyState({ onAsk, active = true }: { onAsk: (query: string) => void; active?: boolean }) {
  const { t } = useI18n();
  const Title = active ? "h1" : "h2";
  return (
    <section className="empty-state mx-auto flex max-w-2xl flex-col gap-8 px-1 pt-[6vh] pb-6 sm:pt-[9vh]">
      <div className="space-y-3">
        <Title className="text-[26px] leading-tight font-semibold tracking-tight text-balance sm:text-[32px]">{t("empty.title")}</Title>
        <p className="max-w-[36rem] text-[15px] leading-relaxed text-muted">{t("empty.body")}</p>
      </div>

      {/* The agent pipeline, lit up once in order: the one orchestrated moment on the page. */}
      <ol className="pipeline grid grid-cols-2 gap-2 sm:grid-cols-4" aria-label="pipeline">
        {STAGES.map(({ key, icon: Icon }, i) => (
          <motion.li
            key={key}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.12 + i * 0.14, duration: 0.35, ease: "easeOut" }}
            className="relative flex items-center gap-2 rounded-xl border border-line bg-surface px-3 py-2.5"
          >
            <span className="font-mono text-[11px] text-faint">{i + 1}</span>
            <Icon className="size-4 text-cobalt" aria-hidden />
            <span className="text-[13px] font-medium">{t(key)}</span>
          </motion.li>
        ))}
      </ol>

      <div className="space-y-2.5">
        <h2 className="text-[13px] font-medium text-muted">{t("empty.try")}</h2>
        <div className="grid gap-2 sm:grid-cols-2">
          {EXAMPLES.map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => onAsk(t(key))}
              className="example-question group rounded-xl border border-line bg-surface px-4 py-3 text-left text-[14.5px] leading-snug transition-colors hover:border-cobalt hover:bg-cobalt-soft/60"
            >
              <span className="text-ink group-hover:text-cobalt">{t(key)}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
