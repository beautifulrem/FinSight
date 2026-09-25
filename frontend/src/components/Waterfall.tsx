import { cn } from "@/lib/cn";
import { formatMs } from "@/lib/format";
import { nodeLabel, toolLabel, useI18n } from "@/lib/i18n";
import type { TraceNode } from "@/lib/trace";

interface Bar {
  key: string;
  label: string;
  start: number; // seconds
  ms: number;
  kind: "node" | "tool" | "error";
}

/** Timing waterfall of graph nodes and the tool calls inside them (from `spans` and `tool_calls`). */
export function Waterfall({ nodes }: { nodes: TraceNode[] }) {
  const { lang, t } = useI18n();
  const bars: Bar[] = [];
  for (const node of nodes) {
    if (node.startedAt !== undefined && node.durationMs !== undefined) {
      bars.push({ key: node.id, label: nodeLabel(lang, node.node), start: node.startedAt, ms: node.durationMs, kind: "node" });
    }
    for (const tool of node.tools) {
      if (tool.startedAt === undefined || tool.latencyMs === undefined) continue;
      bars.push({
        key: `${node.id}-${tool.id}`,
        label: toolLabel(lang, tool.tool),
        start: tool.startedAt,
        ms: tool.latencyMs,
        kind: tool.status === "error" ? "error" : "tool",
      });
    }
  }
  if (bars.length < 2) return null;
  const t0 = Math.min(...bars.map((bar) => bar.start));
  const t1 = Math.max(...bars.map((bar) => bar.start + bar.ms / 1000));
  const total = Math.max(t1 - t0, 0.001);
  return (
    <figure className="waterfall space-y-1">
      <figcaption className="mb-1.5 flex justify-between text-[12px] text-muted">
        <span>{t("trace.waterfall")}</span>
        <span className="tabular-nums">{formatMs(total * 1000)}</span>
      </figcaption>
      {bars.map((bar) => {
        const left = ((bar.start - t0) / total) * 100;
        const width = Math.max((bar.ms / 1000 / total) * 100, 0.8);
        return (
          <div key={bar.key} className="grid grid-cols-[6.5rem_1fr_3.6rem] items-center gap-2 text-[11.5px]">
            <span className={cn("truncate", bar.kind === "node" ? "text-ink" : "pl-2 text-muted")}>{bar.label}</span>
            <span className="relative h-2.5 rounded-sm bg-surface-2">
              <span
                className={cn(
                  "absolute inset-y-0 rounded-sm",
                  bar.kind === "node" && "bg-cobalt/80",
                  bar.kind === "tool" && "bg-gilt/80",
                  bar.kind === "error" && "bg-up/80",
                )}
                style={{ left: `${Math.min(left, 99.2)}%`, width: `${Math.min(width, 100 - Math.min(left, 99.2))}%` }}
              />
            </span>
            <span className="text-right text-faint tabular-nums">{formatMs(bar.ms)}</span>
          </div>
        );
      })}
    </figure>
  );
}
