import { Fragment, type ReactNode } from "react";

import { splitCitations } from "@/lib/citations";
import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";

import { Tooltip } from "./ui/tooltip";

export interface CiteHandler {
  index: Map<string, number>;
  titles: Map<string, string>;
  onCite: (id: string) => void;
  active?: string | null;
}

function bold(text: string, key: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={`${key}-${i}`} className="font-semibold text-ink">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <Fragment key={`${key}-${i}`}>{part}</Fragment>
    ),
  );
}

export function CitationChip({ id, index, cite }: { id: string; index: number; cite: CiteHandler }) {
  const { t } = useI18n();
  const title = cite.titles.get(id);
  const active = cite.active === id;
  return (
    <Tooltip content={<span><span className="font-mono">{id}</span>{title ? <><br />{title}</> : null}</span>} side="top">
      <button
        type="button"
        className={cn(
          "citation-chip mx-0.5 inline-flex h-[1.35em] min-w-[1.9em] -translate-y-px items-center justify-center rounded-[5px] px-1 align-middle font-mono text-[0.72em] leading-none font-medium transition-colors",
          active ? "bg-gilt text-surface" : "bg-cobalt-soft text-cobalt hover:bg-cobalt hover:text-cobalt-ink",
        )}
        aria-label={t("answer.cite", { id: `E${index}` })}
        data-evidence-id={id}
        onClick={() => cite.onCite(id)}
      >
        E{index}
      </button>
    </Tooltip>
  );
}

function Inline({ text, cite, keyPrefix }: { text: string; cite?: CiteHandler; keyPrefix: string }) {
  const { t } = useI18n();
  if (!cite) return <>{bold(text, keyPrefix)}</>;
  return (
    <>
      {splitCitations(text, cite.index).map((segment, i) => {
        const key = `${keyPrefix}-${i}`;
        if (segment.type === "text") return <Fragment key={key}>{bold(segment.text, key)}</Fragment>;
        if (segment.type === "cite") return <CitationChip key={key} id={segment.id} index={segment.index} cite={cite} />;
        return (
          <Tooltip key={key} content={t("answer.invalidCite", { id: segment.id })} side="top">
            <span
              tabIndex={0}
              className="mx-0.5 inline-flex h-[1.35em] items-center rounded-[5px] bg-warn-soft px-1 align-middle font-mono text-[0.72em] text-warn line-through"
            >
              ?
            </span>
          </Tooltip>
        );
      })}
    </>
  );
}

const LIST_ITEM = /^\s*(?:[-*•·]|\d+[.、)])\s+/;

/** Plain answer text with paragraphs, simple lists, **bold** and citation chips. No HTML is interpreted. */
export function RichText({ text, cite, className }: { text: string; cite?: CiteHandler; className?: string }) {
  const blocks: { type: "p" | "ul"; lines: string[] }[] = [];
  for (const line of text.split(/\n/)) {
    if (!line.trim()) {
      blocks.push({ type: "p", lines: [] });
      continue;
    }
    const isItem = LIST_ITEM.test(line);
    const last = blocks[blocks.length - 1];
    if (isItem) {
      if (last?.type === "ul") last.lines.push(line.replace(LIST_ITEM, ""));
      else blocks.push({ type: "ul", lines: [line.replace(LIST_ITEM, "")] });
    } else if (last?.type === "p" && last.lines.length) {
      last.lines.push(line);
    } else {
      blocks.push({ type: "p", lines: [line] });
    }
  }
  return (
    <div className={cn("space-y-3", className)}>
      {blocks
        .filter((block) => block.lines.length)
        .map((block, b) =>
          block.type === "ul" ? (
            <ul key={b} className="list-disc space-y-1.5 pl-5 marker:text-faint">
              {block.lines.map((line, i) => (
                <li key={i}>
                  <Inline text={line} cite={cite} keyPrefix={`${b}-${i}`} />
                </li>
              ))}
            </ul>
          ) : (
            <p key={b} className="text-pretty">
              {block.lines.map((line, i) => (
                <Fragment key={i}>
                  {i > 0 && <br />}
                  <Inline text={line} cite={cite} keyPrefix={`${b}-${i}`} />
                </Fragment>
              ))}
            </p>
          ),
        )}
    </div>
  );
}
