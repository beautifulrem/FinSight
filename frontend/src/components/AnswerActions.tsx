import { AnimatePresence, m as motion } from "motion/react";
import { Check, Copy, Download, FileDown, FileText, ThumbsDown, ThumbsUp } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import { useEffect, useId, useRef, useState } from "react";

import type { FeedbackResult } from "@/lib/api";
import { cn } from "@/lib/cn";
import { answerToMarkdown, downloadText, exportFilename } from "@/lib/exportMarkdown";
import { readFeedback, writeFeedback, type FeedbackRecord } from "@/lib/feedback";
import { useI18n } from "@/lib/i18n";
import type { FeedbackRating, FeedbackRequest } from "@/lib/types";
import type { AnswerView } from "@/lib/view";

import { Button } from "./ui/button";
import { Tooltip } from "./ui/tooltip";

export type FeedbackSender = (body: FeedbackRequest) => Promise<FeedbackResult>;

function useFlag(ms: number): [boolean, () => void] {
  const [on, setOn] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return [
    on,
    () => {
      setOn(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setOn(false), ms);
    },
  ];
}

export function CopyButton({ text }: { text: string }) {
  const { t } = useI18n();
  const [copied, flash] = useFlag(1400);
  return (
    <Tooltip content={copied ? t("answer.copied") : t("answer.copy")}>
      <Button
        size="icon-sm"
        className="copy-answer"
        aria-label={t("answer.copy")}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            flash();
          } catch {
            /* clipboard unavailable */
          }
        }}
      >
        {copied ? <Check /> : <Copy />}
      </Button>
    </Tooltip>
  );
}

/** Copy or download the answer as Markdown with its evidence list and the risk disclaimer. */
export function ExportMenu({ view, query }: { view: AnswerView; query: string }) {
  const { lang, t } = useI18n();
  const [notice, setNotice] = useState("");
  const [shown, flash] = useFlag(1800);
  const markdown = () => answerToMarkdown(view, { query, lang, disclaimerFallback: t("answer.disclaimerDefault") });
  const done = (text: string) => {
    setNotice(text);
    flash();
  };
  return (
    <>
      <span role="status" className={cn("text-[11.5px] text-muted", !shown && "sr-only")}>
        {shown ? notice : ""}
      </span>
      <DropdownMenu.Root modal={false}>
        <Tooltip content={t("export.menu")}>
          <DropdownMenu.Trigger asChild>
            <Button size="icon-sm" className="export-trigger" aria-label={t("export.menu")}>
              <FileDown />
            </Button>
          </DropdownMenu.Trigger>
        </Tooltip>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            align="end"
            sideOffset={6}
            collisionPadding={8}
            className="export-menu z-50 min-w-48 rounded-lg border border-line bg-surface p-1 text-[13px] text-ink shadow-card"
          >
            <DropdownMenu.Item
              className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 outline-none data-[highlighted]:bg-surface-2"
              onSelect={async () => {
                try {
                  await navigator.clipboard.writeText(markdown());
                  done(t("export.copied"));
                } catch {
                  /* clipboard unavailable */
                }
              }}
            >
              <FileText className="size-4 text-muted" aria-hidden />
              {t("export.copyMd")}
            </DropdownMenu.Item>
            <DropdownMenu.Item
              className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 outline-none data-[highlighted]:bg-surface-2"
              onSelect={() => {
                downloadText(exportFilename(query), markdown());
                done(t("export.downloaded"));
              }}
            >
              <Download className="size-4 text-muted" aria-hidden />
              {t("export.download")}
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </>
  );
}

interface FeedbackProps {
  traceId: string;
  sessionId: string | null;
  onSend: FeedbackSender;
}

/** Thumbs up/down with an optional comment, stored per trace id and posted to `/agent/feedback`. */
export function FeedbackControls({ traceId, sessionId, onSend }: FeedbackProps) {
  const { t } = useI18n();
  const [record, setRecord] = useState<FeedbackRecord | undefined>(() => readFeedback(traceId));
  const [sending, setSending] = useState(false);
  const [commenting, setCommenting] = useState(false);
  const [comment, setComment] = useState("");
  const textareaId = useId();
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (commenting) textarea.current?.focus();
  }, [commenting]);

  const submit = async (rating: FeedbackRating, text?: string) => {
    const optimistic: FeedbackRecord = { rating, comment: text || record?.comment, state: "sent" };
    setRecord(optimistic);
    setSending(true);
    let next: FeedbackRecord;
    try {
      const result = await onSend({ trace_id: traceId, session_id: sessionId, rating, comment: text?.trim() || null });
      next = result === "ok" ? optimistic : { ...optimistic, state: "local", reason: result };
    } catch {
      next = { ...optimistic, state: "error" };
    }
    setSending(false);
    setRecord(next);
    writeFeedback(traceId, next);
  };

  const rate = (rating: FeedbackRating) => {
    if (sending) return;
    if (record?.rating === rating && record.state !== "error") {
      setCommenting((open) => !open);
      return;
    }
    void submit(rating);
    setCommenting(rating === "down");
  };

  const status = sending
    ? t("feedback.sending")
    : !record
      ? ""
      : record.state === "error"
        ? t("feedback.error")
        : record.state === "local"
          ? t(record.reason === "unknown_trace" ? "feedback.expired" : "feedback.local")
          : t("feedback.thanks");

  return (
    <div className="feedback min-w-0 flex-1" data-rating={record?.rating ?? ""} data-state={record?.state ?? ""}>
      <div className="flex flex-wrap items-center gap-1">
        <div role="group" aria-label={t("feedback.group")} className="flex items-center gap-0.5">
          {(["up", "down"] as const).map((rating) => {
            const Icon = rating === "up" ? ThumbsUp : ThumbsDown;
            const active = record?.rating === rating;
            return (
              <Tooltip key={rating} content={t(rating === "up" ? "feedback.up" : "feedback.down")}>
                <Button
                  size="icon-sm"
                  className={cn(`feedback-${rating}`, active && (rating === "up" ? "text-cobalt" : "text-warn"))}
                  aria-label={t(rating === "up" ? "feedback.up" : "feedback.down")}
                  aria-pressed={active}
                  onClick={() => rate(rating)}
                >
                  <Icon className={cn(active && "fill-current")} />
                </Button>
              </Tooltip>
            );
          })}
        </div>
        <span role="status" className={cn("feedback-status text-[12px]", record?.state === "error" ? "text-up" : "text-muted")}>
          {status}
        </span>
        {record && !sending && !commenting && !record.comment && record.state !== "error" && (
          <button
            type="button"
            className="feedback-add-comment rounded px-1 text-[12px] font-medium text-cobalt hover:underline"
            onClick={() => setCommenting(true)}
          >
            {t("feedback.addComment")}
          </button>
        )}
      </div>
      <AnimatePresence initial={false}>
        {commenting && record && (
          <motion.form
            key="comment"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="feedback-form overflow-hidden"
            onSubmit={(event) => {
              event.preventDefault();
              setCommenting(false);
              void submit(record.rating, comment);
            }}
          >
            <div className="space-y-1.5 pt-2">
              <label htmlFor={textareaId} className="block text-[12px] font-medium text-muted">
                {t("feedback.comment")}
              </label>
              <textarea
                id={textareaId}
                ref={textarea}
                value={comment}
                onChange={(event) => setComment(event.target.value)}
                rows={2}
                maxLength={1000}
                placeholder={t(record.rating === "down" ? "feedback.placeholderDown" : "feedback.placeholderUp")}
                className="block w-full resize-y rounded-lg border border-line bg-bg/60 px-3 py-2 text-[13px] leading-relaxed text-ink outline-none placeholder:text-muted focus:border-cobalt/60"
                onKeyDown={(event) => {
                  if (event.key === "Escape") setCommenting(false);
                }}
              />
              <div className="flex justify-end gap-1.5">
                <Button size="sm" variant="ghost" onClick={() => setCommenting(false)}>
                  {t("feedback.skip")}
                </Button>
                <Button size="sm" variant="primary" type="submit" disabled={!comment.trim()}>
                  {t("feedback.send")}
                </Button>
              </div>
            </div>
          </motion.form>
        )}
      </AnimatePresence>
    </div>
  );
}
