import { AnimatePresence, m as motion } from "motion/react";
import { ArrowUp, HelpCircle, Square, X } from "lucide-react";
import { useImperativeHandle, useLayoutEffect, useRef, useState, type KeyboardEvent, type Ref } from "react";

import { useI18n } from "@/lib/i18n";
import type { Clarification, UiMode } from "@/lib/types";

import { ModeSwitcher } from "./ModeSwitcher";
import { Button } from "./ui/button";

export interface ComposerHandle {
  focus: () => void;
  setValue: (value: string) => void;
}

interface Props {
  busy: boolean;
  mode: UiMode;
  onMode: (mode: UiMode) => void;
  pending: Clarification | null;
  onCancelPending: () => void;
  onSend: (text: string) => void;
  onStop: () => void;
  placeholder?: string;
  submitText?: string;
  ref?: Ref<ComposerHandle>;
}

export function Composer({ busy, mode, onMode, pending, onCancelPending, onSend, onStop, placeholder, submitText, ref }: Props) {
  const { t } = useI18n();
  const [value, setValue] = useState("");
  const [hint, setHint] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const replying = Boolean(pending) && mode !== "classic";

  useImperativeHandle(ref, () => ({
    focus: () => textarea.current?.focus(),
    setValue: (next: string) => {
      setValue(next);
      textarea.current?.focus();
    },
  }));

  useLayoutEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }, [value]);

  const submit = () => {
    if (busy) return;
    const text = value.trim();
    if (!text) {
      setHint(true);
      textarea.current?.focus();
      return;
    }
    setHint(false);
    setValue("");
    onSend(text);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter adds a line; ignore Enter while an IME (pinyin) is composing.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form
      id="chat-form"
      className="composer relative rounded-2xl border border-line bg-surface shadow-card transition-shadow focus-within:border-cobalt/60 focus-within:ring-4 focus-within:ring-cobalt/10"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <AnimatePresence initial={false}>
        {replying && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="clarify-banner flex items-center gap-2 border-b border-line px-3.5 py-2 text-[12.5px] text-gilt">
              <HelpCircle className="size-3.5 shrink-0" aria-hidden />
              <span className="min-w-0 flex-1 truncate">{t("composer.replying")}: {pending?.question}</span>
              <button type="button" onClick={onCancelPending} className="inline-flex items-center gap-1 rounded px-1 text-muted hover:text-ink">
                <X className="size-3.5" aria-hidden />
                {t("composer.cancelReply")}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <label htmlFor="query-input" className="sr-only">
        {replying ? t("composer.replyPlaceholder") : t("composer.placeholder")}
      </label>
      <textarea
        id="query-input"
        ref={textarea}
        rows={1}
        value={value}
        maxLength={2000}
        onChange={(event) => {
          setValue(event.target.value);
          if (hint) setHint(false);
        }}
        onKeyDown={onKeyDown}
        placeholder={replying ? t("composer.replyPlaceholder") : placeholder || t("composer.placeholder")}
        aria-describedby="composer-hint"
        className="block w-full resize-none bg-transparent px-4 pt-3.5 pb-1 text-[15px] leading-relaxed text-ink outline-none placeholder:text-faint"
        autoComplete="off"
        enterKeyHint="send"
      />
      <div className="flex items-center gap-2 px-2.5 pt-1 pb-2.5">
        <ModeSwitcher value={mode} onChange={onMode} disabled={busy} />
        <span id="composer-hint" className={hint ? "text-[12px] text-up" : "hidden text-[12px] text-faint md:inline"}>
          {hint ? t("composer.empty") : t("composer.hint")}
        </span>
        {busy ? (
          <Button id="stop-button" variant="outline" size="sm" className="ml-auto" onClick={onStop}>
            <Square className="fill-current" />
            {t("composer.stop")}
          </Button>
        ) : (
          <Button id="submit-button" type="submit" variant="primary" size="sm" className="ml-auto" aria-label={submitText || t("composer.send")}>
            <ArrowUp />
            <span>{submitText || t("composer.send")}</span>
          </Button>
        )}
      </div>
    </form>
  );
}
