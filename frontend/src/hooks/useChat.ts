import { useCallback, useLayoutEffect, useReducer, useRef } from "react";

import { ApiError, classicChat, resumeClarification, streamAgentChat } from "@/lib/api";
import type {
  AgentResponse,
  Clarification,
  ClassicResponse,
  SessionTurn,
  StreamEvent,
  ToolError,
  UiMode,
} from "@/lib/types";

export interface LiveTool {
  id: string;
  tool: string;
  args: unknown;
  status: "running" | "ok" | "error";
  latencyMs?: number;
  evidenceIds?: string[];
  error?: ToolError | null;
}

export interface LiveStep {
  id: string;
  node: string;
  tools: LiveTool[];
}

export type ErrorKind = "network" | "auth" | "rate" | "timeout" | "generic";

export interface Turn {
  kind: "turn";
  id: string;
  query: string;
  mode: UiMode;
  /** `resume` turns answer a pending clarification through /agent/resume. */
  via: "stream" | "classic" | "resume";
  status: "running" | "done" | "clarify" | "error" | "stopped";
  startedAt: number;
  finishedAt?: number;
  steps: LiveStep[];
  agent?: AgentResponse;
  classic?: ClassicResponse;
  clarification?: Clarification;
  error?: { kind: ErrorKind; message: string };
}

export interface Notice {
  kind: "notice";
  id: string;
  text: string;
}

/** Turns restored from `GET /agent/sessions/{id}` after a reload (query and answer text only). */
export interface HistoryItem {
  kind: "history";
  id: string;
  turns: SessionTurn[];
}

export type ChatItem = Turn | Notice | HistoryItem;

interface State {
  items: ChatItem[];
  pending: Clarification | null;
}

type Action =
  | { type: "start"; turn: Turn }
  | { type: "event"; id: string; event: StreamEvent }
  | { type: "agent"; id: string; response: AgentResponse }
  | { type: "classic"; id: string; response: ClassicResponse }
  | { type: "error"; id: string; error: { kind: ErrorKind; message: string } }
  | { type: "stopped"; id: string }
  | { type: "notice"; notice: Notice }
  | { type: "reset"; notice?: Notice }
  | { type: "pending"; clarification: Clarification | null }
  | { type: "restore"; turns: SessionTurn[]; pending: Clarification | null };

let counter = 0;
const nextId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(counter++).toString(36)}`;

function updateTurn(state: State, id: string, update: (turn: Turn) => Turn): State {
  return {
    ...state,
    items: state.items.map((item) => (item.kind === "turn" && item.id === id ? update(item) : item)),
  };
}

function applyEvent(turn: Turn, event: StreamEvent): Turn {
  switch (event.event) {
    case "step":
      return { ...turn, steps: [...turn.steps, { id: nextId("step"), node: event.data.node, tools: [] }] };
    case "tool_call": {
      const tool: LiveTool = { id: nextId("tool"), tool: event.data.tool, args: event.data.arguments, status: "running" };
      if (!turn.steps.length) return { ...turn, steps: [{ id: nextId("step"), node: "agent_tools", tools: [tool] }] };
      const steps = turn.steps.slice();
      const last = steps[steps.length - 1]!;
      steps[steps.length - 1] = { ...last, tools: [...last.tools, tool] };
      return { ...turn, steps };
    }
    case "tool_result": {
      let matched = false;
      const steps = turn.steps.map((step) => ({
        ...step,
        tools: step.tools.map((tool) => {
          if (matched || tool.status !== "running" || tool.tool !== event.data.tool) return tool;
          matched = true;
          return {
            ...tool,
            status: event.data.ok ? ("ok" as const) : ("error" as const),
            latencyMs: event.data.latency_ms,
            evidenceIds: event.data.evidence_ids,
            error: event.data.error,
          };
        }),
      }));
      return { ...turn, steps };
    }
    default:
      return turn;
  }
}

function withAgent(turn: Turn, response: AgentResponse): Turn {
  const finishedAt = performance.now();
  if (response.status === "needs_clarification") {
    return { ...turn, status: "clarify", finishedAt, clarification: response.clarification ?? { question: "" } };
  }
  return { ...turn, status: "done", finishedAt, agent: response };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "start":
      return { ...state, items: [...state.items, action.turn] };
    case "event":
      return updateTurn(state, action.id, (turn) => applyEvent(turn, action.event));
    case "agent": {
      const next = updateTurn(state, action.id, (turn) => withAgent(turn, action.response));
      const pending =
        action.response.status === "needs_clarification" ? (action.response.clarification ?? { question: "" }) : null;
      return { ...next, pending };
    }
    case "classic":
      return updateTurn(state, action.id, (turn) => ({
        ...turn,
        status: "done",
        finishedAt: performance.now(),
        classic: action.response,
      }));
    case "error":
      return updateTurn(state, action.id, (turn) => ({
        ...turn,
        status: "error",
        finishedAt: performance.now(),
        error: action.error,
      }));
    case "stopped":
      return updateTurn(state, action.id, (turn) =>
        turn.status === "running" ? { ...turn, status: "stopped", finishedAt: performance.now() } : turn,
      );
    case "notice":
      return { ...state, items: [...state.items, action.notice] };
    case "reset":
      return { items: action.notice ? [action.notice] : [], pending: null };
    case "pending":
      return { ...state, pending: action.clarification };
    case "restore":
      if (state.items.length) return state;
      return {
        items: action.turns.length ? [{ kind: "history", id: nextId("history"), turns: action.turns }] : [],
        pending: action.pending,
      };
    default:
      return state;
  }
}

function classify(error: unknown): { kind: ErrorKind; message: string } {
  if (error instanceof ApiError) {
    if (error.status === 401 || error.status === 403) return { kind: "auth", message: error.message };
    if (error.status === 429) return { kind: "rate", message: error.message };
    if (error.status === 504) return { kind: "timeout", message: error.message };
    return { kind: "generic", message: error.message };
  }
  if (error instanceof TypeError) return { kind: "network", message: error.message };
  return { kind: "generic", message: error instanceof Error ? error.message : String(error) };
}

export interface ChatOptions {
  sessionId: string;
  mode: UiMode;
  apiKey: string;
  onSession?: (sessionId: string) => void;
}

interface Run {
  id: string;
  abort: AbortController;
  /** Set by Stop: the UI stops listening, but the request is drained so the server can finish the turn. */
  detached: boolean;
}

export function useChat(options: ChatOptions) {
  const [state, dispatch] = useReducer(reducer, { items: [], pending: null });
  const current = useRef<Run | null>(null);
  const optionsRef = useRef(options);
  const pendingRef = useRef(state.pending);
  useLayoutEffect(() => {
    optionsRef.current = options;
    pendingRef.current = state.pending;
  });

  const busy = state.items.some((item) => item.kind === "turn" && item.status === "running");

  const send = useCallback(async (raw: string) => {
    const query = raw.trim();
    if (!query || current.current) return;
    const { sessionId, mode, apiKey } = optionsRef.current;
    const pending = pendingRef.current;
    const via: Turn["via"] = mode === "classic" ? "classic" : pending ? "resume" : "stream";
    const id = nextId("turn");
    dispatch({
      type: "start",
      turn: { kind: "turn", id, query, mode, via, status: "running", startedAt: performance.now(), steps: [] },
    });
    const run: Run = { id, abort: new AbortController(), detached: false };
    current.current = run;
    const emit = (action: Action) => {
      if (!run.detached) dispatch(action);
    };
    const request = { apiKey, signal: run.abort.signal };
    const runStream = async () => {
      let finished = false;
      for await (const event of streamAgentChat(query, sessionId, mode as Exclude<UiMode, "classic">, request)) {
        if (event.event === "answer") {
          emit({ type: "agent", id, response: event.data });
          finished = true;
        } else if (event.event === "clarification") {
          const { session_id: session, ...clarification } = event.data;
          emit({ type: "agent", id, response: { status: "needs_clarification", session_id: session, clarification } });
          finished = true;
        } else if (event.event === "error") {
          throw new Error(event.data.message);
        } else if (event.event === "session") {
          if (!run.detached) optionsRef.current.onSession?.(event.data.session_id);
        } else {
          emit({ type: "event", id, event });
        }
      }
      if (!finished) throw new Error("stream ended without an answer");
    };
    try {
      if (via === "classic") {
        emit({ type: "classic", id, response: await classicChat(query, request) });
      } else if (via === "resume") {
        try {
          emit({ type: "agent", id, response: await resumeClarification(sessionId, query, request) });
        } catch (error) {
          // 409: the server has nothing pending (e.g. it restarted); ask the reply as a new question.
          if (!(error instanceof ApiError && error.status === 409)) throw error;
          emit({ type: "pending", clarification: null });
          await runStream();
        }
      } else {
        await runStream();
      }
    } catch (error) {
      if (run.abort.signal.aborted) dispatch({ type: "stopped", id });
      else emit({ type: "error", id, error: classify(error) });
    } finally {
      if (current.current === run) current.current = null;
    }
  }, []);

  // Stop listening without cancelling the request: the server serializes turns per session, so the
  // run is allowed to finish (its answer is discarded) and the next question is not blocked.
  const stop = useCallback(() => {
    const run = current.current;
    if (!run) return;
    run.detached = true;
    current.current = null;
    dispatch({ type: "stopped", id: run.id });
  }, []);

  const reset = useCallback((text?: string) => {
    current.current?.abort.abort();
    current.current = null;
    dispatch({ type: "reset", notice: text ? { kind: "notice", id: nextId("notice"), text } : undefined });
  }, []);

  const notify = useCallback((text: string) => dispatch({ type: "notice", notice: { kind: "notice", id: nextId("notice"), text } }), []);

  const setPending = useCallback((clarification: Clarification | null) => dispatch({ type: "pending", clarification }), []);

  const restore = useCallback(
    (turns: SessionTurn[], pending: Clarification | null) => dispatch({ type: "restore", turns, pending }),
    [],
  );

  return { items: state.items, pending: state.pending, busy, send, stop, reset, notify, setPending, restore };
}
