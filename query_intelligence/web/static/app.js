const form = document.getElementById("chat-form");
const input = document.getElementById("query-input");
const button = document.getElementById("submit-button");
const statusPill = document.getElementById("status-pill");
const messagesEl = document.getElementById("chat-messages");
const modeSelect = document.getElementById("mode-select");
const newSessionButton = document.getElementById("new-session");
const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");
const apiKeyInput = document.getElementById("api-key-input");
const sessionIdEl = document.getElementById("session-id");
const submitText = button.textContent;

const STORAGE = { mode: "finsight.mode", session: "finsight.session", apiKey: "finsight.apiKey" };
let pendingClarification = null;

function storageGet(key, fallback = "") {
  try {
    return window.localStorage.getItem(key) ?? fallback;
  } catch (_) {
    return fallback;
  }
}

function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (_) {
    /* storage may be unavailable (private mode); the page still works */
  }
}

function newSessionId() {
  const random = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : String(Date.now());
  return random.replaceAll("-", "");
}

let sessionId = storageGet(STORAGE.session) || newSessionId();
storageSet(STORAGE.session, sessionId);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(text, busy = false) {
  statusPill.textContent = text;
  statusPill.classList.toggle("busy", busy);
}

async function copyBubbleText(buttonEl) {
  const bubble = buttonEl.closest(".bubble");
  if (!bubble) return;
  const clone = bubble.cloneNode(true);
  const copyButton = clone.querySelector(".copy-btn");
  if (copyButton) copyButton.remove();
  const text = clone.innerText.trim();
  try {
    await navigator.clipboard.writeText(text);
    buttonEl.textContent = "Copied";
    setTimeout(() => { buttonEl.textContent = "Copy"; }, 1200);
  } catch (_) {
    buttonEl.textContent = "Copy failed";
    setTimeout(() => { buttonEl.textContent = "Copy"; }, 1200);
  }
}

function attachCopyButton(bubble) {
  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "copy-btn";
  copyButton.textContent = "Copy";
  copyButton.addEventListener("click", () => copyBubbleText(copyButton));
  bubble.appendChild(copyButton);
}

function renderKeyPoints(items) {
  if (!items || !items.length) return "";
  return `<ul class="key-points">${items.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderEvidenceSources(items) {
  if (!items || !items.length) return "";
  const rows = items.map(item => {
    const title = escapeHtml(item.title || item.source_name || item.source_type || "Unknown source");
    const metaParts = [item.evidence_id, item.source_name, item.source_type, item.as_of].filter(Boolean).map(escapeHtml);
    const meta = metaParts.length ? `<span class="evidence-meta">${metaParts.join(" · ")}</span>` : "";
    const url = item.source_url ? String(item.source_url) : "";
    const link = /^https?:\/\//i.test(url)
      ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Open webpage</a>`
      : `<span class="evidence-meta">No web link</span>`;
    return `<li><div class="evidence-title">${title}</div><div>${meta} ${link}</div></li>`;
  }).join("");
  return `<ul class="evidence-list">${rows}</ul>`;
}

function renderSentiment(sentiment) {
  if (!sentiment || !sentiment.label_counts) return "";
  const counts = Object.entries(sentiment.label_counts).map(([label, count]) => `${escapeHtml(label)} ${escapeHtml(count)}`);
  return `<div class="section-title">Document Tone</div><div class="sentiment-line">${escapeHtml(sentiment.overall_label || "")}` +
    ` (${counts.join(", ")}; score ${escapeHtml(sentiment.mean_score)}, ${escapeHtml(sentiment.backend || "")} model)</div>`;
}

function renderNextQuestions(items) {
  if (!items || !items.length) return "";
  const chips = items.map(item => `<button type="button" class="next-question">${escapeHtml(item.question)}</button>`).join("");
  return `<div class="section-title">You could also ask</div><div class="next-questions">${chips}</div>`;
}

function renderDetails(data) {
  const tools = (data.tool_calls || []).map(call => {
    const state = call.ok ? "ok" : `error: ${(call.error && call.error.code) || "failed"}`;
    return `<li>${escapeHtml(call.tool)} <span class="evidence-meta">${escapeHtml(state)} · ${escapeHtml(call.latency_ms)} ms · ${escapeHtml(call.source || "")}</span></li>`;
  }).join("");
  const verification = data.verification && Object.keys(data.verification).length
    ? (data.verification.passed ? "passed" : "repaired")
    : "n/a";
  const parts = [
    `<div>Route: <code>${escapeHtml(data.route || "")}</code> (${escapeHtml((data.route_reasons || []).join(", "))})</div>`,
    `<div>Answer source: ${escapeHtml(data.answer_source || "")} · Evidence check: ${escapeHtml(verification)}</div>`,
    data.degraded && data.degraded.length ? `<div>Degraded: ${escapeHtml(data.degraded.join(", "))}</div>` : "",
    data.llm && data.llm.calls ? `<div>LLM calls: ${escapeHtml(data.llm.calls)} · tokens ${escapeHtml((data.llm.usage || {}).total_tokens)}</div>` : "",
    `<div>Trace: <code>${escapeHtml(data.trace_id || data.run_id || "")}</code></div>`,
    tools ? `<ul class="tool-list">${tools}</ul>` : "",
  ];
  return `<details class="run-details"><summary>How this answer was produced</summary>${parts.join("")}</details>`;
}

function buildBotContent(data) {
  const answer = `<div class="answer-text">${escapeHtml(data.answer || "")}</div>`;
  const keyPoints = data.key_points && data.key_points.length
    ? `<div class="section-title">Key Points</div>${renderKeyPoints(data.key_points)}`
    : "";
  const limitations = data.limitations && data.limitations.length
    ? `<div class="section-title">Limitations</div>${renderKeyPoints(data.limitations)}`
    : "";
  const evidence = data.evidence_sources && data.evidence_sources.length
    ? `<div class="section-title">Evidence Sources</div>${renderEvidenceSources(data.evidence_sources)}`
    : "";
  const disclaimer = data.risk_disclaimer
    ? `<div class="disclaimer">${escapeHtml(data.risk_disclaimer)}</div>`
    : "";
  const agentParts = data.route
    ? `${renderSentiment(data.sentiment)}${renderNextQuestions(data.next_questions)}${renderDetails(data)}`
    : "";
  return `${answer}${keyPoints}${limitations}${evidence}${agentParts}${disclaimer}`;
}

function appendMessage(kind, htmlContent, options = {}) {
  const rawText = Boolean(options.rawText);
  const row = document.createElement("div");
  row.className = `msg-row ${kind === "user" ? "msg-user" : "msg-bot"}`;
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind === "user" ? "bubble-user" : "bubble-bot"}`;
  bubble.innerHTML = rawText ? escapeHtml(htmlContent) : htmlContent;
  attachCopyButton(bubble);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  bubble.querySelectorAll(".next-question").forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.textContent;
      input.focus();
    });
  });
  scrollToBottom();
  return row;
}

function appendTyping() {
  const row = document.createElement("div");
  row.className = "msg-row msg-bot";
  row.id = "typing-row";
  row.innerHTML = '<div class="bubble bubble-bot"><div class="typing-dots"><span></span><span></span><span></span></div><ul class="step-list" id="step-list"></ul></div>';
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function addStep(text) {
  const list = document.getElementById("step-list");
  if (!list) return;
  const item = document.createElement("li");
  item.textContent = text;
  list.appendChild(item);
  scrollToBottom();
}

function removeTyping() {
  const typing = document.getElementById("typing-row");
  if (typing) typing.remove();
}

function appendError(message) {
  const row = document.createElement("div");
  row.className = "msg-row msg-bot";
  row.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function setLoading(loading) {
  button.disabled = loading;
  input.disabled = loading;
  button.textContent = loading ? "Analyzing..." : submitText;
  setStatus(loading ? "Analyzing" : "Ready", loading);
}

function requestHeaders() {
  const headers = { "Content-Type": "application/json" };
  const key = storageGet(STORAGE.apiKey);
  if (key) headers["X-API-Key"] = key;
  return headers;
}

async function readError(res) {
  try {
    const data = await res.json();
    return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
  } catch (_) {
    return `Request failed (${res.status})`;
  }
}

async function runClassic(query) {
  const res = await fetch("/chat", { method: "POST", headers: requestHeaders(), body: JSON.stringify({ query }) });
  if (!res.ok) throw new Error(await readError(res));
  const data = await res.json();
  removeTyping();
  appendMessage("bot", buildBotContent(data));
  if (data.llm && data.llm.status === "fallback") {
    appendMessage("bot", '<div class="answer-text">Using structured-summary fallback.</div>');
  }
}

function handleAgentResult(data) {
  removeTyping();
  if (data.status === "needs_clarification") {
    pendingClarification = data.clarification || {};
    appendMessage("bot", `<div class="answer-text">${escapeHtml(pendingClarification.question || "Could you clarify?")}</div>`);
    setStatus("Waiting for clarification");
    return;
  }
  pendingClarification = null;
  appendMessage("bot", buildBotContent(data));
}

async function runAgentResume(reply) {
  const res = await fetch("/agent/resume", {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify({ session_id: sessionId, reply }),
  });
  if (!res.ok) throw new Error(await readError(res));
  handleAgentResult(await res.json());
}

async function runAgentStream(query, mode) {
  const res = await fetch("/agent/chat/stream", {
    method: "POST",
    headers: requestHeaders(),
    body: JSON.stringify({ query, session_id: sessionId, mode }),
  });
  if (!res.ok || !res.body) throw new Error(await readError(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  while (!finished) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
      const event = /^event: (.*)$/m.exec(block);
      const payload = /^data: (.*)$/m.exec(block);
      if (!event || !payload) continue;
      const data = JSON.parse(payload[1]);
      switch (event[1]) {
        case "step":
          setStatus(data.label, true);
          addStep(data.label);
          break;
        case "tool_call":
          addStep(`→ ${data.tool}`);
          break;
        case "tool_result":
          if (!data.ok) addStep(`✗ ${data.tool}: ${(data.error && data.error.code) || "failed"}`);
          break;
        case "clarification":
          handleAgentResult({ status: "needs_clarification", clarification: data });
          finished = true;
          break;
        case "answer":
          handleAgentResult(data);
          finished = true;
          break;
        case "error":
          throw new Error(data.message || "Agent error");
        default:
          break;
      }
    }
  }
  removeTyping();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) {
    appendError("Please enter a question.");
    return;
  }
  input.value = "";
  appendMessage("user", query, { rawText: true });
  appendTyping();
  setLoading(true);
  const mode = modeSelect.value;
  try {
    if (mode === "classic") {
      await runClassic(query);
    } else if (pendingClarification) {
      await runAgentResume(query);
    } else {
      await runAgentStream(query, mode);
    }
    if (!pendingClarification) setStatus("Complete");
  } catch (error) {
    removeTyping();
    const message = error instanceof TypeError && String(error.message || "").toLowerCase().includes("fetch")
      ? "Cannot reach the local server. Keep the start.bat command window open, then refresh http://127.0.0.1:8765/."
      : (error.message || error);
    appendError(message);
    setStatus("Error");
  } finally {
    button.disabled = false;
    input.disabled = false;
    button.textContent = submitText;
    input.focus();
  }
});

modeSelect.value = storageGet(STORAGE.mode, "auto") || "auto";
modeSelect.addEventListener("change", () => storageSet(STORAGE.mode, modeSelect.value));
newSessionButton.addEventListener("click", () => {
  sessionId = newSessionId();
  storageSet(STORAGE.session, sessionId);
  sessionIdEl.textContent = sessionId;
  pendingClarification = null;
  appendMessage("bot", '<div class="answer-text">Started a new session; earlier questions are no longer used as context.</div>');
});
settingsToggle.addEventListener("click", () => {
  const open = settingsPanel.hidden;
  settingsPanel.hidden = !open;
  settingsToggle.setAttribute("aria-expanded", String(open));
});
apiKeyInput.value = storageGet(STORAGE.apiKey);
apiKeyInput.addEventListener("change", () => storageSet(STORAGE.apiKey, apiKeyInput.value.trim()));
sessionIdEl.textContent = sessionId;

const welcomeBubble = messagesEl.querySelector(".bubble");
if (welcomeBubble) attachCopyButton(welcomeBubble);
input.focus();
