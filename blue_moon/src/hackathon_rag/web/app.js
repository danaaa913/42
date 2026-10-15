"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  chat: $("chat"),
  form: $("composer"),
  input: $("question"),
  send: $("send"),
  error: $("composerError"),
  status: $("status"),
  statusText: $("statusText"),
  newConversation: $("newConversation"),
  emptyState: $("emptyState"),
};

let sessionId = newSessionId();
let busy = false;

function newSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return "sess-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]
  ));
}

function authHeaders(json = false) {
  const token = (sessionStorage.getItem("novacortex_token") || "").trim();
  return {
    ...(json ? { "Content-Type": "application/json" } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error(typeof detail === "string" ? detail : "تعذر إكمال الطلب.");
  }
  return body;
}

/* ---------------------------------------------------------------- rendering */

function hideEmptyState() {
  if (els.emptyState && !els.emptyState.classList.contains("hidden")) {
    els.emptyState.classList.add("hidden");
  }
}

function addMessage(role, text) {
  hideEmptyState();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  els.chat.appendChild(wrap);
  els.chat.scrollTop = els.chat.scrollHeight;
  return { wrap, bubble };
}

function addTyping() {
  hideEmptyState();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
  els.chat.appendChild(wrap);
  els.chat.scrollTop = els.chat.scrollHeight;
  return wrap;
}

function addSources(wrap, contexts) {
  if (!contexts || !contexts.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent = `المصادر المستخدمة (${contexts.length})`;
  details.appendChild(summary);
  for (const context of contexts) {
    const item = document.createElement("div");
    item.className = "source";
    item.textContent = context;
    details.appendChild(item);
  }
  wrap.appendChild(details);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function addMeta(wrap, latencyMs) {
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `زمن الاستجابة: ${(latencyMs / 1000).toFixed(1)} ث`;
  wrap.appendChild(meta);
}

/* -------------------------------------------------------------------- health */

async function checkHealth() {
  try {
    const data = await api("/api/health");
    const ready = data.index_ready && data.ollama_ready && data.chat_model_ready;
    els.status.classList.toggle("ready", ready);
    els.status.classList.toggle("bad", !ready);
    if (ready) {
      els.statusText.textContent = "النظام جاهز";
    } else if (!data.ollama_ready) {
      els.statusText.textContent = "تعذر الاتصال بـ Ollama";
    } else if (!data.index_ready) {
      els.statusText.textContent = "الفهرس غير جاهز";
    } else {
      els.statusText.textContent = "النموذج غير متوفر";
    }
    return data;
  } catch {
    els.status.classList.remove("ready");
    els.status.classList.add("bad");
    els.statusText.textContent = "تعذر الاتصال بالخادم";
    return null;
  }
}

/* ------------------------------------------------------------------ composer */

function setBusy(value) {
  busy = value;
  els.send.disabled = value;
  els.send.querySelector(".send-label").textContent = value ? "جاري التفكير" : "إرسال";
  els.send.querySelector(".spinner").classList.toggle("hidden", !value);
}

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
}

function showError(message) {
  els.error.textContent = message;
  els.error.classList.remove("hidden");
}

function clearError() {
  els.error.classList.add("hidden");
  els.error.textContent = "";
}

async function ask(question) {
  clearError();
  addMessage("user", question);
  const typing = addTyping();
  setBusy(true);
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    if (result.session_id) sessionId = result.session_id;
    typing.remove();
    const { wrap } = addMessage("assistant", result.answer);
    addSources(wrap, result.contexts);
    addMeta(wrap, result.latency_ms || 0);
  } catch (error) {
    typing.remove();
    const { wrap, bubble } = addMessage("assistant", error.message);
    bubble.classList.add("error");
    showError(error.message);
    checkHealth();
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  const question = els.input.value.trim();
  if (!question) return;
  els.input.value = "";
  autoGrow();
  ask(question);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});
els.input.addEventListener("input", autoGrow);

els.newConversation.addEventListener("click", async () => {
  if (busy) return;
  const previous = sessionId;
  sessionId = newSessionId();
  try {
    await api("/api/chat/reset", {
      method: "POST",
      headers: authHeaders(true),
      body: JSON.stringify({ session_id: previous }),
    });
  } catch { /* clearing the view is enough even if the call fails */ }
  els.chat.innerHTML = "";
  els.chat.appendChild(els.emptyState);
  els.emptyState.classList.remove("hidden");
  clearError();
  els.input.value = "";
  autoGrow();
  els.input.focus();
});

checkHealth();
els.input.focus();
