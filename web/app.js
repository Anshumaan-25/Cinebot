// Movie Chatbot UI (Part 10) — streams the /chat orchestration response.
"use strict";

const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const userIdEl = document.getElementById("user-id");

// Markers the /chat stream appends after the answer body.
const META_MARKERS = ["\n\n[route:", "\n[error:"];

// ---- user id persistence (drives long-term memory) ----
userIdEl.value = localStorage.getItem("chat_user_id") || "demo-user";
userIdEl.addEventListener("change", () =>
  localStorage.setItem("chat_user_id", userIdEl.value.trim() || "demo-user")
);

// ---- helpers ----
function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollDown();
  return bubble;
}

function typingBubble() {
  const bubble = addMessage("bot");
  bubble.innerHTML =
    '<span class="typing"><span></span><span></span><span></span></span>';
  return bubble;
}

// Split the raw stream into the answer body and the trailing meta footer.
function splitMeta(full) {
  let cut = -1;
  for (const marker of META_MARKERS) {
    const i = full.indexOf(marker);
    if (i !== -1 && (cut === -1 || i < cut)) cut = i;
  }
  return cut === -1
    ? { body: full, meta: "" }
    : { body: full.slice(0, cut).trimEnd(), meta: full.slice(cut).trim() };
}

function renderBot(bubble, full) {
  const { body, meta } = splitMeta(full);
  bubble.textContent = body;
  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }
}

// ---- send ----
async function send(text) {
  const message = text.trim();
  if (!message || sendBtn.disabled) return;

  addMessage("user").textContent = message;
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  const bubble = typingBubble();
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        user_id: userIdEl.value.trim() || "demo-user",
      }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = "";
    let first = true;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      full += decoder.decode(value, { stream: true });
      if (first) {
        bubble.textContent = "";
        first = false;
      }
      renderBot(bubble, full);
      scrollDown();
    }
    if (first) bubble.textContent = "(no response)";
  } catch (err) {
    bubble.textContent = `⚠️ ${err.message}. Is the server running?`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
    scrollDown();
  }
}

// ---- events ----
form.addEventListener("submit", (e) => {
  e.preventDefault();
  send(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send(input.value);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => send(chip.dataset.q))
);

input.focus();
