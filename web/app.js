// CineMind UI (Part 10) — streams the /chat orchestration response.
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
const scrollDown = () => (messagesEl.scrollTop = messagesEl.scrollHeight);

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// Minimal, safe markdown: escape first, then inline bold / italic / code.
function renderMarkdown(text) {
  let h = escapeHtml(text);
  h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  return h;
}

function addMessage(role) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  if (role === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🍿";
    msg.appendChild(avatar);
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const md = document.createElement("div");
  md.className = "md";
  bubble.appendChild(md);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollDown();
  return { bubble, md };
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

function parseMeta(meta) {
  const route = (meta.match(/\[route:\s*([^\]]+)\]/) || [])[1];
  const tools = /live tools consulted/i.test(meta);
  const error = (meta.match(/\[error:\s*([^\]]+)\]/) || [])[1];
  const sources = [];
  const after = meta.split("Sources:")[1];
  if (after) {
    for (const line of after.split("\n")) {
      const m = line.match(/\[(\d+)\]\s*(.+)/);
      if (m) sources.push({ n: m[1], text: m[2].trim() });
    }
  }
  return { route, tools, error, sources };
}

function renderMeta(bubble, metaText) {
  const { route, tools, error, sources } = parseMeta(metaText);
  if (!route && !tools && !error && !sources.length) return;

  const meta = document.createElement("div");
  meta.className = "meta";

  const badges = document.createElement("div");
  badges.className = "badges";
  if (route) {
    const b = document.createElement("span");
    b.className = `badge route-${route.trim()}`;
    b.textContent = route.trim();
    badges.appendChild(b);
  }
  if (tools) {
    const b = document.createElement("span");
    b.className = "badge tools";
    b.textContent = "live tools";
    badges.appendChild(b);
  }
  if (error) {
    const b = document.createElement("span");
    b.className = "badge tools";
    b.textContent = "error";
    badges.appendChild(b);
  }
  if (badges.children.length) meta.appendChild(badges);

  if (sources.length) {
    const wrap = document.createElement("div");
    wrap.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = `Grounded in ${sources.length} source${sources.length > 1 ? "s" : ""}`;
    wrap.appendChild(title);
    for (const s of sources) {
      const row = document.createElement("div");
      row.className = "source";
      const num = document.createElement("span");
      num.className = "num";
      num.textContent = `[${s.n}]`;
      const txt = document.createElement("span");
      txt.className = "txt";
      txt.textContent = s.text;
      row.append(num, txt);
      wrap.appendChild(row);
    }
    meta.appendChild(wrap);
  }
  bubble.appendChild(meta);
}

// ---- send ----
async function send(text) {
  const message = text.trim();
  if (!message || sendBtn.disabled) return;

  addMessage("user").md.textContent = message;
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  const { bubble, md } = addMessage("bot");
  md.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, user_id: userIdEl.value.trim() || "demo-user" }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = "";
    let started = false;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      full += decoder.decode(value, { stream: true });
      if (!started) {
        bubble.classList.add("streaming");
        started = true;
      }
      md.textContent = splitMeta(full).body; // plain while streaming (no flicker)
      scrollDown();
    }

    // finalize: rich markdown body + structured meta footer
    bubble.classList.remove("streaming");
    const { body, meta } = splitMeta(full);
    md.innerHTML = renderMarkdown(body) || "<em>(no response)</em>";
    if (meta) renderMeta(bubble, meta);
  } catch (err) {
    bubble.classList.remove("streaming");
    md.textContent = `⚠️ ${err.message}. Is the server running?`;
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
  input.style.height = Math.min(input.scrollHeight, 168) + "px";
});
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => send(chip.dataset.q))
);
input.focus();
