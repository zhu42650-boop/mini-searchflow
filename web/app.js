const state = {
  threadId: null,
  running: false,
  typingEl: null,
  researchMode: false,
  directReplyCandidate: "",
  renderedReports: new Set(),
};

const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const subquestionsEl = document.getElementById("subquestions");
const reportFilesEl = document.getElementById("report-files");
const reportFileEmptyEl = document.getElementById("report-file-empty");
const interruptBox = document.getElementById("interrupt-box");
const interruptText = document.getElementById("interrupt-text");
const editInput = document.getElementById("edit-input");
const submitEdit = document.getElementById("submit-edit");
const questionEl = document.getElementById("question");
const researchPanelEl = document.querySelector(".panel.research");

function setStatus(kind, text) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = text;
}

function scrollBottom(el) {
  el.scrollTop = el.scrollHeight;
}

function addMessage(role, name, content) {
  const item = document.createElement("div");
  item.className = `msg ${role || "assistant"}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = name || role || "assistant";

  const body = document.createElement("div");
  body.className = "content";
  body.textContent = String(content || "");

  item.appendChild(meta);
  item.appendChild(body);
  messagesEl.appendChild(item);
  scrollBottom(messagesEl);
}

function addTyping() {
  if (state.typingEl) return;
  const item = document.createElement("div");
  item.className = "msg assistant";
  item.id = "typing-msg";
  item.innerHTML = `
    <div class="meta">system</div>
    <div class="typing"><span></span><span></span><span></span></div>
  `;
  messagesEl.appendChild(item);
  state.typingEl = item;
  scrollBottom(messagesEl);
}

function removeTyping() {
  if (state.typingEl && state.typingEl.parentNode) {
    state.typingEl.parentNode.removeChild(state.typingEl);
  }
  state.typingEl = null;
}

function renderSubquestions(items) {
  subquestionsEl.innerHTML = "";
  items.forEach((it, idx) => {
    const li = document.createElement("li");
    li.className = it.done ? "done" : "";
    const stepType = String(it.step_type || "").replace(/^TaskType\./, "").toLowerCase();
    li.textContent = `${idx + 1}. [${stepType}] ${it.question}`;
    subquestionsEl.appendChild(li);
  });
}

function renderReportFile(reportName, reportPath) {
  reportFileEmptyEl.classList.add("hidden");
  const key = `${reportName || "report.md"}|${state.threadId || ""}|${reportPath || ""}`;
  if (state.renderedReports.has(key)) return;
  state.renderedReports.add(key);

  const card = document.createElement("div");
  card.className = "report-file";
  card.innerHTML = `
    <div class="report-file-name">${reportName || "report.md"}</div>
    <div class="report-file-meta">点击查看 Markdown 渲染结果</div>
  `;
  card.addEventListener("click", () => {
    if (!state.threadId) return;
    const params = new URLSearchParams({
      thread_id: state.threadId,
    });
    if (reportPath) {
      params.set("report_path", reportPath);
    }
    const target = `/web/report.html?${params.toString()}`;
    window.open(target, "_blank");
  });

  reportFilesEl.appendChild(card);
}

function resetView() {
  messagesEl.innerHTML = "";
  subquestionsEl.innerHTML = "";
  reportFilesEl.innerHTML = "";
  reportFileEmptyEl.classList.remove("hidden");
  state.researchMode = false;
  state.directReplyCandidate = "";
  researchPanelEl.classList.remove("active");
  interruptBox.classList.add("hidden");
  editInput.classList.add("hidden");
  submitEdit.classList.add("hidden");
  editInput.value = "";
  state.renderedReports.clear();
}

function parseSSEBlocks(buffer) {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  return {
    complete: blocks.slice(0, -1),
    rest: blocks[blocks.length - 1] || "",
  };
}

function parseSSEBlock(block) {
  let event = "message";
  const dataLines = [];

  block.split("\n").forEach((line) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      return;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  });

  const dataText = dataLines.join("\n");
  if (!dataText) return null;

  try {
    return { event, payload: JSON.parse(dataText) };
  } catch {
    return { event: "error", payload: { error: `SSE JSON parse failed: ${dataText}` } };
  }
}

function handleEvent(event, payload) {
  if (event === "mode" && payload.kind === "research") {
    state.researchMode = true;
    researchPanelEl.classList.add("active");
    return;
  }

  if (event === "thread") {
    state.threadId = payload.thread_id;
    return;
  }

  if (event === "message") {
    const role = payload.role || "assistant";
    const name = payload.name || role;

    // Left panel rules:
    // 1) Ignore streamed user echoes from backend to avoid duplicated user bubbles.
    // 2) Ignore all intermediate agent messages for deep-research mode.
    // 3) Keep coordinator direct reply candidate and render only at the end.
    if (role === "user") {
      return;
    }

    if (!state.researchMode && role === "assistant") {
      state.directReplyCandidate = String(payload.content || "").trim();
    }

    return;
  }

  if (event === "subquestions") {
    state.researchMode = true;
    researchPanelEl.classList.add("active");
    renderSubquestions(payload.items || []);
    return;
  }

  if (event === "interrupt") {
    state.running = false;
    setStatus("idle", "等待反馈");
    removeTyping();
    interruptText.textContent = payload.content || "需要反馈";
    interruptBox.classList.remove("hidden");
    return;
  }

  if (event === "final_report") {
    state.researchMode = true;
    researchPanelEl.classList.add("active");
    renderReportFile(payload.report_name, payload.report_path);
    return;
  }

  if (event === "error") {
    setStatus("error", "发生错误");
    removeTyping();
    addMessage("system", "error", payload.error || "unknown error");
    return;
  }

  if (event === "done") {
    state.running = false;
    removeTyping();
    setStatus("done", "研究完成");

    // Direct response path: only show one system output on the left panel.
    if (!state.researchMode && state.directReplyCandidate) {
      addMessage("system", "system", state.directReplyCandidate);
    }
  }
}

async function consumeStream(payload) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    setStatus("error", "请求失败");
    addMessage("system", "error", `请求失败: ${response.status}`);
    return;
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        const trailing = buffer.trim();
        if (trailing) {
          const parsed = parseSSEBlock(trailing);
          if (parsed) handleEvent(parsed.event, parsed.payload);
        }
        break;
      }

      buffer += value;
      const parsedBlocks = parseSSEBlocks(buffer);
      buffer = parsedBlocks.rest;

      parsedBlocks.complete.forEach((b) => {
        const parsed = parseSSEBlock(b);
        if (parsed) handleEvent(parsed.event, parsed.payload);
      });
    }
  } finally {
    if (state.running) {
      state.running = false;
      removeTyping();
      if (statusEl.classList.contains("running")) {
        setStatus("idle", "待命");
      }
    }
  }
}

async function sendNewQuestion() {
  const message = questionEl.value.trim();
  if (!message || state.running) return;

  // Per-turn reset to avoid previous mode affecting current turn rendering.
  state.researchMode = false;
  state.directReplyCandidate = "";

  state.running = true;
  setStatus("running", "研究中");
  addMessage("user", "user", message);
  addTyping();
  questionEl.value = "";

  await consumeStream({
    message,
    locale: "zh-CN",
    max_son_questions: 5,
    enable_human_feedback: false,
    enable_background_investigation: true,
    thread_id: state.threadId || "__default__",
  });
}

async function sendInterruptFeedback(feedback) {
  if (!state.threadId || state.running) return;

  state.running = true;
  setStatus("running", "继续执行");
  interruptBox.classList.add("hidden");
  addTyping();

  await consumeStream({
    thread_id: state.threadId,
    interrupt_feedback: feedback,
  });
}

document.getElementById("send-btn").addEventListener("click", () => {
  void sendNewQuestion();
});

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    void sendNewQuestion();
  }
});

document.getElementById("clear-btn").addEventListener("click", () => {
  resetView();
  state.threadId = null;
  state.running = false;
  state.researchMode = false;
  state.directReplyCandidate = "";
  setStatus("idle", "待命");
});

document.getElementById("accept-btn").addEventListener("click", () => {
  void sendInterruptFeedback("[ACCEPTED]");
});

document.getElementById("edit-btn").addEventListener("click", () => {
  editInput.classList.remove("hidden");
  submitEdit.classList.remove("hidden");
});

submitEdit.addEventListener("click", () => {
  const edit = editInput.value.trim() || "请重新拆分并给出更可执行的问题";
  void sendInterruptFeedback(`[EDIT_PLAN] ${edit}`);
});

resetView();
