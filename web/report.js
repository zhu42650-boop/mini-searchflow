async function loadReport() {
  const params = new URLSearchParams(window.location.search);
  const threadId = params.get("thread_id");
  const reportPath = params.get("report_path");
  const contentEl = document.getElementById("content");
  const titleEl = document.getElementById("title");
  const metaEl = document.getElementById("meta");

  if (!threadId && !reportPath) {
    contentEl.textContent = "缺少 thread_id";
    return;
  }

  try {
    let response = null;
    let data = null;

    if (threadId) {
      response = await fetch(`/api/report/${encodeURIComponent(threadId)}`);
      data = await response.json();
    }

    if ((!response || !response.ok) && reportPath) {
      response = await fetch(`/api/report/file?path=${encodeURIComponent(reportPath)}`);
      data = await response.json();
    }

    if (!response || !response.ok) {
      throw new Error((data && data.detail) || "failed to load report");
    }

    titleEl.textContent = data.report_name || "研究报告";
    metaEl.textContent = data.report_path || "";

    const markdown = normalizeMarkdown(String(data.content || ""));

    if (window.marked && window.DOMPurify) {
      window.marked.setOptions({
        gfm: true,
        breaks: true,
      });
      const html = window.marked.parse(markdown);
      contentEl.innerHTML = window.DOMPurify.sanitize(html);
      attachImageFallback(contentEl);
    } else {
      // Fallback if CDN scripts fail
      contentEl.textContent = markdown;
    }
  } catch (err) {
    contentEl.textContent = `报告加载失败: ${String(err)}`;
  }
}

function normalizeMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const compact = [];
  for (let i = 0; i < lines.length; i += 1) {
    const prev = i > 0 ? lines[i - 1] : "";
    const next = i + 1 < lines.length ? lines[i + 1] : "";
    const current = lines[i];
    const isTableish = (line) => {
      const s = String(line || "").trim();
      return s && !s.startsWith("```") && (s.match(/\|/g) || []).length >= 2;
    };
    if (!current.trim() && isTableish(prev) && isTableish(next)) {
      continue;
    }
    compact.push(current);
  }

  return compact
    .join("\n")
    .replace(/^\\(#+\s)/gm, "$1")
    .replace(/!\[([^\]]*)\]\(\s*(https?:\/\/[^)\s]+)\s+\)/g, "![$1]($2)");
}

function attachImageFallback(container) {
  const images = container.querySelectorAll("img");
  images.forEach((img) => {
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";
    img.onerror = () => {
      const wrapper = document.createElement("div");
      wrapper.className = "img-fallback";
      const src = img.getAttribute("src") || "";
      wrapper.innerHTML = `图片加载失败：<a href=\"${src}\" target=\"_blank\" rel=\"noopener noreferrer\">${src}</a>`;
      if (img.parentNode) {
        img.parentNode.replaceChild(wrapper, img);
      }
    };
  });
}

void loadReport();
