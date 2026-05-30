// Grid Splitter · Workbench (v3) — vanilla JS, mode: empty / loaded / cut

const $ = (id) => document.getElementById(id);

const els = {
  fileName: $("file-name"),
  fileSize: $("file-size"),
  statusChip: $("status-chip"),
  statusText: $("status-text"),
  btnReset: $("btn-reset"),
  dropZone: $("drop-zone"),
  fileInput: $("file-input"),
  stage: $("stage"),
  stageInner: $("stage-inner"),
  previewImg: $("preview-img"),
  overlay: $("grid-overlay"),
  inspector: $("inspector"),
  autoGrid: $("auto-grid"),
  autoSource: $("auto-source"),
  autoFallback: $("auto-fallback"),
  inputRows: $("input-rows"),
  inputCols: $("input-cols"),
  btnPreview: $("btn-preview"),
  outCell: $("out-cell"),
  outCount: $("out-count"),
  btnSplit: $("btn-split"),
  kbdCut: $("kbd-cut"),
  footerStrip: $("footer-strip"),
  resultCount: $("result-count"),
  resultCell: $("result-cell"),
  clipsStrip: $("clips-strip"),
  btnZip: $("btn-zip"),
  busy: $("busy"),
  busyText: $("busy-text"),
  toast: $("toast"),
};

const state = {
  fileId: null,
  fileName: "",
  imgW: 0,
  imgH: 0,
  borders: null,
  hLines: [],
  vLines: [],
  rows: 0,
  cols: 0,
};
window.state = state;

const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);
els.kbdCut.textContent = IS_MAC ? "⌘ ↵" : "Ctrl ↵";

function setBusy(on, text = "Working…") {
  els.busyText.textContent = text;
  els.busy.hidden = !on;
}

let toastTimer = null;
function toast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.hidden = true), 4000);
}

function setMode(mode) {
  if (mode === "empty") {
    els.dropZone.hidden = false;
    els.stage.hidden = true;
    els.inspector.dataset.state = "empty";
    els.fileSize.hidden = true;
    els.statusChip.hidden = true;
    els.btnReset.hidden = true;
    els.footerStrip.hidden = true;
    els.inputRows.disabled = true;
    els.inputCols.disabled = true;
    els.btnPreview.disabled = true;
    els.btnSplit.disabled = true;
    els.fileName.textContent = "untitled";
    return;
  }
  els.dropZone.hidden = true;
  els.stage.hidden = false;
  els.inspector.dataset.state = "ready";
  els.fileSize.hidden = false;
  els.statusChip.hidden = false;
  els.btnReset.hidden = false;
  els.inputRows.disabled = false;
  els.inputCols.disabled = false;
  els.btnPreview.disabled = false;
  els.btnSplit.disabled = false;
  els.footerStrip.hidden = mode !== "cut";
}

// Upload
els.dropZone.addEventListener("click", () => els.fileInput.click());
els.dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) doUpload(f);
});
["dragover", "dragenter"].forEach((evt) =>
  els.dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropZone.classList.add("is-dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("is-dragover");
  })
);
els.dropZone.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) doUpload(f);
});

async function doUpload(file) {
  if (!/\.(png|jpe?g)$/i.test(file.name)) return toast("仅支持 PNG / JPG / JPEG");
  setBusy(true, "Detecting grid…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await readErr(r));
    onUploaded(await r.json());
  } catch (err) {
    toast(err.message || "上传失败");
  } finally {
    setBusy(false);
  }
}

function onUploaded(d) {
  state.fileId = d.file_id;
  state.fileName = d.original_name;
  state.imgW = d.img_w;
  state.imgH = d.img_h;
  state.borders = d.borders;
  state.hLines = d.h_lines;
  state.vLines = d.v_lines;
  state.rows = d.rows;
  state.cols = d.cols;

  els.fileName.textContent = d.original_name;
  els.fileSize.textContent = `${d.img_w} × ${d.img_h}`;
  els.statusText.textContent = d.auto_detected ? "detected" : "manual";
  els.autoGrid.textContent = `${d.rows} × ${d.cols}`;
  els.autoSource.textContent = d.auto_detected ? "auto" : "fallback";
  els.autoFallback.hidden = d.auto_detected;
  els.inputRows.value = d.rows;
  els.inputCols.value = d.cols;
  updateEstimate();

  els.previewImg.src = d.preview_url;
  els.previewImg.onload = drawOverlay;
  setMode("loaded");
}

function drawOverlay() {
  const svg = els.overlay;
  svg.setAttribute("viewBox", `0 0 ${state.imgW} ${state.imgH}`);
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const { top, bot, left, right } = state.borders || { top: 0, bot: 0, left: 0, right: 0 };
  const STROKE = "rgb(245, 185, 66)";
  const MASK = "rgba(245, 185, 66, 0.14)";

  const addRect = (x, y, w, h) => {
    if (w <= 0 || h <= 0) return;
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", w); r.setAttribute("height", h);
    r.setAttribute("fill", MASK);
    svg.appendChild(r);
  };
  if (top) addRect(0, 0, state.imgW, top);
  if (bot) addRect(0, state.imgH - bot, state.imgW, bot);
  if (left) addRect(0, top, left, state.imgH - top - bot);
  if (right) addRect(state.imgW - right, top, right, state.imgH - top - bot);

  const strokeW = Math.max(1, state.imgW / 1800);
  const addLine = (x1, y1, x2, y2) => {
    const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
    ln.setAttribute("x1", x1); ln.setAttribute("y1", y1);
    ln.setAttribute("x2", x2); ln.setAttribute("y2", y2);
    ln.setAttribute("stroke", STROKE);
    ln.setAttribute("stroke-width", strokeW);
    ln.setAttribute("stroke-linecap", "square");
    svg.appendChild(ln);
  };
  for (const y of state.hLines) addLine(left, y, state.imgW - right, y);
  for (const x of state.vLines) addLine(x, top, x, state.imgH - bot);
}

function updateEstimate() {
  const rows = parseInt(els.inputRows.value, 10);
  const cols = parseInt(els.inputCols.value, 10);
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1) {
    els.outCell.textContent = "—";
    els.outCount.textContent = "—";
    return;
  }
  const { top = 0, bot = 0, left = 0, right = 0 } = state.borders || {};
  const usableW = state.imgW - left - right;
  const usableH = state.imgH - top - bot;
  els.outCell.textContent = `${Math.floor(usableW / cols)} × ${Math.floor(usableH / rows)}`;
  els.outCount.textContent = String(rows * cols);
}

els.inputRows.addEventListener("input", updateEstimate);
els.inputCols.addEventListener("input", updateEstimate);

els.btnPreview.addEventListener("click", doPreview);
[els.inputRows, els.inputCols].forEach((el) =>
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (e.metaKey || e.ctrlKey) doSplit();
      else doPreview();
    }
  })
);
els.btnSplit.addEventListener("click", doSplit);

async function doPreview() {
  if (!state.fileId) return;
  const rows = parseInt(els.inputRows.value, 10);
  const cols = parseInt(els.inputCols.value, 10);
  if (!validateGrid(rows, cols)) return;
  setBusy(true, "Re-previewing…");
  try {
    const r = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: state.fileId, rows, cols }),
    });
    if (!r.ok) throw new Error(await readErr(r));
    const d = await r.json();
    state.rows = rows;
    state.cols = cols;
    state.borders = d.borders;
    state.hLines = d.h_lines;
    state.vLines = d.v_lines;
    drawOverlay();
    updateEstimate();
    els.statusText.textContent = "manual";
  } catch (err) {
    toast(err.message || "预览失败");
  } finally {
    setBusy(false);
  }
}

async function doSplit() {
  if (!state.fileId) return;
  const rows = parseInt(els.inputRows.value, 10);
  const cols = parseInt(els.inputCols.value, 10);
  if (!validateGrid(rows, cols)) return;
  setBusy(true, `Cutting ${rows * cols} cells…`);
  try {
    const r = await fetch("/api/split", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: state.fileId, rows, cols }),
    });
    if (!r.ok) throw new Error(await readErr(r));
    onSplit(await r.json());
  } catch (err) {
    toast(err.message || "切割失败");
  } finally {
    setBusy(false);
  }
}

function onSplit(d) {
  els.resultCount.textContent = `${d.count} clips`;
  els.resultCell.textContent = `${d.cell_w} × ${d.cell_h}`;
  els.btnZip.href = d.zip_url;
  els.btnZip.setAttribute("download", "");

  els.clipsStrip.innerHTML = "";
  for (const clip of d.clips) {
    const card = document.createElement("a");
    card.className = "wb__clip";
    card.href = clip.url;
    card.setAttribute("download", clip.name);
    card.title = clip.name;

    const img = document.createElement("img");
    img.src = clip.url;
    img.loading = "lazy";
    img.alt = clip.name;

    const lbl = document.createElement("span");
    lbl.className = "wb__clip-label";
    lbl.textContent = String(clip.idx).padStart(3, "0");

    const dl = document.createElement("span");
    dl.className = "wb__clip-dl";
    dl.textContent = "↓ download";

    card.append(img, lbl, dl);
    els.clipsStrip.appendChild(card);
  }

  setMode("cut");
  els.statusText.textContent = "cut";
}

els.btnReset.addEventListener("click", doReset);

async function doReset() {
  if (state.fileId) {
    fetch(`/api/session/${state.fileId}`, { method: "DELETE" }).catch(() => {});
  }
  state.fileId = null;
  state.fileName = "";
  state.imgW = state.imgH = 0;
  state.borders = null;
  state.hLines = [];
  state.vLines = [];
  els.fileInput.value = "";
  els.previewImg.removeAttribute("src");
  els.clipsStrip.innerHTML = "";
  els.overlay.innerHTML = "";
  setMode("empty");
}

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !els.btnSplit.disabled) {
    e.preventDefault();
    doSplit();
    return;
  }
  if (e.key === "Escape" && state.fileId && els.busy.hidden) {
    doReset();
  }
});

function validateGrid(rows, cols) {
  if (
    !Number.isInteger(rows) || !Number.isInteger(cols) ||
    rows < 1 || cols < 1 || rows > 20 || cols > 20
  ) {
    toast("Rows / Cols 必须是 1–20 之间的整数");
    return false;
  }
  return true;
}

async function readErr(r) {
  try {
    const j = await r.json();
    return j.detail || `HTTP ${r.status}`;
  } catch {
    return `HTTP ${r.status}`;
  }
}

setMode("empty");
