const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

// ---------- Health ----------
async function refreshHealth() {
  const pill = $("#healthPill");
  const text = $("#healthText");
  try {
    const r = await fetch("/health");
    const d = await r.json();
    pill.classList.toggle("ok", d.status === "ok" && d.models_loaded);
    pill.classList.toggle("bad", !d.models_loaded);
    text.textContent = `${d.device} · v${d.version}`;
  } catch {
    pill.classList.add("bad");
    text.textContent = "offline";
  }
}
refreshHealth();
setInterval(refreshHealth, 10000);

// ---------- Tabs ----------
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// ---------- Drop-zone helper ----------
function setupDrop(zone, fileInput, titleEl, onFile, defaultTitle) {
  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    if (e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  function handleFile(f) {
    titleEl.textContent = f.name;
    zone.classList.add("has-file");
    onFile(f);
  }
  return {
    reset() {
      fileInput.value = "";
      titleEl.textContent = defaultTitle;
      zone.classList.remove("has-file");
    },
  };
}

// ---------- SIGN → TEXT ----------
const signDrop = $("#signDrop");
const signFile = $("#signFile");
const signPreview = $("#signPreview");
const signBtn = $("#signBtn");
const signClear = $("#signClear");
const signResult = $("#signResult");
let signSelected = null;

const signZone = setupDrop(signDrop, signFile, $("#signDropTitle"), (f) => {
  signSelected = f;
  signPreview.src = URL.createObjectURL(f);
  signPreview.classList.remove("hidden");
  signBtn.disabled = false;
  signClear.classList.remove("hidden");
  signResult.innerHTML = "";
}, "Drag a video here");

signClear.addEventListener("click", () => {
  signSelected = null;
  signZone.reset();
  signPreview.classList.add("hidden");
  signPreview.removeAttribute("src");
  signBtn.disabled = true;
  signClear.classList.add("hidden");
  signResult.innerHTML = "";
});

signBtn.addEventListener("click", async () => {
  if (!signSelected) return;
  signBtn.classList.add("loading");
  signBtn.disabled = true;
  signResult.innerHTML = '<div class="spinner-block"><div class="spinner"></div><div>Extracting pose with HRNet, then classifying with R(2+1)D-18 — usually 30–60s on Apple MPS…</div></div>';
  const fd = new FormData();
  fd.append("video", signSelected);
  try {
    const r = await fetch("/sign-to-text", { method: "POST", body: fd });
    if (!r.ok) {
      signResult.innerHTML = `<div class="error">HTTP ${r.status}: ${escape(await r.text())}</div>`;
      return;
    }
    const d = await r.json();
    renderSignResult(d);
  } catch (e) {
    signResult.innerHTML = `<div class="error">${escape(e.message)}</div>`;
  } finally {
    signBtn.classList.remove("loading");
    signBtn.disabled = false;
  }
});

function renderSignResult(d) {
  const conf = (d.confidence * 100).toFixed(1);
  const alts = d.alternatives.map((a) => {
    const pct = (a.confidence * 100).toFixed(1);
    return `
      <div class="alt-row">
        <div class="alt-name">
          <span>${escape(a.text)}</span>
          <div class="alt-bar-wrap"><div class="alt-bar" style="transform: scaleX(${a.confidence})"></div></div>
        </div>
        <div class="alt-conf">${pct}%</div>
      </div>`;
  }).join("");
  signResult.innerHTML = `
    <div class="result-card">
      <div class="result-top">${escape(d.text)}</div>
      <div class="result-status">
        <div class="result-conf">Confidence <strong>${conf}%</strong></div>
        <div class="result-meta">${d.duration_ms} ms</div>
      </div>
      <div class="alts">
        <div class="alt-row" style="border-top: 0;">
          <div class="alt-name">
            <span><strong>${escape(d.text)}</strong> <span style="color: var(--fg-mute); font-size:12px;">(top-1)</span></span>
            <div class="alt-bar-wrap"><div class="alt-bar" style="transform: scaleX(${d.confidence})"></div></div>
          </div>
          <div class="alt-conf">${conf}%</div>
        </div>
        ${alts}
      </div>
    </div>`;
}

// ---------- Profiles ----------
async function refreshProfiles() {
  try {
    const r = await fetch("/profiles");
    const profiles = await r.json();
    const sel = $("#profileSelect");
    const cur = sel.value;
    sel.innerHTML = '<option value="">— None —</option>' + profiles.map((p) => `<option>${escape(p.name)}</option>`).join("");
    sel.value = cur || "";
  } catch {}
}
refreshProfiles();

// ---------- SPEECH → SIGN ----------
const speechDrop = $("#speechDrop");
const speechFile = $("#speechFile");
const speechPreview = $("#speechPreview");
const speechBtn = $("#speechBtn");
const speechClear = $("#speechClear");
const speechResult = $("#speechResult");
const recordBtn = $("#recordBtn");
const recordText = recordBtn.querySelector(".record-text");
let speechSelected = null;
let mediaRecorder = null;
let recordedChunks = [];

const speechZone = setupDrop(speechDrop, speechFile, $("#speechDropTitle"), (f) => {
  setSpeechFile(f);
}, "Upload audio file");

function setSpeechFile(f) {
  speechSelected = f;
  speechPreview.src = URL.createObjectURL(f);
  speechPreview.classList.remove("hidden");
  speechBtn.disabled = false;
  speechClear.classList.remove("hidden");
  speechResult.innerHTML = "";
}

speechClear.addEventListener("click", () => {
  speechSelected = null;
  speechZone.reset();
  speechPreview.classList.add("hidden");
  speechPreview.removeAttribute("src");
  speechBtn.disabled = true;
  speechClear.classList.add("hidden");
  speechResult.innerHTML = "";
});

// In-browser mic recording via MediaRecorder
recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    });
    mediaRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(recordedChunks, { type: mime });
      const file = new File([blob], `recording-${Date.now()}.webm`, { type: mime });
      setSpeechFile(file);
      recordBtn.classList.remove("recording");
      recordText.textContent = "Record from microphone";
    });
    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordText.textContent = "Stop recording";
  } catch (e) {
    speechResult.innerHTML = `<div class="error">Microphone access denied: ${escape(e.message)}</div>`;
  }
});

speechBtn.addEventListener("click", async () => {
  if (!speechSelected) return;
  speechBtn.classList.add("loading");
  speechBtn.disabled = true;
  speechResult.innerHTML = '<div class="spinner-block"><div class="spinner"></div><div>Transcribing with Whisper, lemmatizing with zeyrek, looking up AUTSL signs, stitching with ffmpeg…</div></div>';
  const fd = new FormData();
  fd.append("audio", speechSelected);
  const profile = $("#profileSelect").value;
  if (profile) fd.append("profile", profile);
  try {
    const r = await fetch("/speech-to-sign", { method: "POST", body: fd });
    if (!r.ok) {
      speechResult.innerHTML = `<div class="error">HTTP ${r.status}: ${escape(await r.text())}</div>`;
      return;
    }
    const d = await r.json();
    renderSpeechResult(d);
  } catch (e) {
    speechResult.innerHTML = `<div class="error">${escape(e.message)}</div>`;
  } finally {
    speechBtn.classList.remove("loading");
    speechBtn.disabled = false;
  }
});

function renderSpeechResult(d) {
  const matched = (d.matched_signs || []).map((m) => `
    <span class="matched-pill">
      <span>${escape(m.turkish)}</span>
      <span style="color: var(--fg-mute);">·</span>
      <span style="color: var(--fg-dim);">${escape(m.english)}</span>
      <span class="id">#${m.class_id}</span>
    </span>
  `).join("");
  const missing = (d.missing_words || []).map((w) => `<span class="missing-pill">${escape(w)}</span>`).join("");
  const videoName = d.output_video_path.split("/").pop();
  speechResult.innerHTML = `
    <div class="result-card">
      <div class="result-status">
        <div class="result-meta">${d.duration_ms} ms · ${d.matched_signs.length} matched / ${d.missing_words.length} missing</div>
      </div>
      <div class="transcript">${escape(d.transcript || "(empty transcript)")}</div>
      ${matched ? `<div class="matched-list">${matched}</div>` : ""}
      <video controls autoplay class="result-video" src="/clips/${escape(videoName)}"></video>
      ${missing ? `<div class="missing-list"><span class="missing-label">Not in vocabulary:</span>${missing}</div>` : ""}
    </div>`;
}
