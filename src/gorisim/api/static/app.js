const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const HISTORY_KEY = "gorisim_history_v1";

// ---------- Toasts ----------
function toast(msg, kind = "info", ms = 3500) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toastStack").appendChild(el);
  setTimeout(() => { el.classList.add("fadeout"); setTimeout(() => el.remove(), 300); }, ms);
}

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

// ---------- Drop helper ----------
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

// ---------- Copy ----------
function copyBtn(text) {
  return `<button class="copy-btn" data-copy="${escape(text)}" title="Copy"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`;
}
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-copy]");
  if (btn) {
    navigator.clipboard.writeText(btn.dataset.copy).then(() => toast("Copied", "ok", 1500));
  }
});

// ---------- History ----------
function historyLoad() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}
function historySave(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 30)));
}
function historyAdd(entry) {
  const items = historyLoad();
  items.unshift({ ...entry, ts: Date.now() });
  historySave(items);
  historyRender();
}
function historyRender() {
  const items = historyLoad();
  const body = $("#historyBody");
  if (!items.length) {
    body.innerHTML = '<div class="muted center">No history yet — translations you make appear here.</div>';
    return;
  }
  body.innerHTML = items.map((it) => {
    const time = new Date(it.ts).toLocaleTimeString();
    const tag = it.kind === "sign" ? "Sign→Text" : "Speech→Sign";
    const tagClass = it.kind;
    const sub = it.kind === "sign"
      ? `${(it.confidence*100).toFixed(1)}% · ${it.duration_ms} ms`
      : `${it.matched_count} matched · ${it.duration_ms} ms`;
    return `
      <div class="history-item">
        <div class="history-row">
          <span class="history-tag ${tagClass}">${tag}</span>
          <span class="history-time">${time}</span>
        </div>
        <div class="history-content">${escape(it.summary)}</div>
        <div class="history-sub">${sub}</div>
      </div>`;
  }).join("");
}
historyRender();

$("#historyToggle").addEventListener("click", () => $("#historyDrawer").classList.toggle("open"));
$("#historyClose").addEventListener("click", () => $("#historyDrawer").classList.remove("open"));
$("#historyClear").addEventListener("click", () => {
  if (!confirm("Clear all history?")) return;
  localStorage.removeItem(HISTORY_KEY);
  historyRender();
  toast("History cleared", "info");
});

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

signBtn.addEventListener("click", runSign);
async function runSign() {
  if (!signSelected || signBtn.disabled) return;
  signBtn.classList.add("loading");
  signBtn.disabled = true;
  signResult.innerHTML = '<div class="spinner-block"><div class="spinner"></div><div>Extracting pose with HRNet, then classifying with R(2+1)D-18 — usually 30–60s on Apple MPS…</div></div>';
  const fd = new FormData();
  fd.append("video", signSelected);
  try {
    const r = await fetch("/sign-to-text", { method: "POST", body: fd });
    if (!r.ok) {
      const txt = await r.text();
      signResult.innerHTML = `<div class="error">HTTP ${r.status}: ${escape(txt)}</div>`;
      toast(`Sign→Text failed: HTTP ${r.status}`, "err");
      return;
    }
    const d = await r.json();
    renderSignResult(d);
    historyAdd({ kind: "sign", summary: d.text, confidence: d.confidence, duration_ms: d.duration_ms });
    toast(`Predicted: ${d.text} (${(d.confidence*100).toFixed(0)}%)`, "ok");
  } catch (e) {
    signResult.innerHTML = `<div class="error">${escape(e.message)}</div>`;
    toast(`Network error: ${e.message}`, "err");
  } finally {
    signBtn.classList.remove("loading");
    signBtn.disabled = false;
  }
}

function renderSignResult(d) {
  const conf = (d.confidence * 100).toFixed(1);
  const altRows = [{ text: d.text, confidence: d.confidence, top1: true }, ...d.alternatives];
  const rowsHtml = altRows.map((a, i) => {
    const pct = (a.confidence * 100).toFixed(1);
    return `
      <div class="alt-row" style="animation: fadeIn 0.4s ease ${i*0.06}s backwards;">
        <div class="alt-name">
          <span>${a.top1 ? `<strong>${escape(a.text)}</strong> <span style="color: var(--fg-mute); font-size:12px;">(top-1)</span>` : escape(a.text)}</span>
          <div class="alt-bar-wrap"><div class="alt-bar" style="transform: scaleX(${a.confidence})"></div></div>
        </div>
        <div class="alt-conf">${pct}%</div>
      </div>`;
  }).join("");
  signResult.innerHTML = `
    <div class="result-card">
      <div class="result-top">
        <span>${escape(d.text)}</span>
        ${copyBtn(d.text)}
      </div>
      <div class="result-status">
        <div class="result-conf">Confidence <strong>${conf}%</strong></div>
        <div class="result-meta">${d.duration_ms} ms</div>
      </div>
      <div class="alts">${rowsHtml}</div>
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
    renderProfileList(profiles);
  } catch {}
}
function renderProfileList(profiles) {
  const list = $("#profileList");
  if (!profiles.length) { list.innerHTML = '<div class="muted">No profiles enrolled yet.</div>'; return; }
  list.innerHTML = profiles.map((p) => `
    <div class="profile-item">
      <span class="profile-name">${escape(p.name)}</span>
      <button class="btn btn-danger small" data-delete-profile="${escape(p.name)}">Delete</button>
    </div>
  `).join("");
}
refreshProfiles();

document.addEventListener("click", async (e) => {
  const del = e.target.closest("[data-delete-profile]");
  if (!del) return;
  const name = del.dataset.deleteProfile;
  if (!confirm(`Delete profile "${name}"?`)) return;
  const r = await fetch(`/profiles/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (r.ok) { toast(`Deleted "${name}"`, "ok"); refreshProfiles(); }
  else toast(`Delete failed: HTTP ${r.status}`, "err");
});

// ---------- Profile modal + enrollment ----------
const profileModal = $("#profileModal");
$("#settingsBtn").addEventListener("click", openProfileModal);
$("#manageProfilesBtn").addEventListener("click", openProfileModal);
$("#profileModalClose").addEventListener("click", () => profileModal.classList.add("hidden"));
profileModal.addEventListener("click", (e) => { if (e.target === profileModal) profileModal.classList.add("hidden"); });

function openProfileModal() {
  refreshProfiles();
  enrollSlots = [null, null, null];
  resetEnrollSlots();
  $("#profileName").value = "";
  updateEnrollSubmit();
  profileModal.classList.remove("hidden");
}

let enrollSlots = [null, null, null];
let enrollRecorder = null;
let enrollChunks = [];
let enrollActiveSlot = -1;

function resetEnrollSlots() {
  $$(".enroll-slot").forEach((slot, i) => {
    slot.classList.remove("recording", "done");
    slot.querySelector(".enroll-status").textContent = enrollSlots[i] ? "Recorded ✓" : "Click to record";
    if (enrollSlots[i]) slot.classList.add("done");
  });
}

$$(".enroll-slot").forEach((slot) => {
  slot.addEventListener("click", () => onEnrollSlotClick(parseInt(slot.dataset.slot, 10)));
});

async function onEnrollSlotClick(idx) {
  if (enrollRecorder && enrollRecorder.state === "recording") {
    enrollRecorder.stop();
    return;
  }
  const slot = $$(".enroll-slot")[idx];
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    enrollChunks = [];
    enrollActiveSlot = idx;
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
    enrollRecorder = new MediaRecorder(stream, { mimeType: mime });
    enrollRecorder.addEventListener("dataavailable", (e) => { if (e.data.size > 0) enrollChunks.push(e.data); });
    enrollRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(enrollChunks, { type: mime });
      const file = new File([blob], `slot${idx}.webm`, { type: mime });
      enrollSlots[idx] = file;
      slot.classList.remove("recording");
      resetEnrollSlots();
      updateEnrollSubmit();
    });
    enrollRecorder.start();
    slot.classList.add("recording");
    slot.querySelector(".enroll-status").textContent = "Recording — click to stop";
  } catch (e) {
    toast(`Mic error: ${e.message}`, "err");
  }
}

function updateEnrollSubmit() {
  const name = $("#profileName").value.trim();
  const allFilled = enrollSlots.every(Boolean);
  $("#enrollSubmit").disabled = !(name && allFilled);
}
$("#profileName").addEventListener("input", updateEnrollSubmit);

$("#enrollSubmit").addEventListener("click", async () => {
  const name = $("#profileName").value.trim();
  if (!name || !enrollSlots.every(Boolean)) return;
  const fd = new FormData();
  fd.append("name", name);
  enrollSlots.forEach((f) => fd.append("audio", f));
  $("#enrollSubmit").classList.add("loading");
  $("#enrollSubmit").disabled = true;
  try {
    const r = await fetch("/enroll", { method: "POST", body: fd });
    if (!r.ok) { toast(`Enroll failed: HTTP ${r.status}`, "err"); return; }
    const d = await r.json();
    toast(`Enrolled "${d.profile}" (${d.num_samples} samples)`, "ok");
    enrollSlots = [null, null, null];
    resetEnrollSlots();
    $("#profileName").value = "";
    refreshProfiles();
  } catch (e) {
    toast(`Enroll error: ${e.message}`, "err");
  } finally {
    $("#enrollSubmit").classList.remove("loading");
    updateEnrollSubmit();
  }
});

// ---------- SPEECH → SIGN ----------
const speechDrop = $("#speechDrop");
const speechFile = $("#speechFile");
const speechPreview = $("#speechPreview");
const speechBtn = $("#speechBtn");
const speechClear = $("#speechClear");
const speechResult = $("#speechResult");
const recordBtn = $("#recordBtn");
const recordText = recordBtn.querySelector(".record-text");
const recordTimer = $("#recordTimer");
let speechSelected = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordStart = 0;
let recordTimerHandle = null;

const speechZone = setupDrop(speechDrop, speechFile, $("#speechDropTitle"), (f) => setSpeechFile(f), "Upload audio file");

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

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
    mediaRecorder.addEventListener("dataavailable", (e) => { if (e.data.size > 0) recordedChunks.push(e.data); });
    mediaRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(recordedChunks, { type: mime });
      setSpeechFile(new File([blob], `recording-${Date.now()}.webm`, { type: mime }));
      recordBtn.classList.remove("recording");
      recordText.textContent = "Record from microphone";
      recordTimer.classList.add("hidden");
      clearInterval(recordTimerHandle);
    });
    mediaRecorder.start();
    recordBtn.classList.add("recording");
    recordText.textContent = "Stop recording";
    recordStart = Date.now();
    recordTimer.classList.remove("hidden");
    recordTimer.textContent = "0:00";
    recordTimerHandle = setInterval(() => {
      const s = Math.floor((Date.now() - recordStart) / 1000);
      recordTimer.textContent = `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`;
    }, 250);
  } catch (e) { toast(`Mic error: ${e.message}`, "err"); }
});

speechBtn.addEventListener("click", runSpeech);
async function runSpeech() {
  if (!speechSelected || speechBtn.disabled) return;
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
      const txt = await r.text();
      speechResult.innerHTML = `<div class="error">HTTP ${r.status}: ${escape(txt)}</div>`;
      toast(`Speech→Sign failed: HTTP ${r.status}`, "err");
      return;
    }
    const d = await r.json();
    renderSpeechResult(d);
    historyAdd({ kind: "speech", summary: d.transcript, matched_count: d.matched_signs.length, duration_ms: d.duration_ms });
    toast(`Matched ${d.matched_signs.length}/${d.matched_signs.length + d.missing_words.length} words`, "ok");
  } catch (e) {
    speechResult.innerHTML = `<div class="error">${escape(e.message)}</div>`;
    toast(`Network error: ${e.message}`, "err");
  } finally {
    speechBtn.classList.remove("loading");
    speechBtn.disabled = false;
  }
}

function renderSpeechResult(d) {
  const matched = (d.matched_signs || []).map((m, i) => `
    <span class="matched-pill" style="animation: fadeIn 0.4s ease ${i*0.05}s backwards;">
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
      <div class="transcript">
        ${escape(d.transcript || "(empty transcript)")}
        ${copyBtn(d.transcript || "")}
      </div>
      ${matched ? `<div class="matched-list">${matched}</div>` : ""}
      <video controls autoplay class="result-video" src="/clips/${escape(videoName)}"></video>
      ${missing ? `<div class="missing-list"><span class="missing-label">Not in vocabulary:</span>${missing}</div>` : ""}
    </div>`;
}

// ---------- Keyboard shortcuts ----------
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea, select")) return;
  // Cmd/Ctrl+Enter — translate active tab
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    const activeTab = $(".tab.active").dataset.tab;
    if (activeTab === "sign") runSign();
    else runSpeech();
  }
  // H - history drawer
  if (e.key === "h" || e.key === "H") {
    if (!profileModal.classList.contains("hidden")) return;
    $("#historyDrawer").classList.toggle("open");
  }
  // P - profile modal
  if (e.key === "p" || e.key === "P") {
    if (profileModal.classList.contains("hidden")) openProfileModal();
    else profileModal.classList.add("hidden");
  }
  // Escape - close drawer/modal
  if (e.key === "Escape") {
    $("#historyDrawer").classList.remove("open");
    profileModal.classList.add("hidden");
  }
  // 1/2 - switch tabs
  if (e.key === "1") { $$(".tab")[0].click(); }
  if (e.key === "2") { $$(".tab")[1].click(); }
});
