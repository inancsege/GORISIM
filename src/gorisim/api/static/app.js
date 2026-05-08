const $ = (s) => document.querySelector(s);

// Tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.remove("active", "bg-slate-900", "text-white");
      b.classList.add("bg-slate-200");
    });
    btn.classList.add("active", "bg-slate-900", "text-white");
    btn.classList.remove("bg-slate-200");
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// Sign → Text
$("#signBtn").addEventListener("click", async () => {
  const f = $("#signFile").files[0];
  if (!f) return alert("Bir video seçin");
  const fd = new FormData();
  fd.append("video", f);
  $("#signResult").innerHTML = "Çevriliyor...";
  const r = await fetch("/sign-to-text", { method: "POST", body: fd });
  if (!r.ok) {
    $("#signResult").innerHTML = `<div class="text-red-600">${await r.text()}</div>`;
    return;
  }
  const data = await r.json();
  const alts = data.alternatives.map((a) => `<li>${a.text} <span class="text-slate-500">(${(a.confidence * 100).toFixed(1)}%)</span></li>`).join("");
  $("#signResult").innerHTML = `
    <div class="text-2xl font-bold">${data.text}</div>
    <div class="text-slate-500 text-sm mb-3">Güven: ${(data.confidence * 100).toFixed(1)}% · ${data.duration_ms} ms</div>
    <ol class="list-decimal pl-5">${alts}</ol>
  `;
});

// Profiles
async function refreshProfiles() {
  const r = await fetch("/profiles");
  const profiles = await r.json();
  const sel = $("#profileSelect");
  sel.innerHTML = '<option value="">— Yok —</option>' + profiles.map((p) => `<option>${p.name}</option>`).join("");
}
refreshProfiles();

// Speech → Sign
$("#speechBtn").addEventListener("click", async () => {
  const f = $("#speechFile").files[0];
  if (!f) return alert("Bir ses dosyası seçin");
  const fd = new FormData();
  fd.append("audio", f);
  const profile = $("#profileSelect").value;
  if (profile) fd.append("profile", profile);
  $("#speechResult").innerHTML = "Çevriliyor...";
  const r = await fetch("/speech-to-sign", { method: "POST", body: fd });
  if (!r.ok) {
    $("#speechResult").innerHTML = `<div class="text-red-600">${await r.text()}</div>`;
    return;
  }
  const data = await r.json();
  const missing = data.missing_words.length
    ? `<div class="mt-2"><span class="font-semibold">Eşleşmeyen kelimeler:</span> ${data.missing_words.map((w) => `<span class="inline-block bg-amber-100 text-amber-800 px-2 py-0.5 rounded text-sm mr-1">${w}</span>`).join("")}</div>`
    : "";
  const videoName = data.output_video_path.split("/").pop();
  $("#speechResult").innerHTML = `
    <div class="mt-3"><span class="font-semibold">Transkripsiyon:</span> ${data.transcript}</div>
    <video controls class="mt-3 w-full rounded border" src="/clips/${videoName}"></video>
    ${missing}
  `;
});
