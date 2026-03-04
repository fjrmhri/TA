const BACKEND_URL = "https://fjrmhri-Space_Deteksi_Hoax_TA.hf.space";

const analyzeForm = document.getElementById("analyzeForm");
const inputText = document.getElementById("inputText");
const backendUrlInput = document.getElementById("backendUrl");
const includeNerInput = document.getElementById("includeNer");
const orangeThresholdInput = document.getElementById("orangeThreshold");
const resetBtn = document.getElementById("resetBtn");
const copyBtn = document.getElementById("copyBtn");

const loader = document.getElementById("loader");
const errorBox = document.getElementById("errorBox");
const summaryPanel = document.getElementById("summaryPanel");
const highlightPanel = document.getElementById("highlightPanel");
const confidencePanel = document.getElementById("confidencePanel");
const nerPanel = document.getElementById("nerPanel");

const highlightContent = document.getElementById("highlightContent");
const confidenceContent = document.getElementById("confidenceContent");
const nerContent = document.getElementById("nerContent");

const sumParagraphs = document.getElementById("sumParagraphs");
const sumSentences = document.getElementById("sumSentences");
const sumHoax = document.getElementById("sumHoax");
const sumFakta = document.getElementById("sumFakta");
const sumLowConf = document.getElementById("sumLowConf");

let latestResponse = null;

backendUrlInput.value = BACKEND_URL;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function getBackendUrl() {
  const raw = backendUrlInput.value.trim() || BACKEND_URL;
  return raw.replace(/\/+$/, "");
}

function setLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function clearOutput() {
  latestResponse = null;
  summaryPanel.classList.add("hidden");
  highlightPanel.classList.add("hidden");
  confidencePanel.classList.add("hidden");
  nerPanel.classList.add("hidden");
  highlightContent.innerHTML = "";
  confidenceContent.innerHTML = "";
  nerContent.innerHTML = "";
}

function renderSummary(summary) {
  sumParagraphs.textContent = String(summary.num_paragraphs ?? 0);
  sumSentences.textContent = String(summary.num_sentences ?? 0);
  sumHoax.textContent = String(summary.hoax_sentences ?? 0);
  sumFakta.textContent = String(summary.fakta_sentences ?? 0);
  sumLowConf.textContent = String(summary.low_conf_sentences ?? 0);
  summaryPanel.classList.remove("hidden");
}

function renderHighlights(paragraphs) {
  const blocks = paragraphs
    .map((p) => {
      const spans = p.sentences
        .map(
          (s) =>
            `<span class="hl ${escapeHtml(s.color)}" title="${escapeHtml(
              `${s.label} | conf ${pct(s.confidence)} | Hoaks ${pct(s.prob_hoax)} | Fakta ${pct(s.prob_fakta)}`
            )}">${escapeHtml(s.text)}</span>`
        )
        .join(" ");

      return `
        <article class="paragraph-block">
          <p class="paragraph-title">Paragraf ${p.paragraph_index + 1}</p>
          <p class="paragraph-text">${spans || "<em>(Tidak ada kalimat terdeteksi)</em>"}</p>
        </article>
      `;
    })
    .join("");

  highlightContent.innerHTML = blocks;
  highlightPanel.classList.remove("hidden");
}

function renderConfidence(paragraphs) {
  const html = paragraphs
    .map((p) => {
      const items = p.sentences
        .map(
          (s) => `
          <div class="confidence-item">
            <div class="conf-left">
              <strong>[${escapeHtml(s.label)}]</strong> ${escapeHtml(s.text)}
            </div>
            <div class="conf-right">${pct(s.confidence)}</div>
          </div>
        `
        )
        .join("");

      const ps = p.paragraph_summary || {};
      return `
        <section class="confidence-block">
          <h3>Paragraf ${p.paragraph_index + 1}</h3>
          ${items || "<p><em>(Tidak ada kalimat terdeteksi)</em></p>"}
          <p><small>Hoaks: ${ps.hoax_sentences ?? 0} | Fakta: ${ps.fakta_sentences ?? 0} | Avg conf: ${pct(
            ps.avg_confidence ?? 0
          )} | Max hoaks prob: ${pct(ps.max_hoax_prob ?? 0)}</small></p>
        </section>
      `;
    })
    .join("");

  confidenceContent.innerHTML = html;
  confidencePanel.classList.remove("hidden");
}

function renderNer(ner) {
  if (!ner || !ner.enabled) {
    nerContent.innerHTML = "<p><em>NER tidak diaktifkan.</em></p>";
    nerPanel.classList.remove("hidden");
    return;
  }

  const entities = Array.isArray(ner.entities) ? ner.entities : [];
  if (entities.length === 0) {
    nerContent.innerHTML = "<p><em>Tidak ada entitas terdeteksi.</em></p>";
    nerPanel.classList.remove("hidden");
    return;
  }

  const html = entities
    .map(
      (ent) => `
      <article class="entity-card">
        <span class="entity-type">${escapeHtml(ent.entity_group)}</span>
        <span class="entity-text">${escapeHtml(ent.text)}</span>
        <span class="entity-score">Score: ${pct(ent.score)}</span>
      </article>
    `
    )
    .join("");

  nerContent.innerHTML = html;
  nerPanel.classList.remove("hidden");
}

function formatCopyText(data) {
  const lines = [];
  lines.push("Ringkasan Analisis");
  lines.push(
    `Paragraf=${data.summary.num_paragraphs}, Kalimat=${data.summary.num_sentences}, Hoaks=${data.summary.hoax_sentences}, Fakta=${data.summary.fakta_sentences}, LowConf=${data.summary.low_conf_sentences}`
  );
  lines.push("");

  for (const p of data.paragraphs) {
    lines.push(`Paragraf ${p.paragraph_index + 1}`);
    for (const s of p.sentences) {
      lines.push(`- [${s.label}] conf=${pct(s.confidence)} :: ${s.text}`);
    }
    lines.push("");
  }

  lines.push(`NER enabled: ${data.ner.enabled}`);
  for (const ent of data.ner.entities) {
    lines.push(`- ${ent.entity_group}: ${ent.text} (${pct(ent.score)})`);
  }
  return lines.join("\n");
}

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  clearOutput();

  const text = inputText.value.trim();
  if (!text) {
    showError("Input teks wajib diisi.");
    return;
  }

  const threshold = Number(orangeThresholdInput.value);
  if (Number.isNaN(threshold) || threshold < 0 || threshold > 1) {
    showError("Ambang oranye harus angka 0 sampai 1.");
    return;
  }

  setLoading(true);
  try {
    const endpoint = `${getBackendUrl()}/analyze`;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        include_ner: includeNerInput.checked,
        confidence_orange_threshold: threshold,
      }),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || `HTTP ${response.status}`;
      throw new Error(detail);
    }

    latestResponse = payload;
    renderSummary(payload.summary || {});
    renderHighlights(payload.paragraphs || []);
    renderConfidence(payload.paragraphs || []);
    renderNer(payload.ner || { enabled: false, entities: [] });
  } catch (err) {
    showError(`Gagal memproses: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener("click", () => {
  clearError();
  clearOutput();
  inputText.value = "";
});

copyBtn.addEventListener("click", async () => {
  if (!latestResponse) {
    showError("Belum ada hasil untuk disalin.");
    return;
  }
  clearError();
  const text = formatCopyText(latestResponse);
  try {
    await navigator.clipboard.writeText(text);
    alert("Hasil berhasil disalin.");
  } catch (err) {
    showError(`Gagal menyalin hasil: ${err.message}`);
  }
});
