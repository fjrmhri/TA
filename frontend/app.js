const API_BASE_URL = "https://fjrmhri-space-deteksi-hoax-ta.hf.space";

function resolveApiBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api");
  if (override && override.trim()) {
    return override.trim().replace(/\/+$/, "");
  }
  return API_BASE_URL.replace(/\/+$/, "");
}

const API_BASE = resolveApiBaseUrl();
const ANALYZE_ENDPOINT = `${API_BASE}/analyze`;

const analyzeForm = document.getElementById("analyzeForm");
const inputText = document.getElementById("inputText");
const includeNerInput = document.getElementById("includeNer");
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

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttr(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

// Catatan analisis cepat:
// - Sebelumnya fallback label membentuk string "Label asli model: <TAG>".
// - Tooltip juga menampilkan "Jenis: <TAG> / Label: <raw label> / Skor".
// - Kini tooltip dipadatkan ke "Jenis Indonesia / Kode / Skor" dan label redundan dibersihkan.
const NER_LABEL_ID = {
  CRD: "Angka kardinal",
  ORD: "Angka urutan",
  DAT: "Tanggal",
  TIM: "Waktu",
  MON: "Uang",
  PRC: "Persentase",
  QTY: "Kuantitas/Ukuran",
  PER: "Orang",
  ORG: "Organisasi/Institusi",
  NOR: "Organisasi politik",
  GPE: "Entitas geopolitik",
  LOC: "Lokasi",
  FAC: "Fasilitas",
  LAW: "Entitas hukum",
  PRD: "Produk/Merek",
  EVT: "Peristiwa",
  REG: "Agama",
  LAN: "Bahasa",
  WOA: "Karya seni",
};

function jenisIndonesia(tag) {
  const key = String(tag ?? "").trim().toUpperCase();
  return NER_LABEL_ID[key] || key || "-";
}

function bersihkanLabelNer(rawLabel, tag) {
  if (rawLabel == null) {
    return "";
  }

  const normalizedTag = String(tag ?? "").trim().toUpperCase();
  const cleaned = String(rawLabel).replace(/^label asli model:\s*/i, "").trim();
  if (!cleaned) {
    return "";
  }

  if (cleaned.toUpperCase() === normalizedTag) {
    return "";
  }
  return cleaned;
}

function renderSentenceWithNer(text, nerEntities) {
  const sentenceText = String(text ?? "");
  if (!Array.isArray(nerEntities) || nerEntities.length === 0) {
    return escapeHtml(sentenceText);
  }

  const normalized = nerEntities
    .map((entity) => ({
      start: Number.parseInt(entity?.start, 10),
      end: Number.parseInt(entity?.end, 10),
      entity_group: String(entity?.entity_group ?? "").trim().toUpperCase(),
      score: Number(entity?.score ?? 0),
    }))
    .filter(
      (entity) =>
        Number.isFinite(entity.start) &&
        Number.isFinite(entity.end) &&
        entity.start >= 0 &&
        entity.end > entity.start &&
        entity.end <= sentenceText.length
    )
    .sort((a, b) => {
      if (a.start !== b.start) {
        return a.start - b.start;
      }
      return a.end - b.end;
    });

  if (normalized.length === 0) {
    return escapeHtml(sentenceText);
  }

  const parts = [];
  let cursor = 0;
  let lastEnd = -1;

  for (const entity of normalized) {
    if (entity.start < lastEnd) {
      continue;
    }
    if (entity.start > cursor) {
      parts.push(escapeHtml(sentenceText.slice(cursor, entity.start)));
    }

    const entityText = sentenceText.slice(entity.start, entity.end);
    if (!entityText.trim()) {
      continue;
    }

    const tag = entity.entity_group || "-";
    const jenis = jenisIndonesia(tag);
    const tooltip = [
      `Jenis: ${jenis}`,
      `Kode: ${tag}`,
      `Skor: ${pct(Number.isFinite(entity.score) ? entity.score : 0)}`,
    ].join("\n");
    parts.push(
      `<span class="ner-entity" tabindex="0" data-tooltip="${escapeAttr(tooltip)}">${escapeHtml(entityText)}</span>`
    );
    cursor = entity.end;
    lastEnd = entity.end;
  }

  if (cursor < sentenceText.length) {
    parts.push(escapeHtml(sentenceText.slice(cursor)));
  }

  return parts.join("");
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
    .map((paragraph) => {
      const spans = paragraph.sentences
        .map((sentence) => {
          const renderedSentence = renderSentenceWithNer(sentence.text, sentence.ner_entities);
          return `<span class="hl ${escapeHtml(sentence.color)}" title="${escapeHtml(
            `${sentence.label} | conf ${pct(sentence.confidence)} | Hoaks ${pct(sentence.prob_hoax)} | Fakta ${pct(
              sentence.prob_fakta
            )}`
          )}">${renderedSentence}</span>`;
        })
        .join(" ");

      return `
        <article class="paragraph-block">
          <p class="paragraph-title">Paragraf ${paragraph.paragraph_index + 1}</p>
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
    .map((paragraph) => {
      const items = paragraph.sentences
        .map(
          (sentence) => `
          <div class="confidence-item">
            <div class="conf-left">
              <strong>[${escapeHtml(sentence.label)}]</strong> ${escapeHtml(sentence.text)}
            </div>
            <div class="conf-right">${pct(sentence.confidence)}</div>
          </div>
        `
        )
        .join("");

      const summary = paragraph.paragraph_summary || {};
      return `
        <section class="confidence-block">
          <h3>Paragraf ${paragraph.paragraph_index + 1}</h3>
          ${items || "<p><em>(Tidak ada kalimat terdeteksi)</em></p>"}
          <p><small>Hoaks: ${summary.hoax_sentences ?? 0} | Fakta: ${summary.fakta_sentences ?? 0} | Avg conf: ${pct(
            summary.avg_confidence ?? 0
          )} | Max hoaks prob: ${pct(summary.max_hoax_prob ?? 0)}</small></p>
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
    .map((entity) => {
      const tag = String(entity?.entity_group ?? "").trim().toUpperCase();
      const jenis = jenisIndonesia(tag);
      const cleanedLabel = bersihkanLabelNer(entity?.entity_label_id_id, tag);
      const showCleanedLabel = Boolean(cleanedLabel) && cleanedLabel.toLowerCase() !== jenis.toLowerCase();
      return `
      <article class="entity-card">
        <span class="entity-type">${escapeHtml(entity.entity_group)}</span>
        <span class="entity-text">${escapeHtml(entity.text)}</span>
        <span class="entity-label-id">Jenis: ${escapeHtml(jenis)}</span>
        ${showCleanedLabel ? `<span class="entity-label-id">Label: ${escapeHtml(cleanedLabel)}</span>` : ""}
        <span class="entity-score">Score: ${pct(entity.score)}</span>
      </article>
    `;
    })
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

  for (const paragraph of data.paragraphs) {
    lines.push(`Paragraf ${paragraph.paragraph_index + 1}`);
    for (const sentence of paragraph.sentences) {
      lines.push(`- [${sentence.label}] conf=${pct(sentence.confidence)} :: ${sentence.text}`);
    }
    lines.push("");
  }

  lines.push(`NER enabled: ${data.ner.enabled}`);
  for (const entity of data.ner.entities) {
    const tag = String(entity?.entity_group ?? "").trim().toUpperCase();
    const jenis = jenisIndonesia(tag);
    const cleanedLabel = bersihkanLabelNer(entity?.entity_label_id_id, tag);
    const labelInfo = cleanedLabel && cleanedLabel.toLowerCase() !== jenis.toLowerCase() ? ` | Label: ${cleanedLabel}` : "";
    lines.push(`- ${tag}: ${entity.text} (${pct(entity.score)}) | Jenis: ${jenis}${labelInfo}`);
  }
  return lines.join("\n");
}

function clearPinnedNerEntity(exceptElement = null) {
  highlightContent.querySelectorAll(".ner-entity.pinned").forEach((entityNode) => {
    if (entityNode !== exceptElement) {
      entityNode.classList.remove("pinned");
    }
  });
}

highlightContent.addEventListener("click", (event) => {
  const target = event.target.closest(".ner-entity");
  if (!target) {
    clearPinnedNerEntity();
    return;
  }

  event.stopPropagation();
  const nextPinnedState = !target.classList.contains("pinned");
  clearPinnedNerEntity(target);
  target.classList.toggle("pinned", nextPinnedState);
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".ner-entity")) {
    clearPinnedNerEntity();
  }
});

analyzeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  clearOutput();

  const text = inputText.value.trim();
  if (!text) {
    showError("Input teks wajib diisi.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        include_ner: includeNerInput.checked,
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
  } catch (error) {
    showError(`Gagal memproses: ${error.message}`);
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
  try {
    await navigator.clipboard.writeText(formatCopyText(latestResponse));
    window.alert("Hasil berhasil disalin.");
  } catch (error) {
    showError(`Gagal menyalin hasil: ${error.message}`);
  }
});
