const form = document.getElementById("convert-form");
const statusEl = document.getElementById("status");
const comparison = document.getElementById("comparison");
const afterImg = document.getElementById("after-img");
const outlineImg = document.getElementById("outline-img");
const detailLevel = document.getElementById("detail_level");
const detailOut = document.getElementById("detail_out");
const maxColors = document.getElementById("max_colors");
const maxColorsOut = document.getElementById("max_colors_out");
const colorMerge = document.getElementById("color_merge_threshold");
const colorMergeOut = document.getElementById("color_merge_out");
const imageInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const fileName = fileLabel.querySelector(".file-name");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit");
const downloads = document.getElementById("downloads");
const dlPdf = document.getElementById("dl-pdf");

let originalUrl = null;
let pdfDataUri = null;

// ── AI engine toggle ──
const engineClassicLabel = document.getElementById("engine-classic-label");
const engineSamLabel = document.getElementById("engine-sam-label");
const loadingOverlay = document.getElementById("loading-overlay");

// Probe capabilities on load
fetch("/api/capabilities")
  .then(r => r.json())
  .then(caps => {
    if (!caps.sam_hq) {
      engineSamLabel.classList.add("disabled");
      engineSamLabel.title = "Requires PyTorch + SAM-HQ weights. See README.";
      engineSamLabel.querySelector("input").disabled = true;
    }
  })
  .catch(() => {
    engineSamLabel.classList.add("disabled");
    engineSamLabel.querySelector("input").disabled = true;
  });

// Toggle active state on engine labels
document.querySelectorAll(".engine-option").forEach(label => {
  label.addEventListener("click", () => {
    if (label.querySelector("input").disabled) return;
    document.querySelectorAll(".engine-option").forEach(l => l.classList.remove("active"));
    label.classList.add("active");
  });
});

detailLevel.addEventListener("input", () => {
  detailOut.textContent = detailLevel.value;
});

maxColors.addEventListener("input", () => {
  maxColorsOut.textContent = maxColors.value;
});

colorMerge.addEventListener("input", () => {
  colorMergeOut.textContent = colorMerge.value;
});

imageInput.addEventListener("change", () => {
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  const file = imageInput.files[0];
  if (file) {
    fileName.textContent = file.name;
    originalUrl = URL.createObjectURL(file);
    preview.src = originalUrl;
    preview.hidden = false;
    comparison.hidden = true;
    downloads.hidden = true;
  }
});

function downloadDataUri(dataUri, filename) {
  const a = document.createElement("a");
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

dlPdf.addEventListener("click", () => {
  if (pdfDataUri) downloadDataUri(pdfDataUri, "coloring.pdf");
});

document.addEventListener("click", (e) => {
  if (!e.target.classList.contains("dl-img")) return;
  const img = document.getElementById(e.target.dataset.target);
  if (img && img.src) downloadDataUri(img.src, e.target.dataset.target + ".png");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files[0]) {
    statusEl.textContent = "Please select an image first.";
    return;
  }

  const formData = new FormData(form);
  const numbersChecked = document.querySelector('[name="show_numbers"]').checked;
  formData.set("show_numbers", numbersChecked ? "true" : "false");

  submitBtn.disabled = true;
  const engine = formData.get("engine");
  if (engine === "sam_hq") {
    loadingOverlay.hidden = false;
  }
  statusEl.textContent = "Processing\u2026";
  comparison.hidden = true;
  downloads.hidden = true;

  try {
    const response = await fetch("/api/convert", { method: "POST", body: formData });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || "Error " + response.status);
    }
    const result = await response.json();
    afterImg.src = result.colored;
    outlineImg.src = result.outline;
    pdfDataUri = result.pdf || null;
    comparison.hidden = false;
    if (pdfDataUri) downloads.hidden = false;
    statusEl.textContent = "Done.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
    loadingOverlay.hidden = true;
  }
});
