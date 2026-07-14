const form = document.getElementById("convert-form");
const statusEl = document.getElementById("status");
const comparison = document.getElementById("comparison");
const beforeImg = document.getElementById("before-img");
const afterImg = document.getElementById("after-img");
const palette = document.getElementById("palette_size");
const paletteOut = document.getElementById("palette_out");
const imageInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit");

let originalUrl = null;

palette.addEventListener("input", () => {
  paletteOut.textContent = palette.value;
});

imageInput.addEventListener("change", () => {
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  const file = imageInput.files[0];
  if (file) {
    fileLabel.textContent = file.name;
    originalUrl = URL.createObjectURL(file);
    preview.src = originalUrl;
    preview.hidden = false;
    comparison.hidden = true;
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.files[0]) {
    statusEl.textContent = "Сначала выберите изображение.";
    return;
  }

  const formData = new FormData(form);
  const numbersChecked = document.querySelector('[name="show_numbers"]').checked;
  formData.set("show_numbers", numbersChecked ? "true" : "false");

  submitBtn.disabled = true;
  statusEl.textContent = "Обработка…";
  comparison.hidden = true;

  try {
    const response = await fetch("/api/convert", { method: "POST", body: formData });
    if (!response.ok) {
      const text = await response.text();
      throw new Error("Ошибка " + response.status + ": " + text);
    }
    const blob = await response.blob();
    beforeImg.src = originalUrl;
    afterImg.src = URL.createObjectURL(blob);
    comparison.hidden = false;
    statusEl.textContent = "Готово.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
  }
});
