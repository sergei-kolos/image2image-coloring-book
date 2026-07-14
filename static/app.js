const form = document.getElementById("convert-form");
const statusEl = document.getElementById("status");
const result = document.getElementById("result");
const palette = document.getElementById("palette_size");
const paletteOut = document.getElementById("palette_out");
const imageInput = document.getElementById("image");
const fileLabel = document.getElementById("file-label");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit");

palette.addEventListener("input", () => {
  paletteOut.textContent = palette.value;
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) {
    fileLabel.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
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
  result.hidden = true;

  try {
    const response = await fetch("/api/convert", { method: "POST", body: formData });
    if (!response.ok) {
      const text = await response.text();
      throw new Error("Ошибка " + response.status + ": " + text);
    }
    const blob = await response.blob();
    result.src = URL.createObjectURL(blob);
    result.hidden = false;
    statusEl.textContent = "Готово. Можно скачать или распечатать из просмотра.";
  } catch (err) {
    statusEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
  }
});
