# ТЗ: Улучшение пайплайна — сохранение деталей при лимите цветов

## Контекст для агента

**Ветка:** `feat/pipeline-v2` (уже создана и запушена)
**База:** `feat/paint-by-numbers`
**Язык:** Python 3.8, `from __future__ import annotations` в каждом `.py`
**Фреймворк:** FastAPI, OpenCV, scikit-image, ReportLab
**Тесты:** 48 тестов в `tests/`, запускаются через `python -m pytest tests/ -q --tb=short`

**Главное правило:** НЕ ломать существующий результат. Все изменения должны сохранять обратную совместимость. Если функция меняет поведение — обнови тесты.

---

## Цель

Улучшить пайплайн так, чтобы:
1. **Сохранять больше деталей** (глаза, ресницы, мелкие элементы)
2. **Не ломать результат** (регионы не накладываются, углы не обрезаются)
3. **Сохранять лимитацию по цветам** (palette_size + color_merge_threshold)

---

## Текущая архитектура

```
app/pipeline.py — render_data_from_image() оркестрирует пайплайн
app/quantize.py — k-means + merge_similar_colors (RGB)
app/postprocess.py — smooth_label_boundaries, merge_small_regions, clean_mask
app/regions.py — extract_regions, _simplify_only, _smooth_closed
app/models.py — ConvertParams, PipelineParams, Region, PaletteColor, RenderData
```

### Порядок шагов (app/pipeline.py:16-42)
1. `_load_and_normalize` — resize + pyrMeanShift + CLAHE + unsharp + 2px border
2. `_segment_and_flatten` — Felzenszwalb + flatten to mean
3. `quantize` — k-means (RGB, palette_size из detail_level)
4. `merge_similar_colors` — greedy merge (RGB distance, threshold из UI)
5. `merge_small_regions` — RAG merge (min_area_px из detail_level)
6. `smooth_label_boundaries` — adaptive Gaussian blur on labels
7. `_apply_global_morphology` — MORPH_OPEN per color (kernel из detail_level)
8. `extract_regions` — clean_mask + findContours + simplify
9. `_strip_border_from_regions` — сдвиг контуров на 2px

---

## Задачи (в порядке выполнения)

### Задача 1: k-means в Lab вместо RGB

**Файл:** `app/quantize.py`, функция `quantize()`

**Проблема:** k-means работает в RGB — не перцептуально. Два цвета с малой RGB-дистанцией могут выглядеть по-разному.

**Решение:**
1. Конвертировать `image` из RGB в Lab перед k-means: `lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)`
2. Запустить k-means на Lab пикселях
3. Конвертировать центры обратно в RGB для PaletteColor
4. Labels остаются теми же

**Код:** 
```python
def quantize(image: np.ndarray, palette_size: int):
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels_flat, centers_lab = cv2.kmeans(
        pixels, palette_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    
    # Конвертируем центры обратно в RGB
    centers_lab = centers_lab.reshape(-1, 1, 3).astype(np.uint8)
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    
    palette = [...]
    labels = labels_flat.reshape(h, w).astype(np.int32)
    return palette, labels
```

**Тесты:** Обновить `test_quantize.py` — проверить что палитра всё ещё корректна. Добавить тест `test_quantize_uses_lab_space` — проверить что близкие в Lab, но разные в RGB цвета не сливаются.

---

### Задача 2: CIEDE2000 в merge_similar_colors

**Файл:** `app/quantize.py`, функция `merge_similar_colors()`

**Проблема:** RGB Euclidean distance не соответствует восприятию. Нужно использовать CIEDE2000.

**Решение:**
1. Конвертировать RGB → Lab перед вычислением дистанции
2. Использовать CIEDE2000 формулу для сравнения цветов
3. Threshold теперь в единицах CIEDE2000 (typical: 2.3 = just noticeable difference)

**Код CIEDE2000:** Реализовать функцию `_ciede2000(lab1, lab2) -> float` в `app/quantize.py`. Формула стандартная, см. [Bruce Lindbloom](http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE2000.html).

**Порог:** Изменить default threshold с 15 (RGB) на ~5.0 (CIEDE2000). Обновить диапазон слайдера в UI (`static/index.html` и `static/app.js`): `max=30` вместо `max=80`.

**Тесты:** Обновить `test_quantize.py` — тесты merge должны использовать CIEDE2000 threshold. Добавить тест `test_merge_uses_ciede2000` — два цвета с малой RGB дистанцией, но большой CIEDE2000 не сливаются.

---

### Задача 3: Skip mean-shift при высоком detail_level

**Файл:** `app/pipeline.py`, функция `_load_and_normalize()`

**Проблема:** При detail_level >= 10, `mean_shift_sp = 1` — почти нет фильтрации, но CLAHE+unsharp всё равно применяется и может добавлять шум.

**Решение:** При `mean_shift_sp <= 1` пропустить весь блок (mean-shift + CLAHE + unsharp):
```python
if pp.mean_shift_sp > 1:
    image = cv2.pyrMeanShiftFiltering(...)
    # CLAHE + unsharp...
```

**Тесты:** Не должно ломать существующие. Добавить тест что при detail_level=15 mean-shift не применяется (проверить через mock или сравнить output).

---

### Задача 4: Убрать MORPH_OPEN из пайплайна

**Файл:** `app/pipeline.py`, строки 26-27

**Проблема:** `_apply_global_morphology` обрабатывает каждый цвет независимо, создавая gaps между регионами.

**Решение:** 
1. Удалить вызов `_apply_global_morphology` из `render_data_from_image()`
2. Оставить функцию в файле (не удалять) — может пригодиться
3. В `extract_regions` (`app/regions.py`) использовать `morph_kernel=0` (clean_mask с kernel=1 почти noop)

**Альтернатива:** Заменить на watershed-based границу, но это сложнее. Для начала просто убрать.

**Тесты:** Обновить `test_pipeline.py` если нужно. Проверить что регионы не имеют gaps.

---

### Задача 5: Edge-aware min_size для Felzenszwalb

**Файл:** `app/pipeline.py`, функция `_segment_and_flatten()`

**Проблема:** `min_size = max(10, h*w*0.00005)` — фиксированный, не учитывает локальную сложность. Мелкие детали (глаза) удаляются.

**Решение:**
1. Вычислить edge map: `edges = cv2.Canny(image, 50, 150)`
2. Вычислить локальную плотность границ через `cv2.boxFilter`
3. В зонах с высокой плотностью границ использовать меньший `min_size` (сохранять мелкие сегменты)
4. В однородных зонах использовать больший `min_size` (меньше шума)

**Подход:** 
- Запустить Felzenszwalb дважды: с маленьким min_size (10) и с большим (h*w*0.00005)
- Слить сегменты: в зонах с высокой edge density брать мелкие сегменты, в зонах с низкой — крупные
- Или проще: запустить один раз с min_size=10, потом слить мелкие сегменты только в однородных зонах

**Рекомендация:** Начать с простого варианта — уменьшить `min_size` до `max(5, h*w*0.00002)` и проверить результат. Если слишком много шума — добавить edge-aware merging.

**Тесты:** Проверить что мелкие детали сохраняются. Использовать синтетическое изображение с маленьким кружком.

---

### Задача 6: Перцептуально-упорядоченный merge

**Файл:** `app/quantize.py`, функция `merge_similar_colors()`

**Проблема:** Greedy merge зависит от порядка цветов в палитре. k-means возвращает цвета в произвольном порядке.

**Решение:**
1. Перед merge отсортировать палитру по luminance (L channel в Lab)
2. Якоря идут от тёмных к светлым — детерминированный результат
3. Labels обновляются соответственно

**Код:**
```python
# Перед merge:
lab_colors = cv2.cvtColor(np.array([c.rgb for c in palette]), cv2.COLOR_RGB2LAB)
luminance = lab_colors[:, 0]
sort_order = np.argsort(luminance)
# Переставить palette и remap labels
```

**Тесты:** Добавить тест `test_merge_is_deterministic` — один и тот же input всегда даёт одинаковый output независимо от начального порядка.

---

## Порядок реализации

```
Задача 1 (k-means Lab)     ← не зависит от других
Задача 6 (sort by L)       ← зависит от 1 (нужен Lab)
Задача 2 (CIEDE2000)       ← зависит от 1 (нужен Lab)
Задача 3 (skip mean-shift) ← не зависит
Задача 4 (remove MORPH)    ← не зависит
Задача 5 (edge-aware)      ← не зависит, но тестируется после 3,4
```

**Рекомендуемый порядок:** 1 → 6 → 2 → 3 → 4 → 5

---

## Acceptance criteria

1. **Все 48 тестов проходят** после каждой задачи
2. **Палитра не превышает** `palette_size` (с учётом merge)
3. **Регионов не меньше**, чем при текущей реализации (на тестовом изображении)
4. **Углы не обрезаются** (проверить через `test_pipeline`)
5. **Регионов не накладываются** (проверить визуально на samples/)
6. **UI слайдер color_merge_threshold** работает корректно с новым диапазоном

---

## Файлы для изменения

| Файл | Задачи | Тип изменений |
|---|---|---|
| `app/quantize.py` | 1, 2, 6 | k-means Lab, CIEDE2000, sort by L |
| `app/pipeline.py` | 3, 4, 5 | skip mean-shift, remove morph, edge-aware |
| `static/index.html` | 2 | обновить range слайдера |
| `static/app.js` | 2 | обновить output формат |
| `tests/test_quantize.py` | 1, 2, 6 | новые тесты |
| `tests/test_pipeline.py` | 3, 4, 5 | обновить тесты |
| `app/models.py` | 2 | обновить default color_merge_threshold |

---

## Откат при проблемах

Если что-то ломается:
1. `git checkout feat/paint-by-numbers -- <file>` — откат конкретного файла
2. Каждая задача — отдельный коммит, можно откатить по одной
3. Тесты должны проходить после КАЖДОЙ задачи

## Финальный шаг

После всех задач:
1. Запустить `python -m pytest tests/ -q --tb=short` — все 48+ тестов проходят
2. Запустить `uvicorn app.main:app --reload` — проверить визуально на samples/
3. Закоммитить и запушить в `feat/pipeline-v2`
4. Создать PR: `gh pr create --base feat/paint-by-numbers --head feat/pipeline-v2`
