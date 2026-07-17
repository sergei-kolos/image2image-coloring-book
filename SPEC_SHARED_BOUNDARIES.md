# ТЗ: Shared Boundaries + Detail Preservation

## Контекст для агента

**Ветка:** создать `feat/shared-boundaries` от текущего `feat/pipeline-v2`  
**База:** `feat/pipeline-v2`  
**Язык:** Python 3.8, `from __future__ import annotations` в каждом `.py`  
**Фреймворк:** FastAPI, OpenCV, scikit-image, ReportLab  
**Тесты:** `python -m pytest tests/ -q --tb=short` (все должны проходить после каждой задачи)

**Главное правило:** НЕ ломать существующий результат. Каждая задача — отдельный коммит. Если поведение API/UI меняется — обновить тесты и `AGENTS.md`.

**Референс-проблема (Lion King input):**
1. **Двойные линии** на outline — смежные регионы stroke'ают границу дважды
2. **Gaps / полоски** между зонами — независимое `clean_mask` + сглаживание
3. **Каша мелких зон** (грива, небо) — aggressive merge / flatten
4. **Потеря мелких объектов** (Timon, птицы) — min_size + merge_small_regions

---

## Цель

1. **Одна общая линия** на стыке двух регионов (shared boundary model)
2. **Сохранить больше деталей** без раздувания palette (лимит цветов остаётся)
3. **Не ухудшить** PDF layout, UI, API contract

---

## Текущая архитектура (проблемные места)

```
labels (pixel map, clean shared topology)
    ↓
extract_regions: FOR EACH color independently
    mask = (labels == i)
    clean_mask(mask)          ← shifts boundary independently
    findContours(mask)        ← contour is only "our" side
    organic_smooth(contour)   ← neighbor smoothed differently
    ↓
pdf/visualize: FOR EACH region
    stroke(region.contour)    ← shared edge drawn TWICE
```

**Корневая причина двойных линий:** per-color contour extraction + per-region stroke.  
**Корневая причина gaps:** `clean_mask` / morph / independent smoothing.

---

## Трек A — Shared Boundaries

### Задача A1: Модель SharedEdge

**Файлы:** `app/models.py`

Добавить:

```python
@dataclass
class SharedEdge:
    label_a: int          # 0-based cluster id (or -1 for image border)
    label_b: int          # 0-based cluster id (or -1 for image border)
    polyline: object      # np.ndarray (N, 2) float — shared boundary path
```

Расширить `RenderData`:

```python
@dataclass
class RenderData:
    width: int
    height: int
    palette: list
    regions: list
    edges: list = field(default_factory=list)  # list[SharedEdge]
```

`Region.contour` **оставить** (нужен для fill/centroid/area), но stroke в PDF/outline идёт из `edges`.

**Тесты:** обновить `tests/test_pdf_layout.py`, `tests/test_pipeline.py` — RenderData с `edges=[]` по умолчанию не ломает старые тесты.

---

### Задача A2: Извлечение shared edges из label map

**Файл:** `app/regions.py` (новая функция `extract_shared_edges`)

**Алгоритм (рекомендуемый, OpenCV-совместимый):**

1. Построить binary edge map:
   ```python
   h_diff = labels[:, :-1] != labels[:, 1:]
   v_diff = labels[:-1, :] != labels[1:, :]
   edge = zeros_like(labels, uint8)
   edge[:, :-1] |= h_diff
   edge[:, 1:]  |= h_diff
   edge[:-1, :] |= v_diff
   edge[1:, :]  |= v_diff
   # also mark image outer border as edge
   ```

2. `cv2.findContours(edge, RETR_LIST, CHAIN_APPROX_NONE)` — получить polyline границы.

3. Для каждого polyline-сегмента определить пару labels:
   - Взять midpoints нормали к ребру (±1 px)
   - `label_a = labels[p_left]`, `label_b = labels[p_right]`
   - Если один снаружи изображения → `-1` (outer border)

4. Слить коллинеарные куски с одной парой `(min(a,b), max(a,b))` в один `SharedEdge`.

5. Сгладить **один раз**:
   ```python
   edge.polyline = _simplify_and_smooth(edge.polyline)
   # или adaptive_smooth / organic_smooth
   ```

6. **Не** вызывать `clean_mask` на per-color mask перед этим.

**Важно:**
- Порядок точек в polyline детерминированный (например, start = lexicographically smallest point)
- `label_a <= label_b` всегда (каноническая пара)
- Outer border: `label_b = -1`

**Тесты:** `tests/test_regions.py`
- `test_shared_edges_two_color_halves` — два цвета → одна shared edge по середине + outer border edges
- `test_shared_edge_not_duplicated` — len(edges) для shared pair == 1
- `test_shared_edge_has_both_labels`

---

### Задача A3: Region contours без независимого clean_mask

**Файл:** `app/regions.py`, `extract_regions`

**Изменения:**

1. Убрать `clean_mask(mask, ...)` **или** заменить на:
   ```python
   # only light denoise that does NOT expand/shrink topology
   mask = mask  # no morph
   ```
2. Contours для `Region` по-прежнему из `findContours(mask)` — нужны для fill/centroid.
3. Параллельно вызывать `extract_shared_edges(labels)` и возвращать `(regions, edges)`.

**Сигнатура:**

```python
def extract_regions(labels, palette, morph_kernel: int = 3) -> tuple[list[Region], list[SharedEdge]]:
```

Обновить `pipeline.py`:

```python
regions, edges = extract_regions(labels, palette, morph_kernel=pp.morph_kernel)
return RenderData(..., regions=regions, edges=edges)
```

**Тесты:** обновить все вызовы `extract_regions` в `tests/`.

---

### Задача A4: PDF stroke только shared edges

**Файл:** `app/pdf_layout.py`, `render_pdf`

**Было:**
```python
for region in data.regions:
    draw region.contour
    draw region.holes
```

**Стало:**
```python
# Pass 1: shared edges (single stroke)
if data.edges:
    for edge in data.edges:
        draw edge.polyline
else:
    # fallback for old RenderData without edges
    for region in data.regions:
        draw region.contour + holes
```

Holes: если hole = граница с другим label, она уже в `edges`.  
Если hole = «дыра в маске без другого label» (редко) — оставить fallback из `region.holes`.

**Labels (numbers):** без изменений — centroids из `Region`.

**Тесты:** `test_render_pdf_returns_valid_pdf_bytes` — PDF валиден; при наличии edges файл рисуется.

---

### Задача A5: visualize / outline stroke только shared edges

**Файл:** `app/visualize.py`

```python
def _draw_boundaries(vis, regions, thickness, edge_color, edges=None):
    if edges:
        for edge in edges:
            pts = edge.polyline.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=edge_color, thickness=thickness)
    else:
        # legacy fallback
        for region in regions:
            ...
```

`render_visualization` / `render_outline` передают `data.edges`.

**Fill** в colored preview — по-прежнему `cv2.fillPoly(region.contour)` (fill не stroke).

**Тесты:** pipeline/API тесты — PNG всё ещё генерируется.

---

## Трек B — Detail Preservation

### Задача B1: bilateralFilter вместо pyrMeanShift

**Файл:** `app/pipeline.py`, `_load_and_normalize`

**Было:**
```python
if pp.mean_shift_sp > 1:
    image = cv2.pyrMeanShiftFiltering(image, sp=..., sr=...)
    # CLAHE + unsharp
```

**Стало:**
```python
if pp.mean_shift_sp > 1:
    # edge-preserving denoise; map sp/sr → bilateral params
    d = max(5, min(9, pp.mean_shift_sp))
    sigma_color = float(pp.mean_shift_sr)
    sigma_space = float(pp.mean_shift_sp * 5)
    image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # OPTIONAL: keep mild CLAHE only, DROP unsharp (unsharp reintroduces noise)
```

**Параметры mapping (detail_level 1→15):**
| detail | mean_shift_sp | bilateral d | sigmaColor | sigmaSpace |
|---|---|---|---|---|
| 1 | 12 | 9 | 35 | 60 |
| 5 | ~8 | 7–9 | ~28 | ~40 |
| 15 | 1 (skip) | skip | — | — |

**Тесты:** pipeline tests pass; visual check Lion King — edges of mane/face sharper.

---

### Задача B2: median flatten вместо mean

**Файл:** `app/pipeline.py`, `_segment_and_flatten`

**Было:** mean color per Felzenszwalb segment.  
**Стало:** median color per segment (устойчивее к outliers).

```python
# For each segment id, compute median of RGB pixels
# Vectorized approach: sort or use scipy.ndimage if available
# Simple loop is OK if segment count is moderate
for sid in range(n):
    pixels = flat_img[flat_seg == sid]
    if len(pixels):
        means[sid] = np.median(pixels, axis=0)
```

Для скорости при большом `n` — batch по unique ids.

**Тесты:** pipeline tests pass.

---

### Задача B3: edge-aware merge_small_regions

**Файл:** `app/postprocess.py`, `merge_small_regions`

**Идея:** в зонах с высокой edge density порог площади **ниже** (сохраняем мелкие важные регионы).

```python
def merge_small_regions(labels, palette, min_area_px, edge_density=None):
    ...
    for sid in small_ids:
        # local edge density at segment centroid or mean over mask
        dens = edge_density[seg_mask].mean() if edge_density is not None else 0.0
        # high density → effective min area smaller (harder to merge)
        effective_min = min_area_px * (1.0 - 0.7 * dens)
        if seg_clusters[sid][1] >= effective_min:
            continue  # keep this "small but important" region
        # else merge as before
```

`pipeline.py` передаёт edge_density:
```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150).astype(np.float32)
edge_density = cv2.boxFilter(edges / 255.0, -1, (15, 15), normalize=True)
labels = merge_small_regions(labels, palette, min_area_px, edge_density)
```

**Тесты:** unit test with small high-contrast island that must survive merge.

---

### Задача B4: skip smooth_label_boundaries при низком sigma

**Файл:** `app/pipeline.py`

```python
if pp.boundary_sigma >= 0.5:
    labels = smooth_label_boundaries(labels, sigma=pp.boundary_sigma)
```

При detail_level high (`boundary_sigma` → 0.3) — skip, не размывать тонкие границы.

---

### Задача B5: clean_mask — noop или только median

**Файл:** `app/regions.py` / `postprocess.py`

В `extract_regions` после A3:
```python
# Prefer no topology-changing morph
# If denoise needed: medianBlur(k=3) only, NO MORPH_OPEN/CLOSE
```

`morph_kernel` param может остаться для API compat, но не должен OPEN'ить тонкие линии.

---

## Порядок реализации

```
A1 SharedEdge model          ← foundation
A2 extract_shared_edges      ← core algorithm
A3 extract_regions returns edges, drop clean_mask morph
A4 pdf_layout uses edges
A5 visualize uses edges
B1 bilateralFilter
B2 median flatten
B3 edge-aware merge_small
B4 skip boundary smooth
B5 clean_mask light-only
```

**Рекомендуемый порядок:** A1 → A2 → A3 → A4 → A5 → B1 → B2 → B4 → B5 → B3

(A сначала — чинит «смежные границы не мержатся»; B — детализация)

---

## Acceptance criteria

### Shared boundaries (A)
1. На outline **нет** двойных линий между соседними регионами
2. `len(edges)` для пары labels ≤ 1 (каноническая пара)
3. Outer border рисуется один раз
4. Numbers/centroids на месте
5. PDF валиден, fill preview корректен
6. Все pytest зелёные

### Detail (B)
1. Тонкие/контрастные мелкие регионы не исчезают при default detail_level=5
2. Palette size не превышает cap (`max_colors` / merge threshold)
3. Нет регрессии углов (corners covered)
4. Visual: Lion King — грива/лица читаемее, меньше «мыла»

### Regression
1. API contract: JSON keys `colored`, `outline`, `pdf` без изменений
2. UI params без breaking changes (`color_merge_threshold` 0–30 CIEDE2000)

---

## Файлы для изменения

| Файл | Задачи |
|---|---|
| `app/models.py` | A1 |
| `app/regions.py` | A2, A3, B5 |
| `app/pipeline.py` | A3, B1, B2, B3, B4 |
| `app/postprocess.py` | B3, B5 |
| `app/pdf_layout.py` | A4 |
| `app/visualize.py` | A5 |
| `tests/test_regions.py` | A2, A3 |
| `tests/test_pipeline.py` | A, B |
| `tests/test_pdf_layout.py` | A4 |
| `AGENTS.md` | sync after done |

---

## Не делать (out of scope)

- Менять UI/Liquid Glass
- Менять paper/PDF legend layout
- Переписывать k-means (уже Lab + CIEDE2000)
- Watershed / graph-cut с нуля (можно позже)
- ML superpixels

---

## Откат

- Каждая задача — отдельный коммит
- `git checkout HEAD~1 -- <file>` при регрессии
- Fallback: если `data.edges` пуст — старый per-region stroke

---

## Финальный шаг

1. `python -m pytest tests/ -q --tb=short` — all green  
2. Visual check: Lion King input → outline без double lines, forms readable  
3. Push `feat/shared-boundaries`  
4. PR → `feat/pipeline-v2` (или `master` по договорённости)

```powershell
git checkout -b feat/shared-boundaries
# ... implement ...
git push -u origin feat/shared-boundaries
gh pr create --base feat/pipeline-v2 --head feat/shared-boundaries --title "Shared boundaries + detail preservation"
```

---

## Критерии готовности одной фразой

> Outline = одна линия на стык; мелкие важные формы не съедаются; palette limit соблюдён; 48+ tests green.
