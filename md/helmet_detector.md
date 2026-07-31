# `pipeline/helmet_detector.py`

**Module 2** — checks whether each rider on a `VehicleInstance` is wearing a helmet.

---

## What it does

For every rider in a `VehicleInstance`, it crops the **head region** (top ~45 % of the rider's bounding box) and runs a helmet-classification YOLO on that small crop. The output is a `HelmetResult` reporting how many riders are without a helmet.

Behavioural rules (locked in spec):
- **A.** No detection on a head ⇒ count as a violation (we cannot prove they wore one).
- **B.** Confidence threshold per class (independently tunable for `helmet` vs `no_helmet`).
- **C.** Conflicting boxes on the same head ⇒ highest-confidence label wins.
- **D.** Input crop flagged `small_crop_flag=True` ⇒ run normally but mark `low_confidence_flag=True`.

## Model used

| Model | File | Source |
|-------|------|--------|
| YOLOv8m fine-tuned for helmets | [`models/yolov8m_helmet.pt`](../models/yolov8m_helmet.pt) (50 MB) | Roboflow Universe pretrained |

The model emits classes like `helmet`, `no_helmet`, `with_helmet`, `head`, etc. Roboflow naming is wildly inconsistent across uploads, so we normalise them through the `HELMET_CLASS_MAP` dict to canonical `helmet` / `no_helmet`.

If you swap models, **add new aliases to `HELMET_CLASS_MAP`** — otherwise the detector will silently flag everything as `undetectable`.

## How it's used

```python
from pipeline.two_wheeler_detector import TwoWheelerDetector
from pipeline.helmet_detector       import HelmetDetector, detect_batch

m1 = TwoWheelerDetector("models/yolov8m.pt")
m2 = HelmetDetector("models/yolov8m_helmet.pt")

instances = m1.detect("test_images/01.jpg")
results   = detect_batch(m2, instances)

for inst, res in zip(instances, results):
    print(res.helmet_violations, "/", res.total_riders, res.per_rider_status)
```

## Architecture

```
VehicleInstance
   │
   ├─ for each rider_box_in_crop:
   │     └─► extract head region (top 45% + 10% padding, clamped)
   │            │
   │            ▼
   │       too small? ─► "undetectable"  (rule A → counts as violation)
   │            │
   │            ▼
   │     YOLO helmet model on head crop
   │            │
   │            ▼
   │     parse → normalise class names → filter by per-class threshold
   │            │
   │            ▼
   │     resolve label (highest conf wins, "no_helmet" tiebreak)
   │            │
   │            ▼
   │     status ∈ {helmet, no_helmet, undetectable}
   │
   ▼
HelmetResult(helmet_violations, total_riders, per_rider_status, low_confidence_flag)
```

## Improvements already done

1. **H4 — class-name normalisation** so swapping Roboflow models doesn't silently break the detector.
2. **Init-time class validation** — warns immediately if the loaded model has zero recognisable classes.
3. **Per-class confidence thresholds** — you can require, say, very high confidence for `helmet` but be more permissive for `no_helmet` (a safety-conservative tilt).
4. **"no_helmet" tiebreaker** when two boxes on one head share the same confidence — fail-safe toward flagging.
5. **Per-rider exception isolation** — one bad crop never kills the whole vehicle's result.
6. **Module-level fallback** — on catastrophic failure, all riders are flagged as violations (safety-first default).
7. **`low_confidence_flag` propagation** so downstream reports can mark uncertain readings.

## Suggestions to make it better (simple wins first)

1. **Tighten the head region for tall pillion riders** — currently top 45 %. For a passenger sitting upright on a sportbike, 45 % can include an arm. A simple per-rider tweak: shrink the head region as `rider_h / image_h` grows.
2. **Run the model directly on the rider box** (not a head sub-crop) — many helmet models are trained on full-body crops. Quick A/B test against the current head-crop approach to see which generalises better on your data.
3. **Add a "color-gap" sanity check** — if the head region has uniform skin/hair texture (low std-dev) but the model says "helmet", lower the confidence. Cheap heuristic against false positives on bald people.
4. **Vote across multiple frames in video** — same trick as the OCR ensemble. If 7/10 frames say `no_helmet` and 3 say `helmet`, the rider is not wearing a helmet.
5. **Fine-tune on Indian helmet styles** — many open helmet datasets are dominated by Western full-face helmets. Half-shells and turban-helmet combos common in South Asia trip up generic models.

## Functions / classes

### `Config`
- `HEAD_REGION_FRACTION` (0.45) — top fraction of rider box treated as head.
- `HELMET_CONF` / `NO_HELMET_CONF` (both 0.40) — independent thresholds.
- `MIN_HEAD_SIZE` (20 px) — below this we mark `undetectable`.
- `HEAD_NMS_IOU` (0.45) — NMS within the head region.
- `HEAD_PADDING` (0.10) — fractional pad before the helmet model sees it.

### `HELMET_CLASS_MAP`
Dict mapping every Roboflow naming variant to canonical `helmet` / `no_helmet`. Add to it whenever you load a new model.

### `HelmetResult` (dataclass)
| Field | Meaning |
|---|---|
| `helmet_violations` | riders confirmed without helmet (includes `undetectable`) |
| `total_riders` | mirrors `VehicleInstance.num_riders` |
| `per_rider_status` | list of `"helmet" | "no_helmet" | "undetectable"`, index-aligned |
| `low_confidence_flag` | True when input had `small_crop_flag` |

### `HelmetDetector`

| Method | Purpose |
|---|---|
| `__init__(model_path, config)` | load weights + run `_validate_classes()` |
| `detect(instance) → HelmetResult` | public entry — never raises |
| `_classify_rider(crop, rider_box, idx)` | end-to-end one-rider classification |
| `_extract_head_region(crop, rider_box)` | top 45 % crop with padding + clamping + min-size guard |
| `_parse_detections(results)` | normalise class names, apply per-class threshold |
| `_resolve_label(detections)` | highest-confidence wins, no_helmet tiebreak |
| `_validate_classes()` | init-time warning if model classes don't match the alias map |

### `detect_batch(detector, instances)`
Convenience: index-aligned list comprehension.

### `visualise_helmet_results(instances, results, source_image, output_path)`
Debug overlay — green = helmet, red = no_helmet, yellow = undetectable, orange-bordered crop = low-confidence.

### `__main__` block
```bash
python pipeline/helmet_detector.py \
  models/yolov8m_helmet.pt models/yolov8m.pt test_images/01.jpg
# writes debug_m2.jpg
```
