# `pipeline/plate_reader.py`

**Module 3** — finds the licence plate inside a `VehicleInstance` and reads its text.

This is the most-iterated module in the project. It went through three OCR backends (EasyOCR → planned PARSeq → final FastPlateOCR) and now wraps the OCR call inside a multi-variant ensemble with 6 robustness layers.

---

## What it does

Two stages:

1. **Detect** — YOLOv8s plate detector (TTA on) finds the plate region inside the vehicle crop.
2. **Read** — FastPlateOCR runs on **5–7 preprocessing variants** of the plate (default / tight / wide / CLAHE / sharp / deskew / 2-row split), and the highest-confidence prediction wins.

Then come the rescue layers: deskew correction, per-position majority vote on close ties, character-class confusion correction (e.g. `8 → B` if the result is digit-only), and finally a confidence floor that rejects garbage as `"UNREAD"`.

## Models used

| Model | File | Size | Source |
|---|---|---|---|
| Plate detector — YOLOv8s, fine-tuned | [`models/yolov8s_plates_final.pt`](../models/yolov8s_plates_final.pt) | 21 MB | Trained in `notebooks/finetune_fastplate.ipynb`'s sister notebook (Phase 2 + Phase 3 motorcycle adaptation). Achieves mAP50 ≈ 0.94 on plate detection. |
| OCR — FastPlateOCR `global-plates-mobile-vit-v2` | [`models/global_mobile_vit_v2_ocr.onnx`](../models/global_mobile_vit_v2_ocr.onnx) + `_config.yaml` | 4.8 MB | Pretrained, ONNX runtime |

Why FastPlateOCR over EasyOCR or Tesseract:
- Trained specifically on licence plates (not generic scene text).
- ONNX inference, ~10 ms per plate on CPU.
- Handles a wide character set out of the box.
- Easy to fine-tune (we have a notebook).

## How it's used

```python
from pipeline.plate_reader import PlateReader

reader = PlateReader(
    model_path      = "models/yolov8s_plates_final.pt",
    ocr_onnx_path   = "models/global_mobile_vit_v2_ocr.onnx",
    ocr_config_path = "models/global_mobile_vit_v2_ocr_config.yaml",
)

result = reader.read(vehicle_instance)
print(result.license_plate, result.valid_format, result.ocr_confidence)
```

Hub fallback (auto-downloads to `~/.cache/fast-plate-ocr/`):
```python
reader = PlateReader(model_path="models/yolov8s_plates_final.pt")
```

After fine-tuning your own ONNX weights, just point the constructor at the new files — no other code changes needed.

## Architecture

```
VehicleInstance.crop_image
        │
        ▼
YOLOv8s plate detector (conf≥0.15, TTA on)
        │
        ▼
best plate box (highest conf, min 30×10 px)
        │
        ▼
build preprocessing variants
   default | tight  | wide  | CLAHE  | sharp
                                +
   deskew  (if 4-corner contour found)
                                +
   row_top, row_bottom  (if w/h < 2.5  → 2-row plate)
        │
        ▼
FastPlateOCR  (one batched call)
        │
        ▼
combine row_top + row_bottom → "two_row" candidate
        │
        ▼
sort by mean char-prob ─► leader
        │
        ├─ tied within 0.02 of leader → per-position majority vote
        ▼
clean + validate (4-12 chars, has letters AND digits)
        │
        ├─ invalid? → confusion-rescue (8↔B, 0↔O, 1↔I, 5↔S, 6↔G, 2↔Z)
        ▼
ocr_confidence < 0.45 → "UNREAD"
        │
        ▼
PlateResult
```

## Improvements already done

| Layer | Problem it solves |
|---|---|
| YOLO TTA on plate detector | recovers small / blurry plates the single-pass model misses |
| 5 preprocessing variants | covers loose vs tight detector boxes, low contrast, motion blur — independent of which one was the actual failure mode |
| Batched FastPlateOCR call | runs all variants in one ONNX inference instead of 5 — almost free in time |
| Per-position majority vote on ties | when several variants disagree by < 0.02 in confidence, character-by-character voting picks the most common letter at each position |
| Deskew via 4-corner contour | rectifies tilted plates before OCR — biggest single accuracy gain on motorcycle plates |
| **2-row split-row OCR** | Indian plates (state row + reg-number row) cannot be read by single-row models. We detect aspect ratio < 2.5 and OCR the halves separately, then concatenate |
| Confusion-character rescue | rescues invalid digit-only / letter-only outputs by swapping the most likely confusables at typical positions |
| `OCR_UNREAD_FLOOR = 0.45` | low-confidence garbage becomes `"UNREAD"` instead of a hallucinated plate that pollutes downstream violation logic |

Visible wins on the test set:
- 03.jpg: 1 → 3 vehicles read (vertical-gap fix in Module 1, multi-variant ensemble here).
- 06.jpg: from `(no plate detected)` → `01CA9274` at 0.97 OCR confidence.
- 11.jpg: from 0 vehicles → `01CA9274` at 0.96.

## Suggestions to make it better (simple wins first)

1. **Multi-scale plate detection** — run the YOLO plate model twice on the vehicle crop, once at imgsz 640 and once at 1280, then NMS-merge. Recovers small / distant plates that `augment=True` alone misses (the remaining "no plate detected" cases).
2. **Stack a second OCR model** — FastPlateOCR ships several arches (`cct-xs-v1`, `cct-s-v2`, `argentinian-plates-cnn-synth-model`, …). Loading two and ensembling their predictions is mostly free on CPU and trades model diversity for accuracy.
3. **Better deskew** — current contour-based deskew finds the largest 4-corner shape. It fails on plates with no clear border. Replace with a method that fits a min-area-rect to text contours instead, then use `cv2.getRotationMatrix2D` to rotate.
4. **Region-format validation** — for Indian plates the format is `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$`. If we know the region, we can constrain the result to plates matching that regex and use the regex-best match across variants.
5. **Fine-tune FastPlateOCR on your domain** — already scaffolded in `notebooks/finetune_fastplate.ipynb`. Generate synthetic 1-row + 2-row plates with heavy padding/blur/JPEG-artifact augs, train, drop the new ONNX into `models/`. The reader picks them up with no code change.
6. **Cache + dedup across frames** — for video, collapse a track's per-frame plate predictions into one majority answer over the track.

## Functions / classes

### `Config`
- `PLATE_CONF` (0.15) — YOLO confidence threshold.
- `PLATE_TTA` (True) — toggle plate-detector TTA.
- `PAD_DEFAULT` / `PAD_TIGHT` / `PAD_WIDE` — variant padding fractions.
- `OCR_TIE_EPSILON` (0.02) — confidence margin for tie-break voting.
- `OCR_UNREAD_FLOOR` (0.45) — below this, return `"UNREAD"`.
- `TWO_ROW_RATIO` (2.5) — w/h below this triggers split-row OCR.

### `PlateResult` (dataclass)
| Field | Meaning |
|---|---|
| `license_plate` | the cleaned text, or `"UNREAD"` / `"UNDETECTED"` |
| `plate_box` | `[x1,y1,x2,y2]` in **original image** coords (or None) |
| `confidence` | YOLO plate detection conf |
| `ocr_confidence` | mean char probability of the winning variant |
| `ocr_raw` | raw text from the leading variant before voting / fixing |
| `valid_format` | passed length + alphanumeric mix checks |
| `variant_used` | which preprocessing variant won (e.g. `vote(default,clahe)+fix`) |

### `PlateReader`

| Method | Purpose |
|---|---|
| `__init__(model_path, ocr_onnx_path?, ocr_config_path?, ocr_hub_model?, config?)` | YOLO + FastPlateOCR loader; uses local files if given, else hub |
| `read(instance) → PlateResult` | public entry — never raises |
| `_detect_plate(instance)` | YOLO call with TTA, returns highest-conf valid box |
| `_deskew(plate_bgr)` | adaptive-threshold + `approxPolyDP` to find a 4-corner quad and warp to rectangle. Returns `None` if no clean quad found |
| `_crop_with_pad`, `_to_gray`, `_clahe`, `_unsharp` | preprocessing helpers |
| `_build_variants(crop, plate_box)` | builds the 5–7 variants — adds `deskew` if quad found, adds `row_top` + `row_bottom` if aspect ratio < 2.5 |
| `_read_ensemble(crop, plate_box)` | one batched OCR call → score → tie-break vote → return |
| `_combine_two_row(scored)` | concatenate top + bottom predictions into a single `two_row` candidate |
| `_position_vote(texts)` | per-position majority vote across tied variants |
| `_fix_confusables(clean)` | swap digit↔letter confusables when validation fails |
| `_clean_and_validate(raw)` | strip non-alphanumeric, check 4–12 chars + letter+digit mix |

### `read_batch(reader, instances)`
List-wrapper over `read()`.

### Constants

```python
DIGIT_TO_LETTER = {"0":"O", "1":"I", "2":"Z", "5":"S", "6":"G", "8":"B", "7":"Z"}
LETTER_TO_DIGIT = {"O":"0", "I":"1", "Z":"2", "S":"5", "G":"6", "B":"8"}
```

These are the only confusion pairs we apply — adding more (e.g. `D ↔ 0`) tends to hurt as much as it helps because the model rarely makes those mistakes.
