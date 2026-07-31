# `pipeline/two_wheeler_detector.py`

**Module 1** of the violation pipeline — finds two-wheelers and the riders sitting on them.

---

## What it does

Given a road-scene image, it returns a list of `VehicleInstance` objects. Each instance is one motorcycle that has at least one confirmed rider, plus a padded crop containing the bike + riders for downstream stages.

Parked bikes (no rider) are dropped — they cannot violate anything.

## Model used

| Model | File | Size | Source |
|-------|------|------|--------|
| YOLOv8m (COCO-pretrained) | [`models/yolov8m.pt`](../models/yolov8m.pt) | 50 MB | Ultralytics official weights |

We use the COCO classes:
- class `3` → motorcycle
- class `0` → person

No fine-tuning required; COCO accuracy on these two classes is already strong on street scenes.

## How it's used

```python
from pipeline.two_wheeler_detector import TwoWheelerDetector

detector  = TwoWheelerDetector(model_path="models/yolov8m.pt")
instances = detector.detect("test_images/01.jpg")

for inst in instances:
    print(inst.num_riders, inst.crop_box)
    # inst.crop_image is the BGR np.ndarray fed to plate_reader / helmet_detector
```

The output `VehicleInstance` is the contract every downstream module reads.

## Architecture

```
image
  │
  ├─► YOLOv8m (motorcycle, conf≥0.25, TTA on) ─► moto boxes
  └─► YOLOv8m (person,     conf≥0.40, TTA on) ─► person boxes
                                                       │
              IoU dedup of overlapping motorcycle boxes
                                                       │
              Rider association — Pass 1 then Pass 2
                                                       │
            For each motorcycle with ≥1 confirmed rider
                       │
                       ▼
              union(moto, riders) + 15% pad ─► crop
                       │
                       ▼
                VehicleInstance(crop_image, ...)
```

### Rider association (the hard part)

Pass 1 uses **IoP** (Intersection over Person area):
> What fraction of the *person's bounding box* overlaps the motorcycle box?

Real riders score 0.40–0.90. Bystanders / car occupants score 0.05–0.15. This works across both top-down CCTV angles and street-level cameras — it's a 3D problem expressed in a single 2D ratio.

Pass 2 fallback (for tall riders whose torso extends *above* the motorcycle box) requires three conditions to fire simultaneously:
- some IoP at all (≥ 0.15)
- horizontal overlap ≥ 30 %
- `vertical_gap = max(0, moto_top − person_bottom) ≤ 20 px`

The vertical-gap check (added recently) replaced an older centroid-distance heuristic that broke on close-up shots of tall vehicles.

## Improvements already done

1. **F10 — IoP-based rider association** replacing brittle "containment OR bottom-half overlap" logic that confused car occupants for riders.
2. **Vertical-gap fallback** instead of centroid distance — fixes 11.jpg-style close-ups where the rider's centroid drifts far from the bike's centroid.
3. **Per-motorcycle NMS dedup** so two overlapping motorcycle boxes don't get assigned the same rider twice.
4. **Test-time augmentation** (`augment=True`) on both YOLO calls — adds ~2× cost for a small recall bump on far/blurry vehicles.
5. **Small-crop and fused-bike warnings** flagged in the output for downstream modules to handle.
6. **Module-level exception guard** — never raises; returns `[]` on any failure.

## Suggestions to make it better (simple wins first)

1. **Switch detector model size** — `yolov8s.pt` (22 MB) is ~3× faster than `yolov8m.pt` and only loses ~2 mAP on COCO. If inference time matters more than the last bit of accuracy, swap it. If accuracy matters more, try `yolov8l.pt`.
2. **Filter person boxes by area before association** — tiny person boxes (< 30×30 px) on far-away pedestrians sometimes get matched to nearby bikes via Pass 2. Add a min-area gate.
3. **Re-detect inside the crop** — once you have a motorcycle crop, run a second person-detection pass at higher resolution on just that crop. Catches small pillion riders (back-seat passenger) the full-image pass missed.
4. **Track riders across video frames** — if you ever feed video, a simple IoU tracker (or ByteTrack) gives you stable IDs and lets you smooth out helmet/plate readings over time. One missed frame stops mattering.
5. **Use a motorcycle-specific YOLO** — there are open-weight YOLOv8 models fine-tuned on Indian/SE-Asian street scenes that detect motorcycles + tuk-tuks + scooters as separate classes. Better recall on scooters than COCO.

## Functions / classes

### `Config`
Tunable knobs. Most-edited fields:
- `MOTORCYCLE_CONF` (default 0.25), `PERSON_CONF` (0.40) — raise to drop false positives, lower to gain recall.
- `MIN_FULL_BODY_IOP` (0.30) — Pass 1 threshold. Don't go below 0.15 or above 0.55.
- `MAX_VERTICAL_GAP_PX` (20) — Pass 2 fallback gap.
- `CROP_PADDING` (0.15) — extra context around the union box.
- `TTA` (True) — toggle test-time augmentation.

### `VehicleInstance` (dataclass)
The output contract. Every downstream module reads this and nothing else from Module 1.

| Field | Meaning |
|---|---|
| `motorcycle_box` | `[x1,y1,x2,y2]` in **original image** coords |
| `rider_boxes` | confirmed riders, original-image coords |
| `num_riders` | quick count |
| `crop_box` | padded union — original-image coords |
| `crop_image` | BGR `np.ndarray`, the actual pixels |
| `rider_boxes_in_crop` | rider boxes translated into **crop-local** coords (handy for helmet/plate stages that work inside the crop) |
| `small_crop_flag` | crop < 60×60 px ⇒ downstream accuracy warning |
| `fused_bike_warning` | one detection box covers two real bikes |
| `confidence` | motorcycle detection confidence |

### `TwoWheelerDetector`

| Method | Purpose |
|---|---|
| `__init__(model_path, config)` | loads weights, stores config |
| `detect(image_path) → list[VehicleInstance]` | public entry point — never raises, returns `[]` on failure |
| `_detect_motorcycles(image)` | YOLO call filtered to class 3 |
| `_detect_persons(image)` | YOLO call filtered to class 0 |
| `_deduplicate_motorcycles(boxes, confs)` | greedy NMS by confidence |
| `_associate_riders(motos, persons)` | the IoP / vertical-gap two-pass logic — heart of the module |
| `_build_instances(...)` | builds the padded crops + translates rider boxes to crop space |
| `_iop`, `_iou`, `_horizontal_overlap_ratio` | static geometry helpers |
| `_load_image` | safe `cv2.imread` |

### `visualise_instances(image_path, instances, output_path)`
Debug overlay — blue = motorcycle, green = riders, red = crop, orange = fused-warning crop.

### `__main__` block
Standalone test:
```bash
python pipeline/two_wheeler_detector.py models/yolov8m.pt test_images/01.jpg
# writes debug_m1.jpg
```
