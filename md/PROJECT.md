# Traffic Rule Violation Detection — Project Overview

Course AID 728. Detect motorcycle riders breaking the rules from a single street-scene image, read their licence plate.

Two violations targeted:
1. **No helmet** — any rider on the bike is helmet-less.
2. **Triple riding** — more than 2 people on a two-wheeler.

---

## The pipeline

```
input image
   │
   ▼
┌─────────────────────────────────────────────┐
│ Module 1 — TwoWheelerDetector               │
│   YOLOv8m (COCO) → motorcycles + persons    │
│   IoP-based rider association               │
│   output: VehicleInstance per bike with     │
│           ≥1 rider (parked bikes dropped)   │
└─────────────────────────────────────────────┘
   │
   ├──────► VehicleInstance ──────►
   │                               │
   ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│ Module 2 — Helmet       │   │ Module 3 — PlateReader          │
│   YOLOv8m_helmet on     │   │   YOLOv8s plate detector (TTA)  │
│   top 45 % of each      │   │     ↓ best plate box            │
│   rider box.            │   │   FastPlateOCR (ONNX) on        │
│   output:               │   │     5–7 preprocessing variants  │
│     HelmetResult per    │   │     + deskew + 2-row split      │
│     instance.           │   │   output: PlateResult per       │
│                         │   │     instance.                   │
└─────────────────────────┘   └─────────────────────────────────┘
   │                               │
   └──────────────┬────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ Module 4 — ViolationEngine  (todo)          │
│   pure rule logic over the dataclasses      │
│   output: ViolationReport per vehicle       │
└─────────────────────────────────────────────┘
                  │
                  ▼
            CSV / JSON / debug image
```

Per-module details: [`two_wheeler_detector.md`](./two_wheeler_detector.md) · [`helmet_detector.md`](./helmet_detector.md) · [`plate_reader.md`](./plate_reader.md) · [`violation_engine.md`](./violation_engine.md).

## Models on disk

```
models/
├── yolov8m.pt                              50 MB   COCO YOLOv8m         Module 1
├── yolov8m_helmet.pt                       50 MB   Roboflow helmet      Module 2
├── yolov8s_plates_final.pt                 21 MB   fine-tuned plate det Module 3 stage 1
├── global_mobile_vit_v2_ocr.onnx          4.8 MB   FastPlateOCR         Module 3 stage 2
└── global_mobile_vit_v2_ocr_config.yaml   0.5 KB   OCR charset config   Module 3 stage 2
```

Total ≈ 126 MB. The 250 MB submission budget has plenty of room — you could swap in `yolov8m` for the plate detector or load a second OCR model if accuracy needs it.

## Repository layout

```
cv_project/
├── models/                         
├── pipeline/                        the 4 modules
│   ├── two_wheeler_detector.py
│   ├── helmet_detector.py
│   ├── plate_reader.py
│   ├── violation_engine.py 
│   └── __init__.py
├── test_pipeline.py                 end-to-end runnable test
├── md/               
├── requirements.txt
```

## How to run the test scripts

### One-shot end-to-end test on every image in `test_images/`
```bash
source venv/bin/activate
python test_pipeline.py
```
This runs Modules 1 + 3 (Module 2 not yet wired in by default — see `violation_engine.md`) on every `*.jpg` in `test_images/`, prints per-vehicle output, and writes annotated images into `test_results/`. Console line format:

```
01.jpg  →  2 vehicle(s)
  plate='VA9992' valid=True det=0.20 ocr=0.63 variant=tight
```

Annotated image legend: blue = vehicle crop, green = plate box + OCR text.

### Running individual modules

Each module has a `__main__` block for isolated debugging:

```bash
# Module 1 — vehicle/rider detection only
python pipeline/two_wheeler_detector.py models/yolov8m.pt test_images/01.jpg
# writes debug_m1.jpg

# Module 2 — helmet only (chains Module 1 internally)
python pipeline/helmet_detector.py \
  models/yolov8m_helmet.pt models/yolov8m.pt test_images/01.jpg
# writes debug_m2.jpg
```

(`plate_reader.py` does not have a standalone main; use `test_pipeline.py`.)

## Steps to improve the pipeline (in priority order)

Highest-impact first. Each one is largely independent.

1. **Fine-tune FastPlateOCR.** Run [`notebooks/finetune_fastplate.ipynb`](../notebooks/finetune_fastplate.ipynb) on Colab with synthetic plates (heavy padding + Gaussian + motion blur + JPEG artifacts). Add 2-row Indian-style plates to the synth generator. Drop the resulting `.onnx` + `_config.yaml` into `models/`. Pipeline picks them up with no code change.

2. **Multi-scale plate detection.** Run the YOLO plate model twice on each vehicle crop — once at imgsz 640, once at 1280 — and NMS-merge results. Recovers the small plates currently returning `(no plate detected)` for 3–4 vehicles in the test set.

3. **Build `ViolationEngine`.** Pure rule logic, see [`violation_engine.md`](./violation_engine.md). Required before submission.

4. **Better deskew.** Current contour-based deskew works but fails on plates without a clean border. Replace with a min-area-rect fit on text contours.

5. **Fine-tune the plate detector** on Indian motorcycle plates (a separate Colab job — there's no notebook for this in the repo right now, but the original `finetune_motorcycle_plates.ipynb` workflow described what to do).

6. **Vote across multiple frames** if the input becomes video.

Per-file simple wins are listed at the end of each per-module .md.

## Submission format (assignment-ready)

The assignment expects:

1. A `solution.py` (or equivalent entry point) exposing a single class — usually called `TrafficViolationDetector` — with one mandatory method:

   ```python
   class TrafficViolationDetector:
       def __init__(self):
           # load all models into memory
           ...

       def detect(self, image_path: str) -> list[dict]:
           """
           Returns a list of dicts, one per detected violating vehicle:
             {
               "license_plate"      : str,         # or "UNREAD" / "UNDETECTED"
               "motorcycle_box"     : [x1,y1,x2,y2],
               "num_riders"         : int,
               "violations"         : ["no_helmet", "triple_rider"],
               "helmet_status"      : ["helmet","no_helmet","undetectable"],
               "confidence"         : float,
             }
           """
           ...
   ```

2. The 4 module files in `pipeline/`.
3. The 5 model files in `models/` (≤ 250 MB total).
4. A `requirements.txt`.
5. **No** `venv/`, no notebook output, no test images in the submission zip.

Suggested zip layout:

```
submission/
├── solution.py
├── pipeline/
│   ├── __init__.py
│   ├── two_wheeler_detector.py
│   ├── helmet_detector.py
│   ├── plate_reader.py
│   └── violation_engine.py
├── models/
│   ├── yolov8m.pt
│   ├── yolov8m_helmet.pt
│   ├── yolov8s_plates_final.pt
│   ├── global_mobile_vit_v2_ocr.onnx
│   └── global_mobile_vit_v2_ocr_config.yaml
└── requirements.txt
```

`solution.py` is essentially a thin wrapper that:

```python
from pipeline.two_wheeler_detector import TwoWheelerDetector
from pipeline.helmet_detector       import HelmetDetector, detect_batch
from pipeline.plate_reader          import PlateReader, read_batch
from pipeline.violation_engine      import ViolationEngine

class TrafficViolationDetector:
    def __init__(self):
        self.m1 = TwoWheelerDetector("models/yolov8m.pt")
        self.m2 = HelmetDetector("models/yolov8m_helmet.pt")
        self.m3 = PlateReader(
            model_path     = "models/yolov8s_plates_final.pt",
            ocr_onnx_path  = "models/global_mobile_vit_v2_ocr.onnx",
            ocr_config_path= "models/global_mobile_vit_v2_ocr_config.yaml",
        )
        self.engine = ViolationEngine()

    def detect(self, image_path):
        instances = self.m1.detect(image_path)
        helmets   = detect_batch(self.m2, instances)
        plates    = read_batch(self.m3, instances)
        reports   = self.engine.evaluate(instances, helmets, plates)
        return [r.to_dict() for r in reports]
```

### Pre-submission checklist

- [ ] `du -sh submission/models/` ≤ 250 MB
- [ ] `python -c "from solution import TrafficViolationDetector; d=TrafficViolationDetector(); print(d.detect('test_images/01.jpg'))"` runs end-to-end with no exception
- [ ] `requirements.txt` pins the versions actually in `venv/` (use `pip freeze | grep -iE 'ultralytics|fast-plate-ocr|onnxruntime|opencv|numpy|torch'`)
- [ ] Strip `__pycache__/`, `.DS_Store`, `venv/`, `notebooks/`, `test_results/`, `test_images/` from the zip
- [ ] All 4 modules importable in isolation (no implicit cross-module side-effects on import)
- [ ] `TrafficViolationDetector` returns the spec-defined dict structure (not the internal dataclasses)
- [ ] Catastrophic failure on a single image returns an empty list, not an exception
