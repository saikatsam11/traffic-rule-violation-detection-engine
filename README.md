# Traffic Rule Violation Detection
**AID 728 — Computer Vision Course Project**

Detects traffic rule violations involving two-wheelers from a single street-scene image. The system identifies riders without helmets, over-capacity vehicles (more than 2 riders), and reads the vehicle licence plate for each detected motorcycle.

---

## Directory Structure

```
<MT2025705_716_723>/
├── solution.py                        # Submission entry point
├── test_pipeline.py                   # End-to-end test script with visualisation
├── pipeline/
│   ├── __init__.py
│   ├── two_wheeler_detector.py        # Module 1 — motorcycle + rider detection
│   ├── helmet_detector.py             # Module 2 — per-rider helmet classification
│   ├── plate_reader.py                # Module 3 — plate detection + OCR
│   └── violation_engine.py            # Module 4 — rule-based violation aggregation
├── models/                            # download this folder from below mentioned link 
│   ├── yolov8m.pt                     # 50 MB  — COCO YOLOv8m (Module 1)
│   ├── yolov8m_helmet.pt              # 50 MB  — Roboflow helmet model (Module 2)
│   ├── yolov8s_plates_final.pt        # 21 MB  — Fine-tuned plate detector (Module 3)
│   ├── global_mobile_vit_v2_ocr.onnx  # 4.8 MB — FastPlateOCR (Module 3)
│   └── global_mobile_vit_v2_ocr_config.yaml   # OCR charset config
├── requirements.txt
└── README.md
```
## Download Pretrained Models

Due to LMS upload size limitations, the `models/` directory is hosted separately.

Download all required model files from the following link and place them inside the `models/` folder before running the project:

Model Download Link:  
https://iiitbac-my.sharepoint.com/:f:/g/personal/saikat_pal_iiitb_ac_in/IgBW2uu5ggJmQJnLYj9MvoDrAbFdMUoFYPaoK0Ku4YOIim0?e=r0wWev


> Total model size: **~126 MB** (within the 250 MB limit)

---

## Pipeline

```
Input Image
    │
    ▼
Module 1 — TwoWheelerDetector
    YOLOv8m (COCO) — detects motorcycles and persons
    IoP-based rider association (parked bikes dropped)
    │
    ├─────────────────────────────────┐
    ▼                                 ▼
Module 2 — HelmetDetector     Module 3 — PlateReader
    YOLOv8m (Roboflow)            YOLOv8s + FastPlateOCR ONNX
    Head crop per rider           5–7 preprocessing variants
    helmet / no_helmet /          deskew + 2-row split
    undetectable                  per-position majority vote
    │                                 │
    └──────────────┬──────────────────┘
                   ▼
          Module 4 — ViolationEngine
          Pure rule-based logic
          no_helmet + triple_rider rules
                   │
                   ▼
              JSON Output
```

---

## Installation

```bash
pip install -r requirements.txt
```

All models must be present in the `models/` directory. Internet access is disabled during evaluation — no downloads occur at runtime.

---

## Running `solution.py`

`solution.py` exposes the `TrafficViolationDetector` class which is dynamically imported by the evaluator.

### Programmatic usage

```python
from solution import TrafficViolationDetector

# Step 1 — initialise once (loads all models)
model = TrafficViolationDetector(model_dir="./models")

# Step 2 — run on any image
output = model.predict("path/to/image.jpg")

print(output)
```

### Quick one-liner test

```bash
python -c "
from solution import TrafficViolationDetector
model = TrafficViolationDetector(model_dir='./models')
print(model.predict('path/to/image.jpg'))
"
```

### Expected output format

```json
{
  "violations": [
    {
      "num_riders": 2,
      "helmet_violations": 1,
      "license_plate": "DL8SAB1234"
    },
    {
      "num_riders": 1,
      "helmet_violations": 0,
      "license_plate": "UNDETECTED"
    }
  ]
}
```

Each entry in `"violations"` corresponds to one detected motorcycle. If no motorcycles are found, returns `{"violations": []}`.

| Field | Description |
|-------|-------------|
| `num_riders` | Total riders detected on the vehicle |
| `helmet_violations` | Riders without a confirmed helmet |
| `license_plate` | OCR result, or `"UNREAD"` / `"UNDETECTED"` |

---

## Running `test_pipeline.py`

`test_pipeline.py` is a development/debugging script that runs the full pipeline on a folder of images and saves annotated outputs.

```bash
python test_pipeline.py
```

Annotated images are saved to `test_results/`. Console output format:

```
08.jpg  →  3 vehicle(s)
  vehicle[0] plate='UNDETECTED' violations=[]
  vehicle[1] plate='UNDETECTED' violations=['no_helmet']
  vehicle[2] plate='2IR385'     violations=[]
```

**Annotation legend:**

| Colour | Meaning |
|--------|---------|
| Blue box | Vehicle crop |
| Green box | Detected plate + OCR text |
| Green rider box | Helmet confirmed |
| Red rider box | No helmet |
| Yellow rider box | Undetectable head region |
| Red text | Violation label |

---

## Violation Rules

| Violation | Tag | Condition |
|-----------|-----|-----------|
| Helmet violation | `no_helmet` | Any rider has status `no_helmet` or `undetectable` |
| Over-capacity | `triple_rider` | Number of riders ≥ 3 |

Both violations can fire simultaneously on the same vehicle.

---

## Models

| Module | Model | Size | Source |
|--------|-------|------|--------|
| 1 — Two-wheeler detection | YOLOv8m (COCO) | 50 MB | Ultralytics official |
| 2 — Helmet detection | YOLOv8m | 50 MB | Roboflow Universe, fine-tuned |
| 3 — Plate detection | YOLOv8s | 21 MB | Fine-tuned, mAP50 ≈ 0.94 |
| 3 — Plate OCR | FastPlateOCR MobileViT-v2 | 4.8 MB | Pretrained ONNX |

---

## Known Limitations

- Plates in dense/occluded traffic may return `UNDETECTED`
- Head coverings (scarves, dupattas) may be flagged as helmet violations under the safety-conservative rule
- Very small or distant plates may return truncated OCR results
- Non-Indian helmet styles may reduce helmet detection accuracy
