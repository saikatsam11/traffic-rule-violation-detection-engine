# 🛵 Traffic Rule Violation Detection Engine

> Detects traffic rule violations involving two-wheelers from a single street-scene image.  
> Identifies **riders without helmets**, **over-capacity vehicles**, and **reads the vehicle licence plate** for each detected motorcycle.

---

## 📌 Overview

This project is a four-stage computer vision pipeline processes a street-scene image and produces structured violation reports, integrating object detection, head-region classification, OCR-based plate reading, and rule-based aggregation — all from a single forward pass per module.

---

## 🏗️ Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────────┐
│  Module 1: Two-Wheeler Detector │  YOLOv8m — detects motorcycles + riders
│  + Camera Angle Detection       │  Adaptive: top-down vs street-level
└───────────────┬─────────────────┘
                │  VehicleInstance (crop, rider boxes, camera_angle)
                ▼
┌─────────────────────────────────┐
│  Module 2: Helmet Detector      │  YOLOv8m_helmet — per-rider head crop
│  + Angle-Adaptive Head Region   │  Angle-specific confidence thresholds
└───────────────┬─────────────────┘
                │  HelmetResult (per_rider_status)
                ▼
┌─────────────────────────────────┐
│  Module 3: Plate Reader         │  YOLOv8s plates + FastPlateOCR (ONNX)
└───────────────┬─────────────────┘
                │  PlateResult (license_plate, ocr_confidence)
                ▼
┌─────────────────────────────────┐
│  Module 4: Violation Engine     │  Rule-based aggregation
└───────────────┬─────────────────┘
                │
                ▼
     ViolationReport (JSON / CSV)
```

---

## 📁 Directory Structure

```
<MT2025705_716_723>/
├── solution.py                         # Submission entry point
├── test_pipeline.py                    # End-to-end test script with visualisation
├── pipeline/
│   ├── __init__.py
│   ├── two_wheeler_detector.py         # Module 1 — motorcycle + rider detection
│   ├── helmet_detector.py              # Module 2 — per-rider helmet classification
│   ├── plate_reader.py                 # Module 3 — plate detection + OCR
│   └── violation_engine.py            # Module 4 — rule-based violation aggregation
├── models/                             # See model download instructions below
│   ├── yolov8m.pt                      # COCO YOLOv8m (Module 1)
│   ├── yolov8m_helmet.pt               # Roboflow helmet model (Module 2)
│   ├── yolov8s_plates_final.pt         # Fine-tuned plate detector (Module 3)
│   ├── global_mobile_vit_v2_ocr.onnx  # FastPlateOCR (Module 3)
│   └── global_mobile_vit_v2_ocr_config.yaml
├── requirements.txt
└── README.md
```

---

## ⚙️ Key Technical Features

| Feature | Detail |
|---|---|
| **Rider–Motorcycle Association** | Full-body IoP (Intersection over Person area) — robust to car occupants at road level |
| **Camera Angle Detection** | Automatically detects top-down vs street-level based on motorcycle aspect ratio |
| **Adaptive Head Region** | Head crop fraction and confidence thresholds scale with detected camera angle |
| **Helmet Classification** | Per-rider, with per-class confidence thresholds and safety-first tiebreaking |
| **Licence Plate OCR** | YOLOv8s plate detector + FastPlateOCR ONNX runtime |
| **Violation Rules** | No-helmet (any rider), triple-rider (≥3 riders), with flag propagation |

---

## 🚀 Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pretrained models

> Models are not committed to this repo due to file size (~126 MB total).  
> Download the `models/` folder from the link below and place it at the project root before running.

**[⬇️ Download Models (~126 MB)](https://iiitbac-my.sharepoint.com/:f:/g/personal/saikat_pal_iiitb_ac_in/IgBW2uu5ggJmQJnLYj9MvoDrAbFdMUoFYPaoK0Ku4YOIim0?e=r0wWev)**

```
models/
├── yolov8m.pt                        (~50 MB)
├── yolov8m_helmet.pt                 (~50 MB)
├── yolov8s_plates_final.pt           (~21 MB)
├── global_mobile_vit_v2_ocr.onnx    (~4.8 MB)
└── global_mobile_vit_v2_ocr_config.yaml
```

### 3. Run on a single image

```bash
python solution.py <image_path>
```

### 4. Run end-to-end test with visualisation

```bash
python test_pipeline.py <image_path>
```

---

## 📊 Output

Each detected vehicle produces a `ViolationReport`:

```json
{
  "vehicle_id": 0,
  "license_plate": "DL3CAV2022",
  "plate_valid": true,
  "violations": ["no_helmet"],
  "num_riders": 1,
  "per_rider_status": ["no_helmet"],
  "detection_confidence": 0.81,
  "ocr_confidence": 0.74,
  "flags": []
}
```

Reports are exported as CSV via `report_to_csv()`.

---

## 🧠 Models Used

| Model | Source | Purpose |
|---|---|---|
| `yolov8m.pt` | Ultralytics COCO | Person + motorcycle detection |
| `yolov8m_helmet.pt` | Roboflow Universe | Helmet / no-helmet classification |
| `yolov8s_plates_final.pt` | Fine-tuned | Indian licence plate detection |
| `global_mobile_vit_v2_ocr.onnx` | FastPlateOCR | Plate character recognition |

---

## 📦 Requirements

```
ultralytics
opencv-python
numpy
onnxruntime
fast-plate-ocr
```

---