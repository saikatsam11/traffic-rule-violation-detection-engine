"""
End-to-end OCR pipeline test.

  Image → TwoWheelerDetector (TTA) → VehicleInstance
        → PlateReader (YOLO TTA + FastPlateOCR 5-variant ensemble)
        → annotated output
"""

import sys, cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.two_wheeler_detector import TwoWheelerDetector
from pipeline.helmet_detector       import HelmetDetector, detect_batch
from pipeline.plate_reader          import PlateReader, read_batch
from pipeline.violation_engine      import ViolationEngine

ROOT       = Path(__file__).parent
MODELS     = ROOT / "models"
IMAGES_DIR = ROOT / "test_images"
OUTPUT_DIR = ROOT / "test_results"

DETECTOR_MODEL  = MODELS / "yolov8m.pt"
HELMET_MODEL    = MODELS / "yolov8m_helmet.pt"
PLATE_MODEL     = MODELS / "yolov8s_plates_final.pt"
OCR_ONNX        = MODELS / "global_mobile_vit_v2_ocr.onnx"
OCR_CONFIG      = MODELS / "global_mobile_vit_v2_ocr_config.yaml"

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading models...")
detector = TwoWheelerDetector(model_path=str(DETECTOR_MODEL))
helmet_det = HelmetDetector(model_path=str(HELMET_MODEL))
reader   = PlateReader(
    model_path      = str(PLATE_MODEL),
    ocr_onnx_path   = str(OCR_ONNX),
    ocr_config_path = str(OCR_CONFIG),
)
engine   = ViolationEngine()
print("Models loaded.\n")

for img_path in sorted([p for p in IMAGES_DIR.glob("*.jpg") if not p.name.startswith("._")]):
    original  = cv2.imread(str(img_path))
    if original is None:
        print(f"Warning: Could not load image {img_path}, skipping.")
        continue
    instances = detector.detect(str(img_path))

    # Run downstream modules in batch
    helmet_res = detect_batch(helmet_det, instances)
    plate_res  = read_batch(reader, instances)

    # Combine in Violation Engine
    reports = engine.evaluate(instances, helmet_res, plate_res)

    vis       = original.copy()

    print(f"{img_path.name}  →  {len(instances)} vehicle(s)")

    for i, inst in enumerate(instances):
        report = reports[i]

        # 1. Draw vehicle crop (Blue)
        cx1, cy1, cx2, cy2 = inst.crop_box
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (255, 100, 0), 2)

        # 2. Draw Plate (Green)
        if report.plate_box:
            px1, py1, px2, py2 = report.plate_box
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 220, 0), 2)

            label = f"{report.license_plate}  d={report.ocr_confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (px1, py1 - th - 8), (px1 + tw + 4, py1), (0, 220, 0), -1)
            cv2.putText(vis, label, (px1 + 2, py1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # 3. Draw Riders and Helmet Status
        for r_idx, r_box in enumerate(inst.rider_boxes):
            status = report.per_rider_status[r_idx]
            color = (0, 200, 0) if status == "helmet" else (0, 0, 230) if status == "no_helmet" else (0, 200, 230)
            cv2.rectangle(vis, (r_box[0], r_box[1]), (r_box[2], r_box[3]), color, 2)
            cv2.putText(vis, status, (r_box[0], max(r_box[1] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # 4. Draw Overall Violation Label
        v_text = f"Violations: {', '.join(report.violations) if report.violations else 'None'}"
        cv2.putText(vis, v_text, (cx1, cy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print(f"  vehicle[{i}] plate='{report.license_plate}' violations={report.violations}")

    cv2.imwrite(str(OUTPUT_DIR / img_path.name), vis)

print(f"\nDone. Results saved to {OUTPUT_DIR}/")
print("Blue = vehicle crop | Green = plate | RiderColor = helmet status | RedText = violations")

