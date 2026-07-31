# `pipeline/violation_engine.py`

**Module 4** — combines the upstream module outputs into a final per-vehicle violation report.

> **Status:** placeholder file. The contract is sketched here; implementation is the next task.

---

## What it should do

Takes the list of `VehicleInstance`, `HelmetResult`, and `PlateResult` triples and produces one `ViolationReport` per vehicle that the assignment evaluator can score.

Two violation types must be flagged per the assignment spec:

1. **No-helmet violation** — at least one rider on the bike has `status != "helmet"`.
2. **Triple-rider violation** (overloading) — `num_riders ≥ 3` on a two-wheeler.

Both can fire on the same vehicle. The licence plate is reported either way for citation purposes.

## Models used

**None.** This module is pure rule-based aggregation — no ML, no inference. That's deliberate: the heavy lifting (and failure modes) live upstream. Keeping this layer simple means it stays auditable and the violation logic stays testable without GPUs.

## How it will be used

```python
from pipeline.two_wheeler_detector import TwoWheelerDetector
from pipeline.helmet_detector       import HelmetDetector, detect_batch
from pipeline.plate_reader          import PlateReader, read_batch
from pipeline.violation_engine      import ViolationEngine

# load all 3 models once
m1 = TwoWheelerDetector("models/yolov8m.pt")
m2 = HelmetDetector("models/yolov8m_helmet.pt")
m3 = PlateReader(
    model_path="models/yolov8s_plates_final.pt",
    ocr_onnx_path="models/global_mobile_vit_v2_ocr.onnx",
    ocr_config_path="models/global_mobile_vit_v2_ocr_config.yaml",
)

engine = ViolationEngine()

instances     = m1.detect("test_images/01.jpg")
helmet_res    = detect_batch(m2, instances)
plate_res     = read_batch(m3, instances)
reports       = engine.evaluate(instances, helmet_res, plate_res)

for r in reports:
    print(r.vehicle_id, r.license_plate, r.violations, r.confidence_summary)
```

## Architecture

```
[VehicleInstance,  HelmetResult,  PlateResult]   (one per detected vehicle)
                       │
                       ▼
        ViolationEngine.evaluate(...)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
  helmet rule                   triple-rider rule
  (any status != "helmet"        (num_riders ≥ 3)
   in per_rider_status)
       │                               │
       └────────────► merge ◄──────────┘
                       │
                       ▼
            ViolationReport
            (license_plate, violations, evidence)
```

## Suggested data structures (when implementing)

```python
@dataclass
class ViolationReport:
    vehicle_id          : int           # index into the input lists
    license_plate       : str           # mirrors PlateResult.license_plate
    plate_valid         : bool          # mirrors PlateResult.valid_format
    plate_box           : list | None
    motorcycle_box      : list
    num_riders          : int
    violations          : list[str]     # subset of {"no_helmet", "triple_rider"}
    per_rider_status    : list[str]     # mirrors HelmetResult
    detection_confidence: float         # vehicle conf
    ocr_confidence      : float         # OCR conf (0 if no plate)
    flags               : list[str]     # e.g. "small_crop", "low_confidence", "fused_bike"
```

Why a list of violation strings (not separate booleans): future violations like wrong-way driving, signal-jumping, or no-licence detection slot in cleanly without breaking consumers.

## Suggestions when you implement (simple first)

1. **Be explicit about "no plate"** — if `plate_result.license_plate in ("UNDETECTED", "UNREAD")` you should still emit the violation, with a flag noting unidentified plate. The traffic enforcement use case is exactly to capture violators *without* a readable plate.
2. **Keep the rule logic in one method** with one return path. Don't sprinkle `if violation_x` checks across the pipeline. One place to read = one place to fix.
3. **Echo upstream flags** (`small_crop_flag`, `low_confidence_flag`, `fused_bike_warning`) into `ViolationReport.flags`. Don't re-derive them — that creates two sources of truth.
4. **Add a `confidence_summary` string** like `"vehicle:0.92  helmet:OK  ocr:0.81"` — makes it easy to eyeball borderline cases when reviewing output.
5. **Write a tiny unit test for the rules** — the rules are pure functions of dataclasses. No camera, no model. Trivial to unit-test and worth doing because once this is wrong you'd never notice.
6. **Add a `to_csv_row()` method** to `ViolationReport` so the engine output can be dumped straight to a `.csv` for the evaluator.

## Functions / classes (planned)

### `ViolationReport` (dataclass)
See data structure above.

### `ViolationEngine`

| Method | Purpose |
|---|---|
| `__init__(config?)` | optional thresholds (e.g. min riders for "triple") |
| `evaluate(instances, helmet_results, plate_results) → list[ViolationReport]` | main entry, index-aligns the three input lists |
| `_eval_one(inst, helmet, plate, idx) → ViolationReport` | the rule code for a single vehicle |
| `_check_helmet(helmet_result) → bool` | True if any rider missing helmet |
| `_check_triple(instance) → bool` | `num_riders ≥ 3` |
| `_merge_flags(...)` | collect upstream flags into one list |

### `report_to_csv(reports, output_path)`
Convenience exporter for the assignment submission.

### `__main__` block
Should run all 3 modules + the engine on `test_images/`, write a CSV summary, and save annotated debug images to `test_results/`.
