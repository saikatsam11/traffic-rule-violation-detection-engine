"""
Module 2: Helmet Detection
---------------------------
Accepts a VehicleInstance from Module 1 and returns a HelmetResult containing
the count of riders not wearing helmets.

Behavioral rules locked in spec:
  A - No detection on head region          → count as violation
  B - Confidence threshold                 → HELMET_CONF = 0.40, tunable per class
  C - Conflicting detections on same head  → highest confidence label wins
  D - small_crop_flag=True                 → run normally, set low_confidence_flag=True

Robustness practices: H1–H10 (see checklist in project spec)

Expected model:
  Roboflow Universe pretrained YOLOv8 helmet model.
  Classes must include some form of "helmet" and "no_helmet" labels.
  Class name normalisation (H4) handles all common Roboflow naming variants.
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ultralytics import YOLO

# Import the output contract from Module 1
# When integrating, replace this with your actual import path
try:
    from two_wheeler_detector import VehicleInstance
except ImportError:
    # Fallback stub so this module can be tested in isolation
    from dataclasses import dataclass as _dc
    @_dc
    class VehicleInstance:
        motorcycle_box      : list
        rider_boxes         : list
        num_riders          : int
        crop_box            : list
        crop_image          : object
        rider_boxes_in_crop : list
        small_crop_flag     : bool
        fused_bike_warning  : bool
        confidence          : float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Class name normalisation map (H4) ─────────────────────────────────────────
# Roboflow helmet models use wildly inconsistent class names.
# All variants are mapped to canonical "helmet" or "no_helmet".
# Add new variants here if you switch models.
HELMET_CLASS_MAP = {
    # helmet variants
    "helmet"        : "helmet",
    "Helmet"        : "helmet",
    "with_helmet"   : "helmet",
    "with helmet"   : "helmet",
    "With Helmet"   : "helmet",
    "WithHelmet"    : "helmet",
    "helmet_on"     : "helmet",
    # no_helmet variants
    "no_helmet"     : "no_helmet",
    "no helmet"     : "no_helmet",
    "No Helmet"     : "no_helmet",
    "NoHelmet"      : "no_helmet",
    "without_helmet": "no_helmet",
    "without helmet": "no_helmet",
    "Without Helmet": "no_helmet",
    "head"          : "no_helmet",  # models that detect bare heads
    "no-helmet"     : "no_helmet",
}


# ── Configuration ─────────────────────────────────────────────────────────────
class Config:
    # H1: fraction of rider box height used as head region (top portion)
    HEAD_REGION_FRACTION : float = 0.32   # top 45% of rider box

    # B: confidence thresholds — independently tunable per class
    HELMET_CONF    : float = 0.65
    NO_HELMET_CONF : float = 0.40

    # H3: minimum head region dimension (px) — below this → undetectable → violation
    MIN_HEAD_SIZE  : int   = 20

    # H5: IoU threshold for NMS within a single head region
    HEAD_NMS_IOU   : float = 0.45

    # Padding added around head region before cropping (fraction of head height)
    HEAD_PADDING   : float = 0.10


# ── Output contract ───────────────────────────────────────────────────────────
@dataclass
class HelmetResult:
    """
    Result of helmet detection for one VehicleInstance.

    helmet_violations : int        — riders confirmed without a helmet
                                     includes riders whose head was undetectable (rule A)
    total_riders      : int        — mirrors VehicleInstance.num_riders
    per_rider_status  : List[str]  — "helmet" | "no_helmet" | "undetectable" per rider
                                     index-aligned with VehicleInstance.rider_boxes
    low_confidence_flag : bool     — True when input had small_crop_flag=True (rule D)
    """
    helmet_violations   : int
    total_riders        : int
    per_rider_status    : List[str]
    low_confidence_flag : bool


# ── Core detector ─────────────────────────────────────────────────────────────
class HelmetDetector:
    """
    Detects helmet presence/absence for each rider in a VehicleInstance.
    Operates on the head region (top fraction) of each rider bounding box.
    """

    def __init__(self, model_path: str, config: Config = Config()):
        self.cfg    = config
        self.model  = YOLO(model_path)
        self.names  = self.model.names   # {class_id: raw_class_name}
        self._validate_classes()
        logger.info(f"HelmetDetector loaded: {model_path}")
        logger.info(f"Model classes: {self.names}")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, instance: VehicleInstance) -> HelmetResult:
        """
        Analyse one VehicleInstance and return a HelmetResult.
        Always returns a result — never raises. (H9)
        """
        try:
            per_rider_status: List[str] = []

            for r_idx, rider_box_in_crop in enumerate(instance.rider_boxes_in_crop):
                try:
                    status = self._classify_rider(
                        instance.crop_image, rider_box_in_crop, r_idx
                    )
                except Exception as e:
                    # H8: per-rider fault isolation — one bad crop never kills the vehicle
                    logger.error(
                        f"Rider {r_idx} classification failed: {e} — "
                        "defaulting to undetectable (violation)."
                    )
                    status = "undetectable"

                per_rider_status.append(status)

            # Rule A: "undetectable" counts as violation
            helmet_violations = sum(
                1 for s in per_rider_status if s != "helmet"
            )

            return HelmetResult(
                helmet_violations   = helmet_violations,
                total_riders        = instance.num_riders,
                per_rider_status    = per_rider_status,
                low_confidence_flag = instance.small_crop_flag,  # D
            )

        except Exception as e:   # H9: module-level guard
            logger.error(f"HelmetDetector.detect() failed: {e}", exc_info=True)
            # Safe fallback: treat all riders as violations
            return HelmetResult(
                helmet_violations   = instance.num_riders,
                total_riders        = instance.num_riders,
                per_rider_status    = ["undetectable"] * instance.num_riders,
                low_confidence_flag = instance.small_crop_flag,
            )

    # ── Per-rider classification ───────────────────────────────────────────────
    def _classify_rider(
        self,
        crop_image     : np.ndarray,
        rider_box_crop : List[int],
        rider_idx      : int,
    ) -> str:
        """
        Extract the head region from the rider's bounding box (crop-local coords),
        run the helmet model, and return "helmet", "no_helmet", or "undetectable".
        """
        head_crop, head_valid = self._extract_head_region(
            crop_image, rider_box_crop
        )

        # H3: head region too small → undetectable → violation (rule A)
        if not head_valid or head_crop is None:
            logger.debug(
                f"Rider {rider_idx}: head region below minimum size — undetectable."
            )
            return "undetectable"

        # Run helmet model on head crop
        results = self.model(
            head_crop,
            conf    = min(self.cfg.HELMET_CONF, self.cfg.NO_HELMET_CONF),
            iou     = self.cfg.HEAD_NMS_IOU,   # H5: NMS within head region
            verbose = False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            # No detection at all → undetectable → violation (rule A)
            logger.debug(f"Rider {rider_idx}: no detection — undetectable.")
            return "undetectable"

        # Parse all detections into (canonical_label, confidence) pairs
        detections = self._parse_detections(results)

        if not detections:
            logger.debug(f"Rider {rider_idx}: no recognised class — undetectable.")
            return "undetectable"

        # C: highest confidence label wins (after per-class threshold filtering)
        winner_label, winner_conf = self._resolve_label(detections)

        logger.debug(
            f"Rider {rider_idx}: {winner_label} (conf={winner_conf:.2f})"
        )
        return winner_label

    # ── Head region extraction (H1, H2, H3) ──────────────────────────────────
    def _extract_head_region(
        self,
        crop_image    : np.ndarray,
        rider_box_crop: List[int],
    ) -> Tuple[Optional[np.ndarray], bool]:
        """
        H1: take the top HEAD_REGION_FRACTION of the rider bounding box.
        H2: clamp to crop image boundaries.
        H3: return (None, False) if result is smaller than MIN_HEAD_SIZE.

        Returns (head_crop_image, is_valid).
        """
        H, W = crop_image.shape[:2]
        rx1, ry1, rx2, ry2 = rider_box_crop

        # H1: head region = top fraction of rider height
        rider_h    = ry2 - ry1
        head_h     = int(rider_h * self.cfg.HEAD_REGION_FRACTION)
        head_y2    = ry1 + head_h

        # Optional padding around head region
        pad = int(head_h * self.cfg.HEAD_PADDING)
        hx1 = rx1 - pad
        hy1 = ry1 - pad
        hx2 = rx2 + pad
        hy2 = head_y2 + pad

        # H2: clamp to crop image boundaries
        hx1 = max(0, hx1)
        hy1 = max(0, hy1)
        hx2 = min(W - 1, hx2)
        hy2 = min(H - 1, hy2)

        actual_w = hx2 - hx1
        actual_h = hy2 - hy1

        # H3: minimum size check
        if actual_w < self.cfg.MIN_HEAD_SIZE or actual_h < self.cfg.MIN_HEAD_SIZE:
            logger.debug(
                f"Head region {actual_w}×{actual_h}px below "
                f"minimum ({self.cfg.MIN_HEAD_SIZE}px)."
            )
            return None, False

        head_crop = crop_image[hy1:hy2, hx1:hx2].copy()
        return head_crop, True

    # ── Detection parsing (H4) ────────────────────────────────────────────────
    def _parse_detections(
        self, results
    ) -> List[Tuple[str, float]]:
        """
        H4: normalise raw class names from the model to "helmet" or "no_helmet".
        Filters out detections below the per-class confidence threshold.
        Returns list of (canonical_label, confidence).
        """
        parsed = []
        for box in results.boxes:
            raw_cls  = self.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            label    = HELMET_CLASS_MAP.get(raw_cls)

            if label is None:
                logger.warning(
                    f"Unknown helmet model class: '{raw_cls}' — skipping. "
                    "Add it to HELMET_CLASS_MAP if valid."
                )
                continue

            # Per-class confidence threshold
            threshold = (
                self.cfg.HELMET_CONF if label == "helmet"
                else self.cfg.NO_HELMET_CONF
            )
            if conf >= threshold:
                parsed.append((label, conf))

        return parsed

    # ── Label resolution (C, H5, H6) ─────────────────────────────────────────
    def _resolve_label(
        self, detections: List[Tuple[str, float]]
    ) -> Tuple[str, float]:
        """
        C: among all valid detections on this head region, the highest
           confidence label wins.

        If multiple boxes share the same top confidence (rare), "no_helmet"
        is preferred as a safety-first tiebreaker.
        """
        # Sort by confidence descending; "no_helmet" wins on equal confidence
        detections_sorted = sorted(
            detections,
            key=lambda x: (x[1], x[0] == "no_helmet"),
            reverse=True,
        )
        return detections_sorted[0]   # (label, confidence)

    # ── Class validation ──────────────────────────────────────────────────────
    def _validate_classes(self) -> None:
        """
        Warn at init time if the loaded model has no classes that match
        the known helmet/no_helmet aliases. This catches model mismatches
        before any images are processed.
        """
        recognised = [
            name for name in self.names.values()
            if name in HELMET_CLASS_MAP
        ]
        if not recognised:
            logger.warning(
                "HelmetDetector: none of the model's classes match HELMET_CLASS_MAP. "
                f"Model classes: {list(self.names.values())}. "
                "All riders will be classified as undetectable. "
                "Update HELMET_CLASS_MAP or verify you loaded the correct model."
            )


# ── Batch API ─────────────────────────────────────────────────────────────────
def detect_batch(
    detector  : HelmetDetector,
    instances : List[VehicleInstance],
) -> List[HelmetResult]:
    """
    Convenience wrapper — runs HelmetDetector on a full list of VehicleInstances.
    Returns results index-aligned with the input list.
    """
    return [detector.detect(inst) for inst in instances]


# ── Debug visualiser ──────────────────────────────────────────────────────────
def visualise_helmet_results(
    instances   : List[VehicleInstance],
    results     : List[HelmetResult],
    source_image: np.ndarray,
    output_path : str = "debug_m2.jpg",
) -> None:
    """
    Draws helmet status on the source image using original-image rider coordinates.

    Colour key:
      Green  = helmet confirmed
      Red    = no_helmet confirmed
      Yellow = undetectable (counted as violation)
      Orange border on crop box = low_confidence_flag (small crop)
    """
    vis = source_image.copy()

    for inst, result in zip(instances, results):
        cx1, cy1, cx2, cy2 = inst.crop_box
        crop_colour = (0, 140, 255) if result.low_confidence_flag else (0, 0, 255)
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), crop_colour, 2)

        for r_idx, (rb, status) in enumerate(
            zip(inst.rider_boxes, result.per_rider_status)
        ):
            if status == "helmet":
                colour = (0, 200, 0)       # green
            elif status == "no_helmet":
                colour = (0, 0, 230)       # red
            else:
                colour = (0, 200, 230)     # yellow — undetectable

            cv2.rectangle(vis, (rb[0], rb[1]), (rb[2], rb[3]), colour, 2)
            cv2.putText(
                vis, status,
                (rb[0], max(rb[1] - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2,
            )

        label = (
            f"violations={result.helmet_violations}/{result.total_riders}"
            + (" [LOW-CONF]" if result.low_confidence_flag else "")
        )
        cv2.putText(
            vis, label, (cx1, cy2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, crop_colour, 2,
        )

    cv2.imwrite(output_path, vis)
    logger.info(f"Debug visualisation saved: {output_path}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python helmet_detector.py "
            "<helmet_model_path> <detector_model_path> <image_path>"
        )
        sys.exit(1)

    helmet_model_path   = sys.argv[1]
    detector_model_path = sys.argv[2]
    image_path          = sys.argv[3]

    # Run Module 1 first to get VehicleInstances
    from two_wheeler_detector import TwoWheelerDetector, visualise_instances

    m1        = TwoWheelerDetector(model_path=detector_model_path)
    instances = m1.detect(image_path)

    print(f"\nModule 1: {len(instances)} vehicle instance(s) detected.")

    if not instances:
        print("No vehicles detected — nothing to pass to Module 2.")
        sys.exit(0)

    # Run Module 2
    m2      = HelmetDetector(model_path=helmet_model_path)
    results = detect_batch(m2, instances)

    print(f"\nModule 2 results:\n")
    for i, (inst, res) in enumerate(zip(instances, results)):
        print(
            f"  [{i}] riders={res.total_riders}  "
            f"helmet_violations={res.helmet_violations}  "
            f"low_conf={res.low_confidence_flag}"
        )
        for j, status in enumerate(res.per_rider_status):
            print(f"       rider[{j}] → {status}")

    # Visualise on original image
    source = cv2.imread(image_path)
    if source is not None:
        visualise_helmet_results(instances, results, source, "debug_m2.jpg")
        print("\nDebug → debug_m2.jpg")
        print("Green=helmet  Red=no_helmet  Yellow=undetectable  Orange-border=low-conf")