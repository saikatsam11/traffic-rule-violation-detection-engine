"""
Module 1: Two-Wheeler + Rider Isolation  (v4)
----------------------------------------------
Core structural change vs v3:

  PROBLEM: Pass 1 used (bottom_center_inside_moto OR bottom_half_overlap >= threshold).
           Either condition alone can be satisfied by a non-rider in street-level scenes
           where car occupants appear at the same pixel height as motorcycle riders.
           Parameter tuning cannot fix this — it is a 3D problem in 2D boxes.

  FIX (F10): Replace dual-OR logic with full-body IoP (Intersection over Person area).
             Measures what fraction of the person's ENTIRE bounding box overlaps the
             motorcycle box. True riders score 0.40–0.90. Car-edge occupants score
             0.05–0.15. Works for both top-down and street-level camera angles.

  Pass 1 — full_body_iop >= MIN_FULL_BODY_IOP (primary, replaces old OR logic)
  Pass 2 — fallback: full_body_iop >= MIN_FALLBACK_IOP AND horizontal_overlap >=
             MIN_FALLBACK_H_OVERLAP AND distance <= FALLBACK_DIST_PX (all three required)
           Handles riders whose body extends above the motorcycle box (partial overlap).

All previous fixes F1–F9 and robustness practices R1–R10 retained.
"""

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOTORCYCLE_CLS = 3
PERSON_CLS     = 0


# ── Configuration ─────────────────────────────────────────────────────────────
class Config:
    MOTORCYCLE_CONF : float = 0.25
    PERSON_CONF     : float = 0.40
    NMS_IOU_THRESH  : float = 0.70
    TTA             : bool  = True   # YOLO test-time augmentation

    # F10: full-body IoP thresholds — tune these first before anything else
    # MIN_FULL_BODY_IOP: primary gate. Rider's entire box must overlap moto box
    # by at least this fraction. 0.30 works across camera angles.
    # Raise if you get false positives. Lower if you miss riders on partially
    # visible bikes. Do not go below 0.15 or above 0.55.
    MIN_FULL_BODY_IOP       : float = 0.30

    # Fallback IoP: looser threshold for riders partially outside the moto box
    # (e.g. tall rider, top of body extends above the motorcycle bounding box).
    MIN_FALLBACK_IOP        : float = 0.15

    # Fallback spatial constraints (both required alongside MIN_FALLBACK_IOP)
    MIN_FALLBACK_H_OVERLAP  : float = 0.30   # horizontal overlap fraction
    # Vertical gap = moto_top - person_bottom. 0 ⇒ vertically connected
    # (person extends down into the bike). Positive ⇒ person hovers above
    # the bike entirely. Tolerance allows for 1-2 px detection jitter.
    MAX_VERTICAL_GAP_PX     : int   = 20

    MAX_RIDERS              : int   = 4
    CROP_PADDING            : float = 0.15
    MIN_CROP_SIZE           : int   = 60
    FUSED_BIKE_WIDTH_RATIO  : float = 2.5


# ── Output contract ───────────────────────────────────────────────────────────
@dataclass
class VehicleInstance:
    """
    One motorcycle with ≥1 confirmed rider.

    crop_image / crop_box = padded union of motorcycle + confirmed rider boxes.
    rider_boxes            = rider coordinates in original image space.
    rider_boxes_in_crop    = rider coordinates translated into crop-local space.
    """
    motorcycle_box      : List[int]
    rider_boxes         : List[List[int]]
    num_riders          : int
    crop_box            : List[int]
    crop_image          : object           # np.ndarray (BGR)
    rider_boxes_in_crop : List[List[int]]
    small_crop_flag     : bool
    fused_bike_warning  : bool
    confidence          : float


# ── Core detector ─────────────────────────────────────────────────────────────
class TwoWheelerDetector:

    def __init__(self, model_path: str, config: Config = Config()):
        self.cfg   = config
        self.model = YOLO(model_path)
        logger.info(f"TwoWheelerDetector loaded: {model_path}")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, image_path: str) -> List[VehicleInstance]:
        """Always returns a list. Never raises. (R10)"""
        try:
            image = self._load_image(image_path)
            if image is None:
                return []

            moto_boxes, moto_confs = self._detect_motorcycles(image)
            person_boxes           = self._detect_persons(image)

            if not moto_boxes:
                logger.info("No motorcycles detected.")
                return []

            moto_boxes, moto_confs = self._deduplicate_motorcycles(moto_boxes, moto_confs)
            assignments            = self._associate_riders(moto_boxes, person_boxes)
            instances              = self._build_instances(
                image, moto_boxes, moto_confs, person_boxes, assignments
            )

            logger.info(f"Returning {len(instances)} vehicle instance(s) with riders.")
            return instances

        except Exception as e:
            logger.error(f"TwoWheelerDetector.detect() failed: {e}", exc_info=True)
            return []

    # ── Stage 1: Motorcycle detection ─────────────────────────────────────────
    def _detect_motorcycles(
        self, image: np.ndarray
    ) -> Tuple[List[List[int]], List[float]]:
        results = self.model(
            image,
            conf    = self.cfg.MOTORCYCLE_CONF,
            iou     = self.cfg.NMS_IOU_THRESH,
            classes = [MOTORCYCLE_CLS],
            augment = self.cfg.TTA,
            verbose = False,
        )[0]

        boxes, confs = [], []
        for box in results.boxes:
            if int(box.cls[0]) == MOTORCYCLE_CLS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2, y2])
                confs.append(float(box.conf[0]))

        logger.debug(f"Motorcycle raw detections: {len(boxes)}")
        return boxes, confs

    # ── Stage 2: Person detection ─────────────────────────────────────────────
    def _detect_persons(self, image: np.ndarray) -> List[List[int]]:
        results = self.model(
            image,
            conf    = self.cfg.PERSON_CONF,
            classes = [PERSON_CLS],
            augment = self.cfg.TTA,
            verbose = False,
        )[0]

        boxes = []
        for box in results.boxes:
            if int(box.cls[0]) == PERSON_CLS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2, y2])

        logger.debug(f"Person raw detections: {len(boxes)}")
        return boxes

    # ── Stage 3: Motorcycle deduplication ────────────────────────────────────
    def _deduplicate_motorcycles(
        self,
        boxes: List[List[int]],
        confs: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        if len(boxes) <= 1:
            return boxes, confs

        order    = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        keep     = []
        rejected = set()

        for i in range(len(order)):
            idx = order[i]
            if idx in rejected:
                continue
            keep.append(idx)
            for j in range(i + 1, len(order)):
                jdx = order[j]
                if jdx not in rejected:
                    if self._iou(boxes[idx], boxes[jdx]) > self.cfg.NMS_IOU_THRESH:
                        rejected.add(jdx)

        kept_boxes = [boxes[i] for i in keep]
        kept_confs = [confs[i] for i in keep]
        logger.debug(
            f"After dedup: {len(kept_boxes)} motorcycle(s) "
            f"(removed {len(boxes) - len(kept_boxes)})"
        )
        return kept_boxes, kept_confs

    # ── Stage 4: Rider association (F10) ─────────────────────────────────────
    def _associate_riders(
        self,
        moto_boxes  : List[List[int]],
        person_boxes: List[List[int]],
    ) -> List[List[int]]:
        """
        F10 — full-body IoP replaces the old (containment OR bottom-half overlap) logic.

        Pass 1 (primary):
          full_body_iop >= MIN_FULL_BODY_IOP
          → The person's entire bounding box must substantially overlap the motorcycle.
          → True riders: 0.40–0.90  |  Car occupants / bystanders: 0.00–0.15
          → Works for both top-down and street-level cameras.
          → When multiple motorcycles satisfy the threshold, assign to the highest-IoP one.

        Pass 2 (fallback — partial riders):
          full_body_iop >= MIN_FALLBACK_IOP          (some overlap, not zero)
          AND horizontal_overlap >= MIN_FALLBACK_H_OVERLAP  (horizontally aligned)
          AND vertical_gap <= MAX_VERTICAL_GAP_PX    (vertically connected)
          → Catches riders whose upper body extends above the motorcycle bounding box,
            giving them lower-than-expected full-body IoP.
          → vertical_gap = max(0, moto_top − person_bottom). A real rider always reaches
            down into the bike box, so gap = 0. Bystanders hovering above score positive.
          → Robust to camera framing where centroid distance fails (close-ups, tall riders).
        """
        assignments: List[List[int]] = [[] for _ in moto_boxes]
        assigned = set()

        # Pass 1: full-body IoP
        for p_idx, pbox in enumerate(person_boxes):
            best_moto, best_iop = -1, -1.0

            for m_idx, mbox in enumerate(moto_boxes):
                if len(assignments[m_idx]) >= self.cfg.MAX_RIDERS:
                    continue

                iop = self._iop(pbox, mbox)    # full body intersection over person area

                if iop >= self.cfg.MIN_FULL_BODY_IOP and iop > best_iop:
                    best_iop  = iop
                    best_moto = m_idx

            if best_moto >= 0:
                assignments[best_moto].append(p_idx)
                assigned.add(p_idx)
                logger.debug(
                    f"Person {p_idx} → moto {best_moto} via Pass 1 (iop={best_iop:.2f})"
                )

        # Pass 2: fallback — partial riders (upper body above motorcycle box).
        # Uses vertical-gap (moto_top − person_bottom) instead of centroid distance.
        # For a true rider, person.bottom always reaches into the bike box ⇒ gap = 0.
        # For a bystander above the bike, person.bottom < moto_top ⇒ positive gap.
        # Robust to camera framing (close-ups, tall vehicles) where centroids diverge.
        for p_idx, pbox in enumerate(person_boxes):
            if p_idx in assigned:
                continue

            best_moto, best_gap = -1, float("inf")

            for m_idx, mbox in enumerate(moto_boxes):
                if len(assignments[m_idx]) >= self.cfg.MAX_RIDERS:
                    continue

                iop       = self._iop(pbox, mbox)
                h_overlap = self._horizontal_overlap_ratio(pbox, mbox)
                v_gap     = max(0, mbox[1] - pbox[3])    # moto_top − person_bottom

                if (iop       >= self.cfg.MIN_FALLBACK_IOP
                        and h_overlap >= self.cfg.MIN_FALLBACK_H_OVERLAP
                        and v_gap     <= self.cfg.MAX_VERTICAL_GAP_PX
                        and v_gap     <  best_gap):
                    best_gap  = v_gap
                    best_moto = m_idx

            if best_moto >= 0:
                assignments[best_moto].append(p_idx)
                assigned.add(p_idx)
                logger.debug(
                    f"Person {p_idx} → moto {best_moto} via Pass 2 fallback "
                    f"(v_gap={best_gap}px)"
                )

        return assignments

    # ── Stage 5: Build instances ──────────────────────────────────────────────
    def _build_instances(
        self,
        image       : np.ndarray,
        moto_boxes  : List[List[int]],
        moto_confs  : List[float],
        person_boxes: List[List[int]],
        assignments : List[List[int]],
    ) -> List[VehicleInstance]:
        """
        Crop = padded union of motorcycle + confirmed rider boxes.
        rider_boxes_in_crop = rider coords in crop-local space for downstream use.
        """
        H, W = image.shape[:2]

        all_widths   = [b[2] - b[0] for b in moto_boxes]
        median_width = float(np.median(all_widths)) if all_widths else 200.0

        instances = []

        for m_idx, mbox in enumerate(moto_boxes):
            rider_indices = assignments[m_idx]

            if not rider_indices:
                continue   # parked bike — no riders confirmed

            rider_boxes = [person_boxes[i] for i in rider_indices]
            num_riders  = len(rider_boxes)

            # Union of motorcycle + confirmed rider boxes
            all_boxes = [mbox] + rider_boxes
            ux1 = min(b[0] for b in all_boxes)
            uy1 = min(b[1] for b in all_boxes)
            ux2 = max(b[2] for b in all_boxes)
            uy2 = max(b[3] for b in all_boxes)

            bw    = ux2 - ux1
            bh    = uy2 - uy1
            pad_x = int(bw * self.cfg.CROP_PADDING)
            pad_y = int(bh * self.cfg.CROP_PADDING)

            cx1 = max(0,     ux1 - pad_x)
            cy1 = max(0,     uy1 - pad_y)
            cx2 = min(W - 1, ux2 + pad_x)
            cy2 = min(H - 1, uy2 + pad_y)

            crop_box   = [cx1, cy1, cx2, cy2]
            crop_image = image[cy1:cy2, cx1:cx2].copy()

            rider_boxes_in_crop = [
                [
                    max(0, rb[0] - cx1),
                    max(0, rb[1] - cy1),
                    min(cx2 - cx1, rb[2] - cx1),
                    min(cy2 - cy1, rb[3] - cy1),
                ]
                for rb in rider_boxes
            ]

            crop_w = cx2 - cx1
            crop_h = cy2 - cy1
            small_crop_flag = crop_w < self.cfg.MIN_CROP_SIZE or crop_h < self.cfg.MIN_CROP_SIZE
            if small_crop_flag:
                logger.warning(
                    f"Moto {m_idx}: small crop ({crop_w}×{crop_h}px) — "
                    "downstream accuracy may be reduced."
                )

            box_width          = mbox[2] - mbox[0]
            fused_bike_warning = (
                len(all_widths) > 1
                and box_width > self.cfg.FUSED_BIKE_WIDTH_RATIO * median_width
            )
            if fused_bike_warning:
                logger.warning(
                    f"Moto {m_idx}: box width {box_width}px is "
                    f"{box_width / median_width:.1f}× median — possible fused detection."
                )

            instances.append(VehicleInstance(
                motorcycle_box      = mbox,
                rider_boxes         = rider_boxes,
                num_riders          = num_riders,
                crop_box            = crop_box,
                crop_image          = crop_image,
                rider_boxes_in_crop = rider_boxes_in_crop,
                small_crop_flag     = small_crop_flag,
                fused_bike_warning  = fused_bike_warning,
                confidence          = moto_confs[m_idx],
            ))

        return instances

    # ── Geometry helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _iop(person_box: List[int], moto_box: List[int]) -> float:
        """
        Intersection over Person area (IoP).
        Fraction of the person's bounding box that overlaps the motorcycle box.
        Range: [0.0, 1.0]. Higher = person is more "on" the motorcycle.
        """
        ix1   = max(person_box[0], moto_box[0])
        iy1   = max(person_box[1], moto_box[1])
        ix2   = min(person_box[2], moto_box[2])
        iy2   = min(person_box[3], moto_box[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        p_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
        return inter / p_area if p_area > 0 else 0.0

    @staticmethod
    def _horizontal_overlap_ratio(person_box: List[int], moto_box: List[int]) -> float:
        """Fraction of the person's WIDTH that horizontally overlaps the motorcycle."""
        ix1       = max(person_box[0], moto_box[0])
        ix2       = min(person_box[2], moto_box[2])
        overlap_x = max(0, ix2 - ix1)
        person_w  = person_box[2] - person_box[0]
        return overlap_x / person_w if person_w > 0 else 0.0

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ix1   = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2   = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _load_image(image_path: str) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
        return image


# ── Debug visualiser ──────────────────────────────────────────────────────────
def visualise_instances(
    image_path : str,
    instances  : List[VehicleInstance],
    output_path: str = "debug_m1.jpg",
) -> None:
    """
    Colour key:
      Blue   = motorcycle bounding box
      Green  = confirmed rider bounding boxes
      Red    = crop box (union of moto + confirmed riders + padding)
      Orange = crop box when fused_bike_warning is True
    """
    image = cv2.imread(image_path)
    if image is None:
        return

    for inst in instances:
        mx1, my1, mx2, my2 = inst.motorcycle_box
        cv2.rectangle(image, (mx1, my1), (mx2, my2), (255, 100, 0), 2)

        for rb in inst.rider_boxes:
            cv2.rectangle(image, (rb[0], rb[1]), (rb[2], rb[3]), (0, 200, 0), 2)

        cx1, cy1, cx2, cy2 = inst.crop_box
        colour = (0, 140, 255) if inst.fused_bike_warning else (0, 0, 255)
        cv2.rectangle(image, (cx1, cy1), (cx2, cy2), colour, 2)

        tags  = (["SMALL"]  if inst.small_crop_flag    else [])
        tags += (["FUSED?"] if inst.fused_bike_warning else [])
        label = (
            f"riders={inst.num_riders} conf={inst.confidence:.2f}"
            + (f" [{','.join(tags)}]" if tags else "")
        )
        cv2.putText(
            image, label, (cx1, max(cy1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
        )

    cv2.imwrite(output_path, image)
    logger.info(f"Debug image saved: {output_path}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python two_wheeler_detector.py <model_path> <image_path>")
        sys.exit(1)

    detector  = TwoWheelerDetector(model_path=sys.argv[1])
    instances = detector.detect(sys.argv[2])

    print(f"\nFound {len(instances)} vehicle instance(s) with riders:\n")
    for i, inst in enumerate(instances):
        print(
            f"  [{i}] moto_box={inst.motorcycle_box}  "
            f"riders={inst.num_riders}  conf={inst.confidence:.2f}  "
            f"small={inst.small_crop_flag}  fused={inst.fused_bike_warning}"
        )
        print(
            f"       crop_box={inst.crop_box}  "
            f"crop_size={inst.crop_image.shape[1]}x{inst.crop_image.shape[0]}px"
        )
        for j, (rb, rbc) in enumerate(zip(inst.rider_boxes, inst.rider_boxes_in_crop)):
            print(f"       rider[{j}] orig={rb}  in_crop={rbc}")

    if instances:
        visualise_instances(sys.argv[2], instances, "debug_m1.jpg")
        print("\nDebug → debug_m1.jpg")
        print("Blue=moto  Green=confirmed riders  Red=crop  Orange=fused?")