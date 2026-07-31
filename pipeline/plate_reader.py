"""
Module 3: License Plate Reader (FastPlateOCR + multi-variant ensemble)
----------------------------------------------------------------------
Stage 1: YOLOv8s plate detector (TTA) on VehicleInstance.crop_image
Stage 2: Plate deskew via 4-corner contour detection (perspective warp)
Stage 3: Build OCR variants
            single-row: default | tight | wide | CLAHE | unsharp
            2-row     : top-half + bottom-half concatenated  (when aspect ratio < 2.5)
Stage 4: FastPlateOCR run on all variants, aggregated by mean char probability
Stage 5: Position-aware character-confusion correction (rescue digit-only / letter-only)
Stage 6: Confidence floor → "UNREAD" if winning conf below threshold

Variants targeted at distinct failure modes:
    deskew  : rotated/skewed plates (most motorcycle plates)
    tight   : extra padding from loose detector boxes
    wide    : tight detector boxes that clip plate edges
    CLAHE   : low-contrast / shaded plates
    unsharp : blurry plates (motion / defocus)
    2-row   : Indian-style stacked plates the global model can't read straight
"""

import logging
import re
import cv2
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

try:
    from two_wheeler_detector import VehicleInstance
except ImportError:
    from pipeline.two_wheeler_detector import VehicleInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────
class Config:
    PLATE_CONF        : float = 0.15
    PLATE_NMS_IOU     : float = 0.45
    PLATE_TTA         : bool  = True
    PLATE_MIN_WIDTH   : int   = 30
    PLATE_MIN_HEIGHT  : int   = 10

    PAD_DEFAULT       : float = 0.08
    PAD_TIGHT         : float = 0.0
    PAD_WIDE          : float = 0.20

    OCR_TIE_EPSILON   : float = 0.02
    OCR_UNREAD_FLOOR  : float = 0.45         # below this, return "UNREAD"
    TWO_ROW_RATIO     : float = 2.5          # w/h below this triggers split-row OCR


# Confusion maps for character-class correction.
# Used only as last-resort rescue when validation fails.
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "7": "Z"}
LETTER_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}


# ── Output contract ───────────────────────────────────────────────────────────
@dataclass
class PlateResult:
    license_plate : str
    plate_box     : Optional[List[int]]
    confidence    : float          # plate detection confidence
    ocr_confidence: float          # mean char probability of winning variant
    ocr_raw       : str
    valid_format  : bool
    variant_used  : str            # which preprocessing variant won


# ── Core reader ───────────────────────────────────────────────────────────────
class PlateReader:

    def __init__(
        self,
        model_path        : str,
        ocr_onnx_path     : Optional[str] = None,
        ocr_config_path   : Optional[str] = None,
        ocr_hub_model     : str           = "global-plates-mobile-vit-v2-model",
        config            : Config        = Config(),
    ):
        self.cfg   = config
        self.model = YOLO(model_path)

        if ocr_onnx_path and ocr_config_path:
            self.ocr = LicensePlateRecognizer(
                onnx_model_path   = ocr_onnx_path,
                plate_config_path = ocr_config_path,
                device            = "cpu",
            )
            logger.info(f"FastPlateOCR loaded from local: {ocr_onnx_path}")
        else:
            self.ocr = LicensePlateRecognizer(ocr_hub_model, device="cpu")
            logger.info(f"FastPlateOCR loaded from hub: {ocr_hub_model}")

        logger.info(f"PlateReader ready (plate model: {model_path})")

    # ── Public API ────────────────────────────────────────────────────────────
    def read(self, instance: VehicleInstance) -> PlateResult:
        try:
            best = self._detect_plate(instance)
            if best is None:
                return self._empty_result()

            plate_box, det_conf = best
            text, ocr_conf, raw, variant = self._read_ensemble(
                instance.crop_image, plate_box
            )

            # Position-aware confusion correction (rescue if validation fails)
            clean, valid = self._clean_and_validate(text)
            if not valid and clean:
                fixed = self._fix_confusables(clean)
                fclean, fvalid = self._clean_and_validate(fixed)
                if fvalid:
                    clean, valid = fclean, fvalid
                    variant = variant + "+fix"

            # Confidence-gated UNREAD floor
            if ocr_conf < self.cfg.OCR_UNREAD_FLOOR or not clean:
                final_text = "UNREAD"
                valid = False
            else:
                final_text = clean

            cx1, cy1 = instance.crop_box[0], instance.crop_box[1]
            orig_box = [
                cx1 + plate_box[0], cy1 + plate_box[1],
                cx1 + plate_box[2], cy1 + plate_box[3],
            ]

            logger.info(
                f"Plate: '{final_text}'  valid={valid}  det={det_conf:.2f} "
                f"ocr={ocr_conf:.2f}  variant={variant}  raw='{raw}'"
            )
            return PlateResult(
                license_plate  = final_text,
                plate_box      = orig_box,
                confidence     = det_conf,
                ocr_confidence = ocr_conf,
                ocr_raw        = raw,
                valid_format   = valid,
                variant_used   = variant,
            )

        except Exception as e:
            logger.error(f"PlateReader.read() failed: {e}", exc_info=True)
            return self._empty_result()

    # ── Stage 1: Plate detection (with TTA) ──────────────────────────────────
    def _detect_plate(
        self, instance: VehicleInstance
    ) -> Optional[Tuple[List[int], float]]:
        results = self.model(
            instance.crop_image,
            conf    = self.cfg.PLATE_CONF,
            iou     = self.cfg.PLATE_NMS_IOU,
            augment = self.cfg.PLATE_TTA,
            verbose = False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        H, W = instance.crop_image.shape[:2]
        best_box, best_conf = None, -1.0

        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if (x2 - x1) < self.cfg.PLATE_MIN_WIDTH or (y2 - y1) < self.cfg.PLATE_MIN_HEIGHT:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box  = [x1, y1, x2, y2]

        if best_box is None:
            return None
        return best_box, best_conf

    # ── Stage 2: Deskew via 4-corner contour ─────────────────────────────────
    def _deskew(self, plate_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Try to find the largest 4-corner quadrilateral inside the crop and
        warp it to a clean rectangle. Returns warped image or None on failure.
        """
        h, w = plate_bgr.shape[:2]
        if h < 15 or w < 30:
            return None

        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thr  = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5,
        )
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        crop_area = h * w
        best_quad = None
        best_area = 0

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            area = cv2.contourArea(c)
            if area < 0.20 * crop_area:        # quad must cover decent fraction
                continue
            peri  = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4 and area > best_area:
                best_quad = approx.reshape(4, 2)
                best_area = area

        if best_quad is None:
            return None

        # Order corners: tl, tr, br, bl
        pts = best_quad.astype(np.float32)
        s   = pts.sum(axis=1)
        d   = np.diff(pts, axis=1).reshape(-1)
        tl  = pts[np.argmin(s)]
        br  = pts[np.argmax(s)]
        tr  = pts[np.argmin(d)]
        bl  = pts[np.argmax(d)]
        ordered = np.array([tl, tr, br, bl], dtype=np.float32)

        wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
        out_w, out_h = int(max(wA, wB)), int(max(hA, hB))
        if out_w < 30 or out_h < 15:
            return None

        dst = np.array([[0,0], [out_w-1,0], [out_w-1,out_h-1], [0,out_h-1]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(plate_bgr, M, (out_w, out_h))

    # ── Preprocessing helpers ────────────────────────────────────────────────
    def _crop_with_pad(
        self, image: np.ndarray, box: List[int], pad_frac: float
    ) -> np.ndarray:
        H, W = image.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * pad_frac); py = int(bh * pad_frac)
        cx1 = max(0, x1 - px);  cy1 = max(0, y1 - py)
        cx2 = min(W, x2 + px);  cy2 = min(H, y2 + py)
        return image[cy1:cy2, cx1:cx2].copy()

    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _clahe(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4))
        return clahe.apply(gray)

    @staticmethod
    def _unsharp(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # ── Stage 3 + 4: Build variants & run ensemble ───────────────────────────
    def _build_variants(
        self, crop_image: np.ndarray, plate_box: List[int]
    ) -> List[Tuple[str, np.ndarray]]:
        v_default = self._crop_with_pad(crop_image, plate_box, self.cfg.PAD_DEFAULT)
        v_tight   = self._crop_with_pad(crop_image, plate_box, self.cfg.PAD_TIGHT)
        v_wide    = self._crop_with_pad(crop_image, plate_box, self.cfg.PAD_WIDE)

        g_default = self._to_gray(v_default)
        variants  = [
            ("default", g_default),
            ("tight",   self._to_gray(v_tight)),
            ("wide",    self._to_gray(v_wide)),
            ("clahe",   self._clahe(g_default)),
            ("sharp",   self._unsharp(g_default)),
        ]

        # Deskew variant — use rectified plate if 4-corner contour found
        warped = self._deskew(v_default)
        if warped is not None:
            variants.append(("deskew", self._to_gray(warped)))

        # 2-row variant — split horizontally for stacked plates
        h, w = g_default.shape[:2]
        ratio = w / h if h > 0 else 99.0
        if ratio < self.cfg.TWO_ROW_RATIO:
            mid = h // 2
            top, bot = g_default[:mid], g_default[mid:]
            variants.append(("row_top",    top))
            variants.append(("row_bottom", bot))

        return variants

    def _read_ensemble(
        self, crop_image: np.ndarray, plate_box: List[int]
    ) -> Tuple[str, float, str, str]:
        """
        Run FastPlateOCR on all variants. Single-row variants are aggregated
        by confidence with tie-break voting. 2-row variants (row_top + row_bottom)
        are concatenated and treated as one combined prediction competing with
        the single-row leader.
        """
        variants = self._build_variants(crop_image, plate_box)
        gray_imgs = [g for _, g in variants]
        preds = self.ocr.run(gray_imgs, return_confidence=True)

        scored = []
        for (name, _), p in zip(variants, preds):
            text = p.plate
            conf = float(np.mean(p.char_probs)) if p.char_probs is not None else 0.0
            scored.append((name, text, conf))

        # Combine 2-row halves into one candidate "two_row"
        two_row = self._combine_two_row(scored)
        if two_row is not None:
            scored.append(two_row)

        # Drop the per-half rows from competition (they're partial by definition)
        scored = [s for s in scored if s[0] not in ("row_top", "row_bottom")]
        scored.sort(key=lambda s: s[2], reverse=True)
        if not scored:
            return "", 0.0, "", "none"

        leader_name, leader_text, leader_conf = scored[0]

        # Tie-break: per-position majority vote across variants within ε of leader
        tied = [s for s in scored if leader_conf - s[2] <= self.cfg.OCR_TIE_EPSILON]
        if len(tied) <= 1:
            return leader_text, leader_conf, leader_text, leader_name

        merged   = self._position_vote([s[1] for s in tied])
        avg_conf = float(np.mean([s[2] for s in tied]))
        return merged, avg_conf, leader_text, "vote(" + ",".join(s[0] for s in tied) + ")"

    @staticmethod
    def _combine_two_row(
        scored: List[Tuple[str, str, float]],
    ) -> Optional[Tuple[str, str, float]]:
        top = next((s for s in scored if s[0] == "row_top"),    None)
        bot = next((s for s in scored if s[0] == "row_bottom"), None)
        if top is None or bot is None:
            return None
        combined_text = (top[1] or "") + (bot[1] or "")
        combined_conf = (top[2] + bot[2]) / 2
        return ("two_row", combined_text, combined_conf)

    @staticmethod
    def _position_vote(texts: List[str]) -> str:
        clean = [re.sub(r"[^A-Z0-9]", "", t.upper()) for t in texts if t]
        if not clean:
            return ""
        L = max(len(t) for t in clean)
        out = []
        for i in range(L):
            chars = [t[i] for t in clean if i < len(t)]
            if chars:
                out.append(Counter(chars).most_common(1)[0][0])
        return "".join(out)

    # ── Stage 5: Character-confusion rescue ──────────────────────────────────
    def _fix_confusables(self, clean: str) -> str:
        """
        If the text is all digits or all letters, swap the most likely
        confusables at typical positions to push it into a valid format.
          - all digits → first 2 chars become letters via DIGIT_TO_LETTER
          - all letters → middle/end chars become digits via LETTER_TO_DIGIT
        """
        if not clean:
            return clean

        has_letter = any(c.isalpha() for c in clean)
        has_digit  = any(c.isdigit() for c in clean)

        chars = list(clean)
        if has_digit and not has_letter:
            for i in range(min(2, len(chars))):
                chars[i] = DIGIT_TO_LETTER.get(chars[i], chars[i])
        elif has_letter and not has_digit:
            for i in range(2, len(chars)):
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

        return "".join(chars)

    # ── Text cleaning + validation ────────────────────────────────────────────
    def _clean_and_validate(self, raw: str) -> Tuple[str, bool]:
        clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
        has_letters = bool(re.search(r"[A-Z]", clean))
        has_digits  = bool(re.search(r"[0-9]", clean))
        valid       = (4 <= len(clean) <= 12) and has_letters and has_digits
        return clean, valid

    @staticmethod
    def _empty_result() -> PlateResult:
        return PlateResult(
            license_plate  = "UNDETECTED",
            plate_box      = None,
            confidence     = 0.0,
            ocr_confidence = 0.0,
            ocr_raw        = "",
            valid_format   = False,
            variant_used   = "",
        )


def read_batch(reader: PlateReader, instances: List[VehicleInstance]) -> List[PlateResult]:
    return [reader.read(inst) for inst in instances]
