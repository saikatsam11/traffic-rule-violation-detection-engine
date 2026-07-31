from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ViolationReport:
    vehicle_id          : int           # index into the input lists
    license_plate       : str           # mirrors PlateResult.license_plate
    plate_valid         : bool          # mirrors PlateResult.valid_format
    plate_box           : Optional[List[int]]
    motorcycle_box      : List[int]
    num_riders          : int
    violations          : List[str]     # subset of {"no_helmet", "triple_rider"}
    per_rider_status    : List[str]     # mirrors HelmetResult.per_rider_status
    detection_confidence: float         # vehicle conf
    ocr_confidence      : float         # OCR conf (0 if no plate)
    flags               : List[str]     # e.g. "small_crop", "low_confidence", "fused_bike"

    def to_dict(self):
        """Returns a dictionary representation for the assignment evaluator."""
        return asdict(self)

class ViolationEngine:
    """
    Module 4: ViolationEngine
    Combines the outputs of Vehicle Detection, Helmet Detection, and Plate Reading
    into a final per-vehicle violation report based on predefined rules.
    """

    def __init__(self, triple_rider_threshold: int = 3):
        self.triple_rider_threshold = triple_rider_threshold

    def evaluate(
        self,
        instances,
        helmet_results,
        plate_results
    ) -> List[ViolationReport]:
        """
        Main entry point. Index-aligns the three input lists and evaluates each vehicle.
        """
        reports = []
        for i in range(len(instances)):
            inst = instances[i]
            helmet = helmet_results[i]
            plate = plate_results[i]

            reports.append(self._eval_one(inst, helmet, plate, i))

        return reports

    def _eval_one(self, inst, helmet, plate, idx) -> ViolationReport:
        """Rule code for a single vehicle."""
        violations = []

        # Rule 1: No-helmet violation
        if self._check_helmet(helmet):
            violations.append("no_helmet")

        # Rule 2: Triple-rider violation (overloading)
        if self._check_triple(inst):
            violations.append("triple_rider")

        # Collect flags from upstream modules
        flags = self._merge_flags(inst, helmet)

        return ViolationReport(
            vehicle_id          = idx,
            license_plate       = plate.license_plate,
            plate_valid         = plate.valid_format,
            plate_box           = plate.plate_box,
            motorcycle_box      = inst.motorcycle_box,
            num_riders          = inst.num_riders,
            violations          = violations,
            per_rider_status    = helmet.per_rider_status,
            detection_confidence = inst.confidence,
            ocr_confidence      = plate.ocr_confidence,
            flags               = flags
        )

    def _check_helmet(self, helmet_result) -> bool:
        """True if any rider is missing a helmet or their head was undetectable."""
        # Rule: at least one rider on the bike has status != "helmet"
        return any(status != "helmet" for status in helmet_result.per_rider_status)

    def _check_triple(self, instance) -> bool:
        """True if the number of riders is >= the threshold (usually 3)."""
        return instance.num_riders >= self.triple_rider_threshold

    def _merge_flags(self, inst, helmet) -> List[str]:
        """Collects upstream warnings and flags into a single list."""
        flags = []
        if inst.small_crop_flag:
            flags.append("small_crop")
        if inst.fused_bike_warning:
            flags.append("fused_bike")
        if helmet.low_confidence_flag:
            flags.append("low_confidence_helmet")
        return flags

def report_to_csv(reports: List[ViolationReport], output_path: str):
    """Convenience exporter for the assignment submission."""
    import csv
    if not reports:
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reports[0].to_dict().keys())
        writer.writeheader()
        for r in reports:
            writer.writerow(r.to_dict())
