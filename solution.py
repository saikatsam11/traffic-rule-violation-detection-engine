from pipeline.two_wheeler_detector import TwoWheelerDetector
from pipeline.helmet_detector       import HelmetDetector, detect_batch
from pipeline.plate_reader          import PlateReader, read_batch
from pipeline.violation_engine      import ViolationEngine

class TrafficViolationDetector:
    def __init__(self, model_dir="./models"):
        self.m1 = TwoWheelerDetector(f"{model_dir}/yolov8m.pt")
        self.m2 = HelmetDetector(f"{model_dir}/yolov8m_helmet.pt")
        self.m3 = PlateReader(
            model_path      = f"{model_dir}/yolov8s_plates_final.pt",
            ocr_onnx_path   = f"{model_dir}/global_mobile_vit_v2_ocr.onnx",
            ocr_config_path = f"{model_dir}/global_mobile_vit_v2_ocr_config.yaml",
        )
        self.engine = ViolationEngine()

    def predict(self, image_path: str) -> dict:
        instances = self.m1.detect(image_path)
        helmets   = detect_batch(self.m2, instances)
        plates    = read_batch(self.m3, instances)
        reports   = self.engine.evaluate(instances, helmets, plates)
        return {
            "violations": [
                {
                    "num_riders":        r.num_riders,
                    "helmet_violations": sum(1 for s in r.per_rider_status if s != "helmet"),
                    "license_plate":     r.license_plate,
                }
                for r in reports
            ]
        }