from src.detector import Detector
from utils import load_config, load_zone

if __name__ == "__main__":
    cfg = load_config()
    zone = load_zone(cfg)

    detector = Detector(
        source=cfg["source"],
        model_path=cfg["model_path"],
        zone=zone,
        **cfg["detector"]
    )

    detector.run()