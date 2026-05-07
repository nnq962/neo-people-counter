from src.detector import Detector

MODEL_PATH = "models/person/yolo26m.pt"
SOURCE = "data/test1.mp4"

if __name__ == "__main__":
    detector = Detector(
        source=SOURCE,
        model_path=MODEL_PATH,
        conf=0.65,
        imgsz=640,
        device="cuda",
        show=True
    )

    detector.run()