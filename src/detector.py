class Detector:
    """YOLO-based person detector."""

    def __init__(self, model_path: str, confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

    def load(self):
        raise NotImplementedError

    def detect(self, frame):
        """Returns list of bounding boxes."""
        raise NotImplementedError
