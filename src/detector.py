from ultralytics import YOLO
from pathlib import Path
import time
import cv2
from utils import LOGGER

class Detector:
    """YOLO-based object detector (supports person, head)."""

    # Detect mode from model path
    SUPPORTED_MODES = [
        "person", 
        "head"  
    ]
    
    # Class mapping for each mode
    CLASS_MAP = {
        "person": [0], 
        "head": [0]     
    }

    def __init__(
        self,
        source: str,
        model_path: str,
        conf: float = 0.5,
        imgsz: int = 640,
        device: str = "cuda",
        half: bool = True,
        show: bool = False
    ):

        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.show = show

        # Log info
        LOGGER.info(f"Source: {self.source}")
        LOGGER.info(f"Model path: {self.model_path}")
        LOGGER.info(f"conf: {self.conf}")
        LOGGER.info(f"Image size: {self.imgsz}")
        LOGGER.info(f"Device: {self.device}")
        LOGGER.info(f"Half: {self.half}")
        LOGGER.info(f"Show: {self.show}")

        # Detect mode from model path
        path_parts = Path(self.model_path).parts
        self.detect_mode = "unknown"
        for mode in self.SUPPORTED_MODES:
            if mode in path_parts:
                self.detect_mode = mode
                break
        
        if self.detect_mode == "unknown":
            LOGGER.error(f"Could not detect mode from supported modes: {self.SUPPORTED_MODES}")
            raise ValueError(f"Invalid model path: {self.model_path}")
        else:
            LOGGER.info(f"Detect mode: {self.detect_mode}")

        self.is_running = True

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load the YOLO model."""
        try:
            model = YOLO(
                self.model_path,
                task="detect"
            )
            LOGGER.info(f"Successfully loaded YOLO model from {self.model_path}")
            return model
        except Exception as e:
            LOGGER.error(f"Error loading YOLO model: {e}")
            raise

    def _inference(self):
        target_classes = self.CLASS_MAP.get(self.detect_mode)

        results = self.model.predict(
            source=self.source,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            show=False,
            stream=True,
            verbose=False,
            classes=target_classes
        )

        return results

    def _draw_overlay(self, frame, boxes, fps, count):
        """Vẽ bounding box, FPS và đếm số lượng"""

        # ── Bảng màu ────────────────────────────────────────────────
        COLOR_BOX      = (255, 180, 50)  # Xanh dương nhạt, sáng
        COLOR_LABEL_BG = (255, 180, 50)
        COLOR_LABEL_FG = (255, 255, 255) # Chữ label trắng
        COLOR_HUD_BG   = (15, 15, 15)
        COLOR_KEY      = (200, 200, 200) # Key trắng nhạt
        COLOR_VAL_FPS  = (255, 255, 255) # FPS trắng
        COLOR_VAL_CNT  = (255, 180, 50)  # Count cùng màu bbox
        # ────────────────────────────────────────────────────────────

        frame_h, frame_w = frame.shape[:2]

        # ✅ Copy MỘT LẦN duy nhất ở đây
        overlay = frame.copy()

        # ── 1. Bounding Boxes ────────────────────────────────────────
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Bo góc kiểu bracket
            corner_len = max(12, (x2 - x1) // 6)
            for (sx, sy, dx, dy) in [
                (x1, y1, x1 + corner_len, y1), (x1, y1, x1, y1 + corner_len),
                (x2, y1, x2 - corner_len, y1), (x2, y1, x2, y1 + corner_len),
                (x1, y2, x1 + corner_len, y2), (x1, y2, x1, y2 - corner_len),
                (x2, y2, x2 - corner_len, y2), (x2, y2, x2, y2 - corner_len),
            ]:
                cv2.line(overlay, (sx, sy), (dx, dy), COLOR_BOX, 2)

            # Label pill
            label = f"{self.detect_mode}  {conf:.2f}"
            font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            pad = 4
            lx1, ly1 = x1, max(y1 - lh - pad * 2, 0)
            lx2, ly2 = x1 + lw + pad * 2, ly1 + lh + pad * 2

            # ✅ Vẽ nền label lên overlay
            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), COLOR_LABEL_BG, -1)
            cv2.putText(overlay, label, (lx1 + pad, ly2 - pad),
                        font, font_scale, COLOR_LABEL_FG, font_thick, cv2.LINE_AA)

        # ── 2. HUD Panel ─────────────────────────────────────────────
        rows = [
            ("FPS",                       str(int(fps)),  COLOR_VAL_FPS),
            (f"Total {self.detect_mode}", str(count),     COLOR_VAL_CNT),
        ]
        font_key  = cv2.FONT_HERSHEY_SIMPLEX
        scale_key, scale_val   = 0.55, 0.9
        thick_key, thick_val   = 1, 2
        row_h    = 38
        pad_x, pad_y = 14, 10

        max_w = max(
            cv2.getTextSize(k, font_key, scale_key, thick_key)[0][0] +
            cv2.getTextSize(v, font_key, scale_val, thick_val)[0][0]
            for k, v, _ in rows
        )
        panel_w = max_w + pad_x * 3 + 20
        panel_h = row_h * len(rows) + pad_y * 2
        px1, py1 = frame_w - panel_w - 16, 16
        px2, py2 = frame_w - 16, py1 + panel_h

        # ✅ Vẽ panel lên overlay
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_HUD_BG, -1)
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_BOX, 1)

        for i, (key, val, color_val) in enumerate(rows):
            row_y = py1 + pad_y + (i + 1) * row_h - 8
            cv2.putText(overlay, key + ":", (px1 + pad_x, row_y),
                        font_key, scale_key, COLOR_KEY, thick_key, cv2.LINE_AA)
            (vw, _), _ = cv2.getTextSize(val, font_key, scale_val, thick_val)
            cv2.putText(overlay, val, (px2 - vw - pad_x, row_y),
                        font_key, scale_val, color_val, thick_val, cv2.LINE_AA)

        # ✅ Blend MỘT LẦN duy nhất ở cuối — frame gốc + overlay có vẽ
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        return frame

    def stop(self):
        """Stop the detector"""
        self.is_running = False
        LOGGER.info("Detector stopped")

    def run(self):
        """Run the detection loop."""
        LOGGER.info("Starting detection loop...")
        try:
            results = self._inference()
            
            # Khởi tạo biến đếm thời gian cho FPS
            prev_time = time.time()
            smoothed_fps = 0.0
            
            for result in results:
                if not self.is_running:
                    break
                
                # Tính FPS thực tế (bao gồm cả thời gian vẽ và render)
                current_time = time.time()
                dt = current_time - prev_time
                instant_fps = 1 / dt if dt > 0 else 0
                prev_time = current_time

                if smoothed_fps == 0.0:
                    smoothed_fps = instant_fps
                else:
                    smoothed_fps = 0.9 * smoothed_fps + 0.1 * instant_fps
                
                if self.show:
                    # Lấy khung hình gốc chưa có gì từ YOLO
                    frame = result.orig_img.copy()
                    
                    # Đếm số lượng object (Vì model đã filter qua classes nên len(result.boxes) chính là tổng số cần đếm)
                    count = len(result.boxes)
                    
                    # Gọi hàm vẽ custom
                    annotated_frame = self._draw_overlay(frame, result.boxes, smoothed_fps, count)
                    
                    cv2.imshow(f"Detection - {self.detect_mode}", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.stop() # Giả định bạn có định nghĩa def stop(self): self.is_running = False
                        break
        except Exception as e:
            LOGGER.error(f"Error during detection run: {e}")
        finally:
            if self.show:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            LOGGER.info("Detection loop ended.")