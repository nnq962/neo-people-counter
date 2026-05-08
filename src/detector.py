from typing import Optional, List
from ultralytics import YOLO
from pathlib import Path
import time
import numpy as np
import aidcv as cv2
from utils import LOGGER, restore_level_names
from src.zone import Zone

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
        show: bool = False,
        show_scale: float = 1.0,
        zone: Optional[Zone] = None,
        vid_stride: int = 1,
        verbose: bool = True,
    ):

        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.show = show
        self.show_scale = show_scale
        self.zone = zone
        self.vid_stride = vid_stride
        self.verbose = verbose

        # Log info
        LOGGER.info(f"Source: {self.source}")
        LOGGER.info(f"Model path: {self.model_path}")
        LOGGER.info(f"conf: {self.conf}")
        LOGGER.info(f"Show: {self.show}")
        LOGGER.info(f"Show scale: {self.show_scale}")
        LOGGER.info(f"Zone configured: {self.zone}")
        LOGGER.info(f"Video stride: {self.vid_stride}")
        LOGGER.info(f"Verbose: {self.verbose}")

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
        self._results_gen = None

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

        self._results_gen = self.model.predict(
            source=self.source,
            conf=self.conf,
            show=False,
            stream=True,
            verbose=False,
            classes=target_classes,
            vid_stride=self.vid_stride
        )

    def _draw_overlay(self, frame, in_zone_boxes, out_zone_boxes, fps):
        """Vẽ bounding box, FPS và đếm số lượng.
        
        in_zone_boxes  : boxes nằm trong zone → vẽ màu đỏ
        out_zone_boxes : boxes nằm ngoài zone → vẽ màu xanh
        """

        # ── Bảng màu ────────────────────────────────────────────────
        COLOR_IN_ZONE  = (50,  50,  220) # Đỏ  (BGR) — trong zone
        COLOR_OUT_ZONE = (255, 180,  50) # Xanh nhạt (BGR) — ngoài zone
        COLOR_LABEL_FG = (255, 255, 255) # Chữ label trắng
        COLOR_HUD_BG   = (15,  15,  15)
        COLOR_KEY      = (200, 200, 200) # Key trắng nhạt
        COLOR_VAL_FPS  = (255, 255, 255) # FPS trắng
        COLOR_VAL_CNT  = (50,  50,  220) # Count cùng màu in-zone
        # ────────────────────────────────────────────────────────────

        frame_h, frame_w = frame.shape[:2]

        # ✅ Copy MỘT LẦN duy nhất ở đây
        overlay = frame.copy()

        # ── 1. Bounding Boxes ────────────────────────────────────────
        def _draw_boxes(boxes, color):
            for box in boxes:
                coords_np = box.xyxy[0].cpu().numpy()

                if not np.all(np.isfinite(coords_np)):
                    continue

                x1, y1, x2, y2 = map(int, coords_np)
                conf = float(box.conf[0].cpu().numpy())

                # Bo góc kiểu bracket
                corner_len = max(12, (x2 - x1) // 6)
                for (sx, sy, dx, dy) in [
                    (x1, y1, x1 + corner_len, y1), (x1, y1, x1, y1 + corner_len),
                    (x2, y1, x2 - corner_len, y1), (x2, y1, x2, y1 + corner_len),
                    (x1, y2, x1 + corner_len, y2), (x1, y2, x1, y2 - corner_len),
                    (x2, y2, x2 - corner_len, y2), (x2, y2, x2, y2 - corner_len),
                ]:
                    cv2.line(overlay, (sx, sy), (dx, dy), color, 2)

                # Label pill
                label = f"{self.detect_mode}  {conf:.2f}"
                font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                (lw, lh), _ = cv2.getTextSize(label, font, font_scale, font_thick)
                pad = 4
                lx1, ly1 = x1, max(y1 - lh - pad * 2, 0)
                lx2, ly2 = x1 + lw + pad * 2, ly1 + lh + pad * 2

                cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), color, -1)
                cv2.putText(overlay, label, (lx1 + pad, ly2 - pad),
                            font, font_scale, COLOR_LABEL_FG, font_thick, cv2.LINE_AA)

        # Vẽ out-zone trước (nằm dưới) → in-zone vẽ sau (nằm trên)
        _draw_boxes(out_zone_boxes, COLOR_OUT_ZONE)
        _draw_boxes(in_zone_boxes,  COLOR_IN_ZONE)

        # ── 2. HUD Panel ─────────────────────────────────────────────
        rows = [
            ("FPS",                       str(int(fps)),  COLOR_VAL_FPS),
            (f"Total {self.detect_mode}", str(len(in_zone_boxes)), COLOR_VAL_CNT),
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
        cv2.rectangle(overlay, (px1, py1), (px2, py2), COLOR_IN_ZONE, 1)

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

    def _draw_zones(self, frame):
        """Draw zone on frame: nền bán trong suốt + viền đậm + tên zone."""
        if not self.zone:
            return frame

        pts = self.zone.points.reshape((-1, 1, 2))

        # ── Nền bán trong suốt ──────────────────────────────────
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # ── Viền đậm ────────────────────────────────────────────
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # ── Tên zone: đặt tại centroid của polygon ───────────────
        cx = int(self.zone.points[:, 0].mean())
        cy = int(self.zone.points[:, 1].mean())

        label = self.zone.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thick = 2
        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, font_thick)

        # Nền chữ
        pad = 6
        cv2.rectangle(
            frame,
            (cx - lw // 2 - pad, cy - lh - pad),
            (cx + lw // 2 + pad, cy + pad),
            (0, 255, 0),
            -1
        )
        # Chữ
        cv2.putText(
            frame, label,
            (cx - lw // 2, cy),
            font, font_scale,
            (255, 255, 255),  # chữ trắng trên nền xanh
            font_thick,
            cv2.LINE_AA
        )

        return frame

    def _is_bbox_in_zone(
        self,
        bbox: tuple,
        points: np.ndarray,
        # mode: str = "center",
    ) -> bool:

        # if mode not in ("center", "bottom_center"):
        #     raise ValueError(
        #         f"mode phải là 'center' hoặc 'bottom_center', nhận được: '{mode}'"
        #     )

        x1, y1, x2, y2 = bbox[:4]

        # Bỏ qua bbox có tọa độ NaN/Inf — coi như nằm ngoài zone
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            return False

        point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # if mode == "center":
        #     # Điểm trung tâm của bounding box
        #     point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        # else:  # bottom_center
        #     # Điểm giữa cạnh dưới bbox
        #     point = (int((x1 + x2) / 2), int(y2))

        # cv2.pointPolygonTest trả về:
        #   > 0  : điểm nằm trong đa giác
        #   = 0  : điểm nằm trên cạnh đa giác
        #   < 0  : điểm nằm ngoài đa giác
        result = cv2.pointPolygonTest(points, point, measureDist=False)
        return result >= 0

    def _classify_boxes(self, boxes):
        """Phân loại boxes theo zone.

        Returns:
            in_zone_boxes  : list box có center nằm trong zone
            out_zone_boxes : list box không nằm trong zone
        """
        if not self.zone:
            return list(boxes), []

        in_zone_boxes  = []
        out_zone_boxes = []

        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            in_zone = self._is_bbox_in_zone(coords, self.zone.points)
            if in_zone:
                in_zone_boxes.append(box)
            else:
                out_zone_boxes.append(box)

        return in_zone_boxes, out_zone_boxes

    def _cleanup(self):
        """Gọi từ finally của run() — đúng thread sở hữu generator và RKNN."""

        # 1. Đóng generator
        if self._results_gen is not None:
            try:
                self._results_gen.close()
                LOGGER.info("Inference generator closed.")
            except Exception:
                pass
            self._results_gen = None

        # 2. Đóng LoadStreams (nguyên nhân chính giữ CPU với RTSP)
        if hasattr(self, "model") and self.model is not None:
            predictor = getattr(self.model, "predictor", None)
            if predictor is not None:
                dataset = getattr(predictor, "dataset", None)
                if dataset is not None:
                    try:
                        if hasattr(dataset, "running"):
                            dataset.running = False
                        if hasattr(dataset, "threads"):
                            for t in dataset.threads:
                                if t.is_alive():
                                    t.join(timeout=3)
                        if hasattr(dataset, "caps"):
                            for cap in dataset.caps:
                                if cap and cap.isOpened():
                                    cap.release()
                        if hasattr(dataset, "close"):
                            dataset.close()
                        LOGGER.info("LoadStreams dataset closed.")
                    except Exception as e:
                        LOGGER.warning(f"Error closing dataset: {e}")

                # Reset predictor để lần start sau tạo mới hoàn toàn
                try:
                    self.model.predictor = None
                    LOGGER.info("Predictor reset.")
                except Exception:
                    pass

        # 3. Đóng GUI
        if self.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # 4. GC
        import gc
        gc.collect()

        self.is_running = False
        LOGGER.info("Cleanup complete.")

    def update_detector_params(self, verbose: bool):
        """
        Update detector parameters.
        """
        self.verbose = verbose
        LOGGER.info("Detector params updated")

    def update_zone(self, zone: Optional[Zone]):
        """
        Update the detection zone.
        """
        self.zone = zone
        LOGGER.info("Detector zone updated")

    def stop(self):
        self.is_running = False
        LOGGER.info("Detector stopped")

    def run(self):
        """Run the detection loop."""
        LOGGER.info("Starting detection loop...")
        try:
            self._inference()
            prev_time = time.time()
            # Restore logger level names
            _restored = False

            if self._results_gen is not None:
                for result in self._results_gen:
                    if not self.is_running:
                        break

                    # Restore logger level names
                    if not _restored:
                        restore_level_names()
                        _restored = True

                    # Calculate FPS
                    current_time = time.time()
                    dt = current_time - prev_time
                    fps = 1 / dt if dt > 0 else 0
                    prev_time = current_time

                    # Phân loại boxes
                    in_zone_boxes, out_zone_boxes = self._classify_boxes(result.boxes)

                    # Log verbose
                    if self.verbose:
                        LOGGER.info(
                            f"Total: {len(result.boxes):02d} | "
                            f"In Zone: {len(in_zone_boxes):02d} | "
                            f"Out Zone: {len(out_zone_boxes):02d} | "
                            f"FPS: {fps:.2f}"
                        )

                    if self.show:
                        # Copy original frame
                        frame = result.orig_img.copy()
                        # Draw zones
                        annotated_frame = self._draw_zones(frame)
                        # Draw overlay: đỏ=trong zone, xanh=ngoài zone
                        annotated_frame = self._draw_overlay(annotated_frame, in_zone_boxes, out_zone_boxes, fps)

                        # Resize and add padding if needed
                        if self.show_scale != 1.0:
                            h, w = annotated_frame.shape[:2]
                            new_dim = (int(w * self.show_scale), int(h * self.show_scale))
                            resized_frame = cv2.resize(annotated_frame, new_dim)
                            padded_frame = cv2.copyMakeBorder(
                                resized_frame,
                                20, 20, 20, 20,
                                cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)
                            )
                            cv2.imshow("", padded_frame)
                        else:
                            cv2.imshow("", annotated_frame)

        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received.")
        except Exception as e:
            LOGGER.error(f"Error during detection run: {e}")
        finally:
            self._cleanup()
            LOGGER.info("Detection loop ended.")