from threading import Thread
from typing import Optional
from src.detector import Detector
from src.zone import Zone
from api.services import config as config_service
from utils import LOGGER

_detector: Optional[Detector] = None
_thread: Optional[Thread] = None


def get_status() -> dict:
    return {
        "is_running": _detector.is_running if _detector else False,
        "source": _detector.source if _detector else None,
        "model_path": _detector.model_path if _detector else None,
        "conf": _detector.conf if _detector else None,
        "vid_stride": _detector.vid_stride if _detector else None,
        "verbose": _detector.verbose if _detector else None,
    }


def start() -> dict:
    global _detector, _thread

    if _detector and _detector.is_running:
        return {"status": "already_running", "message": "Detector is already running."}

    try:
        cfg = config_service.get_config_data()
    except FileNotFoundError as e:
        raise RuntimeError(f"Config not found: {e}")

    # Load zone từ config
    zone_cfg = cfg.get("zone")
    zone = None
    if zone_cfg:
        try:
            zone = Zone.from_dict(zone_cfg)
        except ValueError as e:
            LOGGER.warning(f"Bỏ qua zone lỗi: {e}")

    _detector = Detector(
        zone=zone,
        **cfg.get("detector", {})
    )

    _thread = Thread(target=_detector.run, daemon=True)
    _thread.start()
    LOGGER.info("Detector started.")

    return {"status": "started", "message": "Detector started successfully."}


def stop() -> dict:
    global _detector, _thread

    if not _detector or not _detector.is_running:
        return {"status": "not_running", "message": "Detector is not running."}

    # 1. Signal dừng + cleanup resource trong detector
    _detector.stop()

    # 2. Chờ thread kết thúc hẳn (timeout 10s để tránh treo)
    if _thread and _thread.is_alive():
        _thread.join(timeout=10)
        if _thread.is_alive():
            LOGGER.warning("Thread did not stop within timeout.")

    # 3. Xóa reference để GC có thể thu hồi memory
    _detector = None
    _thread = None

    LOGGER.info("Detector stopped.")
    return {"status": "stopped", "message": "Detector stopped successfully."}