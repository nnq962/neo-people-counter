from typing import Optional
from src.zone import Zone
from .logger import LOGGER

def load_zone(cfg: dict) -> Optional[Zone]:
    z = cfg.get("zone")
    if z:
        try:
            return Zone.from_dict(z)
        except ValueError as e:
            LOGGER.warning(f"Bỏ qua zone lỗi: {e}")
    return None