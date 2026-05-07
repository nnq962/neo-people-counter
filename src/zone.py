from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(eq=False)
class Zone:
    name: str
    points: np.ndarray  # shape (N, 2), dtype int32
    meta: Optional[dict] = None

    def __post_init__(self):
        self.points = np.array(self.points, dtype=np.int32)
        if len(self.points) < 3:
            raise ValueError(
                f"Zone '{self.name}' cần ít nhất 3 điểm để tạo polygon, "
                f"hiện có {len(self.points)} điểm."
            )

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(
            name=data["name"],
            points=data["points"],
            meta=data.get("meta")
        )