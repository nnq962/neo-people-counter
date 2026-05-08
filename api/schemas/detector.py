from pydantic import BaseModel
from typing import Optional

class DetectorStatus(BaseModel):
    is_running: bool
    source: Optional[str] = None
    model_path: Optional[str] = None