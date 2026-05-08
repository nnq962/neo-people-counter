from pydantic import BaseModel, Field
from typing import List, Optional

class DetectorConfig(BaseModel):
    source: str = "rtsp://admin:phenikaaneo%40@192.168.0.150:554/Streaming/Channels/101"
    model_path: str = "models/head/yolo8n_rknn_model"
    conf: float = Field(0.70, ge=0.0, le=1.0)
    vid_stride: int = Field(1, ge=1)
    verbose: bool = True

class UartConfig(BaseModel):
    port: str = "/dev/ttyS4"
    baudrate: int = 115200

class ZoneMetaConfig(BaseModel):
    max_capacity: Optional[int] = 10
    alert_threshold: Optional[int] = 8

class ZoneConfig(BaseModel):
    name: str = "Cabin 1"
    points: List[List[int]] = Field(
        default=[[63, 576], [1517, 614], [1500, 1066], [24, 1065]]
    )
    meta: Optional[ZoneMetaConfig] = Field(default_factory=ZoneMetaConfig)

class AppConfig(BaseModel):
    auto_start: bool = False
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    uart: UartConfig = Field(default_factory=UartConfig)
    zone: Optional[ZoneConfig] = Field(default_factory=ZoneConfig)
