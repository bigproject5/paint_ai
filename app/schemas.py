from pydantic import BaseModel
from typing import Optional, List

class PaintRequest(BaseModel):
    image_url: str
    style: Optional[str] = "default"
    
class PaintResponse(BaseModel):
    result_url: str
    processing_time: float
    status: str
