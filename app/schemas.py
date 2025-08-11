from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DefectType(str, Enum):
    SCRATCH = "scratch"
    PDR_DENT = "pdr_dent"
    PAINT = "paint"
    
class QualityGrade(str, Enum):
    PASS = "pass"
    MINOR_DEFECT = "minor_defect"
    MAJOR_DEFECT = "major_defect"
    REJECT = "reject"

# 도장면 검사 요청
class PaintInspectionRequest(BaseModel):
    car_id: str = Field(..., description="차량 ID")
    part_code: str = Field(..., description="부품 코드")
    image_url: str = Field(..., description="검사할 이미지 URL")
    inspector_id: str = Field(..., description="검사자 ID")

# 검출된 결함 정보
class DetectedDefect(BaseModel):
    defect_type: DefectType
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")
    bbox: Dict[str, float] = Field(..., description="바운딩 박스 {x, y, width, height}")
    severity: float = Field(..., ge=0, le=1, description="심각도")

# 도장면 검사 응답
class PaintInspectionResponse(BaseModel):
    car_id: str
    part_code: str
    overall_grade: QualityGrade
    quality_score: float = Field(..., ge=0, le=1, description="전체 품질 점수")
    defects_found: List[DetectedDefect] = Field(default_factory=list)
    total_defects: int
    processing_time: float = Field(..., description="처리 시간 (초)")
    inspection_date: datetime

# 헬스체크 응답
class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
