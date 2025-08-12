from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum

# --- 서비스 내부에서 사용 ---
class DefectType(str, Enum):
    SCRATCH = "scratch"
    PDR_DENT = "pdr_dent"
    PAINT = "paint"

class QualityGrade(str, Enum):
    PASS = "pass"
    MINOR_DEFECT = "minor_defect"
    MAJOR_DEFECT = "major_defect"
    REJECT = "reject"

class DetectedDefect(BaseModel):
    defect_type: DefectType
    confidence: float = Field(..., ge=0, le=1)
    bbox: Dict[str, float]
    severity: float = Field(..., ge=0, le=1)

class PaintInspectionRequest(BaseModel):
    car_id: str
    part_code: str
    image_url: str
    inspector_id: str
    audit_id: Optional[int] = None
    inspection_id: Optional[int] = None
    inspection_type: Optional[str] = "PAINT"

class PaintInspectionResponse(BaseModel):
    car_id: str
    part_code: str
    overall_grade: QualityGrade
    quality_score: float = Field(..., ge=0, le=1)
    defects_found: List[DetectedDefect] = []
    total_defects: int
    processing_time: float
    inspection_date: str  # ISO 문자열로 돌려줌

# --- Kafka 이벤트 DTO (자바 DTO와 1:1로 키 맞춤) ---
class AiDiagnosisCompletedEventDTO(BaseModel):
    auditId: int
    inspectionId: int
    inspectionType: str
    isDefect: bool
    collectDataPath: str
    resultDataPath: str
    diagnosisResult: str
