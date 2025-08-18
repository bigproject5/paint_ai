from pydantic import BaseModel, Field, ConfigDict
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

class TestStartedEventDTO(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    audit_id: int = Field(..., alias="auditId")
    model: str
    line_code: str = Field(..., alias="lineCode")
    inspection_id: int = Field(..., alias="inspectionId")
    inspection_type: str = Field(..., alias="inspectionType")
    collect_data_path: str = Field(..., alias="collectDataPath")

class AiDiagnosisCompletedEventDTO(BaseModel):
    audit_id: int
    inspection_id: int
    inspection_type: str
    is_defect: bool
    collect_data_path: str
    result_data_path: Optional[str] = None
    diagnosis_result: str

class SimpleDiagnosisResult(BaseModel):
    grade: str
    score: float
    defect_count: int
    defect_types: List[str]  
    processing_time: float
    error: Optional[str] = None

class PaintInspectionRequest(BaseModel):
    car_id: str
    part_code: str
    image_url: str
    inspector_id: str

class DetectedDefect(BaseModel):
    defect_type: DefectType
    confidence: float = Field(..., ge=0, le=1)
    bbox: Dict[str, float]
    severity: float = Field(..., ge=0, le=1)

class PaintInspectionResponse(BaseModel):
    car_id: str
    part_code: str
    overall_grade: QualityGrade
    quality_score: float = Field(..., ge=0, le=1)
    defects_found: List[DetectedDefect] = Field(default_factory=list)
    total_defects: int
    processing_time: float
    inspection_date: datetime

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.now)

class DiagnosisResultDetail(BaseModel):
    overall_grade: str
    quality_score: float
    defects_found: List[Dict[str, Any]]
    total_defects: int
    processing_time: float
    inspection_date: str
    error: Optional[str] = None