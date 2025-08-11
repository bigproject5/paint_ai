# schemas.py
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

# Java DTO에 대응하는 이벤트 DTO들
class TestStartedEventDTO(BaseModel):
    """TestStartedEventDTO에 대응하는 파이썬 모델"""
    audit_id: int = Field(..., description="감사 ID")
    model: str = Field(..., description="차량 모델")
    line_code: str = Field(..., description="라인 코드")
    inspection_id: int = Field(..., description="검사 ID")
    inspection_type: str = Field(..., description="검사 유형")
    collect_data_path: str = Field(..., description="수집된 데이터 경로")

class AiDiagnosisCompletedEventDTO(BaseModel):
    """AiDiagnosisCompletedEventDTO에 대응하는 파이썬 모델"""
    audit_id: int = Field(..., description="감사 ID")
    inspection_id: int = Field(..., description="검사 ID")
    inspection_type: str = Field(..., description="검사 유형")
    is_defect: bool = Field(..., description="결함 여부")
    collect_data_path: str = Field(..., description="수집된 데이터 경로")
    result_data_path: str = Field(..., description="결과 데이터 경로")
    diagnosis_result: str = Field(..., description="진단 결과 (JSON 문자열)")

# 기존 스키마들 (레거시 호환용)
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

# 도장면 검사 응답 (레거시 호환용)
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
    timestamp: datetime = Field(default_factory=datetime.now)

# AI 진단 결과 상세 정보 (JSON으로 저장될 데이터)
class DiagnosisResultDetail(BaseModel):
    """diagnosis_result JSON에 포함될 상세 정보"""
    overall_grade: str
    quality_score: float
    defects_found: List[Dict[str, Any]]
    total_defects: int
    processing_time: float
    inspection_date: str
    error: Optional[str] = None