import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import app.config as config
from app.schemas import (
    DefectType, QualityGrade, DetectedDefect,
    PaintInspectionRequest, PaintInspectionResponse,
    AiDiagnosisCompletedEventDTO
)
from app.kafka_client import publish

# -------------------------------------------------
# 로깅
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paint-ai")

# -------------------------------------------------
# 더미 검출기 (USE_DUMMY=True일 때 사용)
# -------------------------------------------------
class DummyDetector:
    model_loaded = True

    def detect_defects(self, image_url: str) -> Dict[str, Any]:
        start = time.time()
        random.seed(hash(image_url) % 2**32)

        n = random.randint(0, 3)
        defects: List[DetectedDefect] = []
        for _ in range(n):
            dtype = random.choice([DefectType.SCRATCH, DefectType.PDR_DENT, DefectType.PAINT])
            conf = round(random.uniform(0.55, 0.95), 2)
            w, h = random.randint(20, 120), random.randint(20, 120)
            x, y = random.randint(0, 500), random.randint(0, 300)
            area = w * h
            sev = min(round((conf + min(area / 10000, 1.0) * 0.3) * (1.3 if dtype == DefectType.PDR_DENT else 1.1 if dtype == DefectType.PAINT else 1.0), 2), 1.0)
            defects.append(
                DetectedDefect(
                    defect_type=dtype,
                    confidence=conf,
                    bbox={"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                    severity=sev,
                )
            )

        if not defects:
            score = 1.0
            grade = QualityGrade.PASS
        else:
            total_ded = 0.0
            for d in defects:
                base = d.severity * d.confidence * 0.1
                if d.defect_type == DefectType.PDR_DENT:
                    base *= 1.2
                elif d.defect_type == DefectType.PAINT:
                    base *= 1.1
                total_ded += base
            score = max(1.0 - total_ded, 0.0)
            grade = (
                QualityGrade.REJECT if any(d.severity > 0.85 for d in defects) else
                QualityGrade.MAJOR_DEFECT if score < 0.6 else
                QualityGrade.MINOR_DEFECT if score < 0.8 else
                QualityGrade.PASS
            )

        return {
            "defects": defects,
            "quality_score": round(score, 3),
            "overall_grade": grade,
            "processing_time": round(time.time() - start, 3),
        }

# 실제 모델 사용 분기
detector = DummyDetector()
if not config.USE_DUMMY:
    try:
        from app import inference  # 실제 추론기 구현돼 있으면 사용
        detector = inference.detector
    except Exception as e:
        logger.error(f"실제 모델 초기화 실패, 더미로 대체: {e}")
        detector = DummyDetector()

# -------------------------------------------------
# FastAPI
# -------------------------------------------------
app = FastAPI(
    title="Paint AI Service (Demo Ready)",
    version="1.0.0",
    description="도장면 결함 검출 서비스 - 더미 데이터 모드 지원",
    docs_url="/paint-ai/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/paint-ai/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": getattr(detector, "model_loaded", False),
        "dummy_mode": config.USE_DUMMY,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/paint-ai/detect", response_model=PaintInspectionResponse)
async def detect(req: PaintInspectionRequest):
    if not getattr(detector, "model_loaded", False):
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
    r = detector.detect_defects(req.image_url)
    return PaintInspectionResponse(
        car_id=req.car_id,
        part_code=req.part_code,
        overall_grade=r["overall_grade"],
        quality_score=r["quality_score"],
        defects_found=r["defects"],
        total_defects=len(r["defects"]),
        processing_time=r["processing_time"],
        inspection_date=datetime.now().isoformat(),
    )

# vehicleAudit HTTP 전송
def _forward_to_vehicle_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.VEHICLE_AUDIT_API_KEY:
        headers["Authorization"] = f"Bearer {config.VEHICLE_AUDIT_API_KEY}"
    try:
        resp = requests.post(config.VEHICLE_AUDIT_URL, json=payload, headers=headers, timeout=config.VEHICLE_AUDIT_TIMEOUT)
        resp.raise_for_status()
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {"ok": True, "status": resp.status_code, "response": body}
    except requests.RequestException as e:
        logger.error(f"vehicleAudit 전송 실패: {e}")
        return {"ok": False, "error": str(e)}

# Kafka DTO 빌드/발행
def _build_ai_completed_event(req: PaintInspectionRequest, res: PaintInspectionResponse) -> AiDiagnosisCompletedEventDTO:
    is_defect = res.total_defects > 0
    collect_path = req.image_url
    result_path = f"{config.RESULT_BASE_URL}{res.car_id}_{res.part_code}_{int(datetime.now().timestamp())}.json"
    diagnosis = res.overall_grade.value

    audit_id = req.audit_id if req.audit_id is not None else int(datetime.now().timestamp())
    inspection_id = req.inspection_id if req.inspection_id is not None else int(datetime.now().timestamp() * 1000)
    inspection_type = req.inspection_type or "PAINT"

    return AiDiagnosisCompletedEventDTO(
        auditId=audit_id,
        inspectionId=inspection_id,
        inspectionType=inspection_type,
        isDefect=is_defect,
        collectDataPath=collect_path,
        resultDataPath=result_path,
        diagnosisResult=diagnosis,
    )

def _publish_ai_completed_event(dto: AiDiagnosisCompletedEventDTO):
    publish(config.KAFKA_TOPIC_AI_COMPLETED, dto.dict(), key=str(dto.inspectionId))

@app.post("/paint-ai/inspect-and-forward", response_model=PaintInspectionResponse)
async def inspect_and_forward(req: PaintInspectionRequest):
    res: PaintInspectionResponse = await detect(req)
    # Kafka
    dto = _build_ai_completed_event(req, res)
    _publish_ai_completed_event(dto)
    # 기존 HTTP 전송(선택)
    payload = {
        "car_id": res.car_id,
        "part_code": res.part_code,
        "overall_grade": res.overall_grade.value,
        "quality_score": res.quality_score,
        "total_defects": res.total_defects,
        "processing_time": res.processing_time,
        "inspection_date": res.inspection_date,
        "defects": [
            {"type": d.defect_type.value, "confidence": d.confidence, "severity": d.severity, "bbox": d.bbox}
            for d in res.defects_found
        ],
        "inspector_id": req.inspector_id,
        "source_service": "paint-ai",
        "dummy_mode": config.USE_DUMMY,
    }
    _ = _forward_to_vehicle_audit(payload)
    return res

from fastapi import BackgroundTasks

@app.post("/paint-ai/inspect-and-forward/bg", response_model=PaintInspectionResponse)
async def inspect_and_forward_bg(req: PaintInspectionRequest, bg: BackgroundTasks):
    res: PaintInspectionResponse = await detect(req)
    dto = _build_ai_completed_event(req, res)
    bg.add_task(_publish_ai_completed_event, dto)
    # HTTP 전송도 백그라운드
    payload = {
        "car_id": res.car_id,
        "part_code": res.part_code,
        "overall_grade": res.overall_grade.value,
        "quality_score": res.quality_score,
        "total_defects": res.total_defects,
        "processing_time": res.processing_time,
        "inspection_date": res.inspection_date,
        "defects": [
            {"type": d.defect_type.value, "confidence": d.confidence, "severity": d.severity, "bbox": d.bbox}
            for d in res.defects_found
        ],
        "inspector_id": req.inspector_id,
        "source_service": "paint-ai",
        "dummy_mode": config.USE_DUMMY,
    }
    bg.add_task(_forward_to_vehicle_audit, payload)
    return res
