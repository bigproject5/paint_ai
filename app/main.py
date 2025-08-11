from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.schemas import *
from app.inference import process_paint_inspection, get_model_status, process_ai_diagnosis

app = FastAPI(
    title="Paint Defect Inspection Service",
    version="1.0.0",
    description="AI 기반 도장면 결함 검사 서비스",
    docs_url="/paint-ai/docs"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 헬스체크
@app.get("/paint-ai/health", response_model=HealthResponse)
async def health_check():
    model_status = get_model_status()
    return HealthResponse(
        status="healthy",
        service="paint-defect-inspection",
        model_loaded=model_status["model_loaded"]
    )

# 이벤트 기반 AI 진단 처리 (메인 API)
@app.post("/ai-service/diagnosis", response_model=AiDiagnosisCompletedEventDTO)
async def process_ai_diagnosis_api(event: TestStartedEventDTO):
    """TestStartedEvent를 받아 AI 진단 후 AiDiagnosisCompletedEvent 반환"""
    try:
        result = process_ai_diagnosis(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 도장면 검사 API (vehicleAudit에서 호출)
@app.post("/paint-ai/inspect", response_model=PaintInspectionResponse)
async def inspect_paint_surface(request: PaintInspectionRequest):
    """도장면 결함 검사 - vehicleAudit 연동용"""
    try:
        result = process_paint_inspection(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 모델 상태 확인
@app.get("/paint-ai/model/status")
async def get_model_status_api():
    """AI 모델 상태 확인"""
    status = get_model_status()
    return {
        "model_loaded": status["model_loaded"],
        "yolo_available": status["available"],
        "message": "모델이 정상적으로 로드되었습니다" if status["model_loaded"] else "모델 로드 실패"
    }

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "message": "서버 내부 오류가 발생했습니다"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
