import uvicorn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from datetime import datetime
from app import inference
from app.schemas import *
from app.config import config
from app.kafka_producer import publish_diagnosis
from app.kafka_consumer import start_background_consumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Paint AI Service",
    description="도장면 결함 검출 서비스",
    title="Paint AI Service",
    description="도장면 결함 검출 서비스",
    version="1.0.0",
    docs_url="/paint-ai/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/paint-ai/health", response_model=HealthResponse)
async def health_check():
    try:
        model_status = inference.get_model_status()
        return HealthResponse(
            status="healthy",
            service="paint-defect-inspection",
            model_loaded=model_status.get("model_loaded", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paint-ai/model/status")
async def get_model_status_endpoint():
    try:
        return inference.get_model_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ImageUrlRequest(BaseModel):
    image_url: str

class PaintInspectionRequest(BaseModel):
    car_id: str
    part_code: str
    image_url: str
    inspector_id: str

@app.post("/paint-ai/detect")
async def detect_defects_api(request: ImageUrlRequest):
    try:
        if not inference.detector.model_loaded:
            raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았습니다.")
        
        result = inference.detector.detect_defects(request.image_url)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai-service/diagnosis", response_model=AiDiagnosisCompletedEventDTO)
async def process_ai_diagnosis_api(event: TestStartedEventDTO):
    try:
        result = inference.process_ai_diagnosis(event)
        
        publish_diagnosis(
            key_audit_id=event.audit_id,
            payload=result.dict(by_alias=True, exclude_none=True),
        )
        
        return result
    except Exception as e:
        logger.error(f"모델 상태 확인 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "message": "서버 내부 오류가 발생했습니다"})

@app.post("/paint-ai/inspect", response_model=PaintInspectionResponse)
async def inspect_paint_surface(request: PaintInspectionRequest):
    try:
        result = inference.process_paint_inspection(request)
        return result
    except HTTPException as e:
        # FastAPI HTTPException은 그대로 전파
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    start_background_consumer()

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)