import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import threading
import time
import os
from datetime import datetime # datetime 모듈 추가

from app import inference
from app.config import config

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Paint AI Service",
    description="도장면 결함 검출 서비스",
    version="1.0.0",
)

# 헬스체크 엔드포인트
@app.get("/paint-ai/health")
async def health_check():
    """
    서버의 상태와 모델의 로드 상태를 확인합니다.
    """
    logger.info("Health check endpoint called.")
    try:
        model_status = inference.get_model_status()
        return {
            "status": "healthy",
            "service": "paint-defect-inspection",
            "model_loaded": model_status.get("model_loaded", False),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"헬스체크 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "message": "서버 내부 오류가 발생했습니다"})

# 모델 상태 엔드포인트
@app.get("/paint-ai/model/status")
async def get_model_status_endpoint():
    """
    모델의 상세 상태 정보를 반환합니다.
    """
    logger.info("Model status endpoint called.")
    try:
        return inference.get_model_status()
    except Exception as e:
        logger.error(f"모델 상태 확인 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "message": "서버 내부 오류가 발생했습니다"})

# 결함 탐지 API 엔드포인트
class ImageUrlRequest(BaseModel):
    image_url: str

@app.post("/paint-ai/detect")
async def detect_defects_api(request: ImageUrlRequest):
    """
    주어진 이미지 URL에서 도장면 결함을 탐지합니다.
    """
    logger.info(f"Detect endpoint called for URL: {request.image_url}")
    try:
        if not inference.detector.model_loaded:
            raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았습니다.")
        
        result = inference.detector.detect_defects(request.image_url)
        return result
    except HTTPException as e:
        # FastAPI HTTPException은 그대로 전파
        raise e
    except Exception as e:
        logger.error(f"결함 탐지 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e), "message": "결함 탐지 처리 중 서버 내부 오류가 발생했습니다"})

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
