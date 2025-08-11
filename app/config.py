import os

class Config:
    # API 설정
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', 8000))
    
    # YOLO 모델 설정
    MODEL_PATH: str = os.getenv('MODEL_PATH', 'runs_yolo11/car_defect_v2/weights/best.pt')
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE: int = int(os.getenv('MAX_IMAGE_SIZE', 10485760))  # 10MB
    DOWNLOAD_TIMEOUT: int = int(os.getenv('DOWNLOAD_TIMEOUT', 10))  # 10초
    
    # 품질 등급 임계값
    QUALITY_PASS_THRESHOLD: float = float(os.getenv('QUALITY_PASS_THRESHOLD', 0.8))
    QUALITY_MINOR_THRESHOLD: float = float(os.getenv('QUALITY_MINOR_THRESHOLD', 0.6))
    CRITICAL_SEVERITY_THRESHOLD: float = float(os.getenv('CRITICAL_SEVERITY_THRESHOLD', 0.7))
    
    # 로그 설정
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # 디버그 모드
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'

# 싱글톤 설정 객체
config = Config()
