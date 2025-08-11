import os
from typing import Optional

class Config:
    # AWS S3 설정
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME: Optional[str] = os.getenv('S3_BUCKET_NAME')
    
    # Kafka 설정
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_TOPIC: str = os.getenv('KAFKA_TOPIC', 'paint_ai')
    
    # 모델 설정
    MODEL_PATH: str = os.getenv('MODEL_PATH', './models/')
    
    # API 설정
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', 8000))

config = Config()
