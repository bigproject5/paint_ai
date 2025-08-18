import os

class Config:
   API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
   API_PORT: int = int(os.getenv('API_PORT', 8000))
   
   MODEL_PATH: str = os.getenv('MODEL_PATH', '../models/car_damage_model.onnx')
   CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
   
   MAX_IMAGE_SIZE: int = int(os.getenv('MAX_IMAGE_SIZE', 10485760))
   DOWNLOAD_TIMEOUT: int = int(os.getenv('DOWNLOAD_TIMEOUT', 10))
   
   QUALITY_PASS_THRESHOLD: float = float(os.getenv('QUALITY_PASS_THRESHOLD', 0.8))
   QUALITY_MINOR_THRESHOLD: float = float(os.getenv('QUALITY_MINOR_THRESHOLD', 0.6))
   CRITICAL_SEVERITY_THRESHOLD: float = float(os.getenv('CRITICAL_SEVERITY_THRESHOLD', 0.7))
   
   KAFKA_BOOTSTRAP: str = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
   KAFKA_TOPIC: str = os.getenv('KAFKA_TOPIC', 'ai-diagnosis-completed')
   
   DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'

config = Config()