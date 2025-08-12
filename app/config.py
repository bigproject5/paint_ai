import os

# === 기본 서버 설정 ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
USE_DUMMY = os.getenv("USE_DUMMY", "true").lower() == "true"

# vehicle-audit HTTP 전송용 (안 쓰면 기본값)
VEHICLE_AUDIT_URL = os.getenv("VEHICLE_AUDIT_URL", "http://localhost:8080/audit")
VEHICLE_AUDIT_API_KEY = os.getenv("VEHICLE_AUDIT_API_KEY", "")
VEHICLE_AUDIT_TIMEOUT = int(os.getenv("VEHICLE_AUDIT_TIMEOUT", 5))

# === Kafka 설정 ===
KAFKA_ENABLED: bool = os.getenv("KAFKA_ENABLED", "true").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CLIENT_ID: str = os.getenv("KAFKA_CLIENT_ID", "paint-ai")
KAFKA_TOPIC_AI_COMPLETED: str = os.getenv("KAFKA_TOPIC_AI_COMPLETED", "ai.diagnosis.completed")

# 결과 저장 경로(예: 리포트 파일/URL prefix)
RESULT_BASE_URL: str = os.getenv("RESULT_BASE_URL", "https://result.example.com/")
