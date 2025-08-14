from kafka import KafkaProducer
import json
from config import config
from schemas import AiDiagnosisCompletedEventDTO

_producer = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            linger_ms=10,
        )
    return _producer

def publish_diagnosis(key_audit_id: int, payload: dict):
    p = get_producer()
    p.send('ai-diagnosis-completed', key=key_audit_id, value=payload)
    p.flush()

def publish_ai_diagnosis_completed(completed_event: AiDiagnosisCompletedEventDTO):
    """
    AiDiagnosisCompletedEventDTO를 vehicleAudit이 받을 수 있는 camelCase 형식으로 변환하여 전송
    """
    # snake_case를 camelCase로 변환 (vehicleAudit 형식에 맞춤)
    message_data = {
        "auditId": completed_event.audit_id,
        "inspectionId": completed_event.inspection_id,
        "inspectionType": completed_event.inspection_type,
        "isDefect": completed_event.is_defect,
        "collectDataPath": completed_event.collect_data_path,
        "resultDataPath": completed_event.result_data_path,
        "diagnosisResult": completed_event.diagnosis_result
    }
    
    p = get_producer()
    p.send('ai-diagnosis-completed', key=completed_event.audit_id, value=message_data)
    p.flush()