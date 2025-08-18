from kafka import KafkaProducer
import json
from datetime import datetime
from config import config
from schemas import AiDiagnosisCompletedEventDTO

_producer = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False, default=str).encode("utf-8"),
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
    Spring Kafka 호환을 위한 타입 헤더 포함
    """
    # snake_case를 camelCase로 변환 (vehicleAudit 형식에 맞춤)
    message_data = {
        "auditId": int(completed_event.audit_id),
        "inspectionId": str(completed_event.inspection_id) if completed_event.inspection_id else "",
        "inspectionType": str(completed_event.inspection_type) if completed_event.inspection_type else "",
        "isDefect": bool(completed_event.is_defect) if completed_event.is_defect is not None else False,
        "collectDataPath": str(completed_event.collect_data_path) if completed_event.collect_data_path else "",
        "resultDataPath": str(completed_event.result_data_path) if completed_event.result_data_path else "",
        "diagnosisResult": str(completed_event.diagnosis_result) if completed_event.diagnosis_result else "",
        "timestamp": int(datetime.now().timestamp() * 1000)
    }
    
    # Spring Kafka가 필요로 하는 타입 헤더 추가
    headers = [
        ('__TypeId__', b'java.util.LinkedHashMap'),
        ('__ContentTypeId__', b'application/json'),
        ('__KeyTypeId__', b'java.lang.Integer')
    ]
    
    try:
        p = get_producer()
        future = p.send('ai-diagnosis-completed', 
                       key=completed_event.audit_id, 
                       value=message_data,
                       headers=headers)
        p.flush()
        
        # 전송 확인
        record_metadata = future.get(timeout=10)
        print(f"[kafka_producer] 메시지 전송 성공 - audit_id: {completed_event.audit_id}, offset: {record_metadata.offset}")
        
    except Exception as e:
        print(f"[kafka_producer] 메시지 전송 실패 - audit_id: {completed_event.audit_id}, error: {e}")
        raise