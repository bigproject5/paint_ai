import threading
import json
import time
from kafka import KafkaConsumer
from config import config
from schemas import TestStartedEventDTO, AiDiagnosisCompletedEventDTO
from inference import process_ai_diagnosis
from kafka_producer import publish_ai_diagnosis_completed

def _process(event: dict):
    # vehicleAudit TestStartedEventDTO (camelCase) → 필드 매핑
    audit_id = event["auditId"]
    inspection_id = event["inspectionId"]
    inspection_type = event["inspectionType"]
    collect_data_path = event["collectDataPath"]
    
    print(f"[kafka_consumer] Processing audit_id: {audit_id}, inspection_id: {inspection_id}")
    print(f"[kafka_consumer] inspection_type: {inspection_type}, collect_data_path: {collect_data_path}")
    
    # inspectionType 필터링 - PAINT만 처리
    if inspection_type != "PAINT":
        print(f"[paint_ai] Skipping inspection_type: {inspection_type} (audit_id: {audit_id})")
        return
    
    try:
        print(f"[kafka_consumer] Starting AI diagnosis...")
        test_event = TestStartedEventDTO(**event)
        result = process_ai_diagnosis(test_event)
        
        print(f"[kafka_consumer] AI diagnosis completed. is_defect: {result.is_defect}")
        print(f"[kafka_consumer] Publishing diagnosis result...")
        
        # AiDiagnosisCompletedEventDTO 생성
        completed_event = AiDiagnosisCompletedEventDTO(
            audit_id=audit_id,
            inspection_id=inspection_id,
            inspection_type=inspection_type,
            is_defect=result.is_defect,
            collect_data_path=collect_data_path,
            result_data_path=result.result_data_path,
            diagnosis_result=result.diagnosis_result
        )
        
        # 타입 헤더가 포함된 새로운 publish 함수 사용
        publish_ai_diagnosis_completed(completed_event)
        
        print(f"[kafka_consumer] Process completed successfully for audit_id: {audit_id}")
        
    except Exception as e:
        print(f"[kafka_consumer] ERROR in _process for audit_id {audit_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 AiDiagnosisCompletedEventDTO 형식으로 전송
        error_completed_event = AiDiagnosisCompletedEventDTO(
            audit_id=audit_id,
            inspection_id=inspection_id,
            inspection_type=inspection_type,
            is_defect=False,
            collect_data_path=collect_data_path,
            result_data_path=None,
            diagnosis_result=json.dumps({"error": str(e)}, ensure_ascii=False)
        )
        
        publish_ai_diagnosis_completed(error_completed_event)

def run_consumer():
    consumer = KafkaConsumer(
        "test-started",
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        group_id="paint-defect-ai",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    
    print("[kafka_consumer] Starting Kafka consumer for paint-defect-ai...")
    for msg in consumer:
        try:
            _process(msg.value)
        except Exception as e:
            print(f"[kafka_consumer] Consumer error: {e}")

def start_background_consumer():
    t = threading.Thread(target=run_consumer, daemon=True)
    t.start()
    print("[kafka_consumer] Background consumer thread started")