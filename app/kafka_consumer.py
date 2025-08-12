import threading
import json
import time
from kafka import KafkaConsumer
from config import config
from schemas import TestStartedEventDTO
from inference import process_ai_diagnosis
from kafka_producer import publish_diagnosis

def safe_json_loads(v):
   try:
       if v:
           return json.loads(v.decode("utf-8"))
       return None
   except:
       return None

def _process(event: dict):
  try:
      test_event = TestStartedEventDTO(**event)
      
      result = process_ai_diagnosis(test_event)
      
      publish_diagnosis(
          key_audit_id=test_event.audit_id,
          payload=result.dict(by_alias=True, exclude_none=True)
      )
      
  except Exception as e:
      audit_id = event.get("auditId", "unknown")
      inspection_id = event.get("inspectionId", "unknown")
      
      error_payload = {
          "auditId": audit_id,
          "inspectionId": inspection_id,
          "inspectionType": event.get("inspectionType", "PAINT_DEFECT"),
          "isDefect": False,
          "collectDataPath": event.get("collectDataPath", ""),
          "resultDataPath": None,
          "diagnosisResult": json.dumps({"error": str(e)}, ensure_ascii=False)
      }
      
      publish_diagnosis(
          key_audit_id=audit_id,
          payload=error_payload
      )

def run_consumer():
    print("Kafka Consumer 시작...")
    consumer = KafkaConsumer(
        "test-started",
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        group_id="paint-defect-ai",
        value_deserializer=safe_json_loads,
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    
    print("메시지 대기 중...")
    for msg in consumer:
        print(f"메시지 수신: {msg}")
        try:
            if msg.value:
                print(f"메시지 내용: {msg.value}")
                _process(msg.value)
            else:
                print("빈 메시지 수신")
        except Exception as e:
            print(f"메시지 처리 오류: {e}")

def start_background_consumer():
  t = threading.Thread(target=run_consumer, daemon=True)
  t.start()