import threading
import json
import time
from kafka import KafkaConsumer
from app.config import config
from app.schemas import TestStartedEventDTO
from app.inference import process_ai_diagnosis
from app.kafka_producer import publish_diagnosis

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
   consumer = KafkaConsumer(
       "test-started",
       bootstrap_servers=config.KAFKA_BOOTSTRAP,
       group_id="paint-defect-ai",
       value_deserializer=lambda v: json.loads(v.decode("utf-8")),
       key_deserializer=lambda k: k.decode("utf-8") if k else None,
       auto_offset_reset="earliest",
       enable_auto_commit=True,
   )
   
   for msg in consumer:
       try:
           _process(msg.value)
       except Exception as e:
           pass

def start_background_consumer():
   t = threading.Thread(target=run_consumer, daemon=True)
   t.start()