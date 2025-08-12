from kafka import KafkaProducer
import json
from config import config

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