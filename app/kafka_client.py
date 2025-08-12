import json
import logging
from typing import Dict, Any, Optional
from kafka import KafkaProducer
import app.config as config

logger = logging.getLogger("paint-ai")
_producer: Optional[KafkaProducer] = None

def get_producer() -> Optional[KafkaProducer]:
    global _producer
    if not config.KAFKA_ENABLED:
        return None
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS.split(","),
            client_id=config.KAFKA_CLIENT_ID,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda v: (str(v).encode("utf-8") if v is not None else None),
            linger_ms=5,
            acks="all",
        )
    return _producer

def publish(topic: str, value: Dict[str, Any], key: Optional[str] = None) -> bool:
    producer = get_producer()
    if producer is None:
        logger.info("[Kafka] disabled; skip publish")
        return False
    try:
        fut = producer.send(topic, key=key, value=value)
        meta = fut.get(timeout=10)
        logger.info(f"[Kafka] sent to {meta.topic}@{meta.partition} offset={meta.offset}")
        return True
    except Exception as e:
        logger.error(f"[Kafka] publish failed: {e}")
        return False
