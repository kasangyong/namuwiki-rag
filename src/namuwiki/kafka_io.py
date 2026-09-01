"""Kafka 헬퍼 (confluent-kafka / librdkafka 기반).

왜 kafka-python이 아닌가:
    kafka-python은 Python 3.14의 selectors 모듈과 맞지 않는다. 소켓이 닫힌 뒤
    `selector.unregister()`가 `ValueError: Invalid file descriptor: -1`로 터져
    컨슈머 폴 루프가 죽는다. 그룹 없는 소비는 되는데 컨슈머 그룹 소비만
    조용히 0건을 반환하는 형태로 나타나 원인을 찾기 어렵다.
    confluent-kafka는 C 라이브러리(librdkafka)라 이 문제가 없다.

핵심 규칙 (설계 §5.1): 모든 메시지의 키는 doc_id다. 같은 문서가 항상 같은
파티션으로 가야 순서가 보장되고, 구버전이 신버전을 덮어쓰는 사고가 막힌다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from .config import SETTINGS, TOPIC_SPECS

log = logging.getLogger(__name__)

__all__ = [
    "ensure_topics", "make_producer", "make_consumer", "send", "Record",
]


@dataclass
class Record:
    key: str | None
    value: Any
    topic: str
    partition: int
    offset: int


def ensure_topics(bootstrap: str | None = None) -> list[str]:
    """토픽을 선언한다. 이미 있으면 건너뛴다 (멱등)."""
    admin = AdminClient({"bootstrap.servers": bootstrap or SETTINGS.kafka_bootstrap})
    existing = set(admin.list_topics(timeout=15).topics)

    new = []
    for name, partitions, policy in TOPIC_SPECS:
        if name in existing:
            continue
        cfg = {"cleanup.policy": policy, "max.message.bytes": "10485760"}
        if policy == "compact":
            # 재파싱·재임베딩의 원천이므로 영구 보존한다
            cfg["retention.ms"] = "-1"
        new.append(NewTopic(name, num_partitions=partitions,
                            replication_factor=1, config=cfg))
    if not new:
        return []

    created = []
    for name, fut in admin.create_topics(new).items():
        try:
            fut.result()
            created.append(name)
        except KafkaException as e:
            if e.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
    return created


class _Producer:
    """kafka-python과 호환되는 최소 인터페이스."""

    def __init__(self, bootstrap: str) -> None:
        self._p = Producer({
            "bootstrap.servers": bootstrap,
            "acks": "all",  # compact 토픽이 재처리의 원천이라 유실을 허용하지 않는다
            "linger.ms": 50,
            "compression.type": "gzip",
            "message.max.bytes": 10485760,
            "retries": 5,
            "enable.idempotence": True,
        })

    def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        while True:
            try:
                self._p.produce(topic, key=key.encode("utf-8"), value=payload)
                return
            except BufferError:
                # 로컬 큐가 찼다. 배출될 때까지 기다린다.
                self._p.poll(0.5)

    def flush(self, timeout: float = 30.0) -> int:
        return self._p.flush(timeout)

    def close(self) -> None:
        self._p.flush(30.0)


class _Consumer:
    """kafka-python 스타일 poll()을 제공하는 어댑터.

    confluent-kafka의 poll()은 메시지 하나를 주므로, 서비스 코드가 기대하는
    '배치' 형태로 모아준다.
    """

    def __init__(self, topics: list[str], group_id: str, bootstrap: str,
                 from_beginning: bool, max_poll_records: int) -> None:
        self.max_poll_records = max_poll_records
        self._c = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest" if from_beginning else "latest",
            # 처리 완료 후에만 커밋한다. 크래시 시 재처리되지만 모든 단계가
            # 멱등이므로 중복이 결과를 바꾸지 않는다 (설계 §8).
            "enable.auto.commit": False,
            "fetch.max.bytes": 52428800,
            "max.partition.fetch.bytes": 10485760,
            # 임베딩 배치가 오래 걸려도 리밸런스로 쫓겨나지 않게 넉넉히 준다
            "max.poll.interval.ms": 900000,
            "session.timeout.ms": 45000,
        })
        self._c.subscribe(topics)

    def poll(self, timeout_ms: int = 2000) -> dict[tuple[str, int], list[Record]]:
        out: dict[tuple[str, int], list[Record]] = {}
        deadline = timeout_ms / 1000
        n = 0
        first = self._c.poll(deadline)
        msgs = [first] if first is not None else []
        while n + len(msgs) < self.max_poll_records:
            m = self._c.poll(0)  # 남은 건 즉시 회수
            if m is None:
                break
            msgs.append(m)
        for m in msgs:
            if m is None:
                continue
            if m.error():
                if m.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.warning("컨슈머 오류: %s", m.error())
                continue
            key = m.key().decode("utf-8") if m.key() else None
            try:
                value = json.loads(m.value().decode("utf-8"))
            except Exception:  # noqa: BLE001
                log.warning("역직렬화 실패 %s[%d]@%d", m.topic(), m.partition(), m.offset())
                continue
            out.setdefault((m.topic(), m.partition()), []).append(
                Record(key, value, m.topic(), m.partition(), m.offset())
            )
            n += 1
        return out

    def assignment(self) -> list:
        return self._c.assignment()

    def commit(self) -> None:
        try:
            self._c.commit(asynchronous=False)
        except KafkaException as e:
            code = e.args[0].code()
            if code == KafkaError._NO_OFFSET:
                return
            if code == KafkaError._ASSIGNMENT_LOST:
                # 처리가 너무 오래 걸려 그룹에서 축출된 뒤의 커밋 시도.
                # 커밋되지 않은 메시지는 재전달되고 모든 단계가 멱등이므로
                # 중복 처리는 결과를 바꾸지 않는다. 죽을 이유가 없다.
                log.warning(
                    "파티션 할당을 잃어 커밋하지 못했다. "
                    "해당 메시지는 재전달된다 (멱등이라 안전)."
                )
                return
            raise

    def seek_to_beginning(self) -> None:
        from confluent_kafka import TopicPartition, OFFSET_BEGINNING

        parts = [
            TopicPartition(tp.topic, tp.partition, OFFSET_BEGINNING)
            for tp in self._c.assignment()
        ]
        for tp in parts:
            self._c.seek(tp)

    def close(self) -> None:
        self._c.close()


def make_producer(bootstrap: str | None = None) -> _Producer:
    return _Producer(bootstrap or SETTINGS.kafka_bootstrap)


def make_consumer(*topics: str, group_id: str, bootstrap: str | None = None,
                  from_beginning: bool = False,
                  max_poll_records: int = 100) -> _Consumer:
    return _Consumer(
        list(topics), group_id, bootstrap or SETTINGS.kafka_bootstrap,
        from_beginning, max_poll_records,
    )


def send(producer: _Producer, topic: str, key: str, value: dict[str, Any]) -> None:
    producer.send(topic, key, value)
