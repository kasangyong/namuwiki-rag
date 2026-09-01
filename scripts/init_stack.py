"""Postgres 스키마 · Kafka 토픽 · Qdrant 컬렉션을 만든다. 멱등하다.

사용법:
    .venv/Scripts/python.exe scripts/init_stack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient, models  # noqa: E402

from namuwiki.config import SETTINGS, TOPIC_SPECS  # noqa: E402
from namuwiki.db import init_schema  # noqa: E402
from namuwiki.kafka_io import ensure_topics  # noqa: E402


def init_qdrant() -> str:
    client = QdrantClient(url=SETTINGS.qdrant_url, timeout=60)
    name = SETTINGS.collection
    if client.collection_exists(name):
        return f"이미 있음: {name}"

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=SETTINGS.dense_dim,
                distance=models.Distance.COSINE,
                # 900만 청크 × 1024dim float32 = 36.9GB. 원본은 디스크에 두고
                # int8 양자화본만 RAM(9.2GB)에 올린다 (설계 §5.3).
                on_disk=True,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    )
                ),
            )
        },
        sparse_vectors_config={
            "lexical": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True)
            )
        },
        # Phase 3 다중 노드 확장을 대비해 미리 4샤드로 쪼갠다.
        # 나중에 늘리려면 재색인이 필요하므로 지금 정해야 한다.
        shard_number=4,
        on_disk_payload=True,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
    )
    # 필터 검색용 payload 인덱스
    for field, schema in [
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("title", models.PayloadSchemaType.KEYWORD),
        ("categories", models.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(name, field_name=field, field_schema=schema)
    return f"생성: {name}"


def backfill_canary() -> int:
    """기준선이 없는 카나리아를 이미 파싱된 본문 길이로 채운다.

    기준선이 0이면 비교가 불가능해 계속 오경보한다. 파서가 char_len을
    저장해두므로 재크롤 없이 채울 수 있다.
    """
    from namuwiki.db import connect

    with connect() as conn:
        return conn.execute(
            """UPDATE canary SET baseline_chars = d.char_len,
                                 last_chars = d.char_len,
                                 last_checked = now(),
                                 status = 'baseline'
                 FROM documents d
                WHERE d.title = canary.title
                  AND canary.baseline_chars <= 0
                  AND d.char_len > 0"""
        ).rowcount


def main() -> None:
    print("1) Postgres 스키마 …", flush=True)
    init_schema()
    n = backfill_canary()
    print(f"   OK (카나리아 기준선 백필 {n}건)", flush=True)

    print("2) Kafka 토픽 …", flush=True)
    created = ensure_topics()
    for name, parts, policy in TOPIC_SPECS:
        mark = "생성" if name in created else "이미 있음"
        print(f"   {name:<18} p={parts:<3} {policy:<8} {mark}", flush=True)

    print("3) Qdrant 컬렉션 …", flush=True)
    print(f"   {init_qdrant()}", flush=True)

    print("\n부트스트랩 완료.")


if __name__ == "__main__":
    main()
