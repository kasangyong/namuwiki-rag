"""임베딩되지 않은 채 남은 청크를 chunks.ready 로 다시 발행한다.

임베더가 배치를 실패하면 그 메시지는 DLQ 로 보내고 오프셋은 커밋한다.
크래시로 커밋 자체가 유실되면 재전달로 저절로 복구되지만, 커밋이 성공한
경우에는 아무도 그 청크를 다시 주지 않는다. DB 에는 행이 있고
embedded_at 만 NULL 인 상태로 조용히 남아 검색에서 빠진다.

실제로 1,081,931 청크 중 140개가 이렇게 남았다 (Qdrant upsert 타임아웃으로
컨슈머가 축출된 밤에 생긴 것).

청크 본문이 DB 에 그대로 있으므로 재추출 없이 그대로 다시 발행하면 된다.
임베더는 멱등이라 (uuid5 포인트 ID) 중복 실행도 안전하다.

사용법:
    .venv/Scripts/python.exe scripts/repair_unembedded.py --dry-run
    .venv/Scripts/python.exe scripts/repair_unembedded.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.config import Topics  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.kafka_io import make_producer, send  # noqa: E402


def find_unembedded(conn) -> list[dict]:
    return conn.execute(
        """SELECT c.chunk_id, c.doc_id, c.seq, c.section_id, c.heading_path,
                  c.text, c.content_hash,
                  d.title, coalesce(d.categories, '{}') AS categories
             FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.embedded_at IS NULL
            ORDER BY c.doc_id, c.seq"""
    ).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        rows = find_unembedded(conn)

    if not rows:
        print("임베딩 안 된 청크 없음 - 할 일 없다")
        return

    titles = sorted({r["title"] for r in rows})
    print(f"임베딩 안 된 청크 {len(rows):,}개 / 문서 {len(titles)}개")
    for t in titles[:20]:
        n = sum(1 for r in rows if r["title"] == t)
        print(f"  {t[:50]:<50} {n:>4}청크")
    if len(titles) > 20:
        print(f"  … 외 {len(titles) - 20}개 문서")

    if args.dry_run:
        print("\n--dry-run — 발행하지 않음")
        return

    producer = make_producer()
    for r in rows:
        send(producer, Topics.CHUNKS_READY, r["doc_id"], {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "title": r["title"],
            "seq": r["seq"],
            "section_id": r["section_id"],
            "heading_path": r["heading_path"],
            "text": r["text"],
            "content_hash": r["content_hash"],
            "categories": list(r["categories"]),
        })
    producer.flush()
    producer.close()
    print(f"\n{len(rows):,}청크 재발행 완료 — 임베더를 켜면 처리된다")


if __name__ == "__main__":
    main()
