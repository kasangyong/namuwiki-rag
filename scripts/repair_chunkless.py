"""청크 없는 문서를 블롭에서 직접 복구한다.

문서 행은 있는데 청크가 0인 상태는 조용해서 위험하다. 검색에서 그 문서가
통째로 빠지는데 아무 오류도 나지 않는다. 실제로 가장 큰 문서 15개
(미국·일본·한국사·삼성전자…)가 이 상태로 누락돼 있었다.

원인은 여러 가지일 수 있다 — 과거의 청크 INSERT 실패, 스키마 변경 중 실패,
스트리밍 경로에서의 누락. 원인을 특정하지 못해도 **복구는 결정적으로 가능하다.**
원본 HTML이 블롭 저장소에 남아 있기 때문이다 (설계 §5.5 클레임 체크).

Kafka를 거치지 않고 블롭 → 추출 → 청킹 → DB + chunks.ready 순으로 직접 처리한다.

사용법:
    .venv/Scripts/python.exe scripts/repair_chunkless.py --dry-run
    .venv/Scripts/python.exe scripts/repair_chunkless.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.kafka_io import make_producer  # noqa: E402


def find_chunkless(conn) -> list[dict]:
    return conn.execute(
        """SELECT doc_id, title, char_len, blob_ref, content_hash, source, fetched_at
             FROM documents d
            WHERE d.char_len > 0
              AND d.blob_ref IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)
            ORDER BY char_len DESC"""
    ).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import parser as parser_svc  # 파서의 handle()을 그대로 재사용한다

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    producer = make_producer()
    ok = failed = 0

    with connect() as conn:
        rows = find_chunkless(conn)
        print(f"청크 없는 문서 {len(rows)}건\n", flush=True)
        if not rows:
            return
        for r in rows:
            if args.dry_run:
                print(f"  [건너뜀] {r['title']:<16} {r['char_len']:>8,}자")
                continue
            msg = {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "source": r["source"],
                "content_hash": r["content_hash"],
                "blob_ref": r["blob_ref"],
                "fetched_at": r["fetched_at"].isoformat(),
            }
            try:
                result = parser_svc.handle(msg, conn, producer, blobs)
                n = conn.execute(
                    "SELECT count(*) c FROM chunks WHERE doc_id = %s", (r["doc_id"],)
                ).fetchone()["c"]
                if n > 0:
                    ok += 1
                    print(f"  [복구] {r['title']:<16} {r['char_len']:>8,}자 → 청크 {n}")
                else:
                    failed += 1
                    print(f"  [실패] {r['title']:<16} handle()={result}, 청크 0")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  [오류] {r['title']:<16} {type(e).__name__}: {e}")
        producer.flush()

        left = len(find_chunkless(conn))
    print(f"\n복구 {ok}건 / 실패 {failed}건 / 남은 청크없는문서 {left}건")


if __name__ == "__main__":
    main()
