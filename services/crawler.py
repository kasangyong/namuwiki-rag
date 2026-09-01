"""크롤러 — crawl.requests 소비 → HTML 수집 → docs.raw 발행 + 링크 수확.

컨슈머 그룹으로 여러 개 띄울 수 있지만, 성능이 아니라 장애 격리가 목적이다.
동시 요청 수는 프로세스 수와 무관하게 정책으로 묶여 있다 (설계 §3.2, §4.3).

사용법:
    .venv/Scripts/python.exe services/crawler.py [--once N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.config import SETTINGS, Topics  # noqa: E402
from namuwiki.crawl import (  # noqa: E402
    CircuitBreaker,
    CircuitOpen,
    RateLimiter,
    fetch_document,
    iter_links,
    make_client,
)
from namuwiki.db import bump, connect  # noqa: E402
from namuwiki.ids import content_hash, doc_id, normalize_title  # noqa: E402
from namuwiki.kafka_io import make_consumer, make_producer, send  # noqa: E402
from namuwiki.runtime import PollLoop, setup_logging  # noqa: E402

log = setup_logging("crawler")

GROUP = "crawler"
BATCH = 32


async def fetch_batch(titles: list[str], limiter, breaker) -> list:
    async with make_client() as client:
        sem = asyncio.Semaphore(SETTINGS.crawl_concurrency)

        async def one(t: str):
            async with sem:
                try:
                    return await fetch_document(client, t, limiter, breaker)
                except CircuitOpen:
                    return None

        return await asyncio.gather(*(one(t) for t in titles))


def handle_results(results, conn, producer, blobs, seen_titles: set[str]) -> dict:
    stats = {"ok": 0, "missing": 0, "failed": 0, "unchanged": 0, "discovered": 0}
    for r in results:
        if r is None:
            stats["failed"] += 1
            continue
        title = normalize_title(r.title)
        did = doc_id(title)

        if r.status == 404:
            # 삭제되었거나 다른 제목으로 이동한 문서. 옛 내용을 색인에 남겨두면
            # 존재하지 않는 문서를 근거로 답변하게 되므로 하위 단계에 삭제를
            # 지시한다 (tombstone). Kafka compact 토픽에서 value=None은
            # "이 키를 지워라"라는 뜻이라 재생 시에도 삭제가 재현된다.
            stats["missing"] += 1
            conn.execute(
                """UPDATE crawl_state SET last_fetched = now(), http_status = 404,
                       next_due_at = now() + interval '90 days', queued_at = NULL
                   WHERE doc_id = %s""",
                (did,),
            )
            send(producer, Topics.DOCS_RAW, did, {
                "doc_id": did,
                "title": title,
                "source": "crawl",
                "fetched_at": datetime.now(UTC).isoformat(),
                "deleted": True,
                "http_status": 404,
            })
            continue

        if r.html is None:
            stats["failed"] += 1
            conn.execute(
                """UPDATE crawl_state
                      SET fail_count = fail_count + 1,
                          http_status = %s,
                          next_due_at = now() + (interval '10 minutes' * (fail_count + 1)),
                          queued_at = NULL
                    WHERE doc_id = %s""",
                (r.status, did),
            )
            send(producer, Topics.CRAWL_DLQ, did,
                 {"title": title, "status": r.status, "error": r.error})
            continue

        html_text = r.html.decode("utf-8", errors="replace")
        chash = content_hash(html_text)

        row = conn.execute(
            "SELECT last_hash FROM crawl_state WHERE doc_id = %s", (did,)
        ).fetchone()
        if row and row["last_hash"] == chash:
            # 내용이 그대로면 하위 단계로 흘리지 않는다 (설계 §8 멱등성)
            stats["unchanged"] += 1
            conn.execute(
                """UPDATE crawl_state SET last_fetched = now(), http_status = 200,
                          fail_count = 0, queued_at = NULL,
                          next_due_at = now() + interval '14 days'
                    WHERE doc_id = %s""",
                (did,),
            )
            continue

        ref = blobs.put(did, r.html)
        send(producer, Topics.DOCS_RAW, did, {
            "doc_id": did,
            "title": title,
            "source": "crawl",
            "fetched_at": datetime.now(UTC).isoformat(),
            "content_hash": chash,
            "blob_ref": ref,
            "http_status": 200,
            "byte_size": len(r.html),
        })
        conn.execute(
            """UPDATE crawl_state
                  SET last_fetched = now(), last_hash = %s, http_status = 200,
                      fail_count = 0, queued_at = NULL,
                      next_due_at = now() + interval '14 days'
                WHERE doc_id = %s""",
            (chash, did),
        )
        stats["ok"] += 1

        # BFS: 링크 수확 → 미발견 문서를 프런티어에 넣는다 (설계 §4.2)
        new = iter_links(html_text) - seen_titles
        if new:
            seen_titles.update(new)
            rows = [(doc_id(t), normalize_title(t)) for t in new]
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO crawl_state (doc_id, title, discovered_by, priority)
                       VALUES (%s, %s, 'bfs', 200)
                       ON CONFLICT (doc_id) DO NOTHING""",
                    rows,
                )
            stats["discovered"] += len(rows)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", type=int, default=0,
                    help="N개 처리 후 종료 (0이면 무한)")
    args = ap.parse_args()

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    limiter = RateLimiter(SETTINGS.crawl_min_interval_ms / 1000)
    breaker = CircuitBreaker(SETTINGS.breaker_threshold, SETTINGS.breaker_cooldown_s)
    producer = make_producer()
    consumer = make_consumer(
        Topics.CRAWL_REQUESTS, group_id=GROUP, from_beginning=True,
        max_poll_records=BATCH,
    )
    seen: set[str] = set()
    total = 0
    log.info("시작 — 동시 %d, 최소간격 %dms",
             SETTINGS.crawl_concurrency, SETTINGS.crawl_min_interval_ms)

    try:
        with connect() as conn:
            for records in PollLoop(consumer, once=bool(args.once)).batches():
                if not records:
                    continue
                titles = [rec.value["title"] for rec in records]
                if breaker.is_open:
                    log.warning("서킷 열림 — %.0f초 대기", breaker.remaining())
                    import time as _t
                    _t.sleep(min(breaker.remaining(), 30))
                    continue

                results = asyncio.run(fetch_batch(titles, limiter, breaker))
                stats = handle_results(results, conn, producer, blobs, seen)
                producer.flush()
                consumer.commit()  # 처리 완료 후에만 커밋
                bump(conn, "crawler", "fetched", stats["ok"])
                bump(conn, "crawler", "discovered", stats["discovered"])
                total += len(titles)
                log.info("배치 %d건 | %s | 누적 %d", len(titles), stats, total)

                if args.once and total >= args.once:
                    break
    except KeyboardInterrupt:
        log.info("중단 요청")
    finally:
        producer.flush()
        consumer.close()
        producer.close()
        log.info("종료 — 총 %d건", total)


if __name__ == "__main__":
    main()
