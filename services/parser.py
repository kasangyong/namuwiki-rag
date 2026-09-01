"""파서 — docs.raw 소비 → 본문 추출 → 청킹 → chunks.ready 발행.

원본 HTML은 블롭 저장소에 있으므로, 파싱 로직을 고쳤을 때 재크롤 없이
docs.raw를 오프셋 0부터 재생하면 전량 재파싱된다 (설계 §8).

추출 수율 카나리아(§6.2)를 내장한다. 나무위키의 CSS 클래스명이 난독화되어
있어 추출기는 언젠가 조용히 깨진다. 감시하지 않으면 빈 문자열로 900만 청크를
오염시켜도 알아채지 못한다.

사용법:
    .venv/Scripts/python.exe services/parser.py [--once]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.chunker import chunk_document  # noqa: E402
from namuwiki.config import SETTINGS, Topics  # noqa: E402
from namuwiki.db import bump, connect  # noqa: E402
from namuwiki.crawl import iter_links  # noqa: E402
from namuwiki.extract import ExtractionError, extract  # noqa: E402
from namuwiki.ids import doc_id, normalize_title  # noqa: E402
from namuwiki.kafka_io import make_consumer, make_producer, send  # noqa: E402
from namuwiki.runtime import PollLoop, setup_logging  # noqa: E402

log = setup_logging("parser")

GROUP = "parser"


def _canary_title(conn, title: str) -> str | None:
    """이 문서가 감시 대상인지 판정한다. 별칭도 정식 제목으로 따라간다.

    '파이썬'이 'Python'으로 접힌 뒤에도 카나리아가 '파이썬'을 추적하면
    문서가 없는 제목을 재는 셈이라 영구 오경보가 난다.
    """
    row = conn.execute("SELECT 1 FROM canary WHERE title = %s", (title,)).fetchone()
    if row:
        return title
    row = conn.execute(
        """SELECT c.title FROM canary c
             JOIN redirects r ON r.canonical = %s
            WHERE c.title = r.alias""",
        (title,),
    ).fetchone()
    return row["title"] if row else None


def check_canary(conn, title: str, chars: int) -> None:
    """기준선 대비 ±30% 이탈 시 경보. 추출기 무성 실패 감지 (설계 §6.2)."""
    title = _canary_title(conn, title) or title
    row = conn.execute(
        "SELECT baseline_chars FROM canary WHERE title = %s", (title,)
    ).fetchone()
    if not row:
        return
    base = row["baseline_chars"]
    if base <= 0:
        # 첫 성공 파싱 결과를 기준선으로 삼는다. 기준선이 없으면 비교 자체가
        # 불가능하므로 경보하지 않는다.
        conn.execute(
            """UPDATE canary SET baseline_chars = %s, last_chars = %s,
                      last_checked = now(), status = 'baseline'
                WHERE title = %s""",
            (chars, chars, title),
        )
        log.info("카나리아 기준선 확립: %r = %d자", title, chars)
        return
    lo, hi = base * (1 - SETTINGS.canary_tolerance), base * (1 + SETTINGS.canary_tolerance)
    status = "ok" if lo <= chars <= hi else "ALERT"
    conn.execute(
        """UPDATE canary SET last_chars = %s, last_checked = now(), status = %s
            WHERE title = %s""",
        (chars, status, title),
    )
    if status == "ALERT":
        log.error(
            "카나리아 경보! %r 추출 길이 %d자 (기준 %d자). 추출기가 깨졌을 수 있다.",
            title, chars, base,
        )


def delete_document(conn, producer, doc_id: str, title: str) -> int:
    """삭제·이동된 문서의 청크를 제거하고 임베더에 tombstone을 흘린다."""
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = %s", (doc_id,)
    ).fetchall()
    for r in rows:
        send(producer, Topics.CHUNKS_READY, doc_id,
             {"chunk_id": r["chunk_id"], "doc_id": doc_id, "deleted": True})
    conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
    if rows:
        log.info("삭제 전파: %r (%d청크)", title, len(rows))
    return len(rows)


def handle(msg: dict, conn, producer, blobs) -> str:
    did = msg["doc_id"]
    title = msg["title"]

    if msg.get("deleted"):
        delete_document(conn, producer, did, title)
        return "deleted"

    # 이미 같은 해시로 파싱했으면 건너뛴다 (멱등).
    # 단 카나리아 문서는 예외다. 추출기가 깨졌는지 감시하려면 문서 내용이
    # 그대로여도 매번 다시 추출해 길이를 재봐야 한다. 단축 경로를 타면
    # 안 바뀌는 문서에서는 카나리아가 영영 재검증되지 않는다. (문서 5개뿐)
    is_canary = _canary_title(conn, title) is not None
    # 해시가 같아도 **청크가 실제로 있는지** 확인한다. 문서 행만 남고 청크가
    # 없는 상태(과거 청크 INSERT 실패의 흔적)에서 해시만 보고 건너뛰면
    # 그 문서는 영원히 복구되지 않는다. 실제로 그렇게 큰 문서 15개가
    # 조용히 누락됐던 적이 있다.
    row = conn.execute(
        """SELECT d.content_hash,
                  EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id) AS has_chunks
             FROM documents d WHERE d.doc_id = %s""",
        (did,),
    ).fetchone()
    if (
        row
        and row["content_hash"] == msg["content_hash"]
        and row["has_chunks"]
        and not is_canary
    ):
        return "unchanged"

    raw = blobs.get(msg["blob_ref"])
    html_text = raw.decode("utf-8", errors="replace")
    try:
        ex = extract(html_text)
    except ExtractionError as e:
        send(producer, Topics.PARSE_DLQ, did, {"title": title, "error": str(e)})
        log.warning("추출 실패 %r: %s", title, e)
        return "failed"

    body_len = len(ex.text)
    check_canary(conn, title, body_len)
    if body_len < 50:
        send(producer, Topics.PARSE_DLQ, did,
             {"title": title, "error": f"본문이 너무 짧다: {body_len}자"})
        return "failed"

    # 리다이렉트 접기 (설계 §13.1 SAME_AS).
    # 크롤러는 리다이렉트를 따라가므로 '픽사'를 요청해도 '픽사 애니메이션
    # 스튜디오'의 내용이 온다. 요청한 제목으로 저장하면 정식 제목까지 수집됐을
    # 때 같은 내용이 두 문서가 되어 임베딩을 두 번 하고 그래프 노드도 갈라진다.
    # 표본 300개 중 5.3%가 이 경우였다.
    canonical = normalize_title(ex.title) if ex.title else title
    if canonical != title:
        conn.execute(
            """INSERT INTO redirects (alias, canonical) VALUES (%s, %s)
               ON CONFLICT (alias) DO UPDATE SET canonical = EXCLUDED.canonical""",
            (title, canonical),
        )
        log.info("리다이렉트: %r → %r", title, canonical)
        # 정식 제목을 문서 정체성으로 삼는다. 별칭으로는 문서를 만들지 않는다.
        did = doc_id(canonical)
        title = canonical

    chunks = chunk_document(did, title, ex)
    # 지식그래프 엣지. 크롤러도 링크를 뽑지만 그건 BFS용으로 즉시 소비되고
    # 버려진다. 여기서 뽑아 저장해야 그래프 질의(이웃 확장·역링크·PageRank)에
    # 쓸 수 있다. 원본 HTML이 블롭에 남아 있으므로 나중에 재생해도 복구된다.
    # 본문 구간에서만 링크를 뽑는다. 전체 HTML을 쓰면 푸터의 라이선스
    # 링크와 툴바가 섞여 PageRank 상위를 오염시킨다.
    outlinks = sorted(iter_links(ex.link_html or html_text))

    # 문서 행과 청크는 **한 트랜잭션**으로 쓴다. autocommit 상태에서 따로 쓰면
    # 청크 INSERT가 실패했을 때 문서 행만 커밋되어, 청크 없는 문서가 남는다.
    # 그 상태에서 재파싱하면 해시가 같아 건너뛰므로 영구 손실이 된다.
    with conn.transaction():
        _write(conn, producer, msg, did, title, ex, chunks, body_len, outlinks)
    return "parsed"


def _write(conn, producer, msg, did, title, ex, chunks, body_len, outlinks) -> None:
    conn.execute(
        """INSERT INTO documents
             (doc_id, title, source, content_hash, blob_ref, char_len,
              last_modified, fetched_at, parsed_at, categories, outlinks)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now(),%s,%s)
           ON CONFLICT (doc_id) DO UPDATE SET
             content_hash = EXCLUDED.content_hash,
             blob_ref     = EXCLUDED.blob_ref,
             char_len     = EXCLUDED.char_len,
             last_modified= EXCLUDED.last_modified,
             fetched_at   = EXCLUDED.fetched_at,
             parsed_at    = now(),
             categories   = EXCLUDED.categories,
             outlinks     = EXCLUDED.outlinks,
             source       = EXCLUDED.source""",
        (did, title, msg.get("source", "crawl"), msg["content_hash"],
         msg.get("blob_ref"), body_len, ex.last_modified, msg["fetched_at"],
         ex.categories, outlinks),
    )

    # 재파싱으로 청크 수가 줄면 꼬리 청크가 남는다. 먼저 지운다.
    stale = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = %s AND seq >= %s",
        (did, len(chunks)),
    ).fetchall()
    for r in stale:
        send(producer, Topics.CHUNKS_READY, did,
             {"chunk_id": r["chunk_id"], "doc_id": did, "deleted": True})
    conn.execute(
        "DELETE FROM chunks WHERE doc_id = %s AND seq >= %s", (did, len(chunks))
    )

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks
                 (chunk_id, doc_id, seq, section_id, heading_path, text, content_hash)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO UPDATE SET
                 section_id   = EXCLUDED.section_id,
                 heading_path = EXCLUDED.heading_path,
                 text         = EXCLUDED.text,
                 content_hash = EXCLUDED.content_hash,
                 embedded_at  = CASE WHEN chunks.content_hash = EXCLUDED.content_hash
                                     THEN chunks.embedded_at ELSE NULL END""",
            [(c.chunk_id, c.doc_id, c.seq, c.section_id, c.heading_path,
              c.text, c.content_hash) for c in chunks],
        )

    for c in chunks:
        send(producer, Topics.CHUNKS_READY, did, {
            "chunk_id": c.chunk_id,
            "doc_id": did,
            "title": ex.title or title,
            "seq": c.seq,
            "section_id": c.section_id,
            "heading_path": c.heading_path,
            "text": c.text,
            "content_hash": c.content_hash,
            "categories": ex.categories,
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="큐가 비면 종료")
    ap.add_argument("--from-beginning", action="store_true",
                    help="오프셋 0부터 재생 (전량 재파싱)")
    args = ap.parse_args()

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    producer = make_producer()
    consumer = make_consumer(
        Topics.DOCS_RAW, group_id=GROUP,
        from_beginning=True, max_poll_records=50,
    )
    if args.from_beginning:
        consumer.poll(timeout_ms=3000)
        consumer.seek_to_beginning()
        log.info("오프셋 0부터 재생 — 전량 재파싱")

    tally = {"parsed": 0, "unchanged": 0, "failed": 0, "deleted": 0}
    log.info("시작")
    try:
        with connect() as conn:
            for records in PollLoop(consumer, once=args.once).batches():
                if not records:
                    continue
                n = 0
                for rec in records:
                    try:
                        tally[handle(rec.value, conn, producer, blobs)] += 1
                    except Exception as e:  # noqa: BLE001
                        tally["failed"] += 1
                        log.exception("처리 실패: %s", e)
                        send(producer, Topics.PARSE_DLQ, rec.key or "?",
                             {"error": f"{type(e).__name__}: {e}"})
                    n += 1
                producer.flush()
                consumer.commit()
                bump(conn, "parser", "parsed", tally["parsed"])
                log.info("배치 %d건 | %s", n, tally)
    except KeyboardInterrupt:
        log.info("중단 요청")
    finally:
        producer.flush()
        consumer.close()
        producer.close()
        log.info("종료 — %s", tally)


if __name__ == "__main__":
    main()
