"""프런티어 — 무엇을 언제 크롤할지 정한다. 신선도의 엔진.

세 갈래로 crawl.requests를 채운다 (설계 §4.2, §4.3):

  1. 실시간 : /sidebar.json 을 10초마다 폴링.
              실측상 항목 15개가 약 32초를 덮는 슬라이딩 윈도우라
              30초를 넘겨 폴링하면 그 사이 편집이 영구 유실된다.
              나무위키는 하루 33,600개 고유 문서가 바뀐다(실측).
  2. 재방문 : next_due_at 이 지난 문서. 1번이 놓친 편집을 결국 잡는다.
  3. 신규   : 아직 한 번도 수집하지 않은 문서(덤프 시드 + BFS 발견분).

단일 리더로 돌린다. 여러 개 띄우면 같은 문서를 중복 큐잉한다.

사용법:
    .venv/Scripts/python.exe services/frontier.py
    .venv/Scripts/python.exe services/frontier.py --drain   # 미수집분만 밀어넣고 종료
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.config import SETTINGS, Topics  # noqa: E402
from namuwiki.db import bump, connect  # noqa: E402
from namuwiki.ids import doc_id, normalize_title  # noqa: E402
from namuwiki.runtime import setup_logging  # noqa: E402
from namuwiki.kafka_io import make_producer, send  # noqa: E402

log = setup_logging("frontier")

SIDEBAR_URL = f"{SETTINGS.base_url}/sidebar.json"
# 큐가 이보다 길면 더 넣지 않는다. 크롤러가 소화하는 속도에 맞춘다.
INFLIGHT_LIMIT = 2000
NON_ARTICLE_NS = (
    "분류:", "틀:", "파일:", "사용자:", "나무위키:", "휴지통:",
    "투표:", "토론:", "위키운영:", "템플릿:",
)


def poll_sidebar() -> list[str]:
    """최근 변경된 문서 제목. robots.txt가 명시적으로 허용한 엔드포인트다."""
    req = urllib.request.Request(
        SIDEBAR_URL, headers={"User-Agent": SETTINGS.user_agent}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    return [
        d["document"]
        for d in data
        if d.get("document") and not d["document"].startswith(NON_ARTICLE_NS)
    ]


def enqueue(conn, producer, rows: list[tuple[str, str]], reason: str) -> int:
    """(doc_id, title) 목록을 crawl.requests로 보낸다. queued_at으로 중복을 막는다."""
    sent = 0
    for did, title in rows:
        send(producer, Topics.CRAWL_REQUESTS, did,
             {"doc_id": did, "title": title, "reason": reason})
        sent += 1
    if rows:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE crawl_state SET queued_at = now() WHERE doc_id = %s",
                [(r[0],) for r in rows],
            )
        producer.flush()
    return sent


def mark_changed(conn, titles: list[str]) -> list[tuple[str, str]]:
    """sidebar가 알려준 변경 문서를 즉시 만료시킨다. 없으면 새로 만든다."""
    out: list[tuple[str, str]] = []
    for raw in titles:
        title = normalize_title(raw)
        did = doc_id(title)
        conn.execute(
            """INSERT INTO crawl_state (doc_id, title, discovered_by, priority, next_due_at)
               VALUES (%s, %s, 'sidebar', 10, now())
               ON CONFLICT (doc_id) DO UPDATE
                 SET next_due_at = now(),
                     priority = LEAST(crawl_state.priority, 10),
                     fail_count = 0""",
            (did, title),
        )
        out.append((did, title))
    return out


def due_rows(conn, limit: int) -> list[tuple[str, str]]:
    """재방문 기한이 지났거나 아직 수집한 적 없는 문서."""
    rows = conn.execute(
        """SELECT doc_id, title FROM crawl_state
            WHERE fail_count < 5
              AND next_due_at <= now()
              AND (queued_at IS NULL OR queued_at < now() - interval '1 hour')
            ORDER BY priority, next_due_at
            LIMIT %s""",
        (limit,),
    ).fetchall()
    return [(r["doc_id"], r["title"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drain", action="store_true",
                    help="미수집분을 전부 큐잉하고 종료 (전수 크롤 시작용)")
    ap.add_argument("--limit", type=int, default=0, help="--drain 시 최대 건수")
    args = ap.parse_args()

    producer = make_producer()
    with connect() as conn:
        if args.drain:
            total = 0
            while True:
                take = 5000 if not args.limit else min(5000, args.limit - total)
                if take <= 0:
                    break
                rows = due_rows(conn, take)
                if not rows:
                    break
                total += enqueue(conn, producer, rows, "drain")
                log.info("큐잉 누적 %d", total)
            log.info("드레인 완료 — %d건", total)
            producer.close()
            return

        log.info("시작 — sidebar %d초 주기 폴링", SETTINGS.sidebar_poll_s)
        last_refill = 0.0
        try:
            while True:
                # 1) 실시간 변경 감지
                try:
                    titles = poll_sidebar()
                    rows = mark_changed(conn, titles)
                    n = enqueue(conn, producer, rows, "sidebar")
                    if n:
                        bump(conn, "frontier", "sidebar_enqueued", n)
                        log.info("sidebar 변경 %d건 큐잉", n)
                except Exception as e:  # noqa: BLE001
                    log.warning("sidebar 폴링 실패: %s", e)

                # 2) 재방문·신규 보충 (1분마다, 큐가 짧을 때만)
                now = time.monotonic()
                if now - last_refill > 60:
                    last_refill = now
                    inflight = conn.execute(
                        """SELECT count(*) AS c FROM crawl_state
                            WHERE queued_at > now() - interval '1 hour'
                              AND (last_fetched IS NULL OR last_fetched < queued_at)"""
                    ).fetchone()["c"]
                    if inflight < INFLIGHT_LIMIT:
                        rows = due_rows(conn, INFLIGHT_LIMIT - inflight)
                        if rows:
                            n = enqueue(conn, producer, rows, "due")
                            bump(conn, "frontier", "due_enqueued", n)
                            log.info("재방문·신규 %d건 큐잉 (대기 %d)", n, inflight)

                time.sleep(SETTINGS.sidebar_poll_s)
        except KeyboardInterrupt:
            log.info("중단 요청")
        finally:
            producer.flush()
            producer.close()


if __name__ == "__main__":
    main()
