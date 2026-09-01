"""시드 로더 — 프런티어에 출발점을 넣는다.

설계 §4.1에서 덤프의 역할을 재정의했다. 덤프 텍스트는 헤딩 구조가 제거되어
있고 4년 묵었으며 현재 문서의 46%만 덮으므로 **콘텐츠 소스로 쓰지 않는다**.
쓰임새는 565,293개의 제목 시드뿐이다.

두 가지 모드:
  --from-dump    : HuggingFace 덤프를 받아 제목 565K개를 적재 (전수 수집용)
  --from-starter : 내장 시드 목록만 적재 (소규모 시험용).
                   문서당 링크가 1,000~1,500개라 BFS가 금방 퍼진다.

사용법:
    .venv/Scripts/python.exe services/seed_loader.py --from-starter
    .venv/Scripts/python.exe services/seed_loader.py --from-dump
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.db import connect  # noqa: E402
from namuwiki.ids import doc_id, normalize_title  # noqa: E402
from namuwiki.runtime import setup_logging  # noqa: E402

log = setup_logging("seed")

DUMP_REPO = "heegyu/namuwiki-extracted"

# 링크 밀도가 높은 허브 문서들. BFS가 여기서 빠르게 퍼진다.
STARTER = [
    "대한민국", "한국사", "서울특별시", "부산광역시", "제주도",
    "김연아", "손흥민", "아이유", "삼성전자", "네이버",
    "파이썬", "컴퓨터", "인공지능", "수학", "물리학",
    "지구", "우주", "역사", "영화", "음악",
    "축구", "야구", "게임", "만화", "애니메이션",
    "일본", "미국", "중국", "유럽", "아시아",
]

# 추출 수율 카나리아 기준 문서 (설계 §6.2)
CANARY_TITLES = ["대한민국", "김연아", "파이썬", "서울특별시", "축구"]


def load_titles(conn, titles, source: str, priority: int = 100) -> int:
    rows = []
    seen = set()
    for t in titles:
        t = normalize_title(t)
        if not t or t in seen:
            continue
        seen.add(t)
        rows.append((doc_id(t), t, source, priority))
    inserted = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), 5000):
            chunk = rows[i : i + 5000]
            cur.executemany(
                """INSERT INTO crawl_state (doc_id, title, discovered_by, priority)
                   VALUES (%s,%s,%s,%s) ON CONFLICT (doc_id) DO NOTHING""",
                chunk,
            )
            inserted += len(chunk)
            if i and i % 50000 == 0:
                log.info("  적재 %d/%d", i, len(rows))
    return inserted


def dump_titles() -> list[str]:
    """HuggingFace 덤프에서 제목만 뽑는다. 본문은 쓰지 않는다."""
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    log.info("덤프 내려받는 중 (약 2.4GB, 제목만 쓴다)…")
    local = snapshot_download(
        repo_id=DUMP_REPO, repo_type="dataset", allow_patterns=["*.parquet"]
    )
    titles: list[str] = []
    for path in sorted(Path(local).rglob("*.parquet")):
        tbl = pq.read_table(path, columns=["title"])
        titles.extend(tbl.column("title").to_pylist())
        log.info("  %s → 누적 %d", path.name, len(titles))
    return titles


def seed_canary(conn) -> None:
    """카나리아 기준선은 처음 파싱될 때 채워진다. 여기선 자리만 만든다."""
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO canary (title, baseline_chars) VALUES (%s, 0)
               ON CONFLICT (title) DO NOTHING""",
            [(t,) for t in CANARY_TITLES],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-dump", action="store_true")
    g.add_argument("--from-starter", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        if args.from_dump:
            titles = dump_titles()
            n = load_titles(conn, titles, "dump", priority=100)
        else:
            n = load_titles(conn, STARTER, "starter", priority=1)
        seed_canary(conn)
        total = conn.execute("SELECT count(*) AS c FROM crawl_state").fetchone()["c"]
        log.info("시드 %d건 처리 | crawl_state 총 %d건", n, total)


if __name__ == "__main__":
    main()
