"""이미 저장된 리다이렉트 문서를 정식 제목으로 접는다 (설계 §13.1 SAME_AS).

크롤러는 리다이렉트를 따라가므로 '픽사'를 요청해도 '픽사 애니메이션 스튜디오'의
내용이 온다. 요청한 제목으로 저장하면:

  ① 정식 제목까지 수집됐을 때 같은 내용이 두 문서가 되어 **임베딩을 두 번** 한다
  ② 그래프 노드가 갈라져 PageRank가 나뉜다
  ③ 검색에 같은 문단이 두 번 나온다

표본 300개 중 5.3%가 이 경우였다.

원본 HTML이 블롭에 있으므로 재크롤 없이 고칠 수 있다. 페이지의 <h1>이
정식 제목이다.

사용법:
    .venv/Scripts/python.exe scripts/fold_redirects.py --dry-run
    .venv/Scripts/python.exe scripts/fold_redirects.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.config import SETTINGS, Topics  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.extract import ExtractionError, extract  # noqa: E402
from namuwiki.ids import doc_id, normalize_title  # noqa: E402
from namuwiki.kafka_io import make_producer, send  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    producer = make_producer()
    found = folded = kept = 0

    with connect() as conn:
        rows = conn.execute(
            """SELECT doc_id, title, blob_ref, char_len
                 FROM documents WHERE blob_ref IS NOT NULL
                ORDER BY title"""
        ).fetchall()
        if args.limit:
            rows = rows[: args.limit]
        print(f"검사 대상 {len(rows):,}개\n", flush=True)

        for i, r in enumerate(rows, 1):
            if i % 500 == 0:
                print(f"  … {i:,}/{len(rows):,} (리다이렉트 {found})", flush=True)
            try:
                ex = extract(blobs.get(r["blob_ref"]).decode("utf-8", "replace"))
            except (ExtractionError, OSError):
                continue
            if not ex.title:
                continue
            canonical = normalize_title(ex.title)
            if canonical == r["title"]:
                continue

            found += 1
            if args.dry_run:
                print(f"  [건너뜀] {r['title']!r} → {canonical!r}")
                continue

            conn.execute(
                """INSERT INTO redirects (alias, canonical) VALUES (%s, %s)
                   ON CONFLICT (alias) DO UPDATE SET canonical = EXCLUDED.canonical""",
                (r["title"], canonical),
            )

            # 정식 제목 문서가 이미 있으면 별칭 문서는 지운다.
            # 없으면 별칭 문서를 정식 제목으로 개명한다 (내용은 이미 정식 것이다).
            canon_id = doc_id(canonical)
            exists = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = %s", (canon_id,)
            ).fetchone()

            chunk_ids = [
                x["chunk_id"] for x in conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = %s", (r["doc_id"],)
                ).fetchall()
            ]
            if exists:
                # 중복 — 별칭 쪽 청크를 색인에서 내리고 문서를 삭제한다
                for cid in chunk_ids:
                    send(producer, Topics.CHUNKS_READY, r["doc_id"],
                         {"chunk_id": cid, "doc_id": r["doc_id"], "deleted": True})
                conn.execute("DELETE FROM documents WHERE doc_id = %s", (r["doc_id"],))
                folded += 1
            else:
                # 정식 문서가 없으니 이 행을 정식 제목으로 옮긴다.
                # doc_id가 바뀌므로 옛 청크는 색인에서 내리고 재파싱시킨다.
                for cid in chunk_ids:
                    send(producer, Topics.CHUNKS_READY, r["doc_id"],
                         {"chunk_id": cid, "doc_id": r["doc_id"], "deleted": True})
                conn.execute("DELETE FROM documents WHERE doc_id = %s", (r["doc_id"],))
                # 정식 제목으로 다시 수집·파싱되도록 프런티어에 넣는다
                conn.execute(
                    """INSERT INTO crawl_state (doc_id, title, discovered_by, priority)
                       VALUES (%s, %s, 'redirect', 30)
                       ON CONFLICT (doc_id) DO UPDATE SET next_due_at = now()""",
                    (canon_id, canonical),
                )
                kept += 1
        producer.flush()

        left = conn.execute("SELECT count(*) c FROM redirects").fetchone()["c"]

    print(f"\n리다이렉트 {found}개 발견")
    if not args.dry_run:
        print(f"  중복 제거 {folded}개 · 정식 제목으로 재수집 예약 {kept}개")
        print(f"  redirects 테이블: {left}개")


if __name__ == "__main__":
    main()
