"""보관된 HTML 에서 타입 있는 관계를 뽑아 relations 테이블을 채운다 (KG-2).

재크롤이 필요 없다. 클레임체크 설계 덕에 원본 HTML 이 전부 블롭 저장소에
남아 있어, 새 규칙을 만들 때마다 디스크만 다시 읽으면 된다.

네 갈래를 모두 채운다:
    infobox  인포박스 표의 라벨-값     (문서당 0~60건)
    title    A/B 슬래시 제목 → PART_OF
    category 분류 → INSTANCE_OF
    redirect 리다이렉트 → SAME_AS

사용법:
    .venv/Scripts/python.exe scripts/extract_relations.py --limit 500   # 맛보기
    .venv/Scripts/python.exe scripts/extract_relations.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.relations import Relation, from_infobox, from_title  # noqa: E402

BATCH = 500

# 분류 중 타입이 아닌 것. 둘러보기 틀이 분류 자리에 섞여 들어온다.
_CAT_NOISE = ("※", "나무위키", "문서가 ", "토론", "편집")


def _cat_ok(cat: str) -> bool:
    return bool(cat) and not cat.startswith(_CAT_NOISE) and "둘러보기" not in cat


def write(conn, rels: list[Relation]) -> int:
    if not rels:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO relations
                 (subject, predicate, object, obj_kind, source, evidence)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (subject, predicate, object, source)
               DO UPDATE SET evidence = EXCLUDED.evidence""",
            [(r.subject, r.predicate, r.object, r.obj_kind, r.source, r.evidence)
             for r in rels],
        )
    return len(rels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="문서 N개만 처리")
    ap.add_argument("--skip-infobox", action="store_true",
                    help="블롭을 읽지 않는다 (제목·분류·리다이렉트만)")
    args = ap.parse_args()

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    t0 = time.time()
    counts = {"infobox": 0, "title": 0, "category": 0, "redirect": 0}

    with connect() as conn:
        # --- 리다이렉트: SAME_AS ---
        rows = conn.execute("SELECT alias, canonical FROM redirects").fetchall()
        counts["redirect"] = write(conn, [
            Relation(r["alias"], "SAME_AS", r["canonical"], "page", "redirect", None)
            for r in rows
        ])
        print(f"redirect  {counts['redirect']:>8,}건", flush=True)

        # --- 제목·분류 ---
        rows = conn.execute("SELECT title, categories FROM documents").fetchall()
        title_rels: list[Relation] = []
        cat_rels: list[Relation] = []
        for r in rows:
            title_rels.extend(from_title(r["title"]))
            for cat in (r["categories"] or []):
                if _cat_ok(cat):
                    cat_rels.append(Relation(r["title"], "INSTANCE_OF", cat,
                                             "literal", "category", None))
        counts["title"] = write(conn, title_rels)
        print(f"title     {counts['title']:>8,}건", flush=True)
        for i in range(0, len(cat_rels), 5000):
            counts["category"] += write(conn, cat_rels[i:i + 5000])
        print(f"category  {counts['category']:>8,}건", flush=True)

        if args.skip_infobox:
            print(f"\n완료 ({time.time()-t0:.0f}s) — 인포박스 생략")
            return

        # --- 인포박스: 블롭을 읽는다 ---
        q = ("SELECT title, blob_ref FROM documents "
             "WHERE blob_ref IS NOT NULL ORDER BY pagerank DESC NULLS LAST")
        if args.limit:
            q += f" LIMIT {args.limit}"
        docs = conn.execute(q).fetchall()
        print(f"\n인포박스 {len(docs):,}문서 …", flush=True)

        buf: list[Relation] = []
        done = failed = 0
        for r in docs:
            try:
                html = blobs.get(r["blob_ref"]).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            buf.extend(from_infobox(r["title"], html))
            done += 1
            if len(buf) >= BATCH:
                counts["infobox"] += write(conn, buf)
                buf.clear()
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"  {done:,}/{len(docs):,} 문서 · 관계 {counts['infobox']:,} "
                      f"· {done/el:.0f}문서/s", flush=True)
        counts["infobox"] += write(conn, buf)

        print(f"\ninfobox   {counts['infobox']:>8,}건 (문서 {done:,}, 실패 {failed})")
        total = conn.execute("SELECT count(*) n FROM relations").fetchone()["n"]
        print(f"\n총 relations {total:,}건 · {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
