"""문서를 온톨로지 클래스로 분류해 documents.entity_type 을 채운다 (KG-2).

관계가 문서 사이를 잇는다면 타입은 문서 자체가 무엇인지 말한다. 있으면
검색을 "인물 중에서"로 좁힐 수 있고, 관계의 정합성 검사도 가능해진다 —
사건 문서에 '학력'이 붙어 있으면 추출이 잘못된 것이다.

인포박스 술어와 분류 이름만 본다. 본문은 읽지 않으므로 블롭 접근이 없고
전체가 몇 초에 끝난다.

사용법:
    .venv/Scripts/python.exe scripts/classify_entities.py --dry-run
    .venv/Scripts/python.exe scripts/classify_entities.py
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.db import connect  # noqa: E402
from namuwiki.entity_types import CLASSES, classify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    with connect() as conn:
        preds: dict[str, set[str]] = collections.defaultdict(set)
        for r in conn.execute(
            "SELECT subject, predicate FROM relations WHERE source = 'infobox'"
        ).fetchall():
            preds[r["subject"]].add(r["predicate"])
        print(f"인포박스를 가진 문서 {len(preds):,}", flush=True)

        rows = conn.execute("SELECT title, categories FROM documents").fetchall()
        assigned = [(classify(list(r["categories"] or []),
                              preds.get(r["title"], set()), r["title"]),
                     r["title"]) for r in rows]

        counts = collections.Counter(c for c, _ in assigned)
        print(f"\n문서 {len(assigned):,}")
        for cls in CLASSES:
            n = counts.get(cls, 0)
            print(f"  {cls:<4} {n:>8,}  {n / len(assigned) * 100:5.1f}%")

        if args.dry_run:
            print("\n--dry-run - 저장하지 않음")
            return

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE documents SET entity_type = %s WHERE title = %s", assigned
            )
        print(f"\n저장 완료 ({time.time() - t0:.0f}s)")

        print("\n=== 클래스별 상위 문서 (PageRank) ===")
        for cls in CLASSES:
            top = conn.execute(
                """SELECT title FROM documents WHERE entity_type = %s
                    ORDER BY pagerank DESC NULLS LAST LIMIT 6""",
                (cls,),
            ).fetchall()
            if top:
                print(f"  {cls:<4} {', '.join(r['title'][:18] for r in top)}")


if __name__ == "__main__":
    main()
