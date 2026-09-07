"""프런티어를 인링크 수로 정렬한다.

BFS는 발견 순서가 가치 순서가 아니다. 목록 문서 하나에 걸리면 거기 나열된
수백 개 항목이 한꺼번에 프런티어에 들어오는데, 대부분 존재하지 않거나
(빨간 링크) 토막글이다. 실제로 생물 분류 목록에 빠져서 '긴다리타원안갑윤충',
'트리코케르카 콜라리스' 같은 윤충 종 이름을 긁던 중 404율이 2% → 11%로 올랐다.

인링크 수가 그 문서의 중요도를 대신한다. 여러 문서가 가리키는 제목은 실재할
가능성도 높고 검색될 가능성도 높다. 목록 하나에서만 링크된 제목은 그 반대다.

PageRank 를 쓰지 않는 이유: PageRank 는 **이미 수집한** 문서에만 값이 있다.
아직 수집하지 않은 프런티어 항목의 우선순위를 정하려면 인링크 수가 맞다.

사용법:
    .venv/Scripts/python.exe scripts/prioritize_frontier.py --dry-run
    .venv/Scripts/python.exe scripts/prioritize_frontier.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.db import connect  # noqa: E402

# priority 는 낮을수록 먼저다. sidebar 가 알려준 실시간 변경(10)보다는 뒤에 둔다.
#   인링크 50+ → 40, 20+ → 60, 5+ → 90, 2+ → 130, 1 → 180, 0 → 220
TIERS = [(50, 40), (20, 60), (5, 90), (2, 130), (1, 180)]
FLOOR = 220


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with connect() as conn:
        print("인링크 집계 중 …", flush=True)
        conn.execute("DROP TABLE IF EXISTS _indeg")
        conn.execute(
            """
            CREATE TEMP TABLE _indeg AS
            SELECT t AS title, count(*)::int AS n
              FROM documents d, unnest(d.outlinks) AS t
             GROUP BY t
            """
        )
        conn.execute("CREATE INDEX ON _indeg (title)")
        total = conn.execute("SELECT count(*) c FROM _indeg").fetchone()["c"]
        print(f"  링크 대상 제목 {total:,}개", flush=True)

        print("\n미수집 프런티어의 인링크 분포:", flush=True)
        for lo, _ in TIERS:
            n = conn.execute(
                """SELECT count(*) c FROM crawl_state cs
                     JOIN _indeg i ON i.title = cs.title
                    WHERE cs.last_fetched IS NULL AND i.n >= %s""",
                (lo,),
            ).fetchone()["c"]
            print(f"  인링크 {lo:>3}+ : {n:>8,}건", flush=True)
        orphan = conn.execute(
            """SELECT count(*) c FROM crawl_state cs
                WHERE cs.last_fetched IS NULL
                  AND NOT EXISTS (SELECT 1 FROM _indeg i WHERE i.title = cs.title)"""
        ).fetchone()["c"]
        print(f"  인링크   0 : {orphan:>8,}건  (수집한 문서 중 아무도 안 가리킴)")

        if args.dry_run:
            print("\n--dry-run — 변경하지 않음")
            return

        print("\n우선순위 갱신 …", flush=True)
        updated = 0
        # 낮은 문턱부터 적용한다. 높은 문턱을 먼저 적용하면 뒤이은 낮은
        # 문턱이 같은 행을 다시 덮어써서 마지막 티어가 이긴다 —
        # 인링크 607개인 '홍콩'이 priority 180 을 받는 식이다.
        for lo, pri in sorted(TIERS):
            n = conn.execute(
                """UPDATE crawl_state cs SET priority = %s
                     FROM _indeg i
                    WHERE i.title = cs.title
                      AND i.n >= %s
                      AND cs.priority > 10        -- 실시간 변경분은 건드리지 않는다
                      AND cs.priority <> %s""",
                (pri, lo, pri),
            ).rowcount
            updated += n
            print(f"  인링크 {lo:>3}+ → priority {pri:>3}  ({n:,}건)", flush=True)

        n = conn.execute(
            """UPDATE crawl_state cs SET priority = %s
                WHERE cs.priority > 10
                  AND cs.priority <> %s
                  AND NOT EXISTS (SELECT 1 FROM _indeg i WHERE i.title = cs.title)""",
            (FLOOR, FLOOR),
        ).rowcount
        updated += n
        print(f"  인링크   0 → priority {FLOOR}  ({n:,}건)", flush=True)
        print(f"\n총 {updated:,}건 갱신")

        print("\n다음에 크롤될 문서 (우선순위 순):")
        for r in conn.execute(
            """SELECT cs.title, cs.priority, coalesce(i.n,0) AS indeg
                 FROM crawl_state cs LEFT JOIN _indeg i ON i.title = cs.title
                WHERE cs.last_fetched IS NULL AND cs.fail_count < 5
                ORDER BY cs.priority, i.n DESC NULLS LAST
                LIMIT 12"""
        ).fetchall():
            print(f"  p{r['priority']:<4} 인링크 {r['indeg']:>4}  {r['title'][:40]}")


if __name__ == "__main__":
    main()
