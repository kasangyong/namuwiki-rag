"""링크 구조로 주제 덩어리를 찾는다 — 커뮤니티 탐지 (KG-3 준비).

PageRank 가 "어느 문서가 중요한가"를 답한다면, 커뮤니티는 "어떤 문서들이
한 주제인가"를 답한다. 분류(카테고리)와 다른 점은 사람이 붙인 라벨이 아니라
실제 링크 밀도에서 나온다는 것이다 — 분류가 없거나 엉성한 문서도 묶인다.

Leiden 을 쓴다. Louvain 의 후신으로, 연결되지 않은 덩어리가 한 커뮤니티로
묶이는 Louvain 의 알려진 결함이 없다. 55,000 노드 / 1,700만 엣지를 C 구현
으로 처리하므로 파이썬 루프로는 불가능한 규모가 몇 분에 끝난다.

방향은 버리고 무향으로 본다. 커뮤니티는 "누가 누구를 가리키는가"가 아니라
"함께 등장하는가"의 문제다.

사용법:
    .venv/Scripts/python.exe scripts/communities.py
    .venv/Scripts/python.exe scripts/communities.py --resolution 1.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import igraph as ig  # noqa: E402

from namuwiki.config import CHROME_TITLES  # noqa: E402
from namuwiki.db import connect  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="높이면 잘게, 낮추면 크게 쪼갠다")
    ap.add_argument("--min-size", type=int, default=5,
                    help="이보다 작은 커뮤니티는 NULL 로 둔다")
    args = ap.parse_args()

    t0 = time.time()
    with connect() as conn:
        rows = conn.execute("SELECT title FROM documents ORDER BY title").fetchall()
        titles = [r["title"] for r in rows]
        index = {t: i for i, t in enumerate(titles)}
        for chrome in CHROME_TITLES:
            index.pop(chrome, None)
        n = len(titles)
        print(f"노드 {n:,}", flush=True)

        # 별칭도 정식 문서로 접는다 (PageRank 와 같은 처리).
        for r in conn.execute("SELECT alias, canonical FROM redirects").fetchall():
            tgt = index.get(r["canonical"])
            if tgt is not None and r["alias"] not in index:
                index[r["alias"]] = tgt

        edges: list[tuple[int, int]] = []
        with conn.transaction(), conn.cursor(name="cm_scan") as cur:
            cur.itersize = 2000
            cur.execute("SELECT title, outlinks FROM documents ORDER BY title")
            for row in cur:
                src = index.get(row["title"])
                if src is None:
                    continue
                for t in row["outlinks"]:
                    dst = index.get(t)
                    if dst is not None and dst != src:
                        edges.append((src, dst) if src < dst else (dst, src))
        print(f"엣지 {len(edges):,} 수집 ({time.time()-t0:.0f}s)", flush=True)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)   # 무향이라 A→B, B→A 는 한 엣지다
    print(f"단순화 후 엣지 {g.ecount():,} ({time.time()-t0:.0f}s) — Leiden 실행",
          flush=True)

    part = g.community_leiden(objective_function="modularity",
                              resolution=args.resolution, n_iterations=-1)
    sizes = part.sizes()
    keep = {c for c, sz in enumerate(sizes) if sz >= args.min_size}
    print(f"커뮤니티 {len(sizes):,}개 (>= {args.min_size}: {len(keep):,}) "
          f"· 모듈성 {part.modularity:.4f} · {time.time()-t0:.0f}s", flush=True)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE documents SET community = %s WHERE title = %s",
                [(part.membership[i] if part.membership[i] in keep else None,
                  titles[i]) for i in range(n)],
            )
        print("\n=== 큰 커뮤니티 (대표 문서는 PageRank 상위) ===")
        for r in conn.execute(
            """SELECT community, count(*) n,
                      (array_agg(title ORDER BY pagerank DESC NULLS LAST))[1:6] AS top
                 FROM documents WHERE community IS NOT NULL
                GROUP BY community ORDER BY n DESC LIMIT 12"""
        ).fetchall():
            print(f"  #{r['community']:<5} {r['n']:>6,}건  {', '.join(r['top'])[:88]}")
    print(f"\n완료 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
