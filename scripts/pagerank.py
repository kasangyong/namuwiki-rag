"""PageRank — 링크 중심성 계산 (설계 §13.1 KG-1).

용도:
  - 검색 순위 보정. 같은 관련도면 중심 문서를 위로 올린다.
  - 크롤 우선순위. 중요한 문서를 먼저·자주 수집한다.

규모 주의:
  최종 코퍼스는 문서 124만 개 / 엣지 약 10억 개다. 엣지를 파이썬 객체로
  들면 메모리가 터지므로 처음부터 numpy CSR(int32)로 만든다.
  int32 타깃 배열만 10억 × 4B = 4GB이니, 32GB 머신에서 여유가 빠듯하다.
  --max-edges로 상한을 두어 초과 시 명시적으로 실패한다 (조용한 스왑 방지).

사용법:
    .venv/Scripts/python.exe scripts/pagerank.py
    .venv/Scripts/python.exe scripts/pagerank.py --iters 30 --damping 0.85
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.db import connect  # noqa: E402

FETCH = 5000


def build_graph(conn, max_edges: int):
    """(제목→인덱스, indptr, indices) CSR 그래프. 아는 문서 사이의 엣지만."""
    rows = conn.execute("SELECT doc_id, title FROM documents ORDER BY doc_id").fetchall()
    n = len(rows)
    if n == 0:
        raise SystemExit("문서가 없다. 먼저 수집·파싱할 것.")
    doc_ids = [r["doc_id"] for r in rows]
    index = {r["title"]: i for i, r in enumerate(rows)}

    # 별칭도 정식 문서를 가리키게 한다 (설계 §13.1 SAME_AS).
    # 이게 없으면 '3.1절'로 향한 링크와 '삼일절'로 향한 링크가 별개 노드로
    # 갈려 중심성이 나뉜다.
    aliases = 0
    for r in conn.execute("SELECT alias, canonical FROM redirects").fetchall():
        tgt = index.get(r["canonical"])
        if tgt is not None and r["alias"] not in index:
            index[r["alias"]] = tgt
            aliases += 1
    if aliases:
        print(f"별칭 {aliases:,}개를 정식 문서로 해석", flush=True)

    indptr = np.zeros(n + 1, dtype=np.int64)
    targets: list[np.ndarray] = []
    total = 0

    # 서버 사이드 커서는 트랜잭션 안에서만 선언할 수 있다. 124만 행의
    # outlinks를 한 번에 가져오면 메모리가 터지므로 스트리밍이 필요하다.
    with conn.transaction(), conn.cursor(name="pr_scan") as cur:
        cur.itersize = FETCH
        cur.execute("SELECT doc_id, outlinks FROM documents ORDER BY doc_id")
        pos = 0
        for row in cur:
            hits = [index[t] for t in row["outlinks"] if t in index]
            if hits:
                targets.append(np.fromiter(hits, dtype=np.int32, count=len(hits)))
                total += len(hits)
                if total > max_edges:
                    raise SystemExit(
                        f"엣지가 상한 {max_edges:,}을 넘었다 ({total:,}). "
                        "--max-edges를 올리거나 메모리를 확인할 것."
                    )
            pos += 1
            indptr[pos] = total

    indices = (
        np.concatenate(targets) if targets else np.zeros(0, dtype=np.int32)
    )
    return doc_ids, indptr, indices, n


def pagerank(indptr, indices, n: int, damping: float, iters: int, tol: float):
    outdeg = np.diff(indptr).astype(np.float64)
    dangling = outdeg == 0
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - damping) / n

    for it in range(iters):
        # 나가는 링크가 없는 문서(dangling)의 점수는 전체에 고르게 나눈다.
        # 이 처리를 빼면 점수 총합이 반복마다 새어나가 결과가 왜곡된다.
        leaked = damping * pr[dangling].sum() / n
        contrib = np.zeros(n, dtype=np.float64)
        src_scores = np.where(dangling, 0.0, pr / np.maximum(outdeg, 1.0))
        # 각 소스의 기여를 타깃에 누적. np.add.at은 느리므로 bincount를 쓴다.
        if indices.size:
            src_of_edge = np.repeat(
                np.arange(n, dtype=np.int64), np.diff(indptr)
            )
            contrib = np.bincount(
                indices, weights=src_scores[src_of_edge], minlength=n
            )
        new = base + leaked + damping * contrib
        delta = np.abs(new - pr).sum()
        pr = new
        print(f"  iter {it + 1:>2}  L1 변화 {delta:.3e}", flush=True)
        if delta < tol:
            break
    return pr / pr.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--damping", type=float, default=0.85)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-edges", type=int, default=1_500_000_000)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    with connect() as conn:
        t0 = time.time()
        doc_ids, indptr, indices, n = build_graph(conn, args.max_edges)
        print(f"그래프: 노드 {n:,} / 엣지 {indices.size:,} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if indices.size == 0:
            raise SystemExit("아는 문서 사이의 엣지가 없다. 더 수집해야 한다.")

        t0 = time.time()
        pr = pagerank(indptr, indices, n, args.damping, args.iters, args.tol)
        print(f"계산 완료 ({time.time() - t0:.1f}s)", flush=True)

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE documents SET pagerank = %s WHERE doc_id = %s",
                [(float(pr[i]), doc_ids[i]) for i in range(n)],
            )

        # 크롤 우선순위에 반영한다 (설계 §13.1 KG-1). 중심 문서일수록 먼저,
        # 자주 다시 본다. priority는 낮을수록 우선이며, sidebar가 알려준
        # 실시간 변경(10)보다는 뒤에 두어 신선도를 앞세운다.
        updated = conn.execute(
            """
            WITH ranked AS (
              SELECT doc_id, ntile(4) OVER (ORDER BY pagerank DESC NULLS LAST) AS q
                FROM documents WHERE pagerank IS NOT NULL
            )
            UPDATE crawl_state cs
               SET priority = 20 + (ranked.q - 1) * 20   -- 20 / 40 / 60 / 80
              FROM ranked
             WHERE cs.doc_id = ranked.doc_id
               AND cs.priority > 10                       -- sidebar 우선분은 건드리지 않는다
            """
        ).rowcount
        print(f"크롤 우선순위 갱신: {updated:,}건", flush=True)

        print(f"\n=== 상위 {args.top}개 ===")
        for r in conn.execute(
            """SELECT title, pagerank, coalesce(array_length(outlinks,1),0) outdeg
                 FROM documents ORDER BY pagerank DESC NULLS LAST LIMIT %s""",
            (args.top,),
        ).fetchall():
            print(f"  {r['pagerank']:.6f}  {r['title']:<24} (out {r['outdeg']:,})")


if __name__ == "__main__":
    main()
