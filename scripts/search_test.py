"""검색 품질 눈으로 확인하기.

서버를 띄우지 않고 검색 경로(하이브리드 → RRF → 리랭킹 → PageRank 동점 보정)를
그대로 태워 결과를 출력한다. 임베딩이 진행 중이어도 지금까지 색인된 범위에서
동작을 확인할 수 있다.

사용법:
    .venv/Scripts/python.exe scripts/search_test.py
    .venv/Scripts/python.exe scripts/search_test.py --q "김연아 밴쿠버 올림픽" --expand
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services"))

from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402

DEFAULT_QUERIES = [
    "대한민국의 수도는 어디이고 인구는 얼마나 되나",
    "김연아가 올림픽에서 딴 메달",
    "부산의 지리적 특징",
    "파이썬은 어떤 프로그래밍 언어인가",
    "축구 경기 규칙",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", action="append", help="질의 (여러 번 지정 가능)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=30,
                    help="리랭커에 넘길 후보 수. CPU에선 작게 잡는다.")
    ap.add_argument("--expand", action="store_true", help="그래프 이웃 확장")
    ap.add_argument("--chars", type=int, default=200, help="본문 미리보기 길이")
    args = ap.parse_args()

    queries = args.q or DEFAULT_QUERIES

    with connect() as conn:
        n = conn.execute(
            "SELECT count(*) c FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()["c"]
        docs = conn.execute(
            """SELECT count(DISTINCT doc_id) c FROM chunks
                WHERE embedded_at IS NOT NULL"""
        ).fetchone()["c"]
    print(f"색인 범위: 청크 {n:,}개 / 문서 {docs:,}개  ·  device={SETTINGS.device or 'auto'}\n",
          flush=True)
    if n == 0:
        raise SystemExit("아직 색인된 청크가 없다.")

    t0 = time.time()
    import api  # noqa: E402 — 모델은 여기서 지연 로드된다

    print(f"모델 로드 중… ", end="", flush=True)
    api._dense(); api._sparse(); api._reranker()
    print(f"{time.time() - t0:.0f}s\n", flush=True)

    for q in queries:
        t0 = time.time()
        hits = api.hybrid_search(q, candidates=args.candidates, top_k=args.k,
                                 expand=args.expand)
        dt = time.time() - t0
        print("=" * 78)
        print(f"질의: {q}   ({dt:.1f}s, {len(hits)}건)")
        print("=" * 78)
        if not hits:
            print("  결과 없음 — 아직 색인되지 않은 범위일 수 있다.\n")
            continue
        for h in hits:
            path = " > ".join(h.heading_path) if h.heading_path else "(섹션 없음)"
            body = " ".join(h.text.split())[:args.chars]
            print(f"\n  [{h.rank}] {h.title}  ›  {path}")
            print(f"      점수 {h.score:.3f}   수정 {(h.last_modified or '?')[:10]}")
            print(f"      {body}…")
        print()


if __name__ == "__main__":
    main()
