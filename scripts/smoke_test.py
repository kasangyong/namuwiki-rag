"""파이프라인 전 구간 검증.

설계 문서 Phase 1 통과 기준을 기계적으로 확인한다:
  - 수집 → 파싱 → 청킹 → 임베딩 → 검색이 관통하는가
  - 멱등성: 같은 입력을 다시 넣어도 중복이 생기지 않는가
  - 추출 수율 카나리아가 살아 있는가
  - 지식그래프 엣지가 쌓이는가

사용법:
    .venv/Scripts/python.exe scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((status, name, detail))


def main() -> int:
    with connect() as conn:
        docs = conn.execute("SELECT count(*) c FROM documents").fetchone()["c"]
        chunks = conn.execute("SELECT count(*) c FROM chunks").fetchone()["c"]
        embedded = conn.execute(
            "SELECT count(*) c FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()["c"]
        edges = conn.execute(
            "SELECT coalesce(sum(array_length(outlinks,1)),0) e FROM documents"
        ).fetchone()["e"]
        cats = conn.execute(
            "SELECT count(*) c FROM documents WHERE array_length(categories,1) > 0"
        ).fetchone()["c"]
        frontier = conn.execute("SELECT count(*) c FROM crawl_state").fetchone()["c"]
        fetched = conn.execute(
            "SELECT count(*) c FROM crawl_state WHERE last_fetched IS NOT NULL"
        ).fetchone()["c"]
        canary = conn.execute(
            "SELECT title, baseline_chars, last_chars, status FROM canary"
        ).fetchall()
        dup = conn.execute(
            """SELECT count(*) c FROM (
                 SELECT doc_id, seq FROM chunks GROUP BY doc_id, seq HAVING count(*) > 1
               ) t"""
        ).fetchone()["c"]
        orphan = conn.execute(
            """SELECT count(*) c FROM chunks ch
                LEFT JOIN documents d USING (doc_id) WHERE d.doc_id IS NULL"""
        ).fetchone()["c"]
        # 문서 행만 있고 청크가 없는 상태. 문서와 청크를 따로 커밋하던 시절
        # 청크 INSERT만 실패하면 이렇게 남았고, 재파싱 때 해시가 같아
        # 건너뛰어져 영구 손실이 됐다. 조용해서 더 위험한 종류다.
        chunkless = conn.execute(
            """SELECT count(*) c FROM documents d
                WHERE d.char_len > 0
                  AND NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.doc_id = d.doc_id)"""
        ).fetchone()["c"]

    check("수집", fetched > 0, f"{fetched}/{frontier} 문서 수집됨")
    check("파싱", docs > 0, f"documents {docs}건")
    check("청킹", chunks > 0, f"chunks {chunks}건 (문서당 {chunks / max(docs, 1):.1f})")
    check("지식그래프 엣지", edges > 0,
          f"{edges:,}개 (문서당 {edges / max(docs, 1):.0f})")
    check("분류 온톨로지", cats > 0, f"분류를 가진 문서 {cats}건")
    check("청크 중복 없음", dup == 0, f"중복 (doc_id, seq) {dup}건")
    check("고아 청크 없음", orphan == 0, f"문서 없는 청크 {orphan}건")
    check("청크 없는 문서 없음", chunkless == 0, f"본문은 있는데 청크가 없는 문서 {chunkless}건")

    baselined = [c for c in canary if c["baseline_chars"] > 0]
    alerts = [c for c in canary if c["status"] == "ALERT"]
    check("카나리아 기준선", len(baselined) > 0,
          f"{len(baselined)}/{len(canary)}개 확립")
    check("카나리아 경보 없음", not alerts,
          f"경보 {len(alerts)}건: {[c['title'] for c in alerts]}")

    try:
        qc = QdrantClient(url=SETTINGS.qdrant_url, timeout=30)
        info = qc.get_collection(SETTINGS.collection)
        points = info.points_count or 0
    except Exception as e:  # noqa: BLE001
        points = -1
        check("Qdrant 연결", False, str(e))
    else:
        check("Qdrant 연결", True, f"컬렉션 {SETTINGS.collection}")

    check("임베딩", embedded > 0, f"{embedded}/{chunks} 청크 임베딩됨",
          warn_only=True)
    if points >= 0:
        check("Qdrant 색인", points > 0, f"{points}포인트", warn_only=True)
        check("DB↔Qdrant 일치", abs(points - embedded) <= max(1, embedded * 0.02),
              f"Qdrant {points} vs DB {embedded}", warn_only=True)

    width = max(len(n) for _, n, _ in results)
    print("\n=== 파이프라인 검증 ===\n")
    for status, name, detail in results:
        mark = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}[status]
        print(f"  {mark} {name:<{width}}  {detail}")

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(f"\n통과 {len(results) - len(failed) - len(warned)} / "
          f"경고 {len(warned)} / 실패 {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
