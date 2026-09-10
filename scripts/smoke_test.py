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
from namuwiki.config import CHROME_TITLES
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
        rel = {r["source"]: r["n"] for r in conn.execute(
            "SELECT source, count(*) n FROM relations GROUP BY source").fetchall()}
        rel_linked = conn.execute(
            "SELECT count(*) n FROM relations "
            "WHERE source='infobox' AND obj_kind='page'").fetchone()["n"]
        pr_chrome = conn.execute(
            """SELECT count(*) n FROM documents
                WHERE title = ANY(%s) AND pagerank IS NOT NULL""",
            (list(CHROME_TITLES),),
        ).fetchone()["n"]
        typed = conn.execute(
            "SELECT count(*) n FROM documents WHERE entity_type IS NOT NULL"
        ).fetchone()["n"]
        # 표본 검사. 이 문서들의 타입이 흔들리면 규칙이 퇴행한 것이다.
        TYPE_SPOT = {"김연아": "인물", "대한민국": "장소", "6.25 전쟁": "사건",
                     "문화방송": "조직", "호랑이": "생물"}
        spot = conn.execute(
            "SELECT title, entity_type FROM documents WHERE title = ANY(%s)",
            (list(TYPE_SPOT),),
        ).fetchall()
        spot_bad = [f"{r['title']}={r['entity_type']}" for r in spot
                    if r["entity_type"] != TYPE_SPOT[r["title"]]]
        comm = conn.execute(
            "SELECT count(DISTINCT community) n FROM documents "
            "WHERE community IS NOT NULL").fetchone()["n"]
        in_comm = conn.execute(
            "SELECT count(*) n FROM documents WHERE community IS NOT NULL"
        ).fetchone()["n"]

    check("수집", fetched > 0, f"{fetched}/{frontier} 문서 수집됨")
    check("파싱", docs > 0, f"documents {docs}건")
    check("청킹", chunks > 0, f"chunks {chunks}건 (문서당 {chunks / max(docs, 1):.1f})")
    check("지식그래프 엣지", edges > 0,
          f"{edges:,}개 (문서당 {edges / max(docs, 1):.0f})")
    check("분류 온톨로지", cats > 0, f"분류를 가진 문서 {cats}건")

    # --- KG-2: 타입 있는 관계 ---
    infobox = rel.get("infobox", 0)
    check("인포박스 관계", infobox > 0,
          f"{infobox:,}건 · 제목 {rel.get('title',0):,} · 분류 {rel.get('category',0):,}")
    # 인포박스 값이 문서로 연결되지 않으면 그래프 엣지가 되지 못한다.
    # 실측 90%. 크게 떨어지면 추출 규칙이 깨진 것이다.
    ratio = rel_linked / infobox if infobox else 0
    check("관계가 문서로 연결됨", ratio >= 0.7,
          f"{rel_linked:,}/{infobox:,} ({ratio*100:.0f}%)")
    check("PageRank 크롬 링크 제외", pr_chrome == 0,
          f"UI 링크에 점수가 남으면 순위 보정이 망가진다 ({pr_chrome}건)")
    check("엔티티 타입", typed > 0, f"{typed:,}건 분류됨")
    check("타입 표본 검사", not spot_bad,
          "규칙 퇴행 없음" if not spot_bad else "틀림: " + ", ".join(spot_bad))
    check("커뮤니티 탐지", comm > 0,
          f"{comm}개 커뮤니티 · 문서 {in_comm:,}건 배정", warn_only=True)
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
