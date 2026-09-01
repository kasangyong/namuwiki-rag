"""검색 API — 하이브리드 검색 + 리랭킹 (설계 §7).

    질의 ─┬─▶ dense  top-100 ─┐
          └─▶ sparse top-100 ─┴─▶ RRF 융합 ─▶ 리랭커 ─▶ top-k

RRF를 쓰는 이유는 dense와 sparse의 점수 스케일이 달라 단순 가중합이
불안정하기 때문이다. Qdrant가 융합을 서버 측에서 해준다.

라이선스 의무(§11): 모든 결과에 원문 링크와 CC BY-NC-SA 고지를 포함한다.
나무위키는 계속 바뀌므로 문서의 '최근 수정 시각'과 수집 시각도 함께 준다.

사용법:
    .venv/Scripts/python.exe -m uvicorn api:app --app-dir services --port 8000
"""

from __future__ import annotations

import logging
import sys
import urllib.parse
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.models import DenseEncoder, Reranker, SparseEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="나무위키 RAG 검색", version="0.1")

LICENSE_NOTE = (
    "출처: 나무위키 · CC BY-NC-SA 2.0 KR. "
    "이 검색 결과는 비영리 연구 목적으로만 제공됩니다."
)


class Hit(BaseModel):
    rank: int
    score: float
    title: str
    heading_path: list[str]
    text: str
    url: str
    doc_id: str
    chunk_id: str
    last_modified: str | None = None
    fetched_at: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]
    license: str = LICENSE_NOTE
    took_ms: int


@lru_cache(maxsize=1)
def _dense() -> DenseEncoder:
    return DenseEncoder()


@lru_cache(maxsize=1)
def _sparse() -> SparseEncoder:
    return SparseEncoder()


@lru_cache(maxsize=1)
def _reranker() -> Reranker:
    return Reranker()


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(url=SETTINGS.qdrant_url, timeout=60)


def _url(title: str) -> str:
    return f"{SETTINGS.base_url}/w/{urllib.parse.quote(title, safe='')}"


def _search_points(dvec, svec, candidates: int, flt):
    return _qdrant().query_points(
        collection_name=SETTINGS.collection,
        prefetch=[
            models.Prefetch(query=dvec, using="dense", limit=candidates, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=svec.indices, values=svec.values),
                using="lexical", limit=candidates, filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidates,
        with_payload=True,
    ).points


def hybrid_search(query: str, candidates: int = 100, top_k: int = 8,
                  category: str | None = None, expand: bool = False) -> list[Hit]:
    import time

    t0 = time.time()
    dvec = _dense().encode([query])[0]
    svec = _sparse().encode([query])[0]

    flt = None
    if category:
        flt = models.Filter(
            must=[models.FieldCondition(
                key="categories", match=models.MatchValue(value=category)
            )]
        )

    res = _search_points(dvec, svec, candidates, flt)

    if expand and res:
        # 그래프 이웃 확장 (설계 §13). 상위 문서와 상호 링크된 문서에서
        # 한 번 더 검색해 후보를 넓힌다. 벡터 검색만으로는 질의어와 표현이
        # 다른 관련 문서를 놓치는데, 링크 구조가 그걸 메운다.
        seeds = list({p.payload["doc_id"] for p in res[:10]})
        neighbors = neighbor_doc_ids(seeds)
        if neighbors:
            nflt = models.Filter(
                must=[models.FieldCondition(
                    key="doc_id", match=models.MatchAny(any=neighbors)
                )]
            )
            extra = _search_points(dvec, svec, max(candidates // 2, 20), nflt)
            seen = {p.id for p in res}
            res = list(res) + [p for p in extra if p.id not in seen]

    if not res:
        return []

    texts = [p.payload.get("text", "") for p in res]
    # 동점 처리를 위해 top_k보다 넉넉히 받아둔다
    ranked = _reranker().rank(query, texts, top_k=min(len(texts), top_k * 3))

    # 문서 메타 — 수정 시각(신선도 표시용)과 PageRank(동점 보정용)
    doc_ids = list({res[i].payload["doc_id"] for i, _ in ranked})
    meta: dict[str, dict] = {}
    if doc_ids:
        with connect() as conn:
            for row in conn.execute(
                """SELECT doc_id, last_modified, fetched_at, pagerank
                     FROM documents WHERE doc_id = ANY(%s)""",
                (doc_ids,),
            ).fetchall():
                meta[row["doc_id"]] = row

    # PageRank는 **동점 보정에만** 쓴다 (설계 §13.1 KG-1).
    # 관련도 점수에 직접 더하면 인기 문서가 정확한 답을 밀어내기 쉽다.
    # 리랭커 점수가 사실상 같을 때만 링크 중심성으로 순서를 정한다.
    # 더 강한 혼합은 자체 평가 데이터가 쌓인 뒤에 판단한다.
    def sort_key(item):
        idx, score = item
        pr = (meta.get(res[idx].payload["doc_id"], {}) or {}).get("pagerank") or 0.0
        return (round(score, 2), pr)

    ranked = sorted(ranked, key=sort_key, reverse=True)[:top_k]

    hits: list[Hit] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        p = res[idx].payload
        m = meta.get(p["doc_id"], {})
        hits.append(Hit(
            rank=rank,
            score=round(score, 4),
            title=p.get("title", ""),
            heading_path=p.get("heading_path", []),
            text=p.get("text", ""),
            url=_url(p.get("title", "")),
            doc_id=p["doc_id"],
            chunk_id=p.get("chunk_id", ""),
            last_modified=str(m["last_modified"]) if m.get("last_modified") else None,
            fetched_at=str(m["fetched_at"]) if m.get("fetched_at") else None,
        ))
    log.info("검색 %r → %d건 (%.0fms)", query, len(hits), (time.time() - t0) * 1000)
    return hits


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(8, ge=1, le=50),
    candidates: int = Query(100, ge=10, le=500),
    category: str | None = None,
    expand: bool = Query(False, description="그래프 이웃까지 확장 검색"),
):
    import time

    t0 = time.time()
    hits = hybrid_search(q, candidates=candidates, top_k=k, category=category,
                         expand=expand)
    return SearchResponse(
        query=q, hits=hits, took_ms=int((time.time() - t0) * 1000)
    )


def neighbor_doc_ids(doc_ids: list[str], limit: int = 40) -> list[str]:
    """주어진 문서들의 링크 이웃 (설계 §13 이웃 확장 검색).

    나무위키는 문서당 링크가 400~800개라 그대로 쓰면 너무 넓다.
    **상호 링크**(서로 가리키는 쌍)만 취해 관련성이 높은 이웃으로 좁힌다.
    """
    if not doc_ids:
        return []
    with connect() as conn:
        titles = [
            r["title"] for r in conn.execute(
                "SELECT title FROM documents WHERE doc_id = ANY(%s)", (doc_ids,)
            ).fetchall()
        ]
        if not titles:
            return []
        rows = conn.execute(
            """
            WITH seeds AS (
              SELECT doc_id, title, outlinks FROM documents WHERE doc_id = ANY(%s)
            )
            SELECT d.doc_id, d.title,
                   count(*) AS shared
              FROM seeds s
              JOIN documents d ON d.title = ANY(s.outlinks)
             WHERE d.doc_id <> ALL(%s)
               AND d.outlinks && %s          -- 상호 링크: 이웃도 시드를 가리킨다
             GROUP BY d.doc_id, d.title
             ORDER BY shared DESC
             LIMIT %s
            """,
            (doc_ids, doc_ids, titles, limit),
        ).fetchall()
    return [r["doc_id"] for r in rows]


@app.get("/api/graph")
def graph(title: str, limit: int = 20):
    """문서의 그래프 이웃. 지식그래프가 실제로 쌓였는지 확인하는 창구."""
    with connect() as conn:
        doc = conn.execute(
            """SELECT doc_id, title, categories,
                      coalesce(array_length(outlinks,1),0) AS outdeg
                 FROM documents WHERE title = %s""",
            (title,),
        ).fetchone()
        if not doc:
            return {"error": f"수집되지 않은 문서: {title}"}
        outs = conn.execute(
            """SELECT unnest(outlinks) AS t FROM documents WHERE doc_id = %s
               LIMIT %s""",
            (doc["doc_id"], limit),
        ).fetchall()
        backs = conn.execute(
            """SELECT title, coalesce(array_length(outlinks,1),0) AS outdeg
                 FROM documents WHERE outlinks @> ARRAY[%s]
                ORDER BY outdeg DESC LIMIT %s""",
            (title, limit),
        ).fetchall()
    return {
        "title": doc["title"],
        "categories": doc["categories"],
        "outdegree": doc["outdeg"],
        "outlinks": [r["t"] for r in outs],
        "backlinks": [{"title": r["title"], "outdegree": r["outdeg"]} for r in backs],
    }


@app.get("/api/stats")
def stats():
    with connect() as conn:
        docs = conn.execute("SELECT count(*) AS c FROM documents").fetchone()["c"]
        chunks = conn.execute("SELECT count(*) AS c FROM chunks").fetchone()["c"]
        embedded = conn.execute(
            "SELECT count(*) AS c FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()["c"]
        frontier = conn.execute(
            """SELECT count(*) AS total,
                      count(*) FILTER (WHERE last_fetched IS NOT NULL) AS fetched
                 FROM crawl_state"""
        ).fetchone()
        canary = conn.execute(
            "SELECT title, baseline_chars, last_chars, status FROM canary"
        ).fetchall()
        pipe = conn.execute(
            "SELECT stage, metric, value FROM pipeline_stats ORDER BY stage, metric"
        ).fetchall()
    try:
        info = _qdrant().get_collection(SETTINGS.collection)
        points = info.points_count
    except Exception:  # noqa: BLE001
        points = None
    return {
        "documents": docs,
        "chunks": chunks,
        "chunks_embedded": embedded,
        "qdrant_points": points,
        "frontier_total": frontier["total"],
        "frontier_fetched": frontier["fetched"],
        "canary": [dict(c) for c in canary],
        "pipeline": [dict(p) for p in pipe],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")
