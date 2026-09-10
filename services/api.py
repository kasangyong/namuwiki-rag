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
import re
import urllib.parse
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from qdrant_client import QdrantClient, models  # noqa: E402

from namuwiki.config import CHROME_TITLES as SETTINGS_CHROME, SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.models import DenseEncoder, Reranker, SparseEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="나무위키 RAG 검색", version="0.1")

# 페이지 UI에서 나온 링크들. 대상 자체는 실제 문서지만, 링크를 거는 쪽이
# 주제상 무관해서(과학고등학교·중졸·아인슈타인 등) 그래프 시각화에서는
# 노이즈다. 저장된 데이터는 그대로 두고 표시할 때만 감춘다 (설계 §13.1.1).
CHROME_TITLES = list(SETTINGS_CHROME)

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


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]
    # 결과 문서들에 대해 그래프가 아는 사실. 본문 조각은 "무엇이라 쓰여
    # 있는가"를 주지만 이건 "무엇인가"를 준다 — 답을 구성할 때 본문보다
    # 짧고 정확하다.
    facts: list[Fact] = []
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
    entity_type: str | None = Query(None, description="인물·장소·조직·작품·사건·생물·개념"),
    expand: bool = Query(False, description="그래프 이웃까지 확장 검색"),
):
    import time

    t0 = time.time()
    hits = hybrid_search(q, candidates=candidates, top_k=k, category=category,
                         expand=expand)
    if entity_type:
        # 타입은 Qdrant 페이로드에 없다. 색인을 다시 만들지 않고 결과에서
        # 거른다 — 후보를 넉넉히 뽑으므로 상위 k개는 대개 남는다.
        with connect() as conn:
            rows = conn.execute(
                "SELECT title FROM documents WHERE title = ANY(%s) AND entity_type = %s",
                ([h.title for h in hits], entity_type),
            ).fetchall()
        keep = {r["title"] for r in rows}
        hits = [h for h in hits if h.title in keep]
        for i, h in enumerate(hits, 1):
            h.rank = i
    facts = facts_for([h.title for h in hits]) if hits else []
    return SearchResponse(
        query=q, hits=hits, facts=facts, took_ms=int((time.time() - t0) * 1000)
    )


def facts_for(titles: list[str], limit: int = 24) -> list[Fact]:
    """검색 결과 문서들에 걸린 인포박스 사실을 모은다.

    분류(INSTANCE_OF)는 뺀다 — 문서당 수십 개라 실제 사실을 덮어버린다.
    """
    if not titles:
        return []
    seen = list(dict.fromkeys(titles))
    with connect() as conn:
        rows = conn.execute(
            """SELECT r.subject, r.predicate, r.object
                 FROM relations r
                 LEFT JOIN documents d ON d.title = r.subject
                WHERE r.subject = ANY(%s) AND r.source = 'infobox'
                ORDER BY d.pagerank DESC NULLS LAST, r.predicate
                LIMIT %s""",
            (seen, limit),
        ).fetchall()
    return [Fact(subject=r["subject"], predicate=r["predicate"],
                 object=r["object"]) for r in rows]


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


@app.get("/api/entity")
def entity(title: str, limit: int = 40):
    """엔티티 카드 — 그 문서에 대해 아는 모든 것.

    outlinks 가 "연결됐다"만 말하는 데 비해 relations 는 "어떻게 연결됐는가"를
    말한다. 나가는 관계(김연아 --국적--> 대한민국)와 들어오는 관계
    (누가 김연아를 무엇으로 가리키는가)를 함께 준다.
    """
    with connect() as conn:
        doc = conn.execute(
            """SELECT title, categories, pagerank, community, char_len,
                      last_modified, entity_type,
                      coalesce(array_length(outlinks,1),0) AS outdeg
                 FROM documents WHERE title = %s""",
            (title,),
        ).fetchone()
        if not doc:
            return {"error": f"수집되지 않은 문서: {title}"}

        out = conn.execute(
            """SELECT predicate, object, obj_kind, source, evidence
                 FROM relations WHERE subject = %s
                -- 인포박스가 가장 값어치 있다. 분류는 수가 많아 그냥 두면
                -- 국적·소속 같은 사실을 화면 밖으로 밀어낸다.
                ORDER BY CASE source WHEN 'infobox' THEN 0 WHEN 'title' THEN 1
                                     WHEN 'redirect' THEN 2 ELSE 3 END,
                         predicate, object LIMIT %s""",
            (title, limit),
        ).fetchall()
        inc = conn.execute(
            """SELECT r.subject, r.predicate, r.source
                 FROM relations r
                 LEFT JOIN documents d ON d.title = r.subject
                WHERE r.object = %s
                ORDER BY d.pagerank DESC NULLS LAST LIMIT %s""",
            (title, limit),
        ).fetchall()
        peers = conn.execute(
            """SELECT title FROM documents
                WHERE community = %s AND title <> %s
                ORDER BY pagerank DESC NULLS LAST LIMIT 8""",
            (doc["community"], title),
        ).fetchall() if doc["community"] is not None else []

    return {
        "title": doc["title"],
        "entity_type": doc["entity_type"],
        "pagerank": doc["pagerank"],
        "community": doc["community"],
        "community_peers": [r["title"] for r in peers],
        "categories": doc["categories"],
        "outdegree": doc["outdeg"],
        "char_len": doc["char_len"],
        "last_modified": doc["last_modified"],
        "relations_out": [
            {"predicate": r["predicate"], "object": r["object"],
             "kind": r["obj_kind"], "source": r["source"],
             "evidence": r["evidence"]} for r in out
        ],
        "relations_in": [
            {"subject": r["subject"], "predicate": r["predicate"],
             "source": r["source"]} for r in inc
        ],
    }


# 달력 문서. '1월 6일', '1992년', '1990년대' 는 연대순 색인이라 서로 무관한
# 문서를 전부 이어버린다. 그대로 두면 '손흥민 → 1월 6일 → 이순신' 처럼
# 어떤 두 문서든 2홉이 되어 경로가 아무것도 말해주지 않는다.
_CALENDAR = re.compile(r"^(\d{1,4}년(대)?|\d{1,2}월( \d{1,2}일)?|\d{1,2}월 \d{1,2}일)$")


@app.get("/api/graph/path")
def graph_path(source: str, target: str, max_hops: int = 4,
               avoid_calendar: bool = True):
    """두 문서를 잇는 최단 경로.

    양쪽에서 동시에 넓히는 양방향 BFS 다. 단방향 BFS 는 홉마다 분기가 수백
    배로 늘어 3홉이면 이미 수백만 노드가 된다. 양방향이면 각각 절반 깊이만
    가면 되므로 탐색량이 제곱근으로 줄어든다.

    링크 방향은 무시한다 — "무엇을 거쳐 닿는가"가 질문이지 "어느 쪽이
    가리키는가"가 아니다.
    """
    if source == target:
        return {"path": [source], "hops": 0}

    def neighbors(conn, titles: list[str]) -> dict[str, list[str]]:
        """여러 제목의 이웃을 한 번의 질의로 가져온다 (양방향)."""
        rows = conn.execute(
            """SELECT title, outlinks FROM documents WHERE title = ANY(%s)""",
            (titles,),
        ).fetchall()
        out: dict[str, list[str]] = {r["title"]: list(r["outlinks"]) for r in rows}
        back = conn.execute(
            """SELECT d.title AS src, t AS dst
                 FROM documents d, unnest(d.outlinks) t
                WHERE t = ANY(%s)""",
            (titles,),
        ).fetchall()
        for r in back:
            out.setdefault(r["dst"], []).append(r["src"])
        if avoid_calendar:
            for k in out:
                out[k] = [t for t in out[k]
                          if not _CALENDAR.match(t) and t not in CHROME_TITLES]
        return out

    with connect() as conn:
        for t in (source, target):
            if not conn.execute("SELECT 1 FROM documents WHERE title = %s",
                                (t,)).fetchone():
                return {"error": f"수집되지 않은 문서: {t}"}

        # from_side / to_side: 노드 -> 그 노드까지의 경로
        fwd = {source: [source]}
        bwd = {target: [target]}
        for hop in range(max_hops):
            # 작은 쪽을 넓힌다. 한쪽이 폭발해도 다른 쪽이 받쳐준다.
            grow, other = (fwd, bwd) if len(fwd) <= len(bwd) else (bwd, fwd)
            frontier = [t for t, p in grow.items() if len(p) == hop // 2 + 1] or list(grow)
            nb = neighbors(conn, frontier[:400])
            for src, dsts in nb.items():
                base = grow.get(src)
                if base is None:
                    continue
                for d in dsts:
                    if d in grow:
                        continue
                    grow[d] = base + [d]
                    if d in other:
                        a, b = grow[d], other[d]
                        path = a + b[::-1][1:] if grow is fwd else b + a[::-1][1:]
                        return {"path": path, "hops": len(path) - 1}
    return {"error": f"{max_hops}홉 안에서 경로를 찾지 못했다",
            "source": source, "target": target}


@app.get("/api/graph/viz")
def graph_viz(title: str, limit: int = 24):
    """시각화용 부분 그래프.

    중심 문서와 그 이웃, 그리고 **이웃끼리의 엣지**까지 준다. 이웃 간 연결을
    빼면 별 모양만 나와서 구조가 보이지 않는다.

    이웃은 PageRank 순으로 고른다. 문서당 링크가 수백 개라 전부 그리면
    헤어볼이 된다.
    """
    with connect() as conn:
        center = conn.execute(
            """SELECT title, pagerank, categories, outlinks, community,
                      coalesce(array_length(outlinks,1),0) AS outdeg
                 FROM documents WHERE title = %s""",
            (title,),
        ).fetchone()
        if not center:
            return {"error": f"수집되지 않은 문서: {title}"}

        # 중심 문서의 outlinks를 파라미터로 넘긴다. 서브쿼리로 배열을 꺼내
        # ANY에 바로 넣으면 '행 하나에 담긴 배열'이라 타입이 맞지 않는다.
        out = center["outlinks"] or []

        # 실제 문서로 존재하는 이웃만. 양방향(상호 링크)을 표시한다.
        #
        # CHROME_TITLES 는 화면에서만 뺀다. 페이지 UI에서 나온 링크라 주제와
        # 무관한데 인링크가 많아 PageRank 상위에 오른다(설계 §13.1.1).
        # 그래프에서는 시각적으로 특히 방해되지만, 저장된 데이터는 건드리지
        # 않는다 — 판별 기준이 정해지기 전까지는 표시 계층에서만 감춘다.
        neighbors = conn.execute(
            """
            SELECT d.title,
                   d.pagerank,
                   d.community,
                   (d.title = ANY(%s))               AS linked_from_center,
                   (d.outlinks @> ARRAY[%s]::text[]) AS links_to_center
              FROM documents d
             WHERE d.title <> %s
               AND d.title <> ALL(%s)
               AND (d.title = ANY(%s) OR d.outlinks @> ARRAY[%s]::text[])
             -- 같은 커뮤니티를 먼저 고른다. PageRank 만으로 고르면 '손흥민'
             -- 주위가 연도와 국가로 채워진다 — 링크는 많지만 주제가 아니다.
             ORDER BY (d.community IS NOT DISTINCT FROM %s) DESC,
                      d.pagerank DESC NULLS LAST
             LIMIT %s
            """,
            (out, title, title, CHROME_TITLES, out, title,
             center["community"], limit),
        ).fetchall()
        names = [r["title"] for r in neighbors]

        # 이웃끼리의 엣지
        inner: list[dict] = []
        if len(names) > 1:
            for r in conn.execute(
                """SELECT d.title AS src, t AS dst
                     FROM documents d, unnest(d.outlinks) AS t
                    WHERE d.title = ANY(%s) AND t = ANY(%s) AND d.title <> t""",
                (names, names),
            ).fetchall():
                inner.append({"source": r["src"], "target": r["dst"]})

    nodes = [{
        "id": center["title"], "center": True,
        "pagerank": center["pagerank"] or 0,
        "outdeg": center["outdeg"],
        "community": center["community"],
        "categories": center["categories"][:3],
    }]
    edges = []
    for r in neighbors:
        nodes.append({
            "id": r["title"], "center": False,
            "pagerank": r["pagerank"] or 0,
            "community": r["community"],
            "mutual": bool(r["linked_from_center"] and r["links_to_center"]),
        })
        if r["linked_from_center"]:
            edges.append({"source": center["title"], "target": r["title"]})
        if r["links_to_center"]:
            edges.append({"source": r["title"], "target": center["title"]})
    edges.extend(inner)

    # 화면에 그려진 노드 사이에 타입 있는 관계가 있으면 라벨로 얹는다.
    # 링크 엣지는 "연결됐다"까지만 말하지만 이건 "국적"이라고 말해준다.
    shown = [n["id"] for n in nodes]
    labels: dict[tuple[str, str], str] = {}
    with connect() as conn:
        for r in conn.execute(
            """SELECT subject, predicate, object FROM relations
                WHERE source = 'infobox' AND obj_kind = 'page'
                  AND subject = ANY(%s) AND object = ANY(%s)""",
            (shown, shown),
        ).fetchall():
            labels.setdefault((r["subject"], r["object"]), r["predicate"])
    for e in edges:
        lab = labels.get((e["source"], e["target"]))
        if lab:
            e["label"] = lab
    return {"center": center["title"], "nodes": nodes, "edges": edges,
            "relation_labels": len(labels)}


@app.get("/api/communities")
def communities(limit: int = 30):
    """커뮤니티 목록과 각 커뮤니티의 대표 문서."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT community, count(*) n,
                      (array_agg(title ORDER BY pagerank DESC NULLS LAST))[1:6] AS top
                 FROM documents WHERE community IS NOT NULL
                GROUP BY community ORDER BY n DESC LIMIT %s""",
            (limit,),
        ).fetchall()
        total = conn.execute(
            "SELECT count(DISTINCT community) n FROM documents WHERE community IS NOT NULL"
        ).fetchone()["n"]
    return {
        "count": total,
        "communities": [
            {"id": r["community"], "size": r["n"], "top": r["top"]} for r in rows
        ],
    }


@app.get("/api/types")
def types():
    """온톨로지 클래스별 문서 수와 대표 문서."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT entity_type AS t, count(*) n,
                      (array_agg(title ORDER BY pagerank DESC NULLS LAST))[1:8] AS top
                 FROM documents WHERE entity_type IS NOT NULL
                GROUP BY entity_type ORDER BY n DESC"""
        ).fetchall()
    return {"types": [{"name": r["t"], "count": r["n"], "top": r["top"]} for r in rows]}


@app.get("/kg", response_class=HTMLResponse)
def kg_page() -> str:
    return (ROOT / "web" / "kg.html").read_text(encoding="utf-8")


@app.get("/graph", response_class=HTMLResponse)
def graph_page() -> str:
    return (ROOT / "web" / "graph.html").read_text(encoding="utf-8")


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
