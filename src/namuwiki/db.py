"""PostgreSQL 스키마와 접근 헬퍼 (설계 §5.2)."""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from .config import SETTINGS

__all__ = ["connect", "init_schema", "SCHEMA"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id        TEXT PRIMARY KEY,
  title         TEXT NOT NULL UNIQUE,
  source        TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  blob_ref      TEXT,
  char_len      INT,
  last_modified TIMESTAMPTZ,
  fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  parsed_at     TIMESTAMPTZ,
  categories    TEXT[] NOT NULL DEFAULT '{}',
  -- 지식그래프의 엣지. 실측상 문서당 약 817개 (124만 문서 → 약 10억 엣지).
  -- 별도 엣지 테이블은 sha1 40자 × 2 × 10억 = 80GB로 과하다. 배열로 두면
  -- TOAST 압축이 먹어 훨씬 가볍고, GIN 인덱스로 역링크 질의도 된다.
  outlinks      TEXT[] NOT NULL DEFAULT '{}'
);
-- 기존 설치본 마이그레이션. CREATE TABLE IF NOT EXISTS는 이미 있는 테이블에
-- 컬럼을 추가해주지 않으므로, 아래 인덱스보다 반드시 먼저 와야 한다.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS outlinks TEXT[] NOT NULL DEFAULT '{}';
-- 링크 중심성 (KG-1). 검색 순위와 크롤 우선순위에 쓴다.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pagerank REAL;
CREATE INDEX IF NOT EXISTS documents_pagerank_idx
  ON documents (pagerank DESC NULLS LAST);

-- 역링크: WHERE outlinks @> ARRAY['김연아']
CREATE INDEX IF NOT EXISTS documents_outlinks_idx ON documents USING GIN (outlinks);
CREATE INDEX IF NOT EXISTS documents_categories_idx ON documents USING GIN (categories);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      TEXT PRIMARY KEY,
  doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  seq           INT  NOT NULL,
  section_id    TEXT,
  heading_path  TEXT[] NOT NULL DEFAULT '{}',
  text          TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  embedded_at   TIMESTAMPTZ,
  UNIQUE (doc_id, seq)
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_pending_idx
  ON chunks (doc_id) WHERE embedded_at IS NULL;

CREATE TABLE IF NOT EXISTS crawl_state (
  doc_id        TEXT PRIMARY KEY,
  title         TEXT NOT NULL UNIQUE,
  discovered_by TEXT NOT NULL,
  last_fetched  TIMESTAMPTZ,
  last_hash     TEXT,
  http_status   INT,
  fail_count    INT NOT NULL DEFAULT 0,
  priority      INT NOT NULL DEFAULT 100,
  next_due_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  queued_at     TIMESTAMPTZ
);
-- 크롤 대상 선정 쿼리가 이 인덱스를 탄다
CREATE INDEX IF NOT EXISTS crawl_due_idx
  ON crawl_state (priority, next_due_at) WHERE fail_count < 5;

-- 리다이렉트 별칭 (설계 §13.1 KG-2의 SAME_AS).
-- 나무위키 문서의 34.4%가 리다이렉트다. 크롤러가 리다이렉트를 따라가므로
-- 그대로 두면 같은 내용이 별칭 제목으로 한 번 더 저장되어 ① 임베딩을 두 번
-- 하고 ② 그래프 노드가 갈라진다 (픽사 / 픽사 애니메이션 스튜디오).
CREATE TABLE IF NOT EXISTS redirects (
  alias      TEXT PRIMARY KEY,
  canonical  TEXT NOT NULL,
  noticed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS redirects_canonical_idx ON redirects (canonical);

-- 추출 수율 카나리아 (설계 §6.2): 파서가 조용히 깨지는 것을 감지한다
CREATE TABLE IF NOT EXISTS canary (
  title          TEXT PRIMARY KEY,
  baseline_chars INT NOT NULL,
  last_chars     INT,
  last_checked   TIMESTAMPTZ,
  status         TEXT NOT NULL DEFAULT 'ok'
);

-- 파이프라인 관측용 카운터
CREATE TABLE IF NOT EXISTS pipeline_stats (
  stage      TEXT NOT NULL,
  metric     TEXT NOT NULL,
  value      BIGINT NOT NULL DEFAULT 0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (stage, metric)
);
"""


@contextmanager
def connect(dsn: str | None = None, autocommit: bool = True):
    conn = psycopg.connect(dsn or SETTINGS.pg_dsn, row_factory=dict_row)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        conn.close()


def init_schema(dsn: str | None = None) -> None:
    with connect(dsn) as conn:
        conn.execute(SCHEMA)


def bump(conn, stage: str, metric: str, delta: int = 1) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_stats (stage, metric, value, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (stage, metric) DO UPDATE
          SET value = pipeline_stats.value + EXCLUDED.value, updated_at = now()
        """,
        (stage, metric, delta),
    )
