"""환경 설정. 전부 환경변수로 덮어쓸 수 있다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


@dataclass(frozen=True)
class Settings:
    # --- 인프라 ---
    kafka_bootstrap: str = field(
        default_factory=lambda: _env("NW_KAFKA", "localhost:29092")
    )
    pg_dsn: str = field(
        default_factory=lambda: _env(
            "NW_PG", "postgresql://namu:namu@localhost:5433/namuwiki"
        )
    )
    qdrant_url: str = field(
        default_factory=lambda: _env("NW_QDRANT", "http://localhost:6333")
    )
    collection: str = field(
        default_factory=lambda: _env("NW_COLLECTION", "namuwiki_chunks")
    )

    # --- 저장소 ---
    blob_root: Path = field(
        default_factory=lambda: Path(_env("NW_BLOB_ROOT", str(ROOT / "data" / "blobs")))
    )

    # --- 크롤 정책 (설계 §4.3) ---
    # 나무위키는 "크롤링은 서버 부담"이라고 명시했다. 기본값을 보수적으로 둔다.
    crawl_concurrency: int = field(
        default_factory=lambda: _env_int("NW_CRAWL_CONCURRENCY", 4)
    )
    crawl_min_interval_ms: int = field(
        default_factory=lambda: _env_int("NW_CRAWL_INTERVAL_MS", 100)
    )
    crawl_timeout_s: int = field(default_factory=lambda: _env_int("NW_CRAWL_TIMEOUT", 30))
    crawl_max_retries: int = field(default_factory=lambda: _env_int("NW_CRAWL_RETRIES", 3))
    # 서킷 브레이커: 1분간 실패 N회 → 정지
    breaker_threshold: int = field(default_factory=lambda: _env_int("NW_BREAKER", 10))
    breaker_cooldown_s: int = field(
        default_factory=lambda: _env_int("NW_BREAKER_COOLDOWN", 600)
    )
    # HTTP 헤더는 ASCII만 허용된다. 한글을 넣으면 httpx가 UnicodeEncodeError로
    # 죽는다. 상대 서버가 연락할 수 있도록 식별자와 연락처를 남긴다 (설계 §4.3).
    user_agent: str = field(
        default_factory=lambda: _env(
            "NW_UA",
            "namuwiki-rag-research/0.1 (non-commercial personal research; "
            "respects robots.txt; contact: github.com/local/namuwiki-rag)",
        )
    )
    # sidebar.json은 15개/약 32초 윈도우다. 30초를 넘기면 편집을 놓친다 (설계 §2.4).
    sidebar_poll_s: int = field(default_factory=lambda: _env_int("NW_SIDEBAR_POLL", 10))

    # --- 모델 (Phase 0 A/B 후 확정) ---
    dense_model: str = field(
        default_factory=lambda: _env("NW_DENSE_MODEL", "nlpai-lab/KURE-v1")
    )
    reranker_model: str = field(
        default_factory=lambda: _env("NW_RERANKER", "BAAI/bge-reranker-v2-m3")
    )
    embed_batch: int = field(default_factory=lambda: _env_int("NW_EMBED_BATCH", 16))
    # SPLADE 로짓은 (배치 × 길이 × 어휘 25만)이라 dense보다 훨씬 무겁다.
    # 배치 2, 길이 450 기준 CPU(fp32) 약 0.9GB, GPU(fp16) 약 0.45GB.
    # 배치 16으로 두면 8GB를 넘겨 머신 전체가 느려진다 (실측).
    sparse_batch: int = field(default_factory=lambda: _env_int("NW_SPARSE_BATCH", 2))
    # sparse 전용 시퀀스 길이. dense와 분리해 둔 이유:
    # sparse가 임베딩 연산의 83%를 쓰고(실측) VRAM 병목이기도 하다.
    # 256으로 낮추면 연산·메모리가 절반이 되지만 청크가 평균 450토큰이라
    # 뒷부분 어휘 신호를 잃는다. 하이브리드 검색에서 sparse의 역할이
    # 고유명사 정확매칭이므로 **기본값은 낮추지 않는다.** 필요할 때만 당긴다.
    sparse_max_seq: int = field(
        default_factory=lambda: _env_int("NW_SPARSE_MAX_SEQ", 512)
    )
    dense_dim: int = field(default_factory=lambda: _env_int("NW_DENSE_DIM", 1024))
    # 청커가 850자(≈512토큰)로 자르므로 그 이상은 낭비다. BGE-M3 계열의
    # 기본값 8192를 그대로 쓰면 어텐션 비용이 불필요하게 커진다.
    max_seq_length: int = field(default_factory=lambda: _env_int("NW_MAX_SEQ", 512))
    # "cuda" | "cpu" | "" (자동). GPU를 다른 작업에 양보해야 할 때 cpu로 강제한다.
    device: str = field(default_factory=lambda: _env("NW_DEVICE", ""))

    # --- 추출 수율 카나리아 (설계 §6.2) ---
    canary_tolerance: float = 0.30

    base_url: str = "https://namu.wiki"


# 모든 문서에 붙는 UI 링크. 주제 그래프의 노드가 아니다.
#
# 인링크 비율로는 걸러낼 수 없다 — 실측상 '크리에이티브 커먼즈 라이선스'는
# 16.3%로 '서울특별시'(16.5%)와 '영국'(15.2%) 사이에 정확히 끼어 있어,
# 어떤 임계값을 잡아도 진짜 허브 문서를 함께 버린다. 그래서 명시한다.
CHROME_TITLES = ("편집 요청", "크리에이티브 커먼즈 라이선스")

SETTINGS = Settings()


class Topics:
    CRAWL_REQUESTS = "crawl.requests"
    DOCS_RAW = "docs.raw"
    CHUNKS_READY = "chunks.ready"
    CRAWL_DLQ = "crawl.dlq"
    PARSE_DLQ = "parse.dlq"
    EMBED_DLQ = "embed.dlq"


# (토픽, 파티션, cleanup.policy) — 설계 §5.1
TOPIC_SPECS = [
    (Topics.CRAWL_REQUESTS, 6, "delete"),
    (Topics.DOCS_RAW, 6, "compact"),
    (Topics.CHUNKS_READY, 12, "compact"),
    (Topics.CRAWL_DLQ, 1, "delete"),
    (Topics.PARSE_DLQ, 1, "delete"),
    (Topics.EMBED_DLQ, 1, "delete"),
]
