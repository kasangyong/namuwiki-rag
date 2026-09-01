"""문서/청크 식별자.

doc_id는 Kafka 파티션 키로 쓰인다. 같은 문서가 항상 같은 파티션으로 가야
구버전이 신버전을 덮어쓰는 사고를 막을 수 있다 (설계 §5.1).
"""

from __future__ import annotations

import hashlib
import unicodedata

__all__ = ["normalize_title", "doc_id", "chunk_id", "content_hash"]


def normalize_title(title: str) -> str:
    """제목 정규화.

    나무위키 제목은 한글이라 NFC/NFD 차이가 실제로 발생한다. macOS나 일부
    HTTP 클라이언트가 NFD로 넘기면 같은 문서가 다른 doc_id를 갖게 되므로
    NFC로 통일한다. 공백은 접어주되 대소문자는 보존한다 (나무위키는
    대소문자를 구분한다).
    """
    t = unicodedata.normalize("NFC", title)
    t = t.replace("_", " ")
    return " ".join(t.split())


def doc_id(title: str) -> str:
    return hashlib.sha1(normalize_title(title).encode("utf-8")).hexdigest()


def chunk_id(doc: str, seq: int) -> str:
    return f"{doc}:{seq:05d}"


def content_hash(text: str) -> str:
    """멱등성 판정용. 해시가 같으면 하위 단계가 즉시 반환한다 (설계 §8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
