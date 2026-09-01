"""블롭 저장소 — 클레임 체크 패턴 (설계 §5.5).

문서 본문은 최대 16만 자라 Kafka 메시지에 넣을 수 없다. 본문은 여기 두고
Kafka에는 참조(blob_ref)만 흘린다. 파서 로직을 고쳤을 때 재크롤 없이
docs.raw를 재생하려면 원본 HTML이 남아 있어야 한다.

Phase 1~2는 로컬 파일시스템, Phase 3에서 SeaweedFS(S3)로 교체한다.
교체 지점을 이 파일 하나로 가둔다.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Protocol

__all__ = ["BlobStore", "FileSystemBlobStore"]


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> str:
        """저장하고 blob_ref를 반환한다."""

    def get(self, ref: str) -> bytes:
        """blob_ref로 원본 바이트를 되찾는다."""


class FileSystemBlobStore:
    """2단계 샤딩 파일시스템 저장소.

    NTFS는 한 디렉터리에 파일이 수만 개를 넘으면 느려진다. key(=doc_id, sha1 hex)의
    앞 4자로 2단계 샤딩하면 124만 파일이 65,536개 디렉터리에 고르게 흩어져
    디렉터리당 20개 수준이 된다.

    gzip으로 저장한다 (실측 압축률 34%, 설계 §2.3).
    """

    scheme = "blob://"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if len(key) < 4:
            key = key.rjust(4, "0")
        return self.root / key[:2] / key[2:4] / f"{key}.gz"

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".gz.tmp")
        tmp.write_bytes(gzip.compress(data, 6))
        tmp.replace(path)  # 원자적 교체 — 중간에 죽어도 반쪽 파일이 남지 않는다
        return f"{self.scheme}{key}"

    def get(self, ref: str) -> bytes:
        key = ref.removeprefix(self.scheme)
        return gzip.decompress(self._path(key).read_bytes())

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
