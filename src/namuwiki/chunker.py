"""섹션 → 청크.

설계 §6.3:
  - 목표 850자 (512 토큰 ≈ 한국어 1토큰 ≈ 1.6자)
  - 오버랩 15% (~128자)
  - 200자 미만은 인접 청크에 병합
  - 헤딩 경로를 텍스트 앞에 붙여 청크 자체가 문맥을 갖게 한다

문서의 10% 이상(p10=336자)은 단일 청크가 된다. 짧은 문서를 억지로 쪼개지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import ExtractedDoc
from .ids import chunk_id, content_hash

__all__ = ["Chunk", "chunk_document", "TARGET_CHARS", "OVERLAP_CHARS", "MIN_CHARS"]

TARGET_CHARS = 850
OVERLAP_CHARS = 128
MIN_CHARS = 200

_PARA = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    section_id: str | None
    heading_path: list[str]
    text: str
    content_hash: str

    @property
    def char_len(self) -> int:
        return len(self.text)


def _split_long(text: str, target: int, overlap: int) -> list[str]:
    """목표 크기를 넘는 텍스트를 문단→문장 경계 우선으로 자른다."""
    if len(text) <= target:
        return [text]

    # 1차: 문단 경계로 모은다
    units = [p.strip() for p in _PARA.split(text) if p.strip()]
    # 문단 하나가 너무 길면 문장 경계로 더 쪼갠다
    expanded: list[str] = []
    for u in units:
        if len(u) <= target:
            expanded.append(u)
            continue
        sentences = re.split(r"(?<=[.!?。」』\n])\s+", u)
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) + 1 > target:
                expanded.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            expanded.append(buf)

    # 2차: 목표 크기까지 뭉치고, 경계마다 오버랩을 준다
    out: list[str] = []
    buf = ""
    for u in expanded:
        if buf and len(buf) + len(u) + 1 > target:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail} {u}".strip() if tail else u
        else:
            buf = f"{buf}\n{u}".strip() if buf else u
    if buf:
        out.append(buf)
    return out


def chunk_document(doc: str, title: str, extracted: ExtractedDoc) -> list[Chunk]:
    """추출된 문서를 청크 목록으로. doc은 doc_id."""
    pieces: list[tuple[str | None, list[str], str]] = []  # (section_id, path, text)

    for idx, sec in enumerate(extracted.sections):
        if not sec.text.strip():
            continue
        path = extracted.heading_path_of(idx)
        for body in _split_long(sec.text, TARGET_CHARS, OVERLAP_CHARS):
            pieces.append((sec.section_id, path, body))

    # 너무 짧은 조각은 같은 섹션의 직전 조각에 병합
    merged: list[tuple[str | None, list[str], str]] = []
    for sid, path, body in pieces:
        if (
            merged
            and len(body) < MIN_CHARS
            and merged[-1][0] == sid
            and len(merged[-1][2]) + len(body) <= TARGET_CHARS * 1.5
        ):
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1], f"{prev[2]}\n{body}")
        else:
            merged.append((sid, path, body))

    chunks: list[Chunk] = []
    for seq, (sid, path, body) in enumerate(merged):
        prefix = " > ".join([title, *path])
        text = f"{prefix}\n{body}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id(doc, seq),
                doc_id=doc,
                seq=seq,
                section_id=sid,
                heading_path=path,
                text=text,
                content_hash=content_hash(text),
            )
        )
    return chunks
