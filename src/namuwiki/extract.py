"""나무위키 렌더링 HTML에서 본문과 섹션 구조를 추출한다.

설계 제약 (docs/superpowers/specs/2026-08-28-namuwiki-rag-design.md §2.7, §6.1):
나무위키의 CSS 클래스명은 빌드 해시로 난독화되어 있어 배포마다 바뀐다.
따라서 클래스 셀렉터를 절대 쓰지 않고, 아래 의미론적 앵커만 사용한다.

  - 섹션 헤딩: <h2><a id='s-4.1' href='#toc'>4.1.</a><span id='수훈'>수훈...</span></h2>
      * id='s-N.M' 의 점 표기가 곧 헤딩 계층이다.
      * <span> 의 id 속성값이 곧 헤딩 텍스트다.
  - 문서 제목:  <h1>...<a href="/w/..."><span>제목</span></a></h1>
  - 수정 시각:  최근 수정 시각: <time datetime="2025-01-14T11:44:01.000Z">
  - 본문 끝:    "이 저작물은 ... CC BY-NC-SA 2.0 KR 에 따라 이용할 수 있습니다"

이 앵커들은 외부 링크(#s-4)와 라이선스 고지 의무 때문에 함부로 바뀌지 않는다.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field

__all__ = ["Section", "ExtractedDoc", "extract", "ExtractionError"]


class ExtractionError(Exception):
    """본문 경계를 찾지 못했을 때. 호출자는 DLQ로 보내야 한다."""


# --- 제거 대상 -------------------------------------------------------------

_DROP_BLOCKS = re.compile(
    r"<(script|style|svg|noscript)\b[^>]*>.*?</\1\s*>", re.S | re.I
)
_SELF_CLOSING_NOISE = re.compile(r"<(?:img|input|br)\b[^>]*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")

# 접이식 네비게이션 틀: <details><summary>[ 펼치기 · 접기 ]</summary>…</details>
# <details>/<summary>는 표준 HTML이라 CSS 클래스와 달리 안정적이다.
# 본문이 아니라 링크 목록이므로 임베딩 품질을 위해 통째로 버린다.
_NAVBOX = re.compile(
    r"<details\b[^>]*>\s*<summary\b[^>]*>(?:(?!</summary>).)*?(?:펼치기|접기)"
    r".*?</details\s*>",
    re.S,
)

# 문서가 아닌 경로를 가리키는 앵커(툴바·분류·편집 링크)는 텍스트째로 버린다.
# 이 경로들은 robots.txt에 명시된 것이라 CSS 클래스보다 훨씬 안정적이다.
_CHROME_HREF = (
    r"/edit/|/discuss/|/history/|/member/|/backlink/|/title_history/|"
    r"/RecentChanges|/RecentDiscuss|/Search|/action/|/w/%EB%B6%84%EB%A5%98:|/w/분류:"
)
_CHROME_LINK = re.compile(
    rf"""<a\b[^>]*\bhref=['"](?:{_CHROME_HREF})[^'"]*['"][^>]*>.*?</a\s*>""",
    re.S | re.I,
)

# 분류 링크 → 메타데이터로 추출
_CATEGORY_LINK = re.compile(
    r"""<a\b[^>]*\bhref=['"]/w/(?:%EB%B6%84%EB%A5%98%3A|%EB%B6%84%EB%A5%98:|분류:)"""
    r"""([^'"]+)['"][^>]*>(.*?)</a\s*>""",
    re.S | re.I,
)

# 남는 라벨성 잡텍스트 (툴바 링크 제거 후 잔여)
_NOISE_LINE = re.compile(
    r"^(?:분류|편집|토론|역사|최근 수정 시각:?|"
    r"\[?\s*펼치기\s*·?\s*접기\s*\]?|더 보기|접기)$"
)

# 본문 끝 랜드마크. 라이선스 고지는 법적 의무라 반드시 존재한다.
_LICENSE_MARKERS = (
    "이 저작물은",
    "CC BY-NC-SA 2.0 KR",
    "크리에이티브 커먼즈",
)

# 본문 시작 랜드마크
_H1 = re.compile(r"<h1\b[^>]*>(.*?)</h1\s*>", re.S | re.I)
_TIME = re.compile(r'<time[^>]*\bdatetime="([^"]+)"', re.I)

# 섹션 헤딩 전체 블록
_HEADING = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.S | re.I)
# 헤딩 블록 안의 섹션 앵커와 제목
_SEC_ID = re.compile(r"""\bid=['"](s-[0-9]+(?:\.[0-9]+)*)['"]""")
_SEC_TITLE = re.compile(r"""<span[^>]*\bid=['"]([^'"]+)['"]""")

# 본문 노이즈
_EDIT_MARK = re.compile(r"\[\s*편집\s*\]")
_FOOTNOTE_BACKREF = re.compile(r"\[\s*\d+\s*\]")
_WS = re.compile(r"[ \t ​]+")
_BLANKS = re.compile(r"\n{3,}")
# 제어문자 제거. PostgreSQL의 text 타입은 NUL(0x00)을 담을 수 없어서
# 그냥 두면 색인 단계에서 DataError로 문서가 통째로 버려진다.
# 탭과 줄바꿈만 남긴다.
_CTRL = re.compile("[" + "".join(chr(c) for c in [*range(0, 9), 11, 12, *range(14, 32), 127]) + "]")


@dataclass
class Section:
    section_id: str | None  # 's-4.1' / 섹션 없는 문서는 None
    heading: str | None  # '수훈'
    level: int  # 1부터. 섹션 없으면 0
    text: str


@dataclass
class ExtractedDoc:
    title: str | None
    last_modified: str | None  # ISO8601 문자열
    sections: list[Section] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    # 본문 구간의 원본 HTML. 링크 수확은 반드시 여기서만 해야 한다.
    # 전체 HTML에서 뽑으면 푸터의 라이선스 링크와 툴바가 그래프에 섞여
    # PageRank 상위를 오염시킨다 (실측: '편집 요청'과 '크리에이티브 커먼즈
    # 라이선스'가 5·6위로 올라왔다 — 모든 페이지가 링크하므로).
    body_html: str = ""
    # 링크 수확용 범위. body_html 과 달리 **문서 상단부터** 포함한다.
    # 인포박스가 첫 섹션 앵커보다 위에 있어서, body_html 로 링크를 뽑으면
    # 수도·통화·언어 같은 핵심 관계가 통째로 빠진다(실측: 대한민국
    # 1,468 → 839개). 잘라낼 것은 푸터의 UI·라이선스 링크뿐이다.
    link_html: str = ""

    @property
    def text(self) -> str:
        """전체 본문. 수율 카나리아(§6.2)가 이 길이를 감시한다."""
        out = []
        for s in self.sections:
            if s.heading:
                out.append(f"== {s.heading} ==")
            if s.text:
                out.append(s.text)
        return "\n".join(out)

    def heading_path_of(self, idx: int) -> list[str]:
        """idx번째 섹션의 조상 헤딩 경로. 's-4.1' 이면 ['수상 기록', '수훈']."""
        sec = self.sections[idx]
        if not sec.section_id:
            return []
        parts = sec.section_id.removeprefix("s-").split(".")
        path: list[str] = []
        for depth in range(1, len(parts) + 1):
            want = "s-" + ".".join(parts[:depth])
            for s in self.sections:
                if s.section_id == want and s.heading:
                    path.append(s.heading)
                    break
        return path


def _strip_to_text(fragment: str) -> str:
    fragment = _DROP_BLOCKS.sub(" ", fragment)
    fragment = _NAVBOX.sub(" ", fragment)
    fragment = _CHROME_LINK.sub(" ", fragment)
    fragment = _SELF_CLOSING_NOISE.sub(" ", fragment)
    # 블록 경계를 줄바꿈으로 보존한 뒤 태그 제거
    fragment = re.sub(r"</(p|div|li|tr|h[1-6])\s*>", "\n", fragment, flags=re.I)
    fragment = _TAG.sub(" ", fragment)
    text = _html.unescape(fragment)
    text = _CTRL.sub("", text)
    text = _EDIT_MARK.sub(" ", text)
    text = _WS.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _NOISE_LINE.match(ln)]
    text = "\n".join(lines)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def _extract_categories(html_doc: str) -> list[str]:
    """분류 이름공간 링크를 메타데이터로 회수한다 (검색 필터용, 설계 §5.2)."""
    seen: dict[str, None] = {}
    for m in _CATEGORY_LINK.finditer(html_doc):
        name = _html.unescape(_TAG.sub("", m.group(2))).strip()
        if name and name not in seen:
            seen[name] = None
    return list(seen)


def _find_body_bounds(html_doc: str) -> tuple[int, int]:
    """본문의 [시작, 끝) 바이트 오프셋. CSS에 의존하지 않는다."""
    # 끝: 라이선스 고지 직전
    end = -1
    for marker in _LICENSE_MARKERS:
        pos = html_doc.rfind(marker)
        if pos != -1:
            end = pos if end == -1 else min(end, pos)
    if end == -1:
        end = len(html_doc)

    # 시작: 첫 섹션 앵커가 있으면 그 헤딩부터, 없으면 <time> 뒤부터
    start = -1
    for m in _HEADING.finditer(html_doc):
        if m.start() >= end:
            break
        if _SEC_ID.search(m.group(2)):
            start = m.start()
            break
    if start == -1:
        t = _TIME.search(html_doc)
        if t:
            # m.end()는 datetime 속성 뒤(태그 내부)라 </time> 뒤로 넘겨야
            # 'data-v-…>' 같은 태그 잔해가 본문에 새지 않는다.
            close = html_doc.find("</time>", t.end())
            start = (close + len("</time>")) if close != -1 else t.end()
        else:
            start = 0

    if start >= end:
        raise ExtractionError(f"본문 경계 역전: start={start} end={end}")
    return start, end


def extract(html_doc: str) -> ExtractedDoc:
    """렌더링된 나무위키 HTML → 섹션 구조.

    Raises:
        ExtractionError: 본문 경계를 찾지 못한 경우.
    """
    if not html_doc or len(html_doc) < 500:
        raise ExtractionError(f"HTML이 너무 짧다: {len(html_doc)}B")

    # 메타데이터는 잘라내기 전에 뽑는다
    title = None
    if (m := _H1.search(html_doc)) is not None:
        title = _strip_to_text(m.group(1)) or None
    last_modified = None
    if (m := _TIME.search(html_doc)) is not None:
        last_modified = m.group(1)

    start, end = _find_body_bounds(html_doc)
    body = html_doc[start:end]

    # 섹션 헤딩 위치 수집 (섹션 앵커를 가진 헤딩만)
    marks: list[tuple[int, int, str, str, int]] = []  # (시작, 끝, sec_id, 제목, level)
    for m in _HEADING.finditer(body):
        inner = m.group(2)
        sid_m = _SEC_ID.search(inner)
        if not sid_m:
            continue
        sid = sid_m.group(1)
        title_m = _SEC_TITLE.search(inner)
        heading = (
            _html.unescape(title_m.group(1)).strip()
            if title_m
            else _strip_to_text(inner).lstrip("0123456789. ").strip()
        )
        level = sid.removeprefix("s-").count(".") + 1
        marks.append((m.start(), m.end(), sid, heading, level))

    sections: list[Section] = []
    if not marks:
        text = _strip_to_text(body)
        if text:
            sections.append(Section(None, None, 0, text))
    else:
        # 첫 헤딩 앞의 도입부(있으면)
        preamble = _strip_to_text(body[: marks[0][0]])
        if len(preamble) > 40:
            sections.append(Section(None, None, 0, preamble))
        for i, (_, h_end, sid, heading, level) in enumerate(marks):
            body_end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
            sections.append(
                Section(sid, heading, level, _strip_to_text(body[h_end:body_end]))
            )

    return ExtractedDoc(
        title=title,
        last_modified=last_modified,
        sections=sections,
        categories=_extract_categories(html_doc),
        body_html=body,
        link_html=html_doc[:end],
    )
