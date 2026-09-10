"""타입 있는 관계를 규칙으로 뽑는다 (KG-2).

`documents.outlinks` 는 "A 가 B 를 가리킨다"까지만 말한다. 검색에는 그걸로
충분하지만 질문에 답하려면 **어떻게** 연결됐는지가 필요하다 — 김연아의
국적이 대한민국인 것과, 김연아 문서가 대한민국을 언급한 것은 다른 사실이다.

관계를 어디서 얻느냐가 성패를 가른다. 산문에서 LLM 으로 뽑으면 비싸고
정밀도가 낮다. 여기서는 네 갈래 모두 규칙으로 뽑는다:

  infobox  : 인포박스 표의 라벨-값 쌍. 가장 값어치가 크다.
  title    : 제목의 A/B 패턴 → B PART_OF A ('김일성/생애' → '김일성')
  category : 분류 → INSTANCE_OF
  redirect : 리다이렉트 → SAME_AS

인포박스 HTML 구조 (실측):

    <td ...><div class='wiki-paragraph'><strong>국적</strong></div></td>
    <td ...><div class='wiki-paragraph'>
        <a class='wiki-link-internal' href='/w/...' title='대한민국'>…</a>
    </div></td>

`wiki-link-internal`·`wiki-paragraph` 는 빌드 해시(_0MSSsdmG)가 아니라
나무위키가 직접 붙인 의미론적 클래스라 배포마다 바뀌지 않는다.
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse
from dataclasses import dataclass

__all__ = ["Relation", "from_infobox", "from_title", "PREDICATE_WHITELIST"]

# 인포박스 라벨 중 관계로 쓸 만한 것만 고른다. '신체', '재산' 같은 값은
# 리터럴이라 그래프 엣지가 되지 않고, 문서마다 표기가 제각각이라 노이즈다.
PREDICATE_WHITELIST = {
    "국적", "출생", "사망", "소속", "소속팀", "소속사", "직업", "학력",
    "가족", "배우자", "자녀", "부모", "형제", "감독", "제작사", "배급사",
    "개발", "유통", "장르", "원작", "주연", "출연", "작가", "작곡가",
    "소재지", "본사", "설립자", "창립자", "대표", "회장", "수도", "언어",
    "정치체제", "통화", "종교", "제조사", "엔진", "플랫폼", "출시",
    "전신", "후신", "상위 조직", "산하", "링크",
}

# 라벨 셀: <strong> 안의 텍스트. 값 셀: 바로 뒤따르는 <td>.
_ROW = re.compile(
    r"<strong[^>]*>(?P<label>[^<]{1,20})</strong>\s*</div>\s*</td>"
    r"\s*<td[^>]*>(?P<value>.*?)</td>",
    re.S,
)
# 값 안의 내부 문서 링크. title 속성이 곧 문서 제목이다.
_LINK = re.compile(r"""<a[^>]*\bclass=['"]wiki-link-internal['"][^>]*\btitle=['"]([^'"]+)['"]""")
_TAG = re.compile(r"<[^>]+>")
_NON_ARTICLE = ("분류:", "파일:", "틀:", "사용자:")

# 값 셀 안에 딸려오는 부수 링크. 관계의 목적어가 될 수 없다 —
# 이순신의 '출생 → 음력', 손흥민의 '학력 → 전학' 같은 것이 실제로 나왔다.
_OBJECT_STOPLIST = frozenset({
    "음력", "양력", "전학", "유스", "현재", "당시", "예정", "미상", "불명",
    "이하", "이상", "년", "월", "일", "세", "명", "기타", "본문", "링크",
})


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    obj_kind: str      # 'page' | 'literal'
    source: str        # 'infobox' | 'title' | 'category' | 'redirect'
    evidence: str | None = None


def _clean(s: str) -> str:
    return " ".join(_html.unescape(_TAG.sub(" ", s)).split())


def from_infobox(title: str, html: str, *, limit: int = 60) -> list[Relation]:
    """인포박스 표에서 (주어, 술어, 목적어)를 뽑는다.

    값 셀에 내부 링크가 있으면 그 문서를 목적어로 삼고(page), 없으면
    텍스트를 리터럴로 남긴다. 리터럴도 버리지 않는 이유는 '출생 1990년
    9월 5일' 처럼 링크가 걸리지 않는 사실이 많기 때문이다.
    """
    out: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    # 같은 라벨이 문서 뒤쪽 통계표에 다시 나온다. 손흥민 문서에는 시즌별
    # 기록표에도 '국적' 열이 있어 '국적 → 도움', '국적 → 득점 수' 같은
    # 관계가 생겼다. 인포박스는 문서 맨 위에 있으므로 라벨당 첫 행만 쓴다.
    done_labels: set[str] = set()
    for m in _ROW.finditer(html):
        label = _clean(m.group("label"))
        if label not in PREDICATE_WHITELIST or label in done_labels:
            continue
        done_labels.add(label)
        value_html = m.group("value")
        text = _clean(value_html)[:200]
        if not text:
            continue
        targets = [_html.unescape(t) for t in _LINK.findall(value_html)]
        targets = [t for t in targets
                   if t and not t.startswith(_NON_ARTICLE)
                   and t not in _OBJECT_STOPLIST]
        if targets:
            for t in dict.fromkeys(targets):
                key = (label, t, "page")
                if key in seen:
                    continue
                seen.add(key)
                out.append(Relation(title, label, t, "page", "infobox", text))
        else:
            key = (label, text, "literal")
            if key not in seen:
                seen.add(key)
                out.append(Relation(title, label, text, "literal", "infobox", text))
        if len(out) >= limit:
            break
    return out


def from_title(title: str) -> list[Relation]:
    """'김일성/생애' → ('김일성/생애', PART_OF, '김일성').

    나무위키는 하위 문서를 슬래시로 표기한다. 여러 단계면 바로 위 한 칸만
    엮는다 — '북한/경제/역사/2023년' 은 '북한/경제/역사' 의 부분이다.
    """
    if "/" not in title:
        return []
    parent = title.rsplit("/", 1)[0].strip()
    if not parent or parent == title:
        return []
    return [Relation(title, "PART_OF", parent, "page", "title", title)]
