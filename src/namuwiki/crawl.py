"""크롤 정책 — 속도 제한과 서킷 브레이커 (설계 §4.3).

나무위키는 robots.txt로 /w/ 를 허용하면서도 "크롤링은 서버 부담이니 덤프를
받으라"고 안내한다. 허용은 됐지만 환영받는 건 아니므로, 상대 서버를 보호하는
장치를 정책이 아니라 코드로 강제한다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass

import httpx

from .config import SETTINGS

log = logging.getLogger(__name__)

__all__ = ["RateLimiter", "CircuitBreaker", "CircuitOpen", "fetch_document", "iter_links"]

# robots.txt Allow 경로만 크롤한다. 그 외는 전부 Disallow: / 에 걸린다.
_ARTICLE_PATH = re.compile(r"^/w/([^#?]+)$")
# 문서가 아닌 이름공간 — 본문 코퍼스에서 제외한다
_NON_ARTICLE_NS = (
    "분류:", "틀:", "파일:", "사용자:", "나무위키:", "휴지통:",
    "투표:", "토론:", "위키운영:", "템플릿:",
)


class CircuitOpen(Exception):
    """서킷이 열려 있어 요청을 보내지 않았다."""


class RateLimiter:
    """도메인 전역 최소 간격 보장. 동시성과 무관하게 요청 사이 간격을 강제한다.

    asyncio.Lock을 쓰지 않는다. 크롤러는 배치마다 asyncio.run()으로 새 이벤트
    루프를 만드는데, Lock은 처음 만들어진 루프에 묶여 두 번째 배치에서
    "bound to a different event loop"로 죽는다.

    락 없이도 정확하다. 슬롯을 예약하는 구간(now 읽기 → _next_at 갱신)에
    await가 없어서 단일 스레드 이벤트 루프에서는 원자적으로 실행된다.
    각 태스크는 자기 몫의 슬롯을 받아 그때까지만 기다린다.
    """

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval = min_interval_s
        self._next_at = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        slot = max(now, self._next_at)
        self._next_at = slot + self.min_interval
        delay = slot - now
        if delay > 0:
            await asyncio.sleep(delay)


class CircuitBreaker:
    """1분간 실패 N회 → cooldown 동안 정지 (설계 §4.3).

    상대 서버가 429/5xx를 뱉기 시작하면 물러난다. 이게 없으면 장애 중인
    서버를 계속 두드려 상황을 악화시킨다.
    """

    def __init__(self, threshold: int, cooldown_s: float, window_s: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown_s
        self.window = window_s
        self._failures: list[float] = []
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self._open_until

    def remaining(self) -> float:
        return max(0.0, self._open_until - time.monotonic())

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures = [t for t in self._failures if now - t < self.window]
        self._failures.append(now)
        if len(self._failures) >= self.threshold:
            self._open_until = now + self.cooldown
            self._failures.clear()
            log.warning("서킷 열림 — %.0f초 정지", self.cooldown)

    def record_success(self) -> None:
        self._failures.clear()


@dataclass
class FetchResult:
    title: str
    status: int
    html: bytes | None
    elapsed_s: float
    error: str | None = None


def article_url(title: str) -> str:
    return f"{SETTINGS.base_url}/w/{urllib.parse.quote(title, safe='')}"


def iter_links(html: str) -> set[str]:
    """본문에서 내부 문서 링크를 수확한다. BFS 프런티어의 연료다.

    실측상 문서당 1,000~1,500개가 나온다 (설계 §2.7).
    """
    out: set[str] = set()
    for href in re.findall(r"""href=['"](/w/[^'"#]+)['"]""", html):
        m = _ARTICLE_PATH.match(href)
        if not m:
            continue
        title = urllib.parse.unquote(m.group(1))
        if title.startswith(_NON_ARTICLE_NS):
            continue
        if not title.strip():
            continue
        out.add(title)
    return out


async def fetch_document(
    client: httpx.AsyncClient,
    title: str,
    limiter: RateLimiter,
    breaker: CircuitBreaker,
) -> FetchResult:
    if breaker.is_open:
        raise CircuitOpen(f"{breaker.remaining():.0f}초 남음")

    await limiter.acquire()
    t0 = time.monotonic()
    try:
        resp = await client.get(article_url(title))
    except Exception as e:  # noqa: BLE001 — 네트워크 예외 전반
        breaker.record_failure()
        return FetchResult(title, 0, None, time.monotonic() - t0, f"{type(e).__name__}: {e}")

    elapsed = time.monotonic() - t0
    if resp.status_code == 429 or resp.status_code >= 500:
        breaker.record_failure()
        return FetchResult(title, resp.status_code, None, elapsed, "일시 실패")
    breaker.record_success()
    if resp.status_code != 200:
        # 404(삭제된 문서)는 정상 응답이다. 재시도하지 않는다.
        return FetchResult(title, resp.status_code, None, elapsed, None)

    body = resp.content
    # 응답이 정말 HTML인지 확인한다. 콘텐츠 협상이 틀어지면(예: 디코딩할 수
    # 없는 압축 인코딩) 바이너리가 그대로 넘어오는데, 그걸 저장하면 파서가
    # 빈 결과를 뱉을 때까지 아무도 눈치채지 못한다.
    head = body[:1024].lstrip().lower()
    if not (head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<html" in head):
        return FetchResult(
            title, 200, None, elapsed,
            f"HTML이 아닌 응답 (content-encoding={resp.headers.get('content-encoding')!r}, "
            f"{len(body)}B)",
        )
    return FetchResult(title, 200, body, elapsed, None)


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent": SETTINGS.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            # Accept-Encoding은 지정하지 않는다. httpx가 실제로 디코딩할 수 있는
            # 인코딩만 광고한다. 직접 'br'을 넣으면 brotli 디코더가 없을 때
            # 서버가 brotli로 보내고, 압축 바이너리가 그대로 저장된다.
        },
        timeout=SETTINGS.crawl_timeout_s,
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(
            max_connections=SETTINGS.crawl_concurrency,
            max_keepalive_connections=SETTINGS.crawl_concurrency,
        ),
    )
