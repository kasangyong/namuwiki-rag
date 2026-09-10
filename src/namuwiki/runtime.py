"""서비스 공통 런타임 — 로깅과 컨슈머 폴 루프.

Windows 기본 콘솔 인코딩(cp949/ascii)에서 한글 로그가 UnicodeEncodeError로
죽는 것을 막고, Kafka 리밸런스 중 첫 빈 폴에 종료해버리는 실수를 막는다.
"""

from __future__ import annotations

import logging
import sys
import time

__all__ = ["setup_logging", "PollLoop"]


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s %(levelname)s [{name}] %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # kafka-python은 정상 동작 중에도 INFO 로그를 쏟아낸다
    logging.getLogger("kafka").setLevel(logging.WARNING)
    return logging.getLogger(name)


class PollLoop:
    """파티션 할당을 기다린 뒤 도는 컨슈머 루프.

    두 가지 함정을 막는다:
      1. 구독 직후 첫 폴은 그룹 조인·파티션 할당 때문에 거의 항상 비어 있다.
         거기서 종료하면 큐에 메시지가 있어도 0건 처리하고 끝난다.
      2. kafka-python의 poll(timeout_ms=)은 컨슈머가 준비되지 않았을 때
         즉시 반환한다. 그래서 '빈 폴 N회'로 세면 순식간에 소진된다.
         횟수가 아니라 **경과 시간**으로 판단해야 한다.
    """

    def __init__(self, consumer, *, once: bool, idle_exit_s: float = 15.0,
                 assign_timeout_s: float = 30.0, timeout_ms: int = 2000):
        self.consumer = consumer
        self.once = once
        self.idle_exit_s = idle_exit_s
        self.assign_timeout_s = assign_timeout_s
        self.timeout_ms = timeout_ms
        self._buffered: list = []

    def _await_assignment(self) -> bool:
        deadline = time.monotonic() + self.assign_timeout_s
        while time.monotonic() < deadline:
            if self.consumer.assignment():
                return True
            # 이 폴은 그냥 기다리는 게 아니라 실제로 레코드를 가져온다.
            # 버리면 그 메시지는 영영 사라진다 — 커밋은 뒤이어 진행되므로
            # 재전달도 없다. 실제로 재발행한 청크가 시작할 때마다 조용히
            # 먹혔다. 받아둔 건 첫 배치로 넘긴다.
            batch = self.consumer.poll(timeout_ms=500)
            if batch:
                self._buffered.extend(
                    rec for recs in batch.values() for rec in recs
                )
        log = logging.getLogger(__name__)
        log.warning("파티션 할당을 %.0f초 안에 못 받았다.", self.assign_timeout_s)
        return False

    def batches(self):
        """(레코드 리스트)를 순차로 내준다. --once 면 큐가 마르면 멈춘다."""
        self._await_assignment()
        if self._buffered:
            yield self._buffered
            self._buffered = []
        idle_since: float | None = None
        while True:
            batch = self.consumer.poll(timeout_ms=self.timeout_ms)
            if not batch:
                now = time.monotonic()
                if idle_since is None:
                    idle_since = now
                elif self.once and now - idle_since >= self.idle_exit_s:
                    return
                yield []
                continue
            idle_since = None
            yield [rec for recs in batch.values() for rec in recs]
