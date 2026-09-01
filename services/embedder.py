"""임베더 — chunks.ready 소비 → dense+sparse 임베딩 → Qdrant upsert.

이 파이프라인의 유일한 진짜 병목이다 (설계 §3.2). 컨슈머 그룹에 워커를
추가하면 처리량이 선형으로 늘고, chunks.ready는 12파티션이라 최대 12개까지
확장된다.

임베딩 모델을 바꾸면 chunks.ready를 오프셋 0부터 재생하는 것만으로 전량
재임베딩된다. 재크롤도 재파싱도 필요 없다 (설계 §8).

사용법:
    .venv/Scripts/python.exe services/embedder.py [--once] [--from-beginning]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient, models  # noqa: E402

from namuwiki.config import SETTINGS, Topics  # noqa: E402
from namuwiki.db import bump, connect  # noqa: E402
from namuwiki.kafka_io import make_consumer, make_producer, send  # noqa: E402
from namuwiki.models import DenseEncoder, SparseEncoder  # noqa: E402
from namuwiki.runtime import PollLoop, setup_logging  # noqa: E402

log = setup_logging("embedder")

GROUP = "embedder"
# 이만큼 모이면 GPU에 한 번에 태운다.
#
# 크게 잡으면 안 된다. Kafka는 컨슈머가 max.poll.interval.ms(15분) 안에 다시
# poll하지 않으면 죽은 것으로 보고 파티션을 회수한다. 한 배치 처리 시간이
# 길수록 그 한도에 걸릴 위험이 커지는데, 실제로 128로 두었다가 GPU·CPU가
# 다른 작업과 경합하는 동안 한도를 넘겨 **조용히 그룹에서 쫓겨난** 적이 있다.
# 크래시가 아니라 진행이 멈출 뿐이라 알아채기 어렵다.
#
# 32면 정상 속도에서 한 사이클이 10초 내외라 20배 이상 여유가 생긴다.
# 배치를 줄여도 GPU 처리량 손해는 미미하다 (dense 배치는 별도 설정).
FLUSH_EVERY = 32


# 배치 하나가 이 시간을 넘으면 GPU가 굶고 있다고 보고 프로세스를 죽인다.
# 정상 배치는 8~30초다. 넉넉히 잡아도 10분이면 충분하다.
BATCH_TIMEOUT_S = int(os.environ.get("NW_BATCH_TIMEOUT", 600))


@contextmanager
def _watchdog(seconds: int, n: int):
    """제한 시간 안에 블록을 못 끝내면 프로세스를 강제 종료한다.

    CUDA 호출은 파이썬 레벨에서 인터럽트할 수 없으므로 스레드를 띄워
    감시하다가 os._exit()로 내린다. 우아하지 않지만, 조용히 몇 시간
    멈춰 있는 것보다 낫다. 커밋 전이라 유실은 없다.
    """
    done = threading.Event()

    def bark() -> None:
        if done.wait(seconds):
            return
        log.error(
            "배치 %d청크가 %d초를 넘겼다. GPU가 다른 작업에 굶고 있을 수 있다. "
            "프로세스를 종료한다 — 커밋 전이라 메시지는 재전달된다.",
            n, seconds,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(75)  # EX_TEMPFAIL — 재시작하면 되는 실패

    t = threading.Thread(target=bark, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()


def vram_needed_mb() -> float:
    """실제 소요 VRAM 추정.

    가중치만 세면 크게 빗나간다. 실측: 여유 5,075MB 상태에서 dense 모델
    (가중치 1,083MB) 하나를 올렸더니 **여유가 0MB가 됐다.** 가중치 외에
    CUDA 컨텍스트·캐싱 할당자 예약분·cuBLAS 워크스페이스가 그만큼 더 붙는다.

    이걸 빼먹은 추정(2,057MB)이 "여유 2,387MB면 충분"이라고 잘못 판정했고,
    실제로는 페이징 구간에 들어가 배치가 10분을 넘겼다.
    """
    weights = 1083 + 200          # dense 0.5B fp16 + sparse 0.1B fp16
    logits = SETTINGS.sparse_batch * SETTINGS.sparse_max_seq * 250002 * 2 / 1024**2
    activations = 300             # dense 배치 활성값
    runtime = 1200                # CUDA 컨텍스트 + 할당자 예약 + cuBLAS 워크스페이스
    return weights + logits + activations + runtime


def _free_vram_mb() -> float:
    """가용 VRAM(MB). **nvidia-smi를 쓴다.**

    `torch.cuda.mem_get_info()`를 믿으면 안 된다. Windows WDDM은 메모리를
    가상화해서, 다른 프로세스가 4GB를 쥐고 있어도 torch는 "여유 5,075MB"라고
    답한다. 실제로 그 상태에서 1,083MB 모델을 올렸더니 여유가 0이 됐다.
    같은 순간 nvidia-smi는 4,035MB 사용 중이라고 정확히 말하고 있었다.

    측정이 맞다고 확인해준 쪽을 쓴다.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (float(x.strip()) for x in out.split(","))
        return total - used
    except Exception as e:  # noqa: BLE001 — nvidia-smi가 없거나 실패하면 torch로 후퇴
        import torch

        log.warning("nvidia-smi 조회 실패(%s) — torch 값으로 후퇴한다 (과대평가 위험)", e)
        return torch.cuda.mem_get_info()[0] / 1024**2


def wait_for_vram(poll_s: int = 60, timeout_s: int | None = None) -> None:
    """VRAM이 충분해질 때까지 기다린다.

    Windows(WDDM)는 VRAM이 모자라도 OOM을 내지 않고 **호스트 RAM으로 페이징**한다.
    PCIe를 오가느라 100배쯤 느려지는데 에러가 없어서 '멈춘 것처럼' 보인다.
    실제로 32청크 배치가 2.6시간 걸린 적이 있는데, 그때 GPU 자체는 멀쩡했고
    (순수 matmul 10.3 TFLOPS) 여유 VRAM만 모자랐다.

    한 번 확인하고 넘어가면 소용이 없다. 같은 GPU를 쓰는 다른 작업이 메모리를
    0~4GB로 오르내려서, 확인 직후 조건이 바뀐다(실측: preflight 통과 9분 뒤
    모델 로딩이 18배 느려짐). 그래서 확인이 아니라 **대기**로 만든다.
    이러면 지켜보지 않아도 상대 작업이 쉬는 틈마다 알아서 진행된다.
    """
    import torch

    if not torch.cuda.is_available():
        return
    # 마진 1.05. 1.15로 두었더니 실측 2,793MB면 되는 상황에서 3,481MB를
    # 요구해 328MB 차이로 13시간을 대기만 했다. GPU는 그동안 0% 유휴였다.
    need = vram_needed_mb() * float(os.environ.get('NW_VRAM_MARGIN', '1.05'))
    waited = 0
    while True:
        free = _free_vram_mb()
        if free >= need:
            log.info("VRAM 여유 %.0fMB (필요 %.0fMB) — 진행", free, need)
            return
        if timeout_s is not None and waited >= timeout_s:
            log.error("VRAM 대기 %d초 초과 (여유 %.0fMB < 필요 %.0fMB). 포기한다.",
                      waited, free, need)
            raise SystemExit(75)
        log.warning(
            "VRAM 부족 — 여유 %.0fMB < 필요 %.0fMB. %d초 뒤 다시 확인한다. "
            "(지금 돌리면 호스트 RAM으로 페이징되어 100배 느려진다)",
            free, need, poll_s,
        )
        time.sleep(poll_s)
        waited += poll_s


def point_id(chunk_id: str) -> str:
    """Qdrant 포인트 ID는 UUID나 정수만 허용한다. chunk_id를 결정적으로 매핑한다.

    같은 chunk_id는 항상 같은 UUID가 되므로 upsert가 멱등해진다.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class Batcher:
    def __init__(self, dense: DenseEncoder, sparse: SparseEncoder, client: QdrantClient):
        self.dense = dense
        self.sparse = sparse
        self.client = client
        self.pending: list[dict] = []
        self.to_delete: list[str] = []
        self.embedded = 0

    def add(self, msg: dict) -> None:
        if msg.get("deleted"):
            self.to_delete.append(point_id(msg["chunk_id"]))
        else:
            self.pending.append(msg)

    def flush(self, conn) -> int:
        if self.to_delete:
            self.client.delete(
                collection_name=SETTINGS.collection,
                points_selector=models.PointIdsList(points=self.to_delete),
                wait=False,
            )
            log.info("삭제 %d포인트", len(self.to_delete))
            self.to_delete.clear()

        if not self.pending:
            return 0

        texts = [m["text"] for m in self.pending]
        # 실행 중에도 상대 작업이 VRAM을 잡을 수 있다. 배치마다 확인해서
        # 페이징 구간으로 들어가는 대신 기다린다 (호출 자체는 아주 싸다).
        wait_for_vram()
        t0 = time.time()
        # GPU를 다른 작업과 공유하면(Windows에는 CUDA MPS가 없다) CUDA 커널이
        # 굶어 아주 오래 블록될 수 있다. 실제로 32청크 배치가 2.6시간 멈춘 적이
        # 있는데, 크래시가 아니라 진행만 멈춰서 알아채기 어려웠다.
        # 감시견을 두어 조용히 멈추는 대신 시끄럽게 죽인다. 커밋 전이라
        # 메시지는 재전달되고 모든 단계가 멱등이므로 재시작이 안전하다.
        with _watchdog(BATCH_TIMEOUT_S, len(texts)):
            dvecs = self.dense.encode(texts)
            svecs = self.sparse.encode(texts)
        gpu_s = time.time() - t0

        points = []
        for m, dv, sv in zip(self.pending, dvecs, svecs, strict=True):
            points.append(
                models.PointStruct(
                    id=point_id(m["chunk_id"]),
                    vector={
                        "dense": dv,
                        "lexical": models.SparseVector(
                            indices=sv.indices, values=sv.values
                        ),
                    },
                    payload={
                        "chunk_id": m["chunk_id"],
                        "doc_id": m["doc_id"],
                        "title": m.get("title", ""),
                        "heading_path": m.get("heading_path", []),
                        "seq": m.get("seq", 0),
                        "categories": m.get("categories", []),
                        "text": m["text"],
                    },
                )
            )
        self.client.upsert(SETTINGS.collection, points=points, wait=True)

        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedded_at = now() WHERE chunk_id = %s",
                [(m["chunk_id"],) for m in self.pending],
            )

        n = len(points)
        self.embedded += n
        log.info(
            "임베딩 %d청크 | GPU %.1fs (%.1f청크/s) | 누적 %d",
            n, gpu_s, n / max(gpu_s, 1e-6), self.embedded,
        )
        self.pending.clear()
        return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="큐가 비면 종료")
    ap.add_argument("--from-beginning", action="store_true",
                    help="오프셋 0부터 재생 (전량 재임베딩)")
    ap.add_argument("--sparse-model", default="telepix/PIXIE-Splade-v1.5")
    args = ap.parse_args()

    wait_for_vram()

    client = QdrantClient(url=SETTINGS.qdrant_url, timeout=120)
    dense = DenseEncoder()
    sparse = SparseEncoder(args.sparse_model)
    if dense.dim != SETTINGS.dense_dim:
        log.warning(
            "컬렉션 차원(%d)과 모델 차원(%d)이 다르다. init_stack을 다시 돌려야 한다.",
            SETTINGS.dense_dim, dense.dim,
        )

    producer = make_producer()
    consumer = make_consumer(
        Topics.CHUNKS_READY, group_id=GROUP,
        from_beginning=True, max_poll_records=FLUSH_EVERY,
    )
    if args.from_beginning:
        consumer.poll(timeout_ms=3000)
        consumer.seek_to_beginning()
        log.info("오프셋 0부터 재생 — 전량 재임베딩")

    batcher = Batcher(dense, sparse, client)
    log.info("시작 — dense=%s sparse=%s", dense.model_id, sparse.model_id)
    try:
        with connect() as conn:
            for records in PollLoop(consumer, once=args.once).batches():
                if not records:
                    if batcher.pending or batcher.to_delete:
                        batcher.flush(conn)
                        consumer.commit()
                    continue
                for rec in records:
                    batcher.add(rec.value)
                try:
                    n = batcher.flush(conn)
                    consumer.commit()
                    if n:
                        bump(conn, "embedder", "embedded", n)
                except Exception as e:  # noqa: BLE001
                    log.exception("임베딩 실패: %s", e)
                    for m in batcher.pending:
                        send(producer, Topics.EMBED_DLQ, m["doc_id"],
                             {"chunk_id": m["chunk_id"], "error": str(e)})
                    batcher.pending.clear()
                    producer.flush()
                    consumer.commit()
    except KeyboardInterrupt:
        log.info("중단 요청")
    finally:
        producer.flush()
        consumer.close()
        producer.close()
        log.info("종료 — 총 %d청크", batcher.embedded)


if __name__ == "__main__":
    main()
