"""임베딩·리랭킹 모델 래퍼.

설계 §2.8의 결론:
  - dense 단독보다 dense+sparse 하이브리드가, 거기에 리랭커를 얹으면 더 낫다.
    (한국어 MTEB 기준 dense 0.788 → 리랭킹 0.817)
  - sparse는 SPLADE를 쓴다. BM25(0.507)보다 26%p 높고 모델이 0.1B로 작다.
  - 모델 ID는 전부 설정 주입이다. Phase 0 A/B 결과로 갈아끼울 수 있어야 한다.
    (벤치마크 이해충돌 우려 때문에 특정 모델에 코드를 묶지 않는다)

VRAM 6GB 배치: dense 0.5B(fp16) + sparse 0.1B(fp16) ≈ 1.4GB.
리랭커 0.5B는 질의 시점에만 필요하므로 색인 워커는 로드하지 않는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .config import SETTINGS

log = logging.getLogger(__name__)

__all__ = ["DenseEncoder", "SparseEncoder", "Reranker", "SparseVec"]


def _device() -> str:
    """실행 디바이스. NW_DEVICE로 강제할 수 있다.

    이 노트북의 GPU는 다른 장기 작업과 공유되는 경우가 있다. VRAM이 6GB뿐이라
    경합하면 상대 작업이 OOM으로 죽을 수 있으므로, 양보가 필요할 때
    NW_DEVICE=cpu 로 돌린다.
    """
    forced = SETTINGS.device
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class DenseEncoder:
    def __init__(self, model_id: str | None = None, fp16: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id or SETTINGS.dense_model
        self.model = SentenceTransformer(self.model_id, device=_device())
        if fp16 and _device() == "cuda":
            self.model.half()
        # BGE-M3 계열은 기본 max_seq_length가 8192다. 청커가 청크를 850자
        # (≈512토큰)로 제한하므로 8192로 두면 어텐션 비용만 낭비한다.
        if self.model.max_seq_length > SETTINGS.max_seq_length:
            log.info(
                "max_seq_length %d → %d 로 낮춘다 (청크가 그보다 짧다)",
                self.model.max_seq_length, SETTINGS.max_seq_length,
            )
            self.model.max_seq_length = SETTINGS.max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("dense 로드: %s (dim=%d)", self.model_id, self.dim)

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size or SETTINGS.embed_batch,
            normalize_embeddings=True,  # 코사인 거리를 쓰므로 정규화한다
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype("float32").tolist()


class SparseEncoder:
    """SPLADE 계열 희소 인코더.

    MLM 로짓에 log(1+relu(·))를 씌우고 시퀀스 축으로 max pooling 하면
    어휘 공간의 희소 가중치가 나온다. 이게 SPLADE의 전부다.
    """

    def __init__(self, model_id: str = "telepix/PIXIE-Splade-v1.5", fp16: bool = True):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model_id = model_id
        self.dev = _device()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(self.dev)
        if fp16 and self.dev == "cuda":
            self.model.half()
        self.model.eval()
        # dense와 분리된 설정을 쓴다 (설계 §12.2 — sparse가 연산의 83%)
        self.max_len = min(
            getattr(self.tok, "model_max_length", 512) or 512,
            SETTINGS.sparse_max_seq,
        )
        log.info("sparse 로드: %s", model_id)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int | None = None) -> list[SparseVec]:
        """SPLADE 희소 벡터.

        메모리 주의: MLM 로짓은 (배치, 길이, 어휘) 모양이고 어휘가 25만이라
        배치 16 × 512토큰이면 그것만 fp16으로 4GB다. 순진하게 구현하면
        중간 텐서까지 세 벌이 생겨 8GB를 넘긴다.

        핵심 최적화: log1p(relu(·))는 단조증가라 max 풀링과 교환된다.
            max_L log1p(relu(x)) == log1p(relu(max_L x))
        그래서 변환을 (B, L, V)가 아니라 풀링 후 (B, V)에만 적용한다.
        중간 텐서 두 벌이 사라진다. 남은 로짓 한 벌은 배치를 작게 잡아 줄인다.
        """
        bs = batch_size or SETTINGS.sparse_batch
        out: list[SparseVec] = []
        for i in range(0, len(texts), bs):
            enc = self.tok(
                texts[i : i + bs],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.dev)
            logits = self.model(**enc).logits
            # 패딩 위치는 최댓값 후보에서 제외한다
            mask = enc["attention_mask"].unsqueeze(-1).bool()
            logits = logits.masked_fill(~mask, float("-inf"))
            pooled = logits.amax(dim=1)                  # (B, V)
            del logits
            pooled = torch.log1p(torch.relu(pooled)).float()
            for row in pooled:
                nz = torch.nonzero(row, as_tuple=False).squeeze(-1)
                out.append(SparseVec(indices=nz.tolist(), values=row[nz].tolist()))
            del pooled, enc
        return out


class Reranker:
    """교차 인코더 리랭커. 상위 100개를 다시 정렬한다 (설계 §7.1).

    색인 비용이 0이면서 NDCG@10을 +3.5%p 올린다. 비용 대비 효과가 가장 크다.
    """

    def __init__(self, model_id: str | None = None, fp16: bool = True) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id or SETTINGS.reranker_model
        self.model = CrossEncoder(self.model_id, device=_device(), max_length=512)
        if fp16 and _device() == "cuda":
            self.model.model.half()
        log.info("reranker 로드: %s", self.model_id)

    def rank(self, query: str, docs: list[str], top_k: int = 8) -> list[tuple[int, float]]:
        """(원본 인덱스, 점수) 목록을 점수 내림차순으로 반환한다."""
        if not docs:
            return []
        scores = self.model.predict(
            [(query, d) for d in docs], show_progress_bar=False, batch_size=16
        )
        order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
        return [(i, float(scores[i])) for i in order[:top_k]]
