"""Phase 0 — dense 임베딩 모델 A/B 평가 (설계 §7.2).

왜 필요한가:
    공개된 한국어 검색 벤치마크는 특정 벤더 모델이 전 부문 1위인데, 벤치마크
    저장소와 그 벤더의 관계가 명확히 분리 검증되지 않는다. 900만 청크 임베딩에
    100시간 넘게 쓰기 전에 **우리 데이터로 직접** 확인한다.

평가 방법 (라벨 없는 자기검색):
    수집된 문서에서 `제목 + 헤딩 경로`를 질의로 만들고, 그 헤딩이 속한 청크가
    상위에 오는지 본다. 예: 질의 "김연아 수상 기록" → 김연아 문서의 s-4 청크.

    완벽한 평가는 아니다. 헤딩 텍스트가 청크 앞에 접두로 붙어 있어 어휘가
    겹치므로 점수가 낙관적으로 나온다. 하지만 **모델 간 상대 비교**에는 쓸 수
    있고, 실제 나무위키 문체·고유명사 분포를 반영한다는 게 공개 벤치마크보다
    나은 점이다.

사용법:
    .venv/Scripts/python.exe scripts/eval_models.py --n 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.db import connect  # noqa: E402

CANDIDATES = [
    "nlpai-lab/KURE-v1",
    "telepix/PIXIE-Rune-Preview",
    "BAAI/bge-m3",
]


def build_queries(n: int, seed: int = 13) -> list[tuple[str, str]]:
    """(질의, 정답 chunk_id) 목록. 헤딩이 있는 청크만 쓴다."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.chunk_id, d.title, c.heading_path
                 FROM chunks c JOIN documents d USING (doc_id)
                WHERE array_length(c.heading_path, 1) >= 1
                  AND length(c.text) > 400"""
        ).fetchall()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        q = f"{r['title']} {' '.join(r['heading_path'])}".strip()
        out.append((q, r["chunk_id"]))
    return out


def load_corpus() -> tuple[list[str], list[str]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
    return [r["chunk_id"] for r in rows], [r["text"] for r in rows]


def evaluate(model_id: str, ids: list[str], texts: list[str],
             queries: list[tuple[str, str]], batch: int) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    from namuwiki.config import SETTINGS

    dev = SETTINGS.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    model = SentenceTransformer(model_id, device=dev)
    if dev == "cuda":
        model.half()
    if model.max_seq_length > SETTINGS.max_seq_length:
        model.max_seq_length = SETTINGS.max_seq_length
    load_s = time.time() - t0

    t0 = time.time()
    corpus = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                          convert_to_tensor=True, show_progress_bar=False)
    encode_s = time.time() - t0
    qvecs = model.encode([q for q, _ in queries], batch_size=batch,
                         normalize_embeddings=True, convert_to_tensor=True,
                         show_progress_bar=False)

    sims = qvecs @ corpus.T                     # 코사인 (정규화됨)
    ranks: list[int] = []
    index = {cid: i for i, cid in enumerate(ids)}
    order = sims.argsort(dim=1, descending=True).cpu()
    for row, (_, gold) in zip(order, queries, strict=True):
        gold_i = index[gold]
        pos = (row == gold_i).nonzero()
        ranks.append(int(pos[0, 0]) + 1 if len(pos) else len(ids) + 1)

    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / len(ranks)

    mrr = statistics.mean(1 / r for r in ranks)
    del model, corpus, qvecs, sims
    if dev == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_id,
        "device": dev,
        "load_s": round(load_s, 1),
        "corpus_encode_s": round(encode_s, 1),
        "chunks_per_s": round(len(texts) / max(encode_s, 1e-6), 1),
        "recall@1": round(recall_at(1), 4),
        "recall@5": round(recall_at(5), 4),
        "recall@10": round(recall_at(10), 4),
        "MRR": round(mrr, 4),
        "median_rank": statistics.median(ranks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="질의 개수")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    args = ap.parse_args()

    ids, texts = load_corpus()
    queries = build_queries(args.n)
    if not queries:
        raise SystemExit("평가할 질의를 만들지 못했다. 먼저 문서를 수집·파싱할 것.")
    print(f"코퍼스 {len(texts):,}청크 | 질의 {len(queries)}개\n", flush=True)

    results = []
    for mid in args.models:
        print(f"--- {mid} ---", flush=True)
        try:
            r = evaluate(mid, ids, texts, queries, args.batch)
        except Exception as e:  # noqa: BLE001
            r = {"model": mid, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)

    ok = [r for r in results if "error" not in r]
    if ok:
        print("\n=== 요약 (Recall@5 기준) ===")
        width = max(len(r["model"]) for r in ok)
        for r in sorted(ok, key=lambda x: -x["recall@5"]):
            print(f"  {r['model']:<{width}}  R@1 {r['recall@1']:.3f}  "
                  f"R@5 {r['recall@5']:.3f}  R@10 {r['recall@10']:.3f}  "
                  f"MRR {r['MRR']:.3f}  {r['chunks_per_s']:>6.1f}청크/s")

    out = ROOT / "data" / "eval_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
