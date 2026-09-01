"""Phase 0 — 임베딩 모델 처리량·VRAM 실측.

설계 §12 미해결사항 1: 900만 청크 임베딩 소요 시간 추정치(30~60시간)가 미검증이다.
이 스크립트가 그 수치를 확정한다.

실제 나무위키 청크로 측정한다. 합성 텍스트는 토큰 길이 분포가 달라 의미가 없다.

사용법:
    bench_venv/Scripts/python.exe scripts/bench_embed.py
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from namuwiki.chunker import chunk_document  # noqa: E402
from namuwiki.extract import extract  # noqa: E402
from namuwiki.ids import doc_id  # noqa: E402

TOTAL_CHUNKS = 9_000_000  # 설계 §2.2 추정 코퍼스 청크 수

DENSE_CANDIDATES = [
    "nlpai-lab/KURE-v1",
    "telepix/PIXIE-Rune-Preview",
    "BAAI/bge-m3",
]
BATCH_SIZES = [16, 32]

# BGE-M3 계열은 max_seq_length가 8192다. 청크는 850자(≈512토큰)로 제한되므로
# 8192로 두면 어텐션 비용만 낭비한다. 운영 설정에도 이 값을 그대로 쓴다.
MAX_SEQ = 512
SAMPLE_N = 192  # 배치 설정당 인코딩할 청크 수. 노트북 GPU(30W)라 작게 잡는다.


def load_real_chunks(html_dir: Path, want: int = 512) -> list[str]:
    """실제 나무위키 HTML → 청크 텍스트."""
    texts: list[str] = []
    for path in sorted(html_dir.glob("*.html")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            ex = extract(raw)
        except Exception:
            continue
        title = ex.title or path.stem
        for c in chunk_document(doc_id(title), title, ex):
            texts.append(c.text)
    if not texts:
        raise SystemExit(f"청크를 하나도 못 만들었다: {html_dir}")
    # 부족하면 순환 반복 (토큰 길이 분포는 유지된다)
    while len(texts) < want:
        texts.extend(texts[: want - len(texts)])
    return texts[:want]


def bench_model(model_id: str, chunks: list[str]) -> dict:
    from sentence_transformers import SentenceTransformer

    result: dict = {"model": model_id}
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        model = SentenceTransformer(model_id, device="cuda", trust_remote_code=True)
        model.half()
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    result["load_s"] = round(time.time() - t0, 1)
    result["dim"] = model.get_sentence_embedding_dimension()
    result["max_seq_default"] = model.max_seq_length
    model.max_seq_length = MAX_SEQ  # 8192 → 512. 청크가 그보다 짧다.
    result["max_seq_used"] = MAX_SEQ
    result["weights_MB"] = round(torch.cuda.memory_allocated() / 1024**2)
    print(f"    로드 {result['load_s']}s | dim={result['dim']} | "
          f"max_seq {result['max_seq_default']}→{MAX_SEQ} | "
          f"가중치 {result['weights_MB']}MB", flush=True)

    lens = [len(t) for t in chunks]
    result["chunk_chars_mean"] = round(statistics.mean(lens))

    per_batch: dict[str, dict] = {}
    for bs in BATCH_SIZES:
        try:
            print(f"    batch={bs} 워밍업…", flush=True)
            model.encode(chunks[:bs], batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            n = min(len(chunks), SAMPLE_N)
            t0 = time.time()
            model.encode(
                chunks[:n],
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            torch.cuda.synchronize()
            dt = time.time() - t0
            cps = n / dt
            entry = {
                "chunks_per_s": round(cps, 1),
                "peak_VRAM_MB": round(torch.cuda.max_memory_allocated() / 1024**2),
                "full_corpus_hours": round(TOTAL_CHUNKS / cps / 3600, 1),
            }
            per_batch[str(bs)] = entry
            print(f"    batch={bs} → {entry['chunks_per_s']}청크/s, "
                  f"peak {entry['peak_VRAM_MB']}MB, "
                  f"전체 {entry['full_corpus_hours']}시간", flush=True)
        except torch.cuda.OutOfMemoryError:
            per_batch[str(bs)] = {"error": "OOM"}
            print(f"    batch={bs} → OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            per_batch[str(bs)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"    batch={bs} → 실패 {e}", flush=True)
    result["batches"] = per_batch

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 못 찾았다. CPU 빌드 torch가 설치된 것 같다.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  VRAM {props.total_memory / 1024**3:.1f} GB\n", flush=True)

    html_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_html"
    chunks = load_real_chunks(html_dir)
    print(f"실제 청크 {len(chunks)}개 로드 (평균 {statistics.mean(map(len, chunks)):.0f}자)\n", flush=True)

    results = []
    for mid in DENSE_CANDIDATES:
        print(f"--- {mid} ---", flush=True)
        r = bench_model(mid, chunks)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
        print(flush=True)

    out = ROOT / "data" / "bench_embed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
