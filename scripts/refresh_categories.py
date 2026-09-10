"""보관된 HTML 에서 분류만 다시 뽑아 documents.categories 를 갱신한다.

추출기가 분류 이름을 링크 **텍스트**에서 가져오고 있었다. 나무위키는 문서
제목과 겹치는 접두어를 화면에서 생략해 렌더링하므로, '분류:경기도 출신
인물' 이 '출신 인물' 로 저장됐다. 그 결과 경기도 문서에 '출신 인물' 이라는
분류가 붙어 타입 분류기가 경기도를 인물로 판정했다. 이제 href 에서 정식
이름을 가져온다.

전체 재파싱은 청킹과 임베딩까지 다시 하므로 과하다. 분류는 문서 메타데이터
일 뿐이라 블롭만 다시 읽으면 된다.

갱신 뒤에는 분류에서 나온 관계와 타입을 다시 만들어야 한다:
    scripts/extract_relations.py --skip-infobox
    scripts/classify_entities.py

사용법:
    .venv/Scripts/python.exe scripts/refresh_categories.py --limit 200
    .venv/Scripts/python.exe scripts/refresh_categories.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from namuwiki.blobstore import FileSystemBlobStore  # noqa: E402
from namuwiki.config import SETTINGS  # noqa: E402
from namuwiki.db import connect  # noqa: E402
from namuwiki.extract import _extract_categories  # noqa: E402

BATCH = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blobs = FileSystemBlobStore(SETTINGS.blob_root)
    t0 = time.time()

    with connect() as conn:
        q = "SELECT title, blob_ref, categories FROM documents WHERE blob_ref IS NOT NULL"
        if args.limit:
            q += f" LIMIT {args.limit}"
        rows = conn.execute(q).fetchall()
        print(f"문서 {len(rows):,}", flush=True)

        pending: list[tuple[list[str], str]] = []
        changed = failed = done = 0
        for r in rows:
            try:
                html = blobs.get(r["blob_ref"]).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            cats = _extract_categories(html)
            done += 1
            if cats != list(r["categories"] or []):
                pending.append((cats, r["title"]))
                changed += 1
            if len(pending) >= BATCH:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE documents SET categories = %s WHERE title = %s", pending)
                pending.clear()
            if done % 5000 == 0:
                print(f"  {done:,}/{len(rows):,} · 변경 {changed:,} "
                      f"· {done / (time.time() - t0):.0f}문서/s", flush=True)
        if pending:
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE documents SET categories = %s WHERE title = %s", pending)

    print(f"\n{done:,}문서 확인 · {changed:,}건 갱신 · 실패 {failed} "
          f"· {time.time() - t0:.0f}s")
    print("이어서 실행할 것:")
    print("  scripts/extract_relations.py --skip-infobox")
    print("  scripts/classify_entities.py")


if __name__ == "__main__":
    main()
