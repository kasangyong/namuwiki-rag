"""부팅·로그온 때 임베딩을 이어서 돌리고, 다 끝나면 스스로 등록을 해제한다.

이 컴퓨터는 하루에 두 번 꺼졌다 — 한 번은 전원이 끊겼고(Kernel-Power 41),
한 번은 정상 종료였다(109). 그때마다 사람이 손으로 다시 올려야 했다.

작업 스케줄러에 그냥 등록하면 임베딩이 끝난 뒤에도 로그온할 때마다 계속
뜬다. 그래서 할 일이 없으면 **자기 등록을 지우고** 끝낸다. 임시 장치는
임시로 남아야 한다.

흐름:
    1. DB 가 준비될 때까지 기다린다 (도커가 뜨는 데 시간이 걸린다).
    2. 남은 청크가 0 이면 예약 작업을 지우고 종료.
    3. 임베더를 --once 로 돌린다. 큐가 마르면 스스로 끝난다.
    4. 끝난 뒤에도 남아 있으면, DLQ 로 빠진 청크를 다시 발행하고 반복한다.
       무한히 돌지 않도록 횟수를 제한한다.

등록/해제:
    .venv/Scripts/python.exe scripts/autostart_embed.py --install
    .venv/Scripts/python.exe scripts/autostart_embed.py --uninstall
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 작업 스케줄러(schtasks)는 이 컴퓨터에서 관리자 권한을 요구한다.
# 시작 프로그램 폴더는 권한이 필요 없고, 해제가 파일 삭제라 더 투명하다.
STARTUP = (Path(os.environ["APPDATA"]) / "Microsoft" / "Windows"
           / "Start Menu" / "Programs" / "Startup")
LAUNCHER = STARTUP / "namuwiki-embed.cmd"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DB_WAIT_S = 900          # 도커 데스크톱이 뜨고 컨테이너가 healthy 가 될 때까지
MAX_ROUNDS = 6           # 재발행 반복 상한. 넘으면 사람이 볼 문제다.


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def unembedded() -> int | None:
    """남은 청크 수. DB 에 닿지 못하면 None."""
    try:
        from namuwiki.db import connect
        with connect() as conn:
            return conn.execute(
                "SELECT count(*) n FROM chunks WHERE embedded_at IS NULL"
            ).fetchone()["n"]
    except Exception:  # noqa: BLE001
        return None


def wait_for_db(timeout_s: int = DB_WAIT_S) -> int | None:
    waited = 0
    while waited < timeout_s:
        n = unembedded()
        if n is not None:
            return n
        time.sleep(20)
        waited += 20
        if waited % 120 == 0:
            _log(f"DB 대기 {waited}초 …")
    return None


def install() -> None:
    """로그온 때 조용히 뜨도록 시작 프로그램에 넣는다."""
    STARTUP.mkdir(parents=True, exist_ok=True)
    # 출력은 파일로 남긴다. 창을 띄우지 않되, 문제가 생기면 로그로 확인한다.
    log = ROOT / "data" / "autostart.log"
    # PYTHONUTF8 이 없으면 콘솔이 cp949 라 로그의 '—' 한 글자에
    # UnicodeEncodeError 로 죽는다. 실제로 그렇게 한 번 죽었다.
    lines = [
        "@echo off",
        "set PYTHONUTF8=1",
        "set PYTHONIOENCODING=utf-8",
        f'cd /d "{ROOT}"',
        f'"{PY}" "{Path(__file__).resolve()}" >> "{log}" 2>&1',
    ]
    LAUNCHER.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    print(f"등록 완료: {LAUNCHER}")
    print("임베딩이 끝나면 이 파일을 스스로 지운다.")


def uninstall(quiet: bool = False) -> None:
    try:
        LAUNCHER.unlink()
        if not quiet:
            print(f"등록 해제: {LAUNCHER}")
    except FileNotFoundError:
        if not quiet:
            print("등록되어 있지 않다")


def run_embedder() -> int:
    env = dict(os.environ,
               NW_DEVICE="cuda", NW_SPARSE_BATCH="1",
               PYTHONUNBUFFERED="1", PYTHONUTF8="1", OMP_NUM_THREADS="4")
    log = ROOT / "data" / "embed_autostart.log"
    with log.open("a", encoding="utf-8") as fh:
        return subprocess.run(
            [str(PY), str(ROOT / "services" / "embedder.py"), "--once"],
            cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
        ).returncode


def republish() -> None:
    subprocess.run([str(PY), str(ROOT / "scripts" / "repair_unembedded.py")],
                   cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()

    if args.install:
        install()
        return
    if args.uninstall:
        uninstall()
        return

    left = wait_for_db()
    if left is None:
        _log("DB 에 닿지 못했다. 등록은 그대로 두고 종료한다.")
        raise SystemExit(1)

    for rnd in range(1, MAX_ROUNDS + 1):
        if left == 0:
            _log("남은 청크 0 — 시작 프로그램 등록을 지우고 끝낸다.")
            uninstall(quiet=True)
            return
        _log(f"{rnd}회차: 남은 {left:,}청크 — 임베더 시작")
        run_embedder()
        after = unembedded()
        if after is None:
            _log("DB 연결이 끊겼다. 다음 로그온에 이어서 한다.")
            return
        if after == 0:
            _log("완료 — 시작 프로그램 등록을 지운다.")
            uninstall(quiet=True)
            return
        if after == left:
            # 큐는 말랐는데 DB 에는 남아 있다. DLQ 로 빠진 것들이다.
            _log(f"큐가 말랐는데 {after:,}청크가 남았다 — 다시 발행한다.")
            republish()
        left = after

    _log(f"{MAX_ROUNDS}회를 돌고도 {left:,}청크가 남았다. 등록은 유지한다.")


if __name__ == "__main__":
    main()
