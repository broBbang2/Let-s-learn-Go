"""실행 중인 작업(리그전·추정 검증·서버) 진행상황 확인.

    python -m adaptive_go.progress            # 한 번 출력
    python -m adaptive_go.progress --watch 10 # 10초마다 갱신 (Ctrl+C로 종료)
    python -m adaptive_go.progress -f         # 작업이 시작되면 자동으로 로그 실시간 추적

리그전·추정 검증은 실행하면 터미널 출력을 logs/<종류>-<시각>.log에도 같이 쓴다 (tee_stdout).
lsof로 작업이 열어 둔 그 로그(없으면 리다이렉트된 stdout 파일)를 찾아 마지막 `[i/N]` 줄을 읽는다.
둘 다 없으면 리그전은 games.jsonl 줄 수로 대신 센다.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARENA_DIR = ROOT / "data/arena"
LOG_DIR = ROOT / "logs"
JOB_PATTERN = re.compile(r"-m adaptive_go\.(arena|estimate_eval|server|play)\b(.*)")
STEP_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""


def _parse_etime(s):
    """ps etime 형식 [[dd-]hh:]mm:ss → 초."""
    days, _, rest = s.strip().rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec


def _fmt_secs(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {sec:02d}초"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return self.streams[0].isatty()


def tee_stdout(kind):
    """stdout·stderr를 터미널과 logs/<kind>-<시각>.log에 동시에 쓴다. 로그 경로 반환."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    f = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f"로그: {path}  (다른 터미널에서 python -m adaptive_go.progress -f)")
    return path


def bar(done, total, width=30):
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def status(done, total, started):
    """진행 막대 + 퍼센트 + 남은 시간 (경과 시간 ÷ 완료 수로 추정)."""
    eta = (time.time() - started) / done * (total - done) if done else 0
    return f"{bar(done, total, 20)} {100 * done / total:.0f}%  남은 {_fmt_secs(eta)}"


def find_jobs():
    jobs = []
    for line in _run(["ps", "-axo", "pid=,etime=,command="]).splitlines():
        pid, etime, cmd = line.strip().split(None, 2)
        m = JOB_PATTERN.search(cmd)
        # 셸 래퍼(zsh -c "... -m adaptive_go.x ...")와 멀티프로세싱 워커는 제외
        is_python = "python" in Path(cmd.split()[0]).name.lower()
        if m and is_python and "multiprocessing" not in cmd:
            jobs.append({"pid": int(pid), "elapsed": _parse_etime(etime),
                         "kind": m.group(1), "args": m.group(2).strip()})
    return jobs


def job_log(pid):
    """작업이 쓰는 로그: 열어 둔 logs/*.log, 없으면 일반 파일로 리다이렉트된 stdout."""
    fd = None
    stdout_file = None
    for line in _run(["lsof", "-p", str(pid), "-Ffn"]).splitlines():
        if line.startswith("f"):
            fd = line[1:]
        elif line.startswith("n/"):
            path = Path(line[1:])
            if path.parent == LOG_DIR and path.suffix == ".log":
                return path
            if fd == "1" and not line.startswith("n/dev/"):
                stdout_file = path
    return stdout_file


def read_progress(log):
    """로그에서 (완료, 전체, 마지막 내용, 출력 폴더)."""
    done = total = None
    last = out_dir = ""
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "→" in line and not out_dir:
            out_dir = line.split("→", 1)[1].strip()
        m = STEP_PATTERN.match(line)
        if m:
            # 작업이 직접 붙인 "   ███ 50%  남은 …" 부분은 뗀다
            done, total, last = int(m.group(1)), int(m.group(2)), m.group(3).split("   ")[0]
    return done, total, last, out_dir


def latest_run_dir(prefix=""):
    dirs = [d for d in ARENA_DIR.glob(prefix + "*") if d.is_dir()] if ARENA_DIR.exists() else []
    return max(dirs, key=lambda d: d.stat().st_mtime, default=None)


def report_job(job):
    print(f"▶ {job['kind']} (pid {job['pid']})  {job['args']}")
    print(f"  경과: {_fmt_secs(job['elapsed'])}")
    if job["kind"] in ("server", "play"):
        return

    log = job_log(job["pid"])
    done = total = None
    last = out_dir = ""
    if log and log.exists():
        done, total, last, out_dir = read_progress(log)
        print(f"  로그: {log}")
    elif job["kind"] == "arena":
        # 터미널 출력이면 games.jsonl 줄 수로 완료 판 수만 센다
        run = latest_run_dir("2")
        if run and (run / "games.jsonl").exists():
            done = sum(1 for _ in open(run / "games.jsonl", encoding="utf-8"))
            out_dir = str(run)

    if out_dir:
        print(f"  출력: {out_dir}")
    if done is None:
        print("  진행: 아직 끝난 판 없음 (첫 판 진행 중이거나 stdout이 터미널)")
        return
    if total:
        pct = 100 * done / total
        eta = job["elapsed"] / done * (total - done) if done else 0
        print(f"  진행: {bar(done, total)} {done}/{total} ({pct:.0f}%)  남은 시간 약 {_fmt_secs(eta)}")
    else:
        print(f"  진행: {done}판 완료")
    if last:
        print(f"  최근: {last}")


def report_katago():
    lines = [l for l in _run(["ps", "-axo", "pid=,pcpu=,rss=,etime=,command="]).splitlines()
             if "katago analysis" in l]
    busy = [l for l in lines if float(l.split()[1]) > 1.0]
    rss_gb = sum(int(l.split()[2]) for l in lines) / 1024 / 1024
    print(f"KataGo 프로세스: {len(lines)}개 (작동 중 {len(busy)}개, 메모리 {rss_gb:.1f}GB)")


def report_once():
    print(f"=== 진행상황 {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    jobs = find_jobs()
    if not jobs:
        print("실행 중인 작업 없음")
    for job in jobs:
        report_job(job)
    report_katago()

    run = latest_run_dir()
    if run:
        finished = any((run / f).exists() for f in ("summary.md", "result.json"))
        mtime = time.strftime("%H:%M", time.localtime(run.stat().st_mtime))
        print(f"최근 결과 폴더: {run.relative_to(ROOT)} ({'완료' if finished else '진행 중'}, {mtime})")


def _alive(pid):
    return bool(_run(["ps", "-p", str(pid), "-o", "pid="]).strip())


def follow_job(job, log):
    """작업 로그를 tail -f처럼 실시간 출력. 작업 프로세스가 끝나면 돌아온다."""
    started = time.time() - job["elapsed"]
    print(f"▶ {job['kind']} (pid {job['pid']}) {job['args']}\n  로그: {log}\n", flush=True)

    def annotate(line, m):
        # 진행률이 아직 안 붙은 줄(stdout 리다이렉트로 돌린 작업)에만 붙인다
        if "남은 " in line:
            return line
        return line + "   " + status(int(m.group(1)), int(m.group(2)), started)

    with open(log, encoding="utf-8", errors="replace") as f:
        # 이미 쌓인 줄은 그대로 보여주고, 따라잡은 시점의 진행률만 한 번 표시
        last = None
        for line in f.read().splitlines():
            print(f"         {line}")
            last = STEP_PATTERN.match(line) or last
        if last:
            done, total = int(last.group(1)), int(last.group(2))
            print(f"── 지금 {time.strftime('%H:%M:%S')}  {status(done, total, started)}", flush=True)

        while True:
            line = f.readline()
            if not line:
                if not _alive(job["pid"]):
                    print(f"\n■ 작업 종료 (총 {_fmt_secs(time.time() - started)})", flush=True)
                    return
                time.sleep(0.2)
                continue
            line = line.rstrip("\n")
            m = STEP_PATTERN.match(line)
            if m:
                line = annotate(line, m)
            print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)


def follow():
    """작업이 시작되길 기다렸다가 자동으로 따라간다. 끝나면 다음 작업을 기다린다."""
    followed = set()
    waiting = False
    while True:
        job = log = None
        for j in find_jobs():
            if j["kind"] in ("arena", "estimate_eval") and j["pid"] not in followed:
                log = job_log(j["pid"])
                if log:
                    job = j
                    break
        if not job:
            if not waiting:
                print("작업 대기 중… (arena·estimate_eval이 시작되면 자동으로 표시, Ctrl+C로 종료)", flush=True)
                waiting = True
            time.sleep(1)
            continue
        waiting = False
        followed.add(job["pid"])
        follow_job(job, log)
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description="실행 중인 작업 진행상황")
    p.add_argument("--watch", type=float, metavar="초", help="주기적으로 갱신")
    p.add_argument("-f", "--follow", action="store_true", help="작업이 시작되면 자동으로 로그를 실시간 추적")
    args = p.parse_args(argv)
    if args.follow:
        try:
            follow()
        except KeyboardInterrupt:
            pass
        return
    if not args.watch:
        report_once()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            report_once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
