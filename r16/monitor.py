"""Round-16 monitor: watches the omission-test run on Beast.
Run every 30 min (cron). Workers were launched by the main agent
(4 workers: 2 tasks x 2 seed-shards); this script only watches.
When all 80 test lines are present, it runs r16/r16_analyze.py
once and reports. Exits 0 always (a failed check is a status,
not an error).
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = ["planted1", "planted2"]
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"


def log(msg):
    print(msg, flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def test_lines():
    return sum(count_lines(
        os.path.join(HERE, f"r16_{t}_test_{s}.jsonl"))
        for t in TASKS for s in ("a", "b"))


def main():
    for t in TASKS:
        for s in ("a", "b"):
            p = os.path.join(HERE, f"r16_{t}_test_{s}.log")
            if os.path.exists(p) and "Traceback" in open(p).read():
                log(f"WORKER FAILED: {t} shard {s}; see "
                    f"r16/r16_{t}_test_{s}.log. Human fix needed.")
                return
    n = test_lines()
    if os.path.exists(os.path.join(HERE, "r16_analysis.md")):
        log(f"ANALYSIS DONE: {n}/80 test lines. Nothing to do.")
        return
    log(f"TEST RUNNING: {n}/80 lines.")
    if n >= 80:
        r = subprocess.run([PY, os.path.join(HERE, "r16_analyze.py")],
                           cwd=ROOT, capture_output=True, text=True)
        log(r.stdout)
        if r.returncode != 0:
            log("ANALYSIS FAILED:\n" + r.stderr[-2000:])
        else:
            log("ANALYSIS DONE: see r16/r16_analysis.md.")


if __name__ == "__main__":
    main()
