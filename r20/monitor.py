"""Round 20 monitor: val -> pick winner -> test -> analysis.

Pattern: r19/monitor.py. Idempotent via marker files
(r20/markers/{val_launched,test_launched,analysis}). Exits 0
always, so a cron can run it repeatedly and it only moves
forward when the previous phase is complete.

Runs on Beast (PY is the Beast venv, as in r19). The driver is
rpc_round20_aligned.py; the analyzer is r20/r20_analyze.py.

Val: 9 workers, one per (wire type, gain): aligned/learned/
random x {0.1, 0.3, 1.0}, planted1-occ, val seeds 302..306,
one file per (wire, gain), 5 lines each, 45 total.
Pick: per wire type, largest val on-cut omit_m2 gap among
gains passing the state gate (val p1_cos < 0.90) and stable
(r1_max < 10); tie-break smallest gain. If no gain passes the
gate, the largest gap among stable gains wins and the gate
failure is flagged in winner.json (the test still runs; the
gate failure is a finding, not a stop).
Test: 6 workers (3 tasks x 2 shards of 10 seeds), 11 lines
per (task, seed), 660 total.
Analysis at 660 lines: concatenates the test shards into
r20_{task}_test_302-321.jsonl and runs r20_analyze.py.
"""
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R20 = os.path.join(BASE, "r20")
MARK = os.path.join(R20, "markers")
DRIVER = os.path.join(BASE, "rpc_round20_aligned.py")
ANALYZER = os.path.join(R20, "r20_analyze.py")
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"
TASKS = ["planted1-occ", "planted2-occ", "sine"]
VAL_SEEDS = "302-306"
SHARDS = (("a", "302-311"), ("b", "312-321"))
WIRE_TYPES = ["aligned", "learned", "random"]
GAINS = [0.1, 0.3, 1.0]
VAL_LINES = 45
TEST_LINES = 660


def log(msg):
    print(f"[r20-monitor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def val_files():
    return [os.path.join(R20, f"r20_planted1-occ_val_{wt}_g{g}.jsonl")
            for wt in WIRE_TYPES for g in GAINS]


def test_files():
    out = []
    for task in TASKS:
        for sh, _ in SHARDS:
            out.append(os.path.join(R20, f"r20_{task}_test_{sh}.jsonl"))
    return out


def any_traceback():
    for pat in ("r20_*_val_*.jsonl", "r20_*_test_*.jsonl"):
        for path in glob.glob(os.path.join(R20, pat)):
            with open(path, errors="replace") as f:
                if "Traceback" in f.read():
                    return path
    return None


def marker(name):
    return os.path.join(MARK, name)


def launch_val():
    os.makedirs(MARK, exist_ok=True)
    for wt in WIRE_TYPES:
        for g in GAINS:
            out = os.path.join(
                R20, f"r20_planted1-occ_val_{wt}_g{g}.jsonl")
            cmd = [PY, DRIVER, "planted1-occ", "val", "--seeds",
                   VAL_SEEDS, "--out", out, "--wire-type", wt,
                   "--fb-gain", str(g)]
            log(f"launch val {wt} g{g}: {' '.join(cmd)}")
            with open(out + ".log", "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                 start_new_session=True)
    with open(marker("val_launched"), "w") as f:
        f.write("launched\n")
    log("val workers launched (detached)")


def pick_winner():
    rows = []
    for path in val_files():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    winners = {}
    for wt in WIRE_TYPES:
        cands = []
        for g in GAINS:
            rs = [r for r in rows
                  if r["wire_type"] == wt and r["fb_gain"] == g]
            if len(rs) != 5:
                log(f"wire {wt} gain {g}: {len(rs)} val rows, "
                    "expected 5; refusing to pick")
                return None
            gap = sum(r["val_gap_on_cut"] for r in rs) / 5
            cos = sum(r["val_p1_cos"] for r in rs) / 5
            acc = sum(r["val_acc"] for r in rs) / 5
            rmax = max(r["r1_max"] for r in rs)
            cands.append((g, gap, cos, acc, rmax))
            log(f"wire {wt} gain {g}: val gap {gap:+.3f}, "
                f"p1_cos {cos:.3f}, val_acc {acc:.3f}, "
                f"r1_max {rmax:.2f}")
        gated = [c for c in cands if c[2] < 0.90 and c[4] < 10.0]
        pool = gated if gated else [c for c in cands if c[4] < 10.0]
        pool = pool if pool else cands
        pool.sort(key=lambda c: (-c[1], c[0]))
        g, gap, cos, acc, rmax = pool[0]
        winners[wt] = {"fb_gain": g, "val_gap": gap,
                       "val_p1_cos": cos, "val_acc": acc,
                       "gate": cos < 0.90,
                       "gate_failed": not bool(gated)}
        log(f"winner {wt}: fb_gain={g} (val gap {gap:+.3f}, "
            f"p1_cos {cos:.3f}, gate "
            f"{'PASS' if cos < 0.90 else 'FAIL'})")
    with open(os.path.join(R20, "winner.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_test():
    for task in TASKS:
        for sh, seeds in SHARDS:
            out = os.path.join(R20, f"r20_{task}_test_{sh}.jsonl")
            cmd = [PY, DRIVER, task, "test", "--seeds", seeds,
                   "--out", out, "--winners",
                   os.path.join(R20, "winner.json")]
            log(f"launch test {task} shard {sh}: {' '.join(cmd)}")
            with open(out + ".log", "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                 start_new_session=True)
    with open(marker("test_launched"), "w") as f:
        f.write("launched\n")
    log("test workers launched (detached)")


def concat_test():
    for task in TASKS:
        out = os.path.join(R20, f"r20_{task}_test_302-321.jsonl")
        with open(out, "w") as fo:
            for sh, _ in SHARDS:
                path = os.path.join(R20, f"r20_{task}_test_{sh}.jsonl")
                if os.path.exists(path):
                    with open(path) as fi:
                        fo.write(fi.read())


def run_analysis():
    with open(os.path.join(R20, "r20_analysis.log"), "a") as lf:
        r = subprocess.run([PY, ANALYZER], stdout=lf, stderr=lf)
    with open(marker("analysis"), "w") as f:
        f.write(f"rc={r.returncode}\n")
    log(f"analysis rc={r.returncode}")
    return r.returncode


def main():
    os.makedirs(MARK, exist_ok=True)
    tb = any_traceback()
    if tb:
        log(f"Traceback found in {tb}; stopping, no new launches")
        return 0
    val_n = sum(count_lines(p) for p in val_files())
    log(f"val lines: {val_n} / {VAL_LINES}")
    if val_n < VAL_LINES:
        if not os.path.exists(marker("val_launched")):
            launch_val()
        else:
            log("val launched but incomplete; waiting")
        return 0
    wpath = os.path.join(R20, "winner.json")
    if not os.path.exists(wpath):
        w = pick_winner()
        if w is None:
            return 0
    else:
        with open(wpath) as f:
            w = json.load(f)
    test_n = sum(count_lines(p) for p in test_files())
    log(f"test lines: {test_n} / {TEST_LINES}")
    if test_n < TEST_LINES:
        if not os.path.exists(marker("test_launched")):
            launch_test()
        else:
            log("test launched but incomplete; waiting")
        return 0
    if not os.path.exists(marker("analysis")):
        concat_test()
        rc = run_analysis()
        log(f"analysis done rc={rc}")
    else:
        log("analysis already done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
