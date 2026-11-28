"""Round 19 monitor: val -> pick winner -> test -> analysis.

Pattern: r18/monitor.py. Idempotent via marker files
(r19/markers/{val_launched,test_launched,analysis}). Exits 0
always, so a cron can run it repeatedly and it only moves
forward when the previous phase is complete.

Val: 6 workers, one per architecture config (the round-18
fix: one config per worker, so 6 workers write 30 lines,
no duplication). Pick: highest mean val acc, tie-break
first in grid order. Test: 6 workers (3 tasks x 2 shards).
Analysis at 480 lines (420 if the winner is 2 levels,
because hier2_ladder is skipped then).
"""
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R19 = os.path.join(BASE, "r19")
MARK = os.path.join(R19, "markers")
DRIVER = os.path.join(BASE, "rpc_round19_hierarchy.py")
ANALYZER = os.path.join(R19, "r19_analyze.py")
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"
TASKS = ["planted1", "sine", "hier"]
VAL_SEEDS = "282-286"
SHARDS = (("a", "282-291"), ("b", "292-301"))
GRID = [(2, 0.1, None), (2, 0.3, None),
        (3, 0.1, 0.1), (3, 0.1, 0.3),
        (3, 0.3, 0.1), (3, 0.3, 0.3)]
VAL_LINES = 30
TEST_LINES = 480


def log(msg):
    print(f"[r19-monitor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tag(n_levels, g1, g2):
    t = f"L{n_levels}g{int(g1 * 100):03d}"
    if n_levels == 3:
        t += f"_g{int(g2 * 100):03d}"
    return t


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def val_files():
    return [os.path.join(R19, f"r19_hier_val_{tag(*g)}.jsonl")
            for g in GRID]


def test_files():
    out = []
    for task in TASKS:
        for sh, _ in SHARDS:
            out.append(os.path.join(
                R19, f"r19_{task}_test_{sh}.jsonl"))
    return out


def any_traceback():
    for pat in ("r19_hier_val_*.jsonl", "r19_*_test_*.jsonl"):
        for path in glob.glob(os.path.join(R19, pat)):
            with open(path, errors="replace") as f:
                if "Traceback" in f.read():
                    return path
    return None


def marker(name):
    return os.path.join(MARK, name)


def launch_val():
    os.makedirs(MARK, exist_ok=True)
    for g in GRID:
        n_levels, g1, g2 = g
        out = os.path.join(R19, f"r19_hier_val_{tag(*g)}.jsonl")
        cmd = [PY, DRIVER, "hier", "val", "--seeds", VAL_SEEDS,
               "--out", out, "--n-levels", str(n_levels),
               "--fb-gain1", str(g1)]
        if n_levels == 3:
            cmd += ["--fb-gain2", str(g2)]
        log(f"launch val {tag(*g)}: {' '.join(cmd)}")
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
    best = None
    for g in GRID:
        n_levels, g1, g2 = g
        cfg_rows = [r for r in rows
                    if r["n_levels"] == n_levels
                    and r["fb_gain1"] == g1
                    and r["fb_gain2"] == g2]
        if len(cfg_rows) != 5:
            log(f"config {tag(*g)} has {len(cfg_rows)} val rows, "
                "expected 5; refusing to pick")
            return None
        mean = sum(r["val_acc"] for r in cfg_rows) / 5
        log(f"config {tag(*g)}: mean val acc {mean:.3f}")
        if best is None or mean > best[0]:
            best = (mean, n_levels, g1, g2)
    mean, n_levels, g1, g2 = best
    w = {"n_levels": n_levels, "fb_gain1": g1, "fb_gain2": g2,
         "val_acc": mean,
         "tasks": {"planted1": {"alpha": 1.0, "smooth_sigma": 0.0,
                                "eta_d": 0.05},
                   "sine": {"alpha": 1.0, "smooth_sigma": 0.0,
                            "eta_d": 0.2},
                   "hier": {"alpha": 1.0, "smooth_sigma": 0.0,
                            "eta_d": 0.05}}}
    with open(os.path.join(R19, "winner.json"), "w") as f:
        json.dump(w, f, indent=2)
    log(f"winner: {tag(n_levels, g1, g2)} val_acc {mean:.3f}")
    return w


def launch_test():
    for task in TASKS:
        for sh, seeds in SHARDS:
            out = os.path.join(R19, f"r19_{task}_test_{sh}.jsonl")
            cmd = [PY, DRIVER, task, "test", "--seeds", seeds,
                   "--out", out, "--winners",
                   os.path.join(R19, "winner.json")]
            log(f"launch test {task} shard {sh}: {' '.join(cmd)}")
            with open(out + ".log", "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                 start_new_session=True)
    with open(marker("test_launched"), "w") as f:
        f.write("launched\n")
    log("test workers launched (detached)")


def concat_test():
    for task in TASKS:
        out = os.path.join(R19, f"r19_{task}_test_282-301.jsonl")
        with open(out, "w") as fo:
            for sh, _ in SHARDS:
                path = os.path.join(R19, f"r19_{task}_test_{sh}.jsonl")
                if os.path.exists(path):
                    with open(path) as fi:
                        fo.write(fi.read())


def run_analysis():
    with open(os.path.join(R19, "r19_analysis.log"), "a") as lf:
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
    wpath = os.path.join(R19, "winner.json")
    if not os.path.exists(wpath):
        w = pick_winner()
        if w is None:
            return 0
    else:
        with open(wpath) as f:
            w = json.load(f)
    expected = TEST_LINES if w["n_levels"] == 3 else TEST_LINES - 60
    test_n = sum(count_lines(p) for p in test_files())
    log(f"test lines: {test_n} / {expected}")
    if test_n < expected:
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
