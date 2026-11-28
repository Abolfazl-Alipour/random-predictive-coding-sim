"""Round 22 monitor: noise val -> pick c_low -> interaction val ->
pick floors -> test -> analysis.

Pattern: r21/monitor.py. Idempotent via marker files
(r22/markers/{noise_val_launched, interaction_val_launched,
test_launched, analysis}). Exits 0 always, so a cron can run it
repeatedly and it only moves forward when the previous phase is
complete.

Runs on Beast (PY is the Beast venv, as in r20/r21). The driver
is rpc_round22_precision.py; the analyzer is r22/r22_analyze.py.

Noise val: 3 workers, one per task, 5 noise levels x 5 val seeds
347..351, one file per task, 25 lines each, 75 total. Pick: per
task, the SMALLEST c with mean cal_decP1 < 0.75 (calibration
trials: CTL for planted, EXP for sine). Target range 0.50-0.75.
If no c gives decP1 < 0.75, pick 0.50 and flag it. If the
smallest grid c already gives decP1 < 0.50, pick it and flag
"over-corrupted".
Interaction val: 3 workers, one per task, 8 configs x 5 val
seeds, one file per task, 40 lines each, 120 total. Pick: per
task, floors = 50% of the val-measured value, with absolute
minima (0.02 rel_e for P1/P3/W1-W4, 0.05 cosine for P2).
Floors go to winner.json.
Test: 6 workers (3 tasks x 2 shards of 10 seeds 352..371),
8 config lines per (task, seed), 160 lines per task, 480
total. Analysis at the full line count: concatenates the test
shards into r22_{task}_test_352-371.jsonl and runs
r22_analyze.py.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R22 = os.path.join(BASE, "r22")
MARK = os.path.join(R22, "markers")
DRIVER = os.path.join(BASE, "rpc_round22_precision.py")
ANALYZER = os.path.join(R22, "r22_analyze.py")
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"
TASKS = ["planted1", "planted2", "sine"]
VAL_SEEDS = "347-351"
SHARDS = (("a", "352-361"), ("b", "362-371"))
NOISE_GRID = [0.10, 0.15, 0.25, 0.35, 0.50]
NOISE_VAL_LINES = 75
INTERACTION_VAL_LINES = 120
TEST_LINES = 480
# Floor rule (protocol section 7): 50% of the val value, with
# absolute minima.
FLOOR_MIN_RELE = 0.02
FLOOR_MIN_COS = 0.05
T4 = 2.7760  # exact Student-t, 4 df, two-sided 95%.


def log(msg):
    print(f"[r22-monitor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def noise_val_files():
    return [os.path.join(R22, f"r22_{t}_val_noise.jsonl")
            for t in TASKS]


def interaction_val_files():
    return [os.path.join(R22, f"r22_{t}_val_interaction.jsonl")
            for t in TASKS]


def test_files():
    out = []
    for task in TASKS:
        for sh, _ in SHARDS:
            out.append(os.path.join(R22, f"r22_{task}_test_{sh}.jsonl"))
    return out


def any_traceback():
    for pat in ("r22_*_val_*.jsonl", "r22_*_test_*.jsonl"):
        for path in glob.glob(os.path.join(R22, pat)):
            with open(path, errors="replace") as f:
                if "Traceback" in f.read():
                    return path
    return None


def marker(name):
    return os.path.join(MARK, name)


def load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def launch_noise_val():
    os.makedirs(MARK, exist_ok=True)
    for task in TASKS:
        out = os.path.join(R22, f"r22_{task}_val_noise.jsonl")
        cmd = [PY, DRIVER, task, "val", "--seeds", VAL_SEEDS,
               "--out", out, "--val-kind", "noise"]
        log(f"launch noise val {task}: {' '.join(cmd)}")
        with open(out + ".log", "a") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=lf,
                             start_new_session=True)
    with open(marker("noise_val_launched"), "w") as f:
        f.write("launched\n")
    log("noise val workers launched (detached)")


def pick_c_low():
    rows = []
    for path in noise_val_files():
        rows.extend(load_jsonl(path))
    winners = {}
    for task in TASKS:
        dec = {}
        for c in NOISE_GRID:
            rs = [r for r in rows if r["task"] == task
                  and abs(r["c_low"] - c) < 1e-9]
            if len(rs) != 5:
                log(f"{task} c={c:.2f}: {len(rs)} val rows, "
                    "expected 5; refusing to pick")
                return None
            dec[c] = sum(r["cal_decP1"] for r in rs) / 5
            log(f"{task} c={c:.2f}: mean cal_decP1 {dec[c]:.3f}")
        picked = None
        for c in NOISE_GRID:
            if dec[c] < 0.75:
                picked = c
                break
        flag = ""
        if picked is None:
            picked = 0.50
            flag = "no-decode-drop (no c gave decP1 < 0.75)"
        elif dec[picked] < 0.50:
            flag = "over-corrupted (decP1 < 0.50 at picked c)"
        winners[task] = {"c_low": picked,
                         "decP1_by_c": {f"{c:.2f}": dec[c]
                                        for c in NOISE_GRID},
                         "flag": flag}
        log(f"c_low {task}: {picked:.2f} {flag}")
    with open(os.path.join(R22, "winners_noise.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_interaction_val():
    for task in TASKS:
        out = os.path.join(R22, f"r22_{task}_val_interaction.jsonl")
        cmd = [PY, DRIVER, task, "val", "--seeds", VAL_SEEDS,
               "--out", out, "--val-kind", "interaction",
               "--winners", os.path.join(R22, "winners_noise.json")]
        log(f"launch interaction val {task}: {' '.join(cmd)}")
        with open(out + ".log", "a") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=lf,
                             start_new_session=True)
    with open(marker("interaction_val_launched"), "w") as f:
        f.write("launched\n")
    log("interaction val workers launched (detached)")


def val_derived(rec):
    """Per-seed derived quantities from one val/test record:
    interaction and spec_interaction (rel_e / cosine), gap0."""
    ps = rec["p1scale"]
    gap0 = (rec["vio_c0_err"] - rec["exp_c0_err"]) / ps
    gapL = (rec["vio_cl_err"] - rec["exp_cl_err"]) / ps
    spec0 = rec["vio_c0_cePres"] - rec["vio_c0_cePred"]
    specL = rec["vio_cl_cePres"] - rec["vio_cl_cePred"]
    return {"gap0": gap0, "interaction": gap0 - gapL,
            "spec_interaction": spec0 - specL}


def pick_floors():
    with open(os.path.join(R22, "winners_noise.json")) as f:
        noise_w = json.load(f)
    rows = []
    for path in interaction_val_files():
        rows.extend(load_jsonl(path))
    winners = {}
    for task in TASKS:
        by_cfg = {}
        for cfg in ("aligned-on", "aligned-cut", "aligned-shuffle",
                    "random-on", "wire-off", "frozen-aligned"):
            rs = [r for r in rows if r["task"] == task
                  and r["config"] == cfg]
            if len(rs) != 5:
                log(f"{task} {cfg}: {len(rs)} val rows, expected 5; "
                    "refusing to pick")
                return None
            by_cfg[cfg] = [val_derived(r) for r in rs]
        mean = lambda key, cfg: sum(
            d[key] for d in by_cfg[cfg]) / 5
        v_p1 = mean("interaction", "aligned-on")
        v_p2 = mean("spec_interaction", "aligned-on")
        v_p3 = mean("gap0", "aligned-on") - mean("gap0",
                                                 "frozen-aligned")
        v_w1 = v_p1 - mean("interaction", "aligned-cut")
        v_w2 = v_p1 - mean("interaction", "aligned-shuffle")
        v_w3 = v_p1 - mean("interaction", "random-on")
        v_w4 = v_p1 - mean("interaction", "wire-off")
        # Warn if the val interaction itself is not significant
        # on 5 seeds (lower t(4) CI <= 0): the floors fall to
        # their minima and the test will likely fail P1.
        ints = [d["interaction"] for d in by_cfg["aligned-on"]]
        m = sum(ints) / 5
        sd = math.sqrt(sum((x - m) ** 2 for x in ints) / 4)
        lo = m - T4 * sd / math.sqrt(5)
        if lo <= 0:
            log(f"WARNING {task}: val interaction lower t(4) CI "
                f"<= 0 (mean {m:.4f}, CI lo {lo:.4f}); P1 will "
                "likely fail -- that is the correct outcome, "
                "not a design failure")
        w = dict(noise_w[task])
        w["floor_P1"] = max(0.5 * v_p1, FLOOR_MIN_RELE)
        w["floor_P2"] = max(0.5 * v_p2, FLOOR_MIN_COS)
        w["floor_P3"] = max(0.5 * v_p3, FLOOR_MIN_RELE)
        w["floor_W1"] = max(0.5 * v_w1, FLOOR_MIN_RELE)
        w["floor_W2"] = max(0.5 * v_w2, FLOOR_MIN_RELE)
        w["floor_W3"] = max(0.5 * v_w3, FLOOR_MIN_RELE)
        w["floor_W4"] = max(0.5 * v_w4, FLOOR_MIN_RELE)
        w["val"] = {"interaction": v_p1, "spec_interaction": v_p2,
                    "gap_diff": v_p3, "on_cut": v_w1,
                    "on_shuffle": v_w2, "aligned_random": v_w3,
                    "on_off": v_w4}
        winners[task] = w
        log(f"floors {task}: P1 {w['floor_P1']:.4f} "
            f"P2 {w['floor_P2']:.4f} P3 {w['floor_P3']:.4f} "
            f"W1 {w['floor_W1']:.4f} W2 {w['floor_W2']:.4f} "
            f"W3 {w['floor_W3']:.4f} W4 {w['floor_W4']:.4f} "
            f"(val interaction {v_p1:.4f})")
    with open(os.path.join(R22, "winner.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_test():
    for task in TASKS:
        for sh, seeds in SHARDS:
            out = os.path.join(R22, f"r22_{task}_test_{sh}.jsonl")
            cmd = [PY, DRIVER, task, "test", "--seeds", seeds,
                   "--out", out, "--winners",
                   os.path.join(R22, "winner.json")]
            log(f"launch test {task} shard {sh}: {' '.join(cmd)}")
            with open(out + ".log", "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                 start_new_session=True)
    with open(marker("test_launched"), "w") as f:
        f.write("launched\n")
    log("test workers launched (detached)")


def concat_test():
    for task in TASKS:
        out = os.path.join(R22, f"r22_{task}_test_352-371.jsonl")
        with open(out, "w") as fo:
            for sh, _ in SHARDS:
                path = os.path.join(R22, f"r22_{task}_test_{sh}.jsonl")
                if os.path.exists(path):
                    with open(path) as fi:
                        fo.write(fi.read())


def run_analysis():
    with open(os.path.join(R22, "r22_analysis.log"), "a") as lf:
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
    val_n = sum(count_lines(p) for p in noise_val_files())
    log(f"noise val lines: {val_n} / {NOISE_VAL_LINES}")
    if val_n < NOISE_VAL_LINES:
        if not os.path.exists(marker("noise_val_launched")):
            launch_noise_val()
        else:
            log("noise val launched but incomplete; waiting")
        return 0
    npath = os.path.join(R22, "winners_noise.json")
    if not os.path.exists(npath):
        w = pick_c_low()
        if w is None:
            return 0
    ival_n = sum(count_lines(p) for p in interaction_val_files())
    log(f"interaction val lines: {ival_n} / {INTERACTION_VAL_LINES}")
    if ival_n < INTERACTION_VAL_LINES:
        if not os.path.exists(marker("interaction_val_launched")):
            launch_interaction_val()
        else:
            log("interaction val launched but incomplete; waiting")
        return 0
    fpath = os.path.join(R22, "winner.json")
    if not os.path.exists(fpath):
        w = pick_floors()
        if w is None:
            return 0
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
