"""Round 21 monitor: token val -> pick token winners -> bump val ->
pick bump sigmas -> test -> analysis.

Pattern: r20/monitor.py. Idempotent via marker files
(r21/markers/{token_val_launched, bump_val_launched, test_launched,
analysis}). Exits 0 always, so a cron can run it repeatedly and
it only moves forward when the previous phase is complete.

Runs on Beast (PY is the Beast venv, as in r20). The driver is
rpc_round21_continuous.py; the analyzer is r21/r21_analyze.py.

Token val: 4 workers, one per task, 5 grid configs x 5 val seeds
322..326, one file per task, 25 lines each, 100 total. Pick: per
task, lowest mean val prototype MSE; tie-break smallest eta_d,
then smallest smooth_sigma, then smallest alpha.
Bump val: 4 workers, one per task, 2 bump sigmas x 5 val seeds,
using the token winners for alpha/eta_d, 10 lines each, 40 total.
Pick: per task, lowest mean val prototype MSE; tie-break smaller
sigma. Winners (token config + bump sigma + preregistered primary
for lorenz/varsine) go to winners.json.
Test: 8 workers (4 tasks x 2 shards of 10 seeds), all arms per
seed, 7 rows per (task, seed) baseline included (6 arm rows +
baseline), plus 2 conditional primary rows when the val winner
differs from the preregistered primary. Analysis at the full
line count: concatenates the test shards into
r21_{task}_test_327-346.jsonl and runs r21_analyze.py.
"""
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R21 = os.path.join(BASE, "r21")
MARK = os.path.join(R21, "markers")
DRIVER = os.path.join(BASE, "rpc_round21_continuous.py")
ANALYZER = os.path.join(R21, "r21_analyze.py")
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"
TASKS = ["lorenz", "varsine", "narma10", "mackeyglass"]
VAL_SEEDS = "322-326"
SHARDS = (("a", "327-336"), ("b", "337-346"))
TOKEN_VAL_LINES = 100
BUMP_VAL_LINES = 40
PRIMARY = {
    "lorenz": {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.2},
    "varsine": {"alpha": 0.7, "smooth_sigma": 2.0, "eta_d": 0.2},
}


def log(msg):
    print(f"[r21-monitor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def token_val_files():
    return [os.path.join(R21, f"r21_{t}_val_token.jsonl")
            for t in TASKS]


def bump_val_files():
    return [os.path.join(R21, f"r21_{t}_val_bump.jsonl")
            for t in TASKS]


def test_files():
    out = []
    for task in TASKS:
        for sh, _ in SHARDS:
            out.append(os.path.join(R21, f"r21_{task}_test_{sh}.jsonl"))
    return out


def any_traceback():
    for pat in ("r21_*_val_*.jsonl", "r21_*_test_*.jsonl"):
        for path in glob.glob(os.path.join(R21, pat)):
            with open(path, errors="replace") as f:
                if "Traceback" in f.read():
                    return path
    return None


def marker(name):
    return os.path.join(MARK, name)


def launch_token_val():
    os.makedirs(MARK, exist_ok=True)
    for task in TASKS:
        out = os.path.join(R21, f"r21_{task}_val_token.jsonl")
        cmd = [PY, DRIVER, task, "val", "--seeds", VAL_SEEDS,
               "--out", out]
        log(f"launch token val {task}: {' '.join(cmd)}")
        with open(out + ".log", "a") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=lf,
                             start_new_session=True)
    with open(marker("token_val_launched"), "w") as f:
        f.write("launched\n")
    log("token val workers launched (detached)")


def pick_token_winners():
    rows = []
    for path in token_val_files():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    winners = {}
    for task in TASKS:
        cands = []
        cfgs = sorted({r["config"] for r in rows
                       if r["task"] == task})
        for cfg in cfgs:
            rs = [r for r in rows if r["task"] == task
                  and r["config"] == cfg]
            if len(rs) != 5:
                log(f"{task} {cfg}: {len(rs)} val rows, expected 5; "
                    "refusing to pick")
                return None
            mse = sum(r["val_mse"] for r in rs) / 5
            cands.append((cfg, rs[0]["alpha"], rs[0]["smooth_sigma"],
                          rs[0]["eta_d"], mse))
            log(f"{task} {cfg}: val proto MSE {mse:.4f} "
                f"(alpha {rs[0]['alpha']}, smooth "
                f"{rs[0]['smooth_sigma']}, eta_d {rs[0]['eta_d']})")
        # Lowest MSE; tie-break smallest eta_d, then smallest
        # smooth_sigma, then smallest alpha.
        cands.sort(key=lambda c: (c[4], c[3], c[2], c[1]))
        cfg, alpha, smooth, eta_d, mse = cands[0]
        winners[task] = {"alpha": alpha, "smooth_sigma": smooth,
                         "eta_d": eta_d, "val_mse": mse,
                         "val_config": cfg}
        log(f"token winner {task}: {cfg} (val MSE {mse:.4f})")
    with open(os.path.join(R21, "winners_token.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_bump_val():
    for task in TASKS:
        out = os.path.join(R21, f"r21_{task}_val_bump.jsonl")
        cmd = [PY, DRIVER, task, "val", "--seeds", VAL_SEEDS,
               "--out", out, "--val-kind", "bump", "--winners",
               os.path.join(R21, "winners_token.json")]
        log(f"launch bump val {task}: {' '.join(cmd)}")
        with open(out + ".log", "a") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=lf,
                             start_new_session=True)
    with open(marker("bump_val_launched"), "w") as f:
        f.write("launched\n")
    log("bump val workers launched (detached)")


def pick_final_winners():
    with open(os.path.join(R21, "winners_token.json")) as f:
        token = json.load(f)
    rows = []
    for path in bump_val_files():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    winners = {}
    for task in TASKS:
        cands = []
        for sigma in (2.0, 3.0):
            rs = [r for r in rows if r["task"] == task
                  and r["bump_sigma"] == sigma]
            if len(rs) != 5:
                log(f"{task} bump sigma {sigma}: {len(rs)} val rows, "
                    "expected 5; refusing to pick")
                return None
            mse = sum(r["val_mse"] for r in rs) / 5
            cands.append((sigma, mse))
            log(f"{task} bump sigma {sigma}: val proto MSE {mse:.4f}")
        cands.sort(key=lambda c: (c[1], c[0]))
        sigma, mse = cands[0]
        w = dict(token[task])
        w["bump_sigma"] = sigma
        w["bump_val_mse"] = mse
        w["primary"] = PRIMARY.get(task)
        winners[task] = w
        log(f"final winner {task}: alpha {w['alpha']}, smooth "
            f"{w['smooth_sigma']}, eta_d {w['eta_d']}, bump_sigma "
            f"{sigma}, primary={w['primary']}")
    with open(os.path.join(R21, "winner.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def test_lines_expected(winners):
    """7 rows per (task, seed) + 2 conditional primary rows when
    the val winner differs from the preregistered primary."""
    total = 0
    for task in TASKS:
        n = 7
        primary = winners[task].get("primary")
        if primary is not None:
            pcfg = {"alpha": primary["alpha"],
                    "smooth_sigma": primary["smooth_sigma"],
                    "eta_d": primary["eta_d"]}
            wcfg = {"alpha": winners[task]["alpha"],
                    "smooth_sigma": winners[task]["smooth_sigma"],
                    "eta_d": winners[task]["eta_d"]}
            if pcfg != wcfg:
                n += 2
        total += 20 * n
    return total


def launch_test():
    for task in TASKS:
        for sh, seeds in SHARDS:
            out = os.path.join(R21, f"r21_{task}_test_{sh}.jsonl")
            cmd = [PY, DRIVER, task, "test", "--seeds", seeds,
                   "--out", out, "--winners",
                   os.path.join(R21, "winner.json")]
            log(f"launch test {task} shard {sh}: {' '.join(cmd)}")
            with open(out + ".log", "a") as lf:
                subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                 start_new_session=True)
    with open(marker("test_launched"), "w") as f:
        f.write("launched\n")
    log("test workers launched (detached)")


def concat_test():
    for task in TASKS:
        out = os.path.join(R21, f"r21_{task}_test_327-346.jsonl")
        with open(out, "w") as fo:
            for sh, _ in SHARDS:
                path = os.path.join(R21, f"r21_{task}_test_{sh}.jsonl")
                if os.path.exists(path):
                    with open(path) as fi:
                        fo.write(fi.read())


def run_analysis():
    with open(os.path.join(R21, "r21_analysis.log"), "a") as lf:
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
    val_n = sum(count_lines(p) for p in token_val_files())
    log(f"token val lines: {val_n} / {TOKEN_VAL_LINES}")
    if val_n < TOKEN_VAL_LINES:
        if not os.path.exists(marker("token_val_launched")):
            launch_token_val()
        else:
            log("token val launched but incomplete; waiting")
        return 0
    wpath = os.path.join(R21, "winners_token.json")
    if not os.path.exists(wpath):
        w = pick_token_winners()
        if w is None:
            return 0
    else:
        with open(wpath) as f:
            w = json.load(f)
    bump_n = sum(count_lines(p) for p in bump_val_files())
    log(f"bump val lines: {bump_n} / {BUMP_VAL_LINES}")
    if bump_n < BUMP_VAL_LINES:
        if not os.path.exists(marker("bump_val_launched")):
            launch_bump_val()
        else:
            log("bump val launched but incomplete; waiting")
        return 0
    fpath = os.path.join(R21, "winner.json")
    if not os.path.exists(fpath):
        w = pick_final_winners()
        if w is None:
            return 0
    else:
        with open(fpath) as f:
            w = json.load(f)
    expected = test_lines_expected(w)
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
