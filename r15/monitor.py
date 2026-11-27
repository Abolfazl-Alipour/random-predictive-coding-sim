"""Round-15 monitor: drives the validation -> test -> analysis chain
on Beast. Run every 30 min (cron). Idempotent: marker files in r15/
record each phase transition, so a re-run never relaunches workers.

Phases:
  smoke -> (all 3 smoke logs end with SMOKE ... DONE) -> launch val
  val   -> (450 val lines) -> pick winners -> launch test
  test  -> (500 test lines) -> run analysis -> write r15_analysis.md
Prints one status line per phase plus counts. Exits 0 always (a
failed check is a status, not an error).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = ["planted1", "planted2", "sine", "lorenz", "varsine"]
GRID = [(a, s, e) for a in (0.7, 1.0) for s in (0.0, 2.0, 3.0)
        for e in (0.02, 0.05, 0.2)]
CONDITIONS = ["delta", "tempcov", "burst", "frozen"]
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"


def log(msg):
    print(msg, flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def smoke_done():
    for g in ("planted", "sine", "cont"):
        p = os.path.join(HERE, f"smoke-{g}.log")
        if not os.path.exists(p):
            return False
        with open(p) as f:
            tail = f.read()[-200:]
        if f"SMOKE {g} DONE" not in tail:
            return False
        if "Traceback" in open(p).read():
            return False
    return True


def smoke_failed():
    for g in ("planted", "sine", "cont"):
        p = os.path.join(HERE, f"smoke-{g}.log")
        if os.path.exists(p) and "Traceback" in open(p).read():
            return g
    return None


def launch(cmd, logpath):
    # Threading fix (Nov 26, 18:24 PT): one BLAS thread per worker.
    # The default 12 threads per process spin-wait on a loaded Beast
    # and turn a 4-min run into ~85 min wall. See rpc_round15.py.
    env = dict(os.environ)
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[_v] = "1"
    with open(logpath, "w") as lf:
        subprocess.Popen(cmd, cwd=ROOT, stdout=lf,
                         stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)


def launch_val():
    for task in TASKS:
        for shard, seeds in (("a", "201-203"), ("b", "204-205")):
            out = os.path.join(HERE, f"r15_{task}_val_{shard}.jsonl")
            logf = os.path.join(HERE, f"r15_{task}_val_{shard}.log")
            launch([PY, "rpc_round15.py", task, "val",
                    "--seeds", seeds, "--out", out], logf)
    open(os.path.join(HERE, "val_launched"), "w").write("ok")
    log("VAL LAUNCHED: 10 workers (5 tasks x 2 shards)")


def val_lines():
    return sum(count_lines(os.path.join(HERE, f"r15_{t}_val_{s}.jsonl"))
               for t in TASKS for s in ("a", "b"))


def pick_winners():
    winners = {}
    for task in TASKS:
        accs = {}
        for shard in ("a", "b"):
            p = os.path.join(HERE, f"r15_{task}_val_{shard}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    key = (r["alpha"], r["smooth_sigma"], r["eta_d"])
                    accs.setdefault(key, []).append(r["acc"])
        best, best_mean = None, -1.0
        for cfg in GRID:  # GRID order is the tie-break order.
            vals = accs.get(cfg, [])
            if not vals:
                continue
            m = sum(vals) / len(vals)
            if m > best_mean:
                best, best_mean = cfg, m
        assert best is not None, f"no val results for {task}"
        winners[task] = {"alpha": best[0], "smooth_sigma": best[1],
                         "eta_d": best[2], "val_mean_acc": best_mean}
        log(f"WINNER {task}: alpha={best[0]} sigma={best[1]} "
            f"eta_d={best[2]} val_mean={best_mean:.3f}")
    with open(os.path.join(HERE, "winners.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_test():
    for task in TASKS:
        for shard, seeds in (("a", "201-210"), ("b", "211-220")):
            out = os.path.join(HERE, f"r15_{task}_test_{shard}.jsonl")
            logf = os.path.join(HERE, f"r15_{task}_test_{shard}.log")
            launch([PY, "rpc_round15.py", task, "test",
                    "--seeds", seeds, "--out", out,
                    "--winners", os.path.join(HERE, "winners.json")], logf)
    open(os.path.join(HERE, "test_launched"), "w").write("ok")
    log("TEST LAUNCHED: 10 workers (5 tasks x 2 shards)")


def test_lines():
    return sum(count_lines(os.path.join(HERE, f"r15_{t}_test_{s}.jsonl"))
               for t in TASKS for s in ("a", "b"))


def main():
    failed = smoke_failed()
    if failed:
        log(f"SMOKE FAILED in group {failed}; see r15/smoke-{failed}.log. "
            "No workers launched. Human fix needed.")
        return
    if not os.path.exists(os.path.join(HERE, "val_launched")):
        if smoke_done():
            launch_val()
        else:
            log("SMOKE RUNNING: waiting for all 3 smoke logs.")
        return
    n_val = val_lines()
    if not os.path.exists(os.path.join(HERE, "test_launched")):
        log(f"VAL RUNNING: {n_val}/450 lines.")
        if n_val >= 450:
            pick_winners()
            launch_test()
        return
    n_test = test_lines()
    if os.path.exists(os.path.join(HERE, "r15_analysis.md")):
        log(f"ANALYSIS DONE: {n_test}/500 test lines. Nothing to do.")
        return
    log(f"TEST RUNNING: {n_test}/500 lines.")
    if n_test >= 500:
        r = subprocess.run([PY, os.path.join(HERE, "r15_analyze.py")],
                           cwd=ROOT, capture_output=True, text=True)
        log(r.stdout)
        if r.returncode != 0:
            log("ANALYSIS FAILED:\n" + r.stderr[-2000:])
        else:
            log("ANALYSIS DONE: see r15/r15_analysis.md.")


if __name__ == "__main__":
    main()
