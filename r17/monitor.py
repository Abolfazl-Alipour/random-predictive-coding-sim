"""Round-17 monitor: drives the validation -> test -> analysis chain
on Beast. Run every 30 min (cron). Idempotent: marker files in r17/
record each phase transition, so a re-run never relaunches workers.

Phases:
  val   -> (20 val lines, 4 gains x 5 seeds) -> pick winner (guard:
           winner val_acc >= 0.80x gain-0.0 val_acc) -> launch test
           (if the guard fails, write val_failed and stop: the wire
           destabilizes learning, which is the finding)
  test  -> (200 test lines) -> run analysis -> write r17_analysis.md
Prints one status line per phase plus counts. Exits 0 always (a
failed check is a status, not an error).
"""
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAINS = [0.0, 0.1, 0.3, 1.0]
TASKS = ["planted1", "planted2"]
PY = "/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python"


def log(msg):
    print(msg, flush=True)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def launch(cmd, logpath):
    # Threading fix (Nov 26, 18:24 PT): one BLAS thread per worker.
    env = dict(os.environ)
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[_v] = "1"
    with open(logpath, "w") as lf:
        subprocess.Popen(cmd, cwd=ROOT, stdout=lf,
                         stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)


def gain_tag(g):
    return f"g{int(round(g * 100)):03d}"


def launch_val():
    for g in GAINS:
        out = os.path.join(HERE, f"r17_planted1_val_{gain_tag(g)}.jsonl")
        logf = os.path.join(HERE, f"r17_planted1_val_{gain_tag(g)}.log")
        launch([PY, "rpc_round17_clamp.py", "planted1", "val",
                "--seeds", "241-245", "--out", out,
                "--winners", os.path.join(HERE, "..", "r15",
                                          "winners.json"),
                "--fb-gain", str(g)], logf)
    open(os.path.join(HERE, "val_launched"), "w").write("ok")
    log("VAL LAUNCHED: 4 workers (gains 0.0/0.1/0.3/1.0, seeds 241-245)")


def val_lines():
    return sum(count_lines(p) for p in
               glob.glob(os.path.join(HERE, "r17_planted1_val_*.jsonl")))


def pick_winner():
    accs = {}
    for p in glob.glob(os.path.join(HERE, "r17_planted1_val_*.jsonl")):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                accs.setdefault(r["fb_gain"], []).append(r["val_acc"])
    means = {g: sum(v) / len(v) for g, v in accs.items() if v}
    for g in sorted(means):
        log(f"VAL fb_gain={g}: mean val_acc={means[g]:.3f} "
            f"n={len(accs[g])}")
    assert 0.0 in means, "no gain-0.0 val results"
    cands = {g: m for g, m in means.items() if g > 0}
    assert cands, "no non-zero gain val results"
    winner = max(cands, key=cands.get)
    guard = cands[winner] >= 0.80 * means[0.0]
    log(f"WINNER fb_gain={winner}: val_acc={cands[winner]:.3f}; "
        f"gain-0.0 val_acc={means[0.0]:.3f}; guard (>= 0.80x): "
        f"{'PASS' if guard else 'FAIL'}")
    if not guard:
        open(os.path.join(HERE, "val_failed"), "w").write(
            f"winner {winner} val_acc {cands[winner]:.3f} < "
            f"0.80x gain-0.0 {means[0.0]:.3f}")
        log("VAL FAILED: the wire destabilizes learning (guard). "
            "No test phase. This is the finding; report it.")
        return None
    with open(os.path.join(HERE, "winner.json"), "w") as f:
        json.dump({"fb_gain": winner, "val_acc": cands[winner],
                   "gain0_val_acc": means[0.0]}, f, indent=2)
    return winner


def launch_test(winner):
    for task in TASKS:
        for shard, seeds in (("a", "241-250"), ("b", "251-260")):
            out = os.path.join(HERE, f"r17_{task}_test_{shard}.jsonl")
            logf = os.path.join(HERE, f"r17_{task}_test_{shard}.log")
            launch([PY, "rpc_round17_clamp.py", task, "test",
                    "--seeds", seeds, "--out", out,
                    "--winners", os.path.join(HERE, "..", "r15",
                                              "winners.json"),
                    "--fb-gain", str(winner)], logf)
    open(os.path.join(HERE, "test_launched"), "w").write("ok")
    log(f"TEST LAUNCHED: 4 workers (2 tasks x 2 shards, "
        f"fb_gain={winner})")


def test_lines():
    return sum(count_lines(os.path.join(HERE, f"r17_{t}_test_{s}.jsonl"))
               for t in TASKS for s in ("a", "b"))


def main():
    for p in glob.glob(os.path.join(HERE, "r17_*_test_*.log")) + \
            glob.glob(os.path.join(HERE, "r17_*_val_*.log")):
        if os.path.exists(p) and "Traceback" in open(p).read():
            log(f"WORKER FAILED: see {p}. Human fix needed.")
            return
    if os.path.exists(os.path.join(HERE, "val_failed")):
        log("VAL FAILED (guard): the wire destabilizes learning. "
            "No test phase. Report this finding.")
        return
    if not os.path.exists(os.path.join(HERE, "val_launched")):
        launch_val()
        return
    n_val = val_lines()
    if not os.path.exists(os.path.join(HERE, "test_launched")):
        log(f"VAL RUNNING: {n_val}/20 lines.")
        if n_val >= 20:
            winner = pick_winner()
            if winner is not None:
                launch_test(winner)
        return
    n_test = test_lines()
    if os.path.exists(os.path.join(HERE, "r17_analysis.md")):
        log(f"ANALYSIS DONE: {n_test}/200 test lines. Nothing to do.")
        return
    log(f"TEST RUNNING: {n_test}/200 lines.")
    if n_test >= 200:
        r = subprocess.run([PY, os.path.join(HERE, "r17_analyze.py")],
                           cwd=ROOT, capture_output=True, text=True)
        log(r.stdout)
        if r.returncode != 0:
            log("ANALYSIS FAILED:\n" + r.stderr[-2000:])
        else:
            log("ANALYSIS DONE: see r17/r17_analysis.md.")


if __name__ == "__main__":
    main()
