"""Round-18 monitor: drives the validation -> test -> analysis chain
on Beast. Run every 30 min (cron). Idempotent: marker files in r18/
record each phase transition, so a re-run never relaunches workers.

Phases:
  val   -> (30 val lines, 2 tasks x 3 bump sigmas x 5 seeds) ->
           pick bump-sigma winner per task (lowest mean val MSE,
           tie-break first in grid order 1.0, 2.0, 3.0) -> write
           winner.json (round-15 winner alpha/eta_d + bump_sigma)
           -> launch test
  test  -> (360 test lines: 2 tasks x 20 seeds x 9 lines) ->
           concatenate shard files into
           r18_{task}_test_262-281.jsonl -> run analysis ->
           write r18_analysis.md
Prints one status line per phase plus counts. Exits 0 always (a
failed check is a status, not an error).
"""
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUMP_GRID = [1.0, 2.0, 3.0]
TASKS = ["lorenz", "varsine"]
VAL_SEEDS = "262-266"
SHARDS = (("a", "262-271"), ("b", "272-281"))
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


def sigma_tag(s):
    return f"s{int(round(s * 10)):02d}"


def launch_val():
    for task in TASKS:
        for s in BUMP_GRID:
            out = os.path.join(
                HERE, f"r18_{task}_val_{sigma_tag(s)}.jsonl")
            logf = os.path.join(
                HERE, f"r18_{task}_val_{sigma_tag(s)}.log")
            launch([PY, "rpc_round18_continuous.py", task, "val",
                    "--seeds", VAL_SEEDS, "--out", out,
                    "--bump-sigma", str(s)], logf)
    open(os.path.join(HERE, "val_launched"), "w").write("ok")
    log("VAL LAUNCHED: 6 workers (2 tasks x 3 bump sigmas, "
        "seeds 262-266)")


def val_lines():
    return sum(count_lines(p) for p in
               glob.glob(os.path.join(HERE, "r18_*_val_*.jsonl")))


def pick_winner():
    """Per task: mean val MSE per bump sigma over the 5 val seeds.
    Winner is the lowest mean; ties break to the first sigma in
    grid order (1.0, 2.0, 3.0). Writes winner.json with the
    round-15 winner config plus the val-picked bump_sigma.
    The round-15 winners are frozen in the protocol; the driver
    asserts them at test launch, so a typo here fails loudly."""
    r15w = {"lorenz": {"alpha": 0.7, "smooth_sigma": 0.0,
                       "eta_d": 0.2},
            "varsine": {"alpha": 1.0, "smooth_sigma": 2.0,
                        "eta_d": 0.02}}
    winners = {}
    for task in TASKS:
        mses = {}
        for p in glob.glob(os.path.join(
                HERE, f"r18_{task}_val_*.jsonl")):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    mses.setdefault(r["bump_sigma"], []).append(
                        r["val_mse"])
        assert mses, f"no val results for {task}"
        means = {s: sum(v) / len(v) for s, v in mses.items() if v}
        for s in sorted(means):
            log(f"VAL {task} bump_sigma={s}: mean val_mse="
                f"{means[s]:.4f} n={len(mses[s])}")
        winner = min(BUMP_GRID, key=lambda s: (means[s], s))
        log(f"WINNER {task} bump_sigma={winner}: mean val_mse="
            f"{means[winner]:.4f}")
        winners[task] = {"alpha": r15w[task]["alpha"],
                         "smooth_sigma": r15w[task]["smooth_sigma"],
                         "eta_d": r15w[task]["eta_d"],
                         "bump_sigma": winner,
                         "val_mse": means[winner]}
    with open(os.path.join(HERE, "winner.json"), "w") as f:
        json.dump(winners, f, indent=2)
    return winners


def launch_test(winners):
    for task in TASKS:
        for shard, seeds in SHARDS:
            out = os.path.join(HERE, f"r18_{task}_test_{shard}.jsonl")
            logf = os.path.join(HERE, f"r18_{task}_test_{shard}.log")
            launch([PY, "rpc_round18_continuous.py", task, "test",
                    "--seeds", seeds, "--out", out,
                    "--winners", os.path.join(HERE, "winner.json")],
                   logf)
    open(os.path.join(HERE, "test_launched"), "w").write("ok")
    log("TEST LAUNCHED: 4 workers (2 tasks x 2 shards, "
        "seeds 262-281, bump_sigma "
        f"lorenz={winners['lorenz']['bump_sigma']} "
        f"varsine={winners['varsine']['bump_sigma']})")


def test_lines():
    return sum(count_lines(os.path.join(
        HERE, f"r18_{t}_test_{s}.jsonl"))
        for t in TASKS for s, _ in SHARDS)


def concat_test_files():
    """Merge the 4 shard files into the two combined files the
    analyzer reads by default (r18_{task}_test_262-281.jsonl)."""
    for task in TASKS:
        out = os.path.join(HERE, f"r18_{task}_test_262-281.jsonl")
        with open(out, "w") as f:
            for shard, _ in SHARDS:
                p = os.path.join(HERE, f"r18_{task}_test_{shard}.jsonl")
                if os.path.exists(p):
                    with open(p) as g:
                        f.write(g.read())
        log(f"CONCAT: {out} ({count_lines(out)} lines)")


def main():
    for p in glob.glob(os.path.join(HERE, "r18_*_test_*.log")) + \
            glob.glob(os.path.join(HERE, "r18_*_val_*.log")):
        if os.path.exists(p) and "Traceback" in open(p).read():
            log(f"WORKER FAILED: see {p}. Human fix needed.")
            return
    if not os.path.exists(os.path.join(HERE, "val_launched")):
        launch_val()
        return
    n_val = val_lines()
    if not os.path.exists(os.path.join(HERE, "test_launched")):
        log(f"VAL RUNNING: {n_val}/30 lines.")
        if n_val >= 30:
            winners = pick_winner()
            launch_test(winners)
        return
    n_test = test_lines()
    if os.path.exists(os.path.join(HERE, "r18_analysis.md")):
        log(f"ANALYSIS DONE: {n_test}/360 test lines. Nothing to do.")
        return
    log(f"TEST RUNNING: {n_test}/360 lines.")
    if n_test >= 360:
        concat_test_files()
        r = subprocess.run([PY, os.path.join(HERE, "r18_analyze.py")],
                           cwd=ROOT, capture_output=True, text=True)
        log(r.stdout)
        if r.returncode != 0:
            log("ANALYSIS FAILED:\n" + r.stderr[-2000:])
        else:
            log("ANALYSIS DONE: see r18/r18_analysis.md.")


if __name__ == "__main__":
    main()
