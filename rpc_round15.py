"""Round 15 driver: preregistered 20-seed replication of the delta win.

Protocol: PROTOCOL-R15.md (frozen Nov 26, 2026). Seeds 201..220.
Val seeds 201..205, test seeds 201..220. Every (task, phase,
config/condition, seed) writes one JSON line to its own results file;
a worker skips lines already present, so a killed worker resumes.

Usage:
  python rpc_round15.py <task> val --seeds A-B --out FILE
  python rpc_round15.py <task> test --seeds A-B --out FILE --winners FILE
Tasks: planted1, planted2, sine, lorenz, varsine.
Winners file: JSON {task: {alpha, smooth_sigma, eta_d}} from the val phase.
"""
import json
import os
import sys

# Threading fix (Nov 26, 18:24 PT): these runs are tiny-matrix
# (64x200, 200x200) and single-threaded by nature. With the default
# 12 BLAS threads per process on a loaded Beast (load 45), the
# threads spin-wait on barriers and burn CPU without progress: the
# first smoke run took ~85 min wall for ~3.5 min of CPU work at
# 166% CPU. One thread per process restores near-single-threaded
# speed. Must be set before numpy/torch are imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (ESN, accuracy, bin_centre_lorenz, bin_centre_sine,
                       make_data, make_data_lag, make_lorenz,
                       markov_predict, mse_from_bin_preds,
                       persistence_mse, persistence_preds)
from rpc_simple import make_sine_fixed
from rpc_spatial import K as K_SPATIAL
from rpc_spatial import make_sine as make_sine_spatial
from rpc_v2 import RPCNetwork, run

TASKS = ["planted1", "planted2", "sine", "lorenz", "varsine"]
VAL_SEEDS = list(range(201, 206))
TEST_SEEDS = list(range(201, 221))
GRID = [(a, s, e) for a in (0.7, 1.0) for s in (0.0, 2.0, 3.0)
        for e in (0.02, 0.05, 0.2)]
CONDITIONS = ["delta", "tempcov", "burst", "frozen"]
N_TRAIN = {"planted1": 200, "planted2": 200, "sine": 50,
           "lorenz": 200, "varsine": 100}
K_OF = {"planted1": 16, "planted2": 16, "sine": 32,
        "lorenz": 32, "varsine": 32}
MOTIF_M1 = {(1, 2), (5, 6), (9, 10)}
MOTIF_M2 = {(1, 2, 3), (5, 6, 7), (9, 10, 11)}


def make_var_sine_cont(n_seq, length=100, seed=0):
    """rpc_spatial.make_sine with the continuous sine kept (for MSE).

    Identical RNG consumption to make_sine (same default_rng(seed),
    same f then ph draws per sequence), so the bins match make_sine
    exactly; verified in the smoke test."""
    rng = np.random.default_rng(seed)
    bins, cont = [], []
    for _ in range(n_seq):
        f = rng.uniform(0.02, 0.08)
        ph = rng.uniform(0, 2 * np.pi)
        t = np.arange(length)
        x = np.sin(2 * np.pi * f * t + ph)
        b = np.clip(((x + 1) / 2 * K_SPATIAL).astype(int), 0, K_SPATIAL - 1)
        bins.append(b.tolist())
        cont.append(x.tolist())
    return bins, cont


def splits(task, seed):
    """(train, val, test, train_cont, val_cont, test_cont, K, decode).

    cont is None for token tasks. decode maps a bin index to a
    continuous centre (for MSE) or None."""
    K = K_OF[task]
    if task == "planted1":
        np.random.seed(seed)
        return (make_data(200), make_data(50), make_data(50),
                None, None, None, K, None)
    if task == "planted2":
        np.random.seed(seed)
        return (make_data_lag(200, lag=2), make_data_lag(50, lag=2),
                make_data_lag(50, lag=2), None, None, None, K, None)
    if task == "sine":
        return (make_sine_fixed(50, seed=seed),
                make_sine_fixed(20, seed=seed + 2000),
                make_sine_fixed(20, seed=seed + 1000),
                None, None, None, K, None)
    if task == "lorenz":
        np.random.seed(seed)
        tr_b, tr_c = make_lorenz(200)
        va_b, va_c = make_lorenz(50)
        te_b, te_c = make_lorenz(50)
        return (tr_b, va_b, te_b, tr_c, va_c, te_c, K, bin_centre_lorenz)
    if task == "varsine":
        tr_b, tr_c = make_var_sine_cont(100, seed=seed)
        va_b, va_c = make_var_sine_cont(20, seed=seed + 2000)
        te_b, te_c = make_var_sine_cont(20, seed=seed + 1000)
        return (tr_b, va_b, te_b, tr_c, va_c, te_c, K, bin_centre_sine)
    raise ValueError(task)


def net_kw(task, cfg, condition):
    """RPCNetwork kwargs for one (task, config, condition)."""
    kw = dict(K=K_OF[task], n_res=200, n_out=256,
              theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
              proto_n=N_TRAIN[task], alpha=cfg["alpha"],
              smooth_sigma=cfg["smooth_sigma"], full_code=True,
              use_confirmation=False, s_inh=0.0, graded_conf=False,
              norm_rows=False)
    if condition in ("delta", "tempcov", "frozen"):
        kw["learn_rule"] = "delta" if condition != "tempcov" else "tempcov"
        kw["eta_d"] = cfg["eta_d"]
    elif condition == "burst":
        kw["learn_rule"] = "burst"
        kw["norm_rows"] = True
    else:
        raise ValueError(condition)
    return kw


def step_kw(condition):
    return dict(use_confirmation=False,
                plasticity=(condition != "frozen"))


def train_net(task, train, cfg, condition, seed):
    """Build, calibrate, prototype, and train a net; return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, condition)
    skw = step_kw(condition)
    # proto_n and use_confirmation are run/step parameters, not
    # constructor arguments (round-14 gotcha: passing them to
    # RPCNetwork.__init__ raises TypeError).
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=kw["proto_n"])
    for _ in range(5):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **skw)
    return net


def clamp_frac(net):
    return float((net.W_dist1.abs() >= net.w_max - 1e-6).float().mean())


def run_planted(task, train, ev, cfg, condition, seed):
    """Planted runner with m1/m2/oth breakdown."""
    net = train_net(task, train, cfg, condition, seed)
    correct = total = m1c = m1n = m2c = m2n = 0
    bursts = []
    for seq in ev:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            bursts.append(out["burst_rate"])
            if i > 0:
                total += 1
                ok = int(out["pred"] == tok)
                correct += ok
                if (seq[i - 1], tok) in MOTIF_M1:
                    m1n += 1
                    m1c += ok
                if i >= 2 and (seq[i - 2], seq[i - 1], tok) in MOTIF_M2:
                    m2n += 1
                    m2c += ok
    othn = total - m1n - m2n
    othc = correct - m1c - m2c
    return {"acc": correct / max(total, 1),
            "m1": m1c / max(m1n, 1), "m1_n": m1n,
            "m2": m2c / max(m2n, 1), "m2_n": m2n,
            "oth": othc / max(othn, 1), "oth_n": othn,
            "all_motif": (m1c + m2c) / max(m1n + m2n, 1),
            "mean_burst": float(np.mean(bursts)),
            "clamp_frac": clamp_frac(net)}


def run_token(task, train, ev, cfg, condition, seed):
    """Sine runner via rpc_v2.run (returns acc, mean_burst, clamp_frac)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, condition)
    # run() pops use_confirmation and plasticity out of kw itself,
    # so pass plasticity merged in (passing step_kw separately would
    # duplicate use_confirmation).
    r = run(train, ev, epochs=5, **kw,
            plasticity=(condition != "frozen"))
    return {"acc": r["acc"], "mean_burst": r["mean_burst"],
            "clamp_frac": r["clamp_frac"]}


def run_cont(task, train, ev, ev_cont, decode, cfg, condition, seed):
    """Lorenz / var-sine runner: collects per-step predictions."""
    net = train_net(task, train, cfg, condition, seed)
    preds, bursts = [], []
    for seq in ev:
        net.reset()
        p = []
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            bursts.append(out["burst_rate"])
            if i < len(seq) - 1:
                p.append(out["pred"])
        preds.append(p)
    acc = accuracy(preds, ev)
    within1 = float(np.mean([
        abs(pr - seq[t + 1]) <= 1
        for p, seq in zip(preds, ev)
        for t, pr in enumerate(p)]))
    mse = mse_from_bin_preds(preds, ev_cont, decode)
    return {"acc": acc, "within1": within1, "mse": mse,
            "mean_burst": float(np.mean(bursts)),
            "clamp_frac": clamp_frac(net)}


def compute_baselines(task, train, test, test_cont, decode, seed):
    """Markov-1/2, persistence, ESN on the identical test sequences."""
    K = K_OF[task]
    m1 = markov_predict(train, test, K, 1)
    m2 = markov_predict(train, test, K, 2)
    per = persistence_preds(test)
    esn = ESN(K, seed).fit(train).predict(test)
    out = {"markov1_acc": accuracy(m1, test),
           "markov2_acc": accuracy(m2, test),
           "persist_acc": accuracy(per, test),
           "esn_acc": accuracy(esn, test)}
    if test_cont is not None:
        out["markov1_mse"] = mse_from_bin_preds(m1, test_cont, decode)
        out["markov2_mse"] = mse_from_bin_preds(m2, test_cont, decode)
        out["persist_mse"] = persistence_mse(test_cont)
        out["esn_mse"] = mse_from_bin_preds(esn, test_cont, decode)
        out["esn_within1"] = float(np.mean([
            abs(pr - seq[t + 1]) <= 1
            for p, seq in zip(esn, test)
            for t, pr in enumerate(p)]))
    return out


def run_one(task, phase_data, cfg, condition, seed):
    """One model run on the given eval split; returns the metric dict."""
    train, ev, ev_cont, decode = phase_data
    if task in ("planted1", "planted2"):
        return run_planted(task, train, ev, cfg, condition, seed)
    if task == "sine":
        return run_token(task, train, ev, cfg, condition, seed)
    return run_cont(task, train, ev, ev_cont, decode, cfg, condition, seed)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("phase"), r.get("config"), r.get("seed")))
    return done


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task, phase = sys.argv[1], sys.argv[2]
    assert task in TASKS and phase in ("val", "test"), (task, phase)
    seeds, out_path, winners_path = None, None, None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--seeds":
            a, b = sys.argv[i + 1].split("-")
            seeds = list(range(int(a), int(b) + 1))
            i += 2
        elif sys.argv[i] == "--out":
            out_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--winners":
            winners_path = sys.argv[i + 1]
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    if phase == "test":
        assert winners_path, "test phase needs --winners"
        with open(winners_path) as f:
            winners = json.load(f)
        cfg_winner = winners[task]
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            tr, va, te, tr_c, va_c, te_c, K, decode = splits(task, seed)
            if phase == "val":
                ev, ev_cont = va, va_c
                jobs = [("delta", dict(alpha=a, smooth_sigma=s, eta_d=e))
                        for (a, s, e) in GRID]
            else:
                ev, ev_cont = te, te_c
                jobs = [(c, cfg_winner) for c in CONDITIONS]
            for cond, cfg in jobs:
                if cond == "delta" and phase == "val":
                    key = ("val",
                           f"{cfg['alpha']}|{cfg['smooth_sigma']}|{cfg['eta_d']}",
                           seed)
                else:
                    key = (phase, cond, seed)
                if key in done:
                    continue
                r = run_one(task, (tr, ev, ev_cont, decode), cfg, cond, seed)
                rec = {"task": task, "phase": phase,
                       "config": key[1], "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} {phase} {key[1]} seed={seed}: "
                      f"acc={r['acc']:.3f}", flush=True)
            if phase == "test":
                bkey = ("test", "baseline", seed)
                if bkey not in done:
                    b = compute_baselines(task, tr, te, te_c, decode, seed)
                    rec = {"task": task, "phase": "test",
                           "config": "baseline", "seed": seed, **b}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test baseline seed={seed}: "
                          f"markov1={b['markov1_acc']:.3f} "
                          f"markov2={b['markov2_acc']:.3f} "
                          f"persist={b['persist_acc']:.3f} "
                          f"esn={b['esn_acc']:.3f}", flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
