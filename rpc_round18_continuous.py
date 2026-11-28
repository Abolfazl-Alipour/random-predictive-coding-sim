"""Round 18 driver: the continuous-task wall (scoring correction
+ readout/input separation).

Protocol: PROTOCOL-R18-CONTINUOUS.md (frozen Nov 26, 2026).
Seeds 262..281 (fresh; seed 261 was the diagnostic seed).
Tasks: lorenz, varsine. Four nets per (task, seed):
  N1 delta, one-hot input -> C1 (prototype readout),
                            C2 (ridge readout from N1's d1)
  N2 delta, bump input    -> C3 (prototype readout),
                            C4 (ridge readout from N2's d1)
  N3 frozen, one-hot      -> C5 (prototype readout),
                            C2f (ridge readout from N3's d1)
  N4 frozen, bump input   -> C6 (prototype readout),
                            C4f (ridge readout from N4's d1)
plus one baseline line per (task, seed). All metrics use
convention A: pred(t) vs token t, MSE decode(pred(t)) vs
cont[t] (section 3 of the protocol: run_cont scored pred(t)
against token t+1, a two-step-ahead task; this driver scores
the one-step-ahead task the model actually performs).

The ridge probe is a diagnostic (fit on train d1 only, never
test): it tests whether the continuous value is in the drive,
not a model claim. The bump input is assumption A27
(rpc_v2.cont_bump / cont_bin).

Usage:
  python rpc_round18_continuous.py <task> val --seeds A-B --out FILE
  python rpc_round18_continuous.py <task> test --seeds A-B --out FILE --winners FILE [--epochs N]
Val phase: bump-sigma grid {1.0, 2.0, 3.0} on val seeds, winner
is lowest mean val MSE per task. Test phase needs --winners
(winner.json from the monitor: round-15 winner alpha/eta_d
plus the val-picked bump_sigma per task).
"""
import json
import math
import os
import sys

# Threading fix (Nov 26, 18:24 PT, round 15): these runs are
# tiny-matrix (64x200, 200x200) and single-threaded by nature.
# With the default 12 BLAS threads per process on a loaded Beast
# (load 45), the threads spin-wait on barriers and burn CPU
# without progress. One thread per process restores
# near-single-threaded speed. Must be set before numpy/torch are
# imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import bin_centre_lorenz, bin_centre_sine
from rpc_round15 import (K_OF, N_TRAIN, compute_baselines, splits)
from rpc_v2 import RPCNetwork, cont_bin, cont_bump

TASKS = ["lorenz", "varsine"]
VAL_SEEDS = list(range(262, 267))
TEST_SEEDS = list(range(262, 282))
BUMP_GRID = [1.0, 2.0, 3.0]
EPOCHS = 5
RIDGE_LAM = 1.0
# Round-15 winners (r15/winners.json, verified Nov 26 20:44 PT).
# Frozen here: re-validating under the correct scoring would
# change two things at once (scoring + config) and confound H0.
ROUND15_WINNER = {
    "lorenz": {"alpha": 0.7, "smooth_sigma": 0.0, "eta_d": 0.2},
    "varsine": {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.02},
}
LO_HI = {"lorenz": (-25.0, 25.0), "varsine": (-1.0, 1.0)}
CONDITIONS = ["C1_onehot_delta_proto", "C2_onehot_delta_ridge",
              "C3_bump_delta_proto", "C4_bump_delta_ridge",
              "C5_onehot_frozen_proto", "C6_bump_frozen_proto",
              "C2f_onehot_frozen_ridge", "C4f_bump_frozen_ridge"]


def load_winners(path):
    """Winner config: round-15 alpha/eta_d (asserted) plus the
    val-picked bump_sigma per task (from the monitor)."""
    assert path, "test phase needs --winners"
    with open(path) as f:
        w = json.load(f)
    out = {}
    for task in TASKS:
        cfg = w[task]
        got = {"alpha": cfg["alpha"], "smooth_sigma": cfg["smooth_sigma"],
               "eta_d": cfg["eta_d"]}
        assert got == ROUND15_WINNER[task], (
            f"winners.json {task} = {got}, protocol says "
            f"{ROUND15_WINNER[task]}; refusing to run")
        assert "bump_sigma" in cfg, f"winners.json {task} has no bump_sigma"
        out[task] = {"alpha": cfg["alpha"],
                     "smooth_sigma": cfg["smooth_sigma"],
                     "eta_d": cfg["eta_d"],
                     "bump_sigma": float(cfg["bump_sigma"])}
    return out


def bump_pairs(bins_list, cont_list, task, sigma):
    """(seqs_xy, seqs_v) for the bump input: x is the A27 bump
    vector for the true continuous value, y is its bin. seqs_v
    keeps the true values for the ridge fit and MSE."""
    K = K_OF[task]
    lo, hi = LO_HI[task]
    seqs_xy, seqs_v = [], []
    for bins, cont in zip(bins_list, cont_list):
        xy = []
        for v, b in zip(cont, bins):
            assert cont_bin(v, K, lo, hi) == b, (
                f"cont_bin mismatch: value {v} bins to "
                f"{cont_bin(v, K, lo, hi)}, generator says {b}")
            x = cont_bump(v, K, lo, hi, sigma).numpy().astype(np.float32)
            xy.append((x, b))
        seqs_xy.append(xy)
        seqs_v.append(list(cont))
    return seqs_xy, seqs_v


def net_kw(task, cfg, condition, bump):
    """RPCNetwork kwargs for one (task, config, condition)."""
    kw = dict(K=K_OF[task], n_res=200, n_out=256,
              theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
              proto_n=N_TRAIN[task], alpha=cfg["alpha"],
              smooth_sigma=0.0 if bump else cfg["smooth_sigma"],
              full_code=True, use_confirmation=False, s_inh=0.0,
              graded_conf=False, norm_rows=False)
    kw["learn_rule"] = "delta"
    kw["eta_d"] = cfg["eta_d"]
    return kw


def step_kw(condition):
    return dict(use_confirmation=False,
                plasticity=(condition != "frozen"))


def train_onehot(task, train_bins, cfg, condition, seed, epochs):
    """Build, calibrate, prototype, and train a one-hot net."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, condition, bump=False)
    skw = step_kw(condition)
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    net.calibrate(train_bins)
    net.build_prototypes(train_bins, proto_n=kw["proto_n"])
    for _ in range(epochs):
        for seq in train_bins:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **skw)
    return net


def train_bump(task, train_xy, cfg, condition, seed, epochs):
    """Build, calibrate, prototype, and train a bump-input net.

    Uses calibrate_vec / build_prototypes_vec on the (x, y)
    pairs and step_vec for training: the input is the A27 bump
    vector, the target class is the bin. Same init draws as the
    one-hot net at the same seed (smooth_sigma does not affect
    the RNG stream), so the paired comparisons share the init.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, condition, bump=True)
    skw = step_kw(condition)
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    net.calibrate_vec(train_xy)
    net.build_prototypes_vec(train_xy, proto_n=kw["proto_n"])
    for _ in range(epochs):
        for seq in train_xy:
            net.reset()
            for x, y in seq:
                net.step_vec(x, train=True, **skw)
    return net


def clamp_frac(net):
    return float((net.W_dist1.abs() >= net.w_max - 1e-6).float().mean())


def collect(net, seqs, seqs_v, use_vec):
    """Per-step (pred, d1, true value) for t >= 1 on frozen nets.

    Convention A: pred(t) is the prediction of token t, made from
    the code of token t-1, so it is scored against the value at
    the same step t. Returns (preds, d1s, vs, bins) with one entry
    per (sequence, step t >= 1)."""
    K = net.K
    preds, d1s, vs, bins = [], [], [], []
    for seq, cont in zip(seqs, seqs_v):
        net.reset()
        for i, item in enumerate(seq):
            if use_vec:
                x, y = item
                out = net.step_vec(x, train=False,
                                   use_confirmation=False)
            else:
                out = net.step(item, train=False,
                               use_confirmation=False)
                y = item
            if i >= 1:
                preds.append(out["pred"])
                d1s.append(out["d1"].numpy().copy())
                vs.append(cont[i])
                bins.append(y)
    return preds, d1s, vs, bins


def proto_metrics(preds, vs, bins, decode):
    """Exact-bin accuracy, within-1 accuracy, MSE (convention A).

    MSE decodes the predicted bin to its centre and compares
    against the true continuous value at the same step."""
    preds = np.asarray(preds)
    vs = np.asarray(vs, dtype=float)
    bins = np.asarray(bins)
    acc = float(np.mean(preds == bins))
    within1 = float(np.mean(np.abs(preds - bins) <= 1))
    mse = float(np.mean([(decode(int(p)) - v) ** 2
                         for p, v in zip(preds, vs)]))
    return {"acc": acc, "within1": within1, "mse": mse}


def fit_ridge(d1s, vs, lam=RIDGE_LAM):
    """Centred ridge regression d1 -> continuous value.

    Fit on train pairs only. Returns (W, d1_mean, v_mean) so the
    caller can predict v_hat = v_mean + (d1 - d1_mean) @ W.
    lam = 1.0, the same as the ESN baseline ridge.
    """
    D = np.stack(d1s).astype(np.float64)
    V = np.asarray(vs, dtype=np.float64)
    d_mean = D.mean(axis=0)
    v_mean = float(V.mean())
    Dc = D - d_mean
    Vc = V - v_mean
    A = Dc.T @ Dc + lam * np.eye(D.shape[1])
    W = np.linalg.solve(A, Dc.T @ Vc)
    return W, d_mean, v_mean


def ridge_metrics(fit, d1s, vs, bins, task):
    """Ridge readout metrics (convention A) on collected test data."""
    W, d_mean, v_mean = fit
    D = np.stack(d1s).astype(np.float64)
    vs = np.asarray(vs, dtype=float)
    bins = np.asarray(bins)
    lo, hi = LO_HI[task]
    K = K_OF[task]
    v_hat = v_mean + (D - d_mean) @ W
    pred_bins = np.array([cont_bin(v, K, lo, hi) for v in v_hat])
    acc = float(np.mean(pred_bins == bins))
    within1 = float(np.mean(np.abs(pred_bins - bins) <= 1))
    mse = float(np.mean((v_hat - vs) ** 2))
    return {"acc": acc, "within1": within1, "mse": mse}


def eval_net(net, test_seqs, test_v, decode, task, ridge_fit=None):
    """Prototype metrics on test, plus ridge metrics if a fit is
    given. test_seqs are bins (one-hot) or (x, y) pairs (bump)."""
    use_vec = isinstance(test_seqs[0][0], tuple)
    preds, d1s, vs, bins = collect(net, test_seqs, test_v, use_vec)
    out = proto_metrics(preds, vs, bins, decode)
    out["mean_burst"] = None  # filled by the caller from training
    if ridge_fit is not None:
        r = ridge_metrics(ridge_fit, d1s, vs, bins, task)
        out["ridge_acc"] = r["acc"]
        out["ridge_within1"] = r["within1"]
        out["ridge_mse"] = r["mse"]
    out["clamp_frac"] = clamp_frac(net)
    return out


def mean_burst_onehot(net, test_bins):
    bursts = []
    for seq in test_bins:
        net.reset()
        for tok in seq:
            bursts.append(net.step(tok, train=False,
                                   use_confirmation=False)["burst_rate"])
    return float(np.mean(bursts)) if bursts else 0.0


def mean_burst_bump(net, test_xy):
    bursts = []
    for seq in test_xy:
        net.reset()
        for x, y in seq:
            bursts.append(net.step_vec(x, train=False,
                                       use_confirmation=False)["burst_rate"])
    return float(np.mean(bursts)) if bursts else 0.0


def run_val(task, seed, sigma, epochs):
    """One bump-sigma validation run: train bump delta on train,
    score prototype readout on VAL (convention A)."""
    tr, va, te, tr_c, va_c, te_c, K, decode = splits(task, seed)
    cfg = ROUND15_WINNER[task]
    tr_xy, tr_v = bump_pairs(tr, tr_c, task, sigma)
    va_xy, va_v = bump_pairs(va, va_c, task, sigma)
    net = train_bump(task, tr_xy, cfg, "delta", seed, epochs)
    m = eval_net(net, va_xy, va_v, decode, task)
    m["mean_burst"] = mean_burst_bump(net, va_xy)
    return {"val_mse": m["mse"], "val_acc": m["acc"],
            "val_within1": m["within1"], "mean_burst": m["mean_burst"],
            "clamp_frac": m["clamp_frac"]}


def run_test_nets(task, seed, cfg, bump_sigma, epochs):
    """Train the four nets and yield (condition, metric dict)
    pairs in order. Yields as each net finishes, so the caller
    can write lines incrementally and a killed worker resumes
    per condition (retraining only the missing nets)."""
    tr, va, te, tr_c, va_c, te_c, K, decode = splits(task, seed)
    tr_xy, tr_v = bump_pairs(tr, tr_c, task, bump_sigma)
    te_xy, te_v = bump_pairs(te, te_c, task, bump_sigma)
    # N1: delta, one-hot.
    net1 = train_onehot(task, tr, cfg, "delta", seed, epochs)
    r1 = eval_net(net1, te, te_v, decode, task)
    r1["mean_burst"] = mean_burst_onehot(net1, te)
    yield "C1_onehot_delta_proto", r1
    fit1 = fit_ridge(*collect_train(net1, tr, tr_v, use_vec=False))
    r2 = eval_net(net1, te, te_v, decode, task, ridge_fit=fit1)
    yield "C2_onehot_delta_ridge", {
        "acc": r2["ridge_acc"], "within1": r2["ridge_within1"],
        "mse": r2["ridge_mse"], "mean_burst": r1["mean_burst"],
        "clamp_frac": r1["clamp_frac"]}
    # N2: delta, bump.
    net2 = train_bump(task, tr_xy, cfg, "delta", seed, epochs)
    r3 = eval_net(net2, te_xy, te_v, decode, task)
    r3["mean_burst"] = mean_burst_bump(net2, te_xy)
    yield "C3_bump_delta_proto", r3
    fit2 = fit_ridge(*collect_train(net2, tr_xy, tr_v, use_vec=True))
    r4 = eval_net(net2, te_xy, te_v, decode, task, ridge_fit=fit2)
    yield "C4_bump_delta_ridge", {
        "acc": r4["ridge_acc"], "within1": r4["ridge_within1"],
        "mse": r4["ridge_mse"], "mean_burst": r3["mean_burst"],
        "clamp_frac": r3["clamp_frac"]}
    # N3: frozen, one-hot.
    net3 = train_onehot(task, tr, cfg, "frozen", seed, epochs)
    r5 = eval_net(net3, te, te_v, decode, task)
    r5["mean_burst"] = mean_burst_onehot(net3, te)
    yield "C5_onehot_frozen_proto", r5
    fit3 = fit_ridge(*collect_train(net3, tr, tr_v, use_vec=False))
    r6 = eval_net(net3, te, te_v, decode, task, ridge_fit=fit3)
    yield "C2f_onehot_frozen_ridge", {
        "acc": r6["ridge_acc"], "within1": r6["ridge_within1"],
        "mse": r6["ridge_mse"], "mean_burst": r5["mean_burst"],
        "clamp_frac": r5["clamp_frac"]}
    # N4: frozen, bump.
    net4 = train_bump(task, tr_xy, cfg, "frozen", seed, epochs)
    r7 = eval_net(net4, te_xy, te_v, decode, task)
    r7["mean_burst"] = mean_burst_bump(net4, te_xy)
    yield "C6_bump_frozen_proto", r7
    fit4 = fit_ridge(*collect_train(net4, tr_xy, tr_v, use_vec=True))
    r8 = eval_net(net4, te_xy, te_v, decode, task, ridge_fit=fit4)
    yield "C4f_bump_frozen_ridge", {
        "acc": r8["ridge_acc"], "within1": r8["ridge_within1"],
        "mse": r8["ridge_mse"], "mean_burst": r7["mean_burst"],
        "clamp_frac": r7["clamp_frac"]}


def collect_train(net, train_seqs, train_v, use_vec):
    """(d1s, vs) on TRAIN for the ridge fit (t >= 1)."""
    _, d1s, vs, _ = collect(net, train_seqs, train_v, use_vec)
    return d1s, vs


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
    seeds, out_path, winners_path, epochs, bump_sigma = \
        None, None, None, EPOCHS, None
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
        elif sys.argv[i] == "--epochs":
            epochs = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--bump-sigma":
            # FIX (Nov 26, main agent): without this flag every val
            # worker ran all three bump sigmas into its own file, so
            # the 6 workers duplicated each other's work 3x and the
            # monitor's 30-line threshold fired early. With the flag
            # a worker runs only its sigma: 6 workers, 30 lines.
            bump_sigma = float(sys.argv[i + 1])
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    if phase == "test":
        winners = load_winners(winners_path)
        cfg = winners[task]
    else:
        cfg = ROUND15_WINNER[task]
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            if phase == "val":
                sigmas = ([bump_sigma] if bump_sigma is not None
                          else BUMP_GRID)
                for sigma in sigmas:
                    key = ("val", f"bumpsigma|{sigma}", seed)
                    if key in done:
                        continue
                    r = run_val(task, seed, sigma, epochs)
                    rec = {"task": task, "phase": "val",
                           "config": key[1], "seed": seed,
                           "alpha": cfg["alpha"],
                           "smooth_sigma": cfg["smooth_sigma"],
                           "eta_d": cfg["eta_d"],
                           "bump_sigma": sigma, "epochs": epochs, **r}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} val bumpsigma={sigma} seed={seed}: "
                          f"val_mse={r['val_mse']:.4f} "
                          f"val_acc={r['val_acc']:.3f}", flush=True)
            else:
                bump_sigma = cfg["bump_sigma"]
                for cond, r in run_test_nets(task, seed, cfg,
                                             bump_sigma, epochs):
                    key = ("test", cond, seed)
                    if key in done:
                        continue
                    rec = {"task": task, "phase": "test",
                           "config": cond, "seed": seed,
                           "alpha": cfg["alpha"],
                           "smooth_sigma": cfg["smooth_sigma"],
                           "eta_d": cfg["eta_d"],
                           "bump_sigma": bump_sigma,
                           "epochs": epochs, **r}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test {cond} seed={seed}: "
                          f"acc={r['acc']:.3f} mse={r['mse']:.4f}",
                          flush=True)
                bkey = ("test", "baseline", seed)
                if bkey not in done:
                    tr, va, te, tr_c, va_c, te_c, K, decode = splits(
                        task, seed)
                    b = compute_baselines(task, tr, te, te_c, decode,
                                          seed)
                    rec = {"task": task, "phase": "test",
                           "config": "baseline", "seed": seed, **b}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test baseline seed={seed}: "
                          f"markov1={b['markov1_acc']:.3f} "
                          f"persist={b['persist_acc']:.3f} "
                          f"esn={b['esn_acc']:.3f}", flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
