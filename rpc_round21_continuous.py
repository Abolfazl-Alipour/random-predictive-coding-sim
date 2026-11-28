"""Round 21 driver: continuous re-validation (Convention-A + NARMA-10 /
Mackey-Glass), column-standardized ridge probes, centroid readout,
aligned-wire on/off factor.

Protocol: PROTOCOL-R21-CONTINUOUS.md. Seeds: val 322-326, test 327-346
(20 fresh seeds; no overlap with rounds 15-20).

Usage:
    rpc_round21_continuous.py <task> <val|test> --seeds A-B --out FILE
        [--winners FILE] [--val-kind token|bump] [--epochs N]

    task in lorenz, varsine, narma10, mackeyglass.
    val --val-kind token (default): the 5-config token grid on VAL.
    val --val-kind bump: bump sigmas 2.0/3.0 on VAL, alpha/eta_d from
        the token winners file. test: the test arms from winners.json.

    Output is JSONL, one row per (phase, config, seed), resumable:
    rows already in the output file are skipped on re-run.

Wire decision (protocol section 5, the round-20 open question):
the aligned wire at gain 1.0 is tested as a FACTOR (token_w arm),
not carried as the primary. Round 20 measured a 23-point one-step
sine cost for the wire (0.666 vs 0.892 wire-off) and a negative
rollout sign, so carrying it as primary would confound "does delta
learn continuous trajectories" with "does the wire tax one-step
prediction". The wire-off arm is primary; the wire-on arm is the
test of whether the wire helps or hurts continuous prediction.

Readouts (all three computed from the SAME run, same net, same
seed, so the comparisons are exactly matched):
  proto:    hard argmax over the p1 prototypes (the round-15 readout).
  centroid: softmax-weighted prototype centroid, v_hat = sum_k
            softmax(beta * cos(d1, mu_k)) * centre_k, beta = 10 primary,
            beta = 5 and 20 reported as a sensitivity check. No new
            fitted weights beyond the prototypes; beta is fixed in
            advance (protocol section 4).
  ridge:    ridge regression d1 -> continuous value, fit on TRAIN d1
            only. COLUMN-STANDARDIZED (the D8 lesson: never an
            unstandardized fixed-lambda ridge again). The raw
            (unstandardized) ridge is also computed and reported, so
            the round-18 numbers can be reproduced, but every bar
            uses the standardized one.
The frozen control is a TWIN net: same seed, same init, same
calibration and prototypes, never trained. Building it costs two
frozen collection passes instead of a second 5-epoch run, and the
comparison is exactly matched by construction. The smoke test
verifies the twin equals a standalone frozen invocation exactly.

Scoring is Convention A throughout (round 18): pred(t) is the
prediction of the value at step t, made from the code of step t-1,
scored against the value at t, for t >= 1. Baselines use the
baselines.py convention (preds[t] predicts token t+1); both score
the identical L-1 transitions per sequence.
"""
import json
import math
import os
import sys

# Threading fix (Nov 26, 18:24 PT, round 15): these runs are
# tiny-matrix and single-threaded by nature. With the default 12
# BLAS threads per process on a loaded Beast the threads spin-wait
# on barriers and burn CPU without progress. One thread per process.
# Must be set before numpy/torch are imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import (ESN, accuracy, bin_centre_lorenz,
                       bin_centre_mackeyglass, bin_centre_narma10,
                       bin_centre_sine, make_mackeyglass, make_narma10,
                       markov_predict, mean_predictor_mse,
                       mse_from_bin_preds, persistence_mse,
                       persistence_preds)
from rpc_round15 import splits as splits_old
from rpc_v2 import RPCNetwork, cont_bin, cont_bump

TASKS = ["lorenz", "varsine", "narma10", "mackeyglass"]
K_OF = {t: 32 for t in TASKS}
LO_HI = {"lorenz": (-25.0, 25.0), "varsine": (-1.0, 1.0),
         "narma10": (0.0, 1.0), "mackeyglass": (0.0, 1.5)}
DECODE = {"lorenz": bin_centre_lorenz, "varsine": bin_centre_sine,
          "narma10": bin_centre_narma10,
          "mackeyglass": bin_centre_mackeyglass}
# proto_n per task: the round-15 value for the old tasks (lorenz 200,
# varsine 100); the new length-100 tasks take varsine's 100.
PROTO_N = {"lorenz": 200, "varsine": 100, "narma10": 100,
           "mackeyglass": 100}
VAL_SEEDS = list(range(322, 327))
TEST_SEEDS = list(range(327, 347))
EPOCHS = 5
RIDGE_LAM = 1.0
CENTROID_BETA = 10.0
CENTROID_BETAS = [5.0, 10.0, 20.0]
# Val token grid (protocol section 6). G1/G2 are the Convention-A
# primaries (lorenz / varsine); G3/G4 are the sigma neighbours;
# G5 is the Convention-B carryover reference (what round 18 ran).
GRID = [
    ("G1", {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.2}),
    ("G2", {"alpha": 0.7, "smooth_sigma": 2.0, "eta_d": 0.2}),
    ("G3", {"alpha": 0.7, "smooth_sigma": 3.0, "eta_d": 0.2}),
    ("G4", {"alpha": 1.0, "smooth_sigma": 3.0, "eta_d": 0.2}),
    ("G5", {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.02}),
]
BUMP_SIGMAS = [2.0, 3.0]
# Preregistered primaries (protocol section 6). Lorenz: reproduced
# exactly on seeds 262-266 with the correct scorer (MSE 5.073).
# Varsine: reviewer-reported (AGY, MSE 0.0370), NOT independently
# re-run before this round; the smoke test re-runs it on 262-266.
PRIMARY = {
    "lorenz": {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.2},
    "varsine": {"alpha": 0.7, "smooth_sigma": 2.0, "eta_d": 0.2},
}


def splits21(task, seed):
    """(train, val, test, train_cont, val_cont, test_cont, K, decode).

    Lorenz/varsine reuse rpc_round15.splits exactly (same counts:
    lorenz 200/50/50 length 24, varsine 100/20/20 length 100).
    The new tasks mirror the varsine counts (100/20/20, length 100)
    with the global-RNG sequential draw order, like lorenz.
    """
    if task in ("lorenz", "varsine"):
        return splits_old(task, seed)
    if task == "narma10":
        np.random.seed(seed)
        tr_b, tr_c = make_narma10(100)
        va_b, va_c = make_narma10(20)
        te_b, te_c = make_narma10(20)
        return (tr_b, va_b, te_b, tr_c, va_c, te_c, K_OF[task],
                DECODE[task])
    if task == "mackeyglass":
        np.random.seed(seed)
        tr_b, tr_c = make_mackeyglass(100)
        va_b, va_c = make_mackeyglass(20)
        te_b, te_c = make_mackeyglass(20)
        return (tr_b, va_b, te_b, tr_c, va_c, te_c, K_OF[task],
                DECODE[task])
    raise ValueError(task)


def net_kw21(task, cfg, wire_on, bump):
    """RPCNetwork kwargs for one (task, config, wire, input) cell.

    The wire is the ONLY difference between the token and token_w
    arms: wire off passes fb_wire="aligned", fb_gain=0.0 (gain 0
    disables the wire whatever its type, _step_core sets fb=None),
    wire on passes the same wire at gain 1.0. All other flags are
    the round-15 validated delta settings (full code, no
    confirmation, no surround inhibition, no graded confirmation,
    no row renormalisation).
    """
    kw = dict(K=K_OF[task], n_res=200, n_out=256,
              theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
              proto_n=PROTO_N[task], alpha=cfg["alpha"],
              smooth_sigma=0.0 if bump else cfg["smooth_sigma"],
              full_code=True, use_confirmation=False, s_inh=0.0,
              graded_conf=False, norm_rows=False,
              learn_rule="delta", eta_d=cfg["eta_d"],
              fb_wire="aligned", fb_gain=1.0 if wire_on else 0.0)
    return kw


def _build_net(task, kw, train, bump):
    """Construct, calibrate, and prototype one net (no training).

    torch.manual_seed is reset here so two nets built with the
    same seed are bit-identical at init: the frozen twin is built
    by this same function with learn_rule="frozen".
    """
    seed = kw.pop("_seed")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    if bump:
        net.calibrate_vec(train)
        net.build_prototypes_vec(train, proto_n=kw["proto_n"])
    else:
        net.calibrate(train)
        net.build_prototypes(train, proto_n=kw["proto_n"])
    return net


def train_token21(task, train_bins, cfg, condition, seed, epochs,
                  wire_on=False):
    """Build, calibrate, prototype, and train a token-input net.

    condition in ("delta", "frozen"). Returns the net.
    """
    kw = net_kw21(task, cfg, wire_on, bump=False)
    if condition == "frozen":
        kw["learn_rule"] = "frozen"
    kw["_seed"] = seed
    net = _build_net(task, kw, train_bins, bump=False)
    skw = dict(use_confirmation=False,
               plasticity=(condition != "frozen"))
    for _ in range(epochs):
        for seq in train_bins:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **skw)
    return net


def bump_pairs(bins_list, cont_list, task, sigma):
    """(seqs_xy, seqs_v) for the bump input: x is the A27 bump
    vector for the true continuous value, y is its bin. Asserts
    the bin mapping agrees with the generator (a mismatch means
    the generator or the bin range changed)."""
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


def train_bump21(task, train_xy, cfg, condition, seed, epochs,
                 wire_on=False):
    """Build, calibrate, prototype, and train a bump-input net."""
    kw = net_kw21(task, cfg, wire_on, bump=True)
    if condition == "frozen":
        kw["learn_rule"] = "frozen"
    kw["_seed"] = seed
    net = _build_net(task, kw, train_xy, bump=True)
    skw = dict(use_confirmation=False,
               plasticity=(condition != "frozen"))
    for _ in range(epochs):
        for seq in train_xy:
            net.reset()
            for x, y in seq:
                net.step_vec(x, train=True, **skw)
    return net


def clamp_frac(net):
    return float((net.W_dist1.abs() >= net.w_max - 1e-6).float().mean())


def collect21(net, seqs, seqs_v, use_vec):
    """Per-step (pred, d1, cosine vector, true value) for t >= 1.

    Convention A: pred(t) is the prediction of the value at step
    t, made from the code of step t-1, scored against the value
    at t. The cosine vector is cos(d1, mu_k) for every prototype
    k, computed by the same path as predict_from_distal's cosine
    branch (proto_proj / center_proto / dot_proto / knn_readout /
    split_proto are all off in the round-21 net_kw; asserted here
    so a future flag change fails loudly instead of silently
    changing the centroid). Returns (preds, d1s, sims, vs, bins,
    mean_burst, mean_fb_drive, mean_fb_rms).
    """
    assert not (net.proto_proj or net.center_proto or net.dot_proto
                or net.knn_readout or net.split_proto), (
        "round-21 centroid assumes the plain cosine readout path")
    assert net._proto_norm is not None, "prototypes not built"
    preds, d1s, sims, vs, bins = [], [], [], [], []
    bursts, fb_drive, fb_rms = [], [], []
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
            bursts.append(out["burst_rate"])
            fb_drive.append(out["fb_drive_ratio"])
            fb_rms.append(out["fb_rms_ratio"])
            if i >= 1:
                d1 = out["d1"]
                a = torch.nn.functional.normalize(d1[None, :], dim=1)
                s = (a @ net._proto_norm.T)[0].numpy().astype(float)
                sims.append(s)
                preds.append(out["pred"])
                d1s.append(d1.numpy().copy())
                vs.append(cont[i])
                bins.append(y)
    return (preds, d1s, sims, vs, bins, float(np.mean(bursts)),
            float(np.mean(fb_drive)), float(np.mean(fb_rms)))


def proto_metrics(preds, vs, bins, decode):
    """Exact-bin accuracy, within-1 accuracy, MSE (Convention A)."""
    preds = np.asarray(preds)
    vs = np.asarray(vs, dtype=float)
    bins = np.asarray(bins)
    acc = float(np.mean(preds == bins))
    within1 = float(np.mean(np.abs(preds - bins) <= 1))
    mse = float(np.mean([(decode(int(p)) - v) ** 2
                         for p, v in zip(preds, vs)]))
    return {"acc": acc, "within1": within1, "mse": mse}


def centroid_metrics(sims, vs, bins, decode, task, betas=CENTROID_BETAS):
    """Softmax-weighted prototype centroid per beta.

    v_hat = sum_k softmax(beta * cos(d1, mu_k)) * centre_k.
    Returns {beta: {mse, acc, within1}} with the predicted bin
    from cont_bin(v_hat). Beta = 10 is the primary; 5 and 20 are
    the sensitivity check (same collected sims, no extra runs).
    """
    K = K_OF[task]
    lo, hi = LO_HI[task]
    vs = np.asarray(vs, dtype=float)
    bins = np.asarray(bins)
    centres = np.array([decode(k) for k in range(K)], dtype=float)
    out = {}
    for beta in betas:
        v_hat = []
        for s in sims:
            z = beta * (s - s.max())
            w = np.exp(z)
            w = w / w.sum()
            v_hat.append(float(w @ centres))
        v_hat = np.array(v_hat)
        pb = np.array([cont_bin(v, K, lo, hi) for v in v_hat])
        out[beta] = {"mse": float(np.mean((v_hat - vs) ** 2)),
                     "acc": float(np.mean(pb == bins)),
                     "within1": float(np.mean(np.abs(pb - bins) <= 1))}
    return out


def fit_ridge_std(d1s, vs, lam=RIDGE_LAM):
    """Column-standardized ridge d1 -> continuous value.

    Each d1 column is centred and scaled by its TRAIN mean/std
    (ddof=0; columns with std < 1e-12 get sd = 1e-12, so a constant
    column stays finite and gets weight ~0). Fit on train pairs
    only. Returns (W, mu, sd, v_mean) with W in standardized
    units; predict as v_mean + ((D - mu) / sd) @ W.
    """
    D = np.stack(d1s).astype(np.float64)
    V = np.asarray(vs, dtype=np.float64)
    mu = D.mean(axis=0)
    sd = D.std(axis=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    Ds = (D - mu) / sd
    Vc = V - V.mean()
    A = Ds.T @ Ds + lam * np.eye(D.shape[1])
    W = np.linalg.solve(A, Ds.T @ Vc)
    return W, mu, sd, float(V.mean())


def fit_ridge_raw(d1s, vs, lam=RIDGE_LAM):
    """Unstandardized ridge (the round-18 fit_ridge, verbatim).

    Reported only, for backward comparability with round 18's
    C2/C4/C2f/C4f. No bar uses it: the D8 lesson is that a fixed
    lambda on unstandardized features confounds scale with
    information.
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


def ridge_predict_std(fit, d1s):
    W, mu, sd, v_mean = fit
    D = np.stack(d1s).astype(np.float64)
    return v_mean + ((D - mu) / sd) @ W


def ridge_predict_raw(fit, d1s):
    W, d_mean, v_mean = fit
    D = np.stack(d1s).astype(np.float64)
    return v_mean + (D - d_mean) @ W


def ridge_metrics(v_hat, vs, bins, task):
    """MSE / acc / within-1 for a ridge value prediction."""
    K = K_OF[task]
    lo, hi = LO_HI[task]
    vs = np.asarray(vs, dtype=float)
    bins = np.asarray(bins)
    pb = np.array([cont_bin(v, K, lo, hi) for v in v_hat])
    return {"mse": float(np.mean((np.asarray(v_hat) - vs) ** 2)),
            "acc": float(np.mean(pb == bins)),
            "within1": float(np.mean(np.abs(pb - bins) <= 1))}


def eval_net21(net, train_seqs, train_v, test_seqs, test_v, decode,
               task, use_vec):
    """All three readouts for one net: prototype, centroid (3
    betas), standardized ridge + raw ridge. Train d1 is collected
    on a frozen pass (plasticity off) AFTER training, so the ridge
    sees the final W_dist1; test d1 comes from the same pass that
    scores the prototype readout. Returns the metric dict."""
    _, d_tr, _, v_tr, _, _, _, _ = collect21(net, train_seqs,
                                             train_v, use_vec)
    (preds, d_te, sims, v_te, bins, mean_burst, fb_drive,
     fb_rms) = collect21(net, test_seqs, test_v, use_vec)
    out = proto_metrics(preds, v_te, bins, decode)
    cen = centroid_metrics(sims, v_te, bins, decode, task)
    for beta in CENTROID_BETAS:
        tag = f"centroid_mse_{int(beta)}"
        out[tag] = cen[beta]["mse"]
        out[f"centroid_acc_{int(beta)}"] = cen[beta]["acc"]
        out[f"centroid_within1_{int(beta)}"] = cen[beta]["within1"]
    fit_s = fit_ridge_std(d_tr, v_tr)
    r_s = ridge_metrics(ridge_predict_std(fit_s, d_te), v_te, bins,
                        task)
    out["ridge_std_mse"] = r_s["mse"]
    out["ridge_std_acc"] = r_s["acc"]
    out["ridge_std_within1"] = r_s["within1"]
    fit_r = fit_ridge_raw(d_tr, v_tr)
    r_r = ridge_metrics(ridge_predict_raw(fit_r, d_te), v_te, bins,
                        task)
    out["ridge_raw_mse"] = r_r["mse"]
    out["ridge_raw_acc"] = r_r["acc"]
    out["ridge_raw_within1"] = r_r["within1"]
    out["mean_burst"] = mean_burst
    out["fb_drive_ratio"] = fb_drive
    out["fb_rms_ratio"] = fb_rms
    out["clamp_frac"] = clamp_frac(net)
    return out


def esn_scalar_ridge(train_bins, train_cont, test_bins, test_cont,
                     seed, task):
    """ESN with a continuous scalar ridge readout (AGY's strongest
    baseline: Lorenz 3.287 one-hot / 2.097 bump, var-sine 0.0191 /
    0.0072 with the raw ridge). The reservoir is the same
    baselines.py ESN (300 units, RandomState(2000 + seed)). The
    readout is fit on TRAIN states after the washout (t >= 2,
    exactly as ESN._states does) with target cont[t+1]; at test
    time the reservoir runs from zero state over every transition
    (t = 0..L-2), exactly as ESN.predict does. Both the raw
    (lam=1.0, AGY's version) and the column-standardized ridge
    are reported; the standardized one is the primary comparison.
    """
    K = K_OF[task]
    esn = ESN(K, seed)
    X, _ = esn._states(train_bins)
    # Targets aligned with _states rows: per sequence, rows are
    # t = washout..L-2, target cont[t+1].
    L = len(train_bins[0])
    w = esn.washout
    targets = []
    for cont in train_cont:
        for t in range(w, L - 1):
            targets.append(cont[t + 1])
    targets = np.asarray(targets, dtype=float)
    assert len(targets) == len(X), (
        f"ESN scalar target count {len(targets)} != states {len(X)}")
    # Raw ridge (AGY's version).
    Xf = X.astype(np.float64)
    mu_r = Xf.mean(axis=0)
    v_mean = float(targets.mean())
    A = (Xf - mu_r).T @ (Xf - mu_r) + RIDGE_LAM * np.eye(Xf.shape[1])
    W_r = np.linalg.solve(A, (Xf - mu_r).T @ (targets - v_mean))
    # Standardized ridge.
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    Xs = (Xf - mu) / sd
    A = Xs.T @ Xs + RIDGE_LAM * np.eye(Xf.shape[1])
    W_s = np.linalg.solve(A, Xs.T @ (targets - v_mean))
    # Test states: reservoir from zero, every transition t=0..L-2.
    te_states, te_vs = [], []
    for bins, cont in zip(test_bins, test_cont):
        r = np.zeros(esn.n_res)
        for t, tok in enumerate(bins[:-1]):
            x = np.zeros(K)
            x[tok] = 1.0
            r = ((1 - esn.leak) * r
                 + esn.leak * np.tanh(esn.W_in @ x + esn.W_res @ r))
            te_states.append(r.copy())
            te_vs.append(cont[t + 1])
    te_states = np.array(te_states)
    te_vs = np.asarray(te_vs, dtype=float)
    v_raw = v_mean + (te_states - mu_r) @ W_r
    v_std = v_mean + ((te_states - mu) / sd) @ W_s
    return {"esn_scalar_raw_mse": float(np.mean((v_raw - te_vs) ** 2)),
            "esn_scalar_std_mse": float(np.mean((v_std - te_vs) ** 2))}


def compute_baselines21(task, train, test, train_c, test_c, decode,
                        seed):
    """Markov-1/2, persistence, train-mean, ESN argmax, ESN scalar
    ridge, on the identical test sequences. Scoring is the
    baselines.py convention (preds[t] predicts token t+1), which
    scores the same L-1 transitions the RPC collect scores.
    """
    K = K_OF[task]
    m1 = markov_predict(train, test, K, 1)
    m2 = markov_predict(train, test, K, 2)
    per = persistence_preds(test)
    esn = ESN(K, seed).fit(train)
    esn_p = esn.predict(test)
    out = {"markov1_acc": accuracy(m1, test),
           "markov2_acc": accuracy(m2, test),
           "persist_acc": accuracy(per, test),
           "esn_acc": accuracy(esn_p, test),
           "markov1_mse": mse_from_bin_preds(m1, test_c, decode),
           "markov2_mse": mse_from_bin_preds(m2, test_c, decode),
           "persist_mse": persistence_mse(test_c),
           "mean_mse": mean_predictor_mse(train_c, test_c),
           "esn_mse": mse_from_bin_preds(esn_p, test_c, decode)}
    out.update(esn_scalar_ridge(train, train_c, test, test_c, seed,
                                task))
    return out


def run_val_token(task, seed, cfg, epochs):
    """One token-grid validation run: train delta on train, score
    the prototype readout on VAL (Convention A)."""
    tr, va, te, tr_c, va_c, te_c, K, decode = splits21(task, seed)
    net = train_token21(task, tr, cfg, "delta", seed, epochs)
    m = eval_net21(net, tr, tr_c, va, va_c, decode, task,
                   use_vec=False)
    return {"val_mse": m["mse"], "val_acc": m["acc"],
            "val_within1": m["within1"],
            "val_centroid_mse_10": m["centroid_mse_10"],
            "val_ridge_std_mse": m["ridge_std_mse"],
            "mean_burst": m["mean_burst"],
            "clamp_frac": m["clamp_frac"]}


def run_val_bump(task, seed, cfg, sigma, epochs):
    """One bump-sigma validation run: train bump delta on train,
    score the prototype readout on VAL (Convention A). cfg carries
    the token winner's alpha/eta_d; smooth_sigma is ignored for
    bump input (the net uses smooth_sigma=0.0)."""
    tr, va, te, tr_c, va_c, te_c, K, decode = splits21(task, seed)
    tr_xy, tr_v = bump_pairs(tr, tr_c, task, sigma)
    va_xy, va_v = bump_pairs(va, va_c, task, sigma)
    net = train_bump21(task, tr_xy, cfg, "delta", seed, epochs)
    m = eval_net21(net, tr_xy, tr_v, va_xy, va_v, decode, task,
                   use_vec=True)
    return {"val_mse": m["mse"], "val_acc": m["acc"],
            "val_within1": m["within1"],
            "val_centroid_mse_10": m["centroid_mse_10"],
            "val_ridge_std_mse": m["ridge_std_mse"],
            "mean_burst": m["mean_burst"],
            "clamp_frac": m["clamp_frac"]}


def run_test_arm(task, seed, cfg, input_type, bump_sigma, wire_on,
                 epochs, tag=None):
    """One test arm: train the delta net, build its frozen twin
    (same seed/init, never trained), evaluate both on TEST with
    all three readouts. Yields (config, metric dict) for the delta
    net then the frozen twin, in order, so the caller can write
    rows incrementally and a killed worker resumes per row.
    input_type in ("token", "bump"); bump_sigma is None for token.
    tag overrides the config name prefix (the token_primary arm
    needs distinct names from the token arm).
    """
    tr, va, te, tr_c, va_c, te_c, K, decode = splits21(task, seed)
    bump = (input_type == "bump")
    if bump:
        tr_xy, tr_v = bump_pairs(tr, tr_c, task, bump_sigma)
        te_xy, te_v = bump_pairs(te, te_c, task, bump_sigma)
        net = train_bump21(task, tr_xy, cfg, "delta", seed, epochs,
                           wire_on=wire_on)
        twin = train_bump21(task, tr_xy, cfg, "frozen", seed, 0,
                            wire_on=wire_on)
        m = eval_net21(net, tr_xy, tr_v, te_xy, te_v, decode, task,
                       use_vec=True)
        mf = eval_net21(twin, tr_xy, tr_v, te_xy, te_v, decode, task,
                        use_vec=True)
    else:
        net = train_token21(task, tr, cfg, "delta", seed, epochs,
                            wire_on=wire_on)
        twin = train_token21(task, tr, cfg, "frozen", seed, 0,
                             wire_on=wire_on)
        m = eval_net21(net, tr, tr_c, te, te_c, decode, task,
                       use_vec=False)
        mf = eval_net21(twin, tr, tr_c, te, te_c, decode, task,
                        use_vec=False)
    tag = tag or (input_type + ("_w" if wire_on else ""))
    yield f"{tag}_delta", m
    yield f"{tag}_frozen", mf


def load_winners(path):
    assert path, "this phase needs --winners"
    with open(path) as f:
        return json.load(f)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("phase"), r.get("config"),
                          r.get("seed")))
    return done


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task, phase = sys.argv[1], sys.argv[2]
    assert task in TASKS and phase in ("val", "test"), (task, phase)
    seeds, out_path, winners_path, epochs, val_kind = \
        None, None, None, EPOCHS, "token"
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
        elif sys.argv[i] == "--val-kind":
            val_kind = sys.argv[i + 1]
            assert val_kind in ("token", "bump"), val_kind
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            if phase == "val":
                if val_kind == "token":
                    for gname, cfg in GRID:
                        key = ("val", f"grid|{gname}", seed)
                        if key in done:
                            continue
                        r = run_val_token(task, seed, cfg, epochs)
                        rec = {"task": task, "phase": "val",
                               "config": key[1], "seed": seed,
                               "alpha": cfg["alpha"],
                               "smooth_sigma": cfg["smooth_sigma"],
                               "eta_d": cfg["eta_d"],
                               "bump_sigma": None, "epochs": epochs,
                               **r}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"{task} val {gname} seed={seed}: "
                              f"val_mse={r['val_mse']:.4f} "
                              f"val_acc={r['val_acc']:.3f}", flush=True)
                else:
                    w = load_winners(winners_path)
                    cfg = {"alpha": w[task]["alpha"],
                           "smooth_sigma": w[task]["smooth_sigma"],
                           "eta_d": w[task]["eta_d"]}
                    for sigma in BUMP_SIGMAS:
                        key = ("val", f"bumpsigma|{sigma}", seed)
                        if key in done:
                            continue
                        r = run_val_bump(task, seed, cfg, sigma,
                                         epochs)
                        rec = {"task": task, "phase": "val",
                               "config": key[1], "seed": seed,
                               "alpha": cfg["alpha"],
                               "smooth_sigma": cfg["smooth_sigma"],
                               "eta_d": cfg["eta_d"],
                               "bump_sigma": sigma, "epochs": epochs,
                               **r}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"{task} val bump sigma={sigma} "
                              f"seed={seed}: val_mse={r['val_mse']:.4f}",
                              flush=True)
            else:
                w = load_winners(winners_path)
                cfg = {"alpha": w[task]["alpha"],
                       "smooth_sigma": w[task]["smooth_sigma"],
                       "eta_d": w[task]["eta_d"]}
                bump_sigma = float(w[task]["bump_sigma"])
                arms = [("token", None, False),
                        ("token", None, True),
                        ("bump", bump_sigma, False)]
                # Conditional 4th arm: the preregistered primary,
                # only when the val winner differs from it (lorenz /
                # varsine; the new tasks have no separate primary).
                primary = w[task].get("primary")
                if primary is not None:
                    pcfg = {"alpha": primary["alpha"],
                            "smooth_sigma": primary["smooth_sigma"],
                            "eta_d": primary["eta_d"]}
                    if pcfg != cfg:
                        arms.append(("token_primary", None, False))
                for input_type, bsig, wire_on in arms:
                    acfg = ({"alpha": primary["alpha"],
                             "smooth_sigma": primary["smooth_sigma"],
                             "eta_d": primary["eta_d"]}
                            if input_type == "token_primary" else cfg)
                    atag = (input_type if input_type == "token_primary"
                            else None)
                    for cond, r in run_test_arm(
                            task, seed, acfg,
                            "token" if input_type != "bump" else "bump",
                            bsig, wire_on, epochs, tag=atag):
                        key = ("test", cond, seed)
                        if key in done:
                            continue
                        rec = {"task": task, "phase": "test",
                               "config": cond, "seed": seed,
                               "alpha": acfg["alpha"],
                               "smooth_sigma": acfg["smooth_sigma"],
                               "eta_d": acfg["eta_d"],
                               "bump_sigma": bsig,
                               "wire": ("aligned-1.0" if wire_on
                                        else "off"),
                               "epochs": epochs, **r}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"{task} test {cond} seed={seed}: "
                              f"acc={r['acc']:.3f} mse={r['mse']:.4f} "
                              f"ridge_std={r['ridge_std_mse']:.4f}",
                              flush=True)
                bkey = ("test", "baseline", seed)
                if bkey not in done:
                    tr, va, te, tr_c, va_c, te_c, K, decode = \
                        splits21(task, seed)
                    b = compute_baselines21(task, tr, te, tr_c,
                                            te_c, decode, seed)
                    rec = {"task": task, "phase": "test",
                           "config": "baseline", "seed": seed, **b}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test baseline seed={seed}: "
                          f"persist_mse={b['persist_mse']:.4f} "
                          f"markov2_mse={b['markov2_mse']:.4f} "
                          f"esn_scalar_std={b['esn_scalar_std_mse']:.4f}",
                          flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
