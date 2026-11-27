"""Round 16 driver: the omission test with frozen weights.

Protocol: PROTOCOL-R16-OMISSION.md (frozen Nov 26, 2026). Seeds
221..240 (fresh). Tasks: planted1, planted2. Conditions: delta,
frozen. Winner config from round 15 (alpha 1.0, smooth_sigma 0.0,
eta_d 0.05) for both planted tasks; asserted against
r15/winners.json when --winners is given.

One JSON line per (task, condition, seed) in the output file; a
worker skips lines already present, so a killed worker resumes.

Usage:
  python rpc_round16_omission.py <task> --seeds A-B --out FILE [--winners FILE] [--epochs N]
"""
import json
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
import torch.nn.functional as F

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import make_data, make_data_lag
from rpc_v2 import RPCNetwork

TASKS = ["planted1", "planted2"]
CONDITIONS = ["delta", "frozen"]
MOTIFS = ((1, 2, 3), (5, 6, 7), (9, 10, 11))
MOTIF_SET = set(MOTIFS)
MOTIF_TOKENS = {2, 3, 6, 7, 10, 11}
K = 16
# Winner config, round 15 (r15/winners.json, verified Nov 26 20:44
# PT): planted1 and planted2 both picked alpha 1.0, smooth_sigma
# 0.0, eta_d 0.05.
WINNER = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.05}
EPOCHS = 5
N_TRAIN = 200


def load_winners(path):
    """Cross-check the hardcoded winner against r15/winners.json."""
    if not path:
        return dict(WINNER)
    with open(path) as f:
        w = json.load(f)
    for task in TASKS:
        cfg = w[task]
        got = {"alpha": cfg["alpha"], "smooth_sigma": cfg["smooth_sigma"],
               "eta_d": cfg["eta_d"]}
        assert got == WINNER, (
            f"winners.json {task} = {got}, protocol says {WINNER}; "
            "refusing to run with the wrong config")
    return dict(WINNER)


def splits(task, seed):
    """(train, test). Draw order matches round 15: train 200, val
    50 (drawn, unused), test 50, from one RNG stream per seed."""
    np.random.seed(seed)
    if task == "planted1":
        train = make_data(200)
        make_data(50)
        test = make_data(50)
    else:
        train = make_data_lag(200, lag=2)
        make_data_lag(50, lag=2)
        test = make_data_lag(50, lag=2)
    return train, test


def detect_motifs(seq, lag):
    """(m1_positions, m2_positions, motif_positions) in the original
    sequence. lag=1: motif at (i, i+1, i+2). lag=2: motif at
    (i, i+2, i+4). A position that is both m1 and m2 for different
    detected motifs is ambiguous and dropped from both lists."""
    m1, m2, pos = set(), set(), set()
    L = len(seq)
    if lag == 1:
        for i in range(L - 2):
            if (seq[i], seq[i + 1], seq[i + 2]) in MOTIF_SET:
                m1.add(i + 1)
                m2.add(i + 2)
                pos.update((i, i + 1, i + 2))
    else:
        for i in range(L - 4):
            if (seq[i], seq[i + 2], seq[i + 4]) in MOTIF_SET:
                m1.add(i + 2)
                m2.add(i + 4)
                pos.update((i, i + 2, i + 4))
    amb = m1 & m2
    return sorted(m1 - amb), sorted(m2 - amb), pos


def filler_pos(seq, motif_positions):
    """First index >= 2 that is not a motif position (cue, m1, m2)."""
    for i in range(2, len(seq)):
        if i not in motif_positions:
            return i
    return None


def shuffled_perm(seed):
    """Fixed per-seed label permutation with no motif token as a
    fixed point. The shuffled prediction is inv_perm[pred]."""
    rng = np.random.default_rng(seed + 777)
    while True:
        perm = rng.permutation(K).tolist()
        if all(perm[t] != t for t in MOTIF_TOKENS):
            break
    inv = [0] * K
    for k, v in enumerate(perm):
        inv[v] = k
    return perm, inv


def net_kw(cfg, condition):
    kw = dict(K=K, n_res=200, n_out=256,
              theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
              proto_n=N_TRAIN, alpha=cfg["alpha"],
              smooth_sigma=cfg["smooth_sigma"], full_code=True,
              use_confirmation=False, s_inh=0.0, graded_conf=False,
              norm_rows=False)
    kw["learn_rule"] = "delta"
    kw["eta_d"] = cfg["eta_d"]
    return kw


def train_net(train, cfg, condition, seed, epochs):
    """Build, calibrate, prototype, and train a net; return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(cfg, condition)
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=kw["proto_n"])
    skw = dict(use_confirmation=False,
               plasticity=(condition != "frozen"))
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **skw)
    return net


def clamp_frac(net):
    return float((net.W_dist1.abs() >= net.w_max - 1e-6).float().mean())


def cosines(net, d1):
    """Cosine of d1 against every unit prototype, shape (K,)."""
    with torch.no_grad():
        a = F.normalize(d1[None, :], dim=1)
        return (a @ net._proto_norm.T)[0]


def run_omission(task, train, test, cfg, condition, seed, epochs):
    """Train one net, then run the omission passes on the test set.

    Returns the aggregated metric dict for this (task, condition,
    seed). One pass per omission event; the blank is fed only at
    the omitted position, all other tokens are presented normally.
    """
    lag = 1 if task == "planted1" else 2
    net = train_net(train, cfg, condition, seed, epochs)
    proto_norm = net._proto_norm
    _, inv_perm = shuffled_perm(seed)
    blank = torch.zeros(K)

    # Accumulators: per event type (m1, m2, filler), lists of
    # per-event values. "cont" is the generative continuation for
    # m1 events only: the decode of the motif completion c at the
    # step where it occurs (t+1 for lag-1, t+2 for lag-2, because
    # lag-2 has a filler between b and c). The t1/t2 measures
    # count only targets that are NOT motif positions, so they are
    # clean filler-target controls; a t+1/t+2 target that is
    # another motif's element would be predictable and would
    # contaminate the control.
    acc = {k: {"n": 0, "omit": [], "margin": [], "shuf": [],
               "t1": [], "t2": [], "cont": [], "d1n_t": [],
               "d1n_t1": [], "d1n_t2": [], "p1n_t": [], "cosp1": [],
               "errp": []}
           for k in ("m1", "m2", "filler")}
    dt_cont = 1 if lag == 1 else 2
    norm = {"m1": [], "m2": [], "filler": []}
    d1n_norm = {"t": [], "t1": [], "t2": []}
    p1n_norm = []
    bursts = []

    for seq in test:
        m1s, m2s, mpos = detect_motifs(seq, lag)
        fp = filler_pos(seq, mpos)
        events = ([(t, "m1") for t in m1s] +
                  [(t, "m2") for t in m2s] +
                  ([(fp, "filler")] if fp is not None else []))
        # Normal pass: reference predictions and norms.
        net.reset()
        pred_n, d1_n, p1_n = [], [], []
        for tok in seq:
            out = net.step(tok, train=False, use_confirmation=False)
            pred_n.append(out["pred"])
            d1_n.append(out["d1"])
            p1_n.append(out["p1"])
            bursts.append(out["burst_rate"])
        for t, kind in events:
            norm[kind].append(int(pred_n[t] == seq[t]))
        # d1/p1 norm reference at the event positions.
        for t, kind in events:
            d1n_norm["t"].append(float(d1_n[t].norm()))
            p1n_norm.append(float(p1_n[t].norm()))
        # Omission passes: one full pass per event, blank at t.
        for t, kind in events:
            omitted = seq[t]
            net.reset()
            outs = []
            for i, tok in enumerate(seq):
                if i == t:
                    outs.append(net._step_core(blank, train=False,
                                               use_confirmation=False))
                else:
                    outs.append(net.step(tok, train=False,
                                         use_confirmation=False))
            o = outs[t]
            d1t, p1t = o["d1"], o["p1"]
            cos = cosines(net, d1t)
            margin = float(cos[omitted] -
                           cos[[k for k in range(K) if k != omitted]].mean())
            a = acc[kind]
            a["n"] += 1
            a["omit"].append(int(o["pred"] == omitted))
            a["margin"].append(margin)
            a["shuf"].append(int(inv_perm[o["pred"]] == omitted))
            a["d1n_t"].append(float(d1t.norm()))
            a["p1n_t"].append(float(p1t.norm()))
            a["cosp1"].append(float(
                F.cosine_similarity(p1t, proto_norm[omitted], dim=0)))
            err = p1t - d1t
            a["errp"].append(float(err @ proto_norm[omitted]))
            for dt, key in ((1, "t1"), (2, "t2")):
                if t + dt < len(seq) and (t + dt) not in mpos:
                    # Filler target only: a motif-position target
                    # would be predictable and contaminate the
                    # control. (For m1 events the continuation
                    # position is a motif position by construction
                    # and is counted in "cont", not here.)
                    a[key].append(int(outs[t + dt]["pred"] ==
                                      seq[t + dt]))
                    a["d1n_" + key].append(
                        float(outs[t + dt]["d1"].norm()))
                    d1n_norm[key].append(
                        float(d1_n[t + dt].norm()))
            if kind == "m1" and t + dt_cont < len(seq):
                # Generative continuation: decode the motif
                # completion c at the step where it occurs. The
                # drive d1(t+dt_cont) is formed from the post-blank
                # state before c is fed in, so this is a genuine
                # prediction from the context (a, blank[, filler]),
                # not leakage.
                a["cont"].append(int(outs[t + dt_cont]["pred"] ==
                                     seq[t + dt_cont]))
    out = {}
    for kind in ("m1", "m2", "filler"):
        a = acc[kind]
        n = a["n"]
        mean = lambda xs: float(np.mean(xs)) if xs else None
        # None means "structurally empty": e.g. omit_acc_t1_m1 is
        # empty on lag-1 because t+1 after an m1 omission is always
        # the continuation position (counted in "cont"), and
        # omit_acc_cont is empty for m2/filler because only m1
        # events have a continuation. The analyzer skips None.
        out[f"n_{kind}"] = n
        out[f"omit_acc_{kind}"] = mean(a["omit"])
        out[f"margin_{kind}"] = mean(a["margin"])
        out[f"omit_acc_shuf_{kind}"] = mean(a["shuf"])
        out[f"omit_acc_t1_{kind}"] = mean(a["t1"])
        out[f"omit_acc_t2_{kind}"] = mean(a["t2"])
        out[f"omit_acc_cont_{kind}"] = mean(a["cont"])
        out[f"n_t1_{kind}"] = len(a["t1"])
        out[f"n_t2_{kind}"] = len(a["t2"])
        out[f"n_cont_{kind}"] = len(a["cont"])
        out[f"d1_norm_omit_{kind}"] = mean(a["d1n_t"])
        out[f"d1_norm_t1_{kind}"] = mean(a["d1n_t1"])
        out[f"d1_norm_t2_{kind}"] = mean(a["d1n_t2"])
        out[f"p1_norm_omit_{kind}"] = mean(a["p1n_t"])
        out[f"cos_p1_omit_{kind}"] = mean(a["cosp1"])
        out[f"err_proj_{kind}"] = mean(a["errp"])
        out[f"normal_acc_{kind}"] = mean(norm[kind])
    out["d1_norm_normal_t"] = float(np.mean(d1n_norm["t"])) \
        if d1n_norm["t"] else None
    out["d1_norm_normal_t1"] = float(np.mean(d1n_norm["t1"])) \
        if d1n_norm["t1"] else None
    out["d1_norm_normal_t2"] = float(np.mean(d1n_norm["t2"])) \
        if d1n_norm["t2"] else None
    out["p1_norm_normal"] = float(np.mean(p1n_norm)) if p1n_norm \
        else None
    out["mean_burst"] = float(np.mean(bursts)) if bursts else 0.0
    out["clamp_frac"] = clamp_frac(net)
    return out


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("task"), r.get("config"), r.get("seed")))
    return done


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task = sys.argv[1]
    assert task in TASKS, task
    seeds, out_path, winners_path, epochs = None, None, None, EPOCHS
    i = 2
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
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    cfg = load_winners(winners_path)
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            train, test = splits(task, seed)
            for cond in CONDITIONS:
                if (task, cond, seed) in done:
                    continue
                r = run_omission(task, train, test, cfg, cond, seed,
                                 epochs)
                rec = {"task": task, "phase": "test",
                       "config": cond, "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs, **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} {cond} seed={seed}: "
                      f"omit_m2={r['omit_acc_m2']:.3f} "
                      f"omit_m1={r['omit_acc_m1']:.3f} "
                      f"omit_filler={r['omit_acc_filler']:.3f} "
                      f"cont_m1={r['omit_acc_cont_m1']:.3f} "
                      f"normal_m2={r['normal_acc_m2']:.3f} "
                      f"n_m2={r['n_m2']} n_m1={r['n_m1']}",
                      flush=True)
    print(f"done {task} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
