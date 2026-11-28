"""Round 17 driver: the top-down clamp (experiment B).

Protocol: PROTOCOL-R17-CLAMP.md (frozen Nov 26, 2026). Seeds
241..260 (fresh). Tasks: planted1, planted2. Winner config
from round 15 (alpha 1.0, smooth_sigma 0.0, eta_d 0.05) for
both planted tasks; asserted against r15/winners.json when
--winners is given.

Phases:
  val:  fb_gain grid {0.0, 0.1, 0.3, 1.0} on val seeds, one
        worker per gain, one file per gain. Reports val
        normal accuracy; the analyzer picks the winner with
        the 0.80x guard.
  test: conditions wire-on (tested on/cut/shuffle: one net,
        three test passes), wire-off, frozen. One JSON line
        per (task, condition, seed); a worker skips lines
        already present, so a killed worker resumes.

Usage:
  python rpc_round17_clamp.py <task> <val|test> --seeds A-B --out FILE [--winners FILE] [--epochs N] [--fb-gain G]
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
import torch.nn.functional as F

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import make_data, make_data_lag
from rpc_round16_omission import (K, WINNER, clamp_frac, cosines,
                                  detect_motifs, filler_pos,
                                  load_winners, shuffled_perm)
from rpc_v2 import RPCNetwork

TASKS = ["planted1", "planted2"]
# Test conditions: wire-on is tested three ways on ONE trained
# net (on/cut/shuffle); wire-off and frozen are separate nets.
CONDITIONS = ["wire-on", "wire-cut", "wire-shuffle", "wire-off",
              "frozen"]
FB_GAINS = [0.0, 0.1, 0.3, 1.0]
EPOCHS = 5
N_TRAIN = 200


def splits3(task, seed):
    """(train, val, test). Draw order matches rounds 15-16: train
    200, val 50, test 50, from one RNG stream per seed."""
    np.random.seed(seed)
    if task == "planted1":
        train = make_data(200)
        val = make_data(50)
        test = make_data(50)
    else:
        train = make_data_lag(200, lag=2)
        val = make_data_lag(50, lag=2)
        test = make_data_lag(50, lag=2)
    return train, val, test


def net_kw(cfg, fb_gain):
    kw = dict(K=K, n_res=200, n_out=256,
              theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
              proto_n=N_TRAIN, alpha=cfg["alpha"],
              smooth_sigma=cfg["smooth_sigma"], full_code=True,
              use_confirmation=False, s_inh=0.0, graded_conf=False,
              norm_rows=False, fb_gain=fb_gain)
    kw["learn_rule"] = "delta"
    kw["eta_d"] = cfg["eta_d"]
    return kw


def train_net(train, cfg, condition, seed, epochs, fb_gain):
    """Build, calibrate, prototype, and train a net; return
    (net, train_stats). train_stats: err_mag per epoch (mean
    ||p1 - d1||), r1_max over training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(cfg, fb_gain)
    net = RPCNetwork(**{k: v for k, v in kw.items()
                        if k not in ("use_confirmation", "proto_n")})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=kw["proto_n"])
    skw = dict(use_confirmation=False,
               plasticity=(condition != "frozen"))
    err_by_epoch = []
    r1_max = 0.0
    for _ in range(epochs):
        errs = []
        for seq in train:
            net.reset()
            for tok in seq:
                out = net.step(tok, train=True, **skw)
                errs.append(float((out["p1"] - out["d1"]).norm()))
                r1_max = max(r1_max, float(out["r1"].abs().max()))
        err_by_epoch.append(float(np.mean(errs)))
    return net, {"err_mag_by_epoch": err_by_epoch, "r1_max": r1_max}


def normal_acc(net, data, fb_mode="on"):
    """Standard next-token accuracy on data, wire as tested."""
    correct, total = 0, 0
    for seq in data:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False, use_confirmation=False,
                           fb_mode=fb_mode)
            if i > 0:
                total += 1
                correct += int(out["pred"] == tok)
    return correct / max(total, 1)


def paired_p1_cos(net, test):
    """Mean cosine between p1 with the wire on and p1 with the
    wire cut, on identical test sequences. Steps after the first
    only (at step 0 both have y2_prev=0, so d1=0 and fb=0)."""
    cos = []
    for seq in test:
        p1_on, p1_cut = [], []
        for mode, acc in (("on", p1_on), ("cut", p1_cut)):
            net.reset()
            for tok in seq:
                out = net.step(tok, train=False,
                               use_confirmation=False, fb_mode=mode)
                acc.append(out["p1"])
        for a, b in zip(p1_on[1:], p1_cut[1:]):
            cos.append(float(F.cosine_similarity(a, b, dim=0)))
    return float(np.mean(cos)) if cos else None


def run_omission_fb(task, train, test, cfg, condition, seed, epochs,
                    fb_gain, fb_mode, net=None, train_stats=None):
    """Train one net (unless given), then run the round-16 omission
    evaluation with the wire in fb_mode. Returns the metric dict
    plus the new clamp measures. The omission code path is the
    round-16 one, with fb_mode threaded into every step call."""
    lag = 1 if task == "planted1" else 2
    if net is None:
        net, train_stats = train_net(train, cfg, condition, seed,
                                     epochs, fb_gain)
    proto_norm = net._proto_norm
    _, inv_perm = shuffled_perm(seed)
    blank = torch.zeros(K)
    acc = {k: {"n": 0, "omit": [], "margin": [], "shuf": [],
               "t1": [], "t2": [], "cont": [], "d1n_t": [],
               "d1n_t1": [], "d1n_t2": [], "p1n_t": [], "cosp1": [],
               "errp": []}
           for k in ("m1", "m2", "filler")}
    dt_cont = 1 if lag == 1 else 2
    norm = {"m1": [], "m2": [], "filler": []}
    norm_all = []
    d1n_norm = {"t": [], "t1": [], "t2": []}
    p1n_norm = []
    bursts = []
    fb_ratios = []
    r1_max = train_stats["r1_max"] if train_stats else 0.0

    def step_call(tok, blank_step=False):
        if blank_step:
            return net._step_core(blank, train=False,
                                  use_confirmation=False,
                                  fb_mode=fb_mode)
        return net.step(tok, train=False, use_confirmation=False,
                        fb_mode=fb_mode)

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
            out = step_call(tok)
            pred_n.append(out["pred"])
            d1_n.append(out["d1"])
            p1_n.append(out["p1"])
            bursts.append(out["burst_rate"])
            fb_ratios.append(out["fb_rms_ratio"])
            r1_max = max(r1_max, float(out["r1"].abs().max()))
        for t, kind in events:
            norm[kind].append(int(pred_n[t] == seq[t]))
        # Overall normal accuracy over all test positions (for the
        # C5 guard: wire-on vs wire-off on the same seeds).
        for i, tok in enumerate(seq):
            if i > 0:
                norm_all.append(int(pred_n[i] == tok))
        for t, kind in events:
            d1n_norm["t"].append(float(d1_n[t].norm()))
            p1n_norm.append(float(p1_n[t].norm()))
        # Omission passes: one full pass per event, blank at t.
        for t, kind in events:
            omitted = seq[t]
            net.reset()
            outs = []
            for i, tok in enumerate(seq):
                outs.append(step_call(tok, blank_step=(i == t)))
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
                    a[key].append(int(outs[t + dt]["pred"] ==
                                      seq[t + dt]))
                    a["d1n_" + key].append(
                        float(outs[t + dt]["d1"].norm()))
                    d1n_norm[key].append(
                        float(d1_n[t + dt].norm()))
            if kind == "m1" and t + dt_cont < len(seq):
                a["cont"].append(int(outs[t + dt_cont]["pred"] ==
                                     seq[t + dt_cont]))
    out = {}
    for kind in ("m1", "m2", "filler"):
        a = acc[kind]
        n = a["n"]
        mean = lambda xs: float(np.mean(xs)) if xs else None
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
    out["normal_acc"] = float(np.mean(norm_all)) if norm_all else None
    out["fb_rms_ratio"] = float(np.mean(fb_ratios)) if fb_ratios \
        else 0.0
    out["r1_max"] = float(r1_max)
    if train_stats:
        out["err_mag_first"] = train_stats["err_mag_by_epoch"][0]
        out["err_mag_last"] = train_stats["err_mag_by_epoch"][-1]
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
                done.add((r.get("task"), r.get("config"),
                          r.get("seed"), r.get("fb_gain")))
    return done


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task = sys.argv[1]
    phase = sys.argv[2]
    assert task in TASKS and phase in ("val", "test"), (task, phase)
    seeds, out_path, winners_path, epochs, fb_gain = \
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
        elif sys.argv[i] == "--fb-gain":
            fb_gain = float(sys.argv[i + 1])
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    cfg = load_winners(winners_path)
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            train, val, test = splits3(task, seed)
            if phase == "val":
                assert fb_gain is not None, "val needs --fb-gain"
                key = (task, f"val-g{fb_gain}", seed, fb_gain)
                if key in done:
                    continue
                net, ts = train_net(train, cfg, "delta", seed,
                                    epochs, fb_gain)
                vacc = normal_acc(net, val, fb_mode="on")
                rec = {"task": task, "phase": "val",
                       "config": f"val-g{fb_gain}", "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs,
                       "fb_gain": fb_gain, "val_acc": vacc,
                       "err_mag_first": ts["err_mag_by_epoch"][0],
                       "err_mag_last": ts["err_mag_by_epoch"][-1],
                       "r1_max": ts["r1_max"]}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} val-g{fb_gain} seed={seed}: "
                      f"val_acc={vacc:.3f} err {rec['err_mag_first']:.3f}"
                      f"->{rec['err_mag_last']:.3f} "
                      f"r1_max={rec['r1_max']:.2f}", flush=True)
                continue
            # Test phase: fb_gain is the frozen winner; conditions
            # wire-on/cut/shuffle share one trained net.
            assert fb_gain is not None, "test needs --fb-gain"
            # 1-3. Wire-on net, tested on/cut/shuffle.
            net, ts = None, None
            p1_cos = None
            for cond, mode in (("wire-on", "on"), ("wire-cut", "cut"),
                               ("wire-shuffle", "shuffle")):
                if (task, cond, seed, fb_gain) in done:
                    continue
                if net is None:
                    net, ts = train_net(train, cfg, "delta", seed,
                                        epochs, fb_gain)
                    p1_cos = paired_p1_cos(net, test)
                r = run_omission_fb(task, train, test, cfg, "delta",
                                    seed, epochs, fb_gain, mode,
                                    net=net, train_stats=ts)
                r["p1_cos"] = p1_cos
                rec = {"task": task, "phase": "test",
                       "config": cond, "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs,
                       "fb_gain": fb_gain, **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} {cond} seed={seed}: "
                      f"omit_m2={r['omit_acc_m2']:.3f} "
                      f"omit_m1={r['omit_acc_m1']:.3f} "
                      f"cont_m1={r['omit_acc_cont_m1']:.3f} "
                      f"p1_cos={p1_cos:.3f} "
                      f"fb_rms={r['fb_rms_ratio']:.3f}", flush=True)
            # 4. Wire-off net (fb_gain=0).
            if (task, "wire-off", seed, 0.0) not in done:
                r = run_omission_fb(task, train, test, cfg, "delta",
                                    seed, epochs, 0.0, "on")
                r["p1_cos"] = None
                rec = {"task": task, "phase": "test",
                       "config": "wire-off", "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs,
                       "fb_gain": 0.0, **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} wire-off seed={seed}: "
                      f"omit_m2={r['omit_acc_m2']:.3f} "
                      f"omit_m1={r['omit_acc_m1']:.3f} "
                      f"cont_m1={r['omit_acc_cont_m1']:.3f}", flush=True)
            # 5. Frozen net, wire on.
            if (task, "frozen", seed, fb_gain) not in done:
                r = run_omission_fb(task, train, test, cfg, "frozen",
                                    seed, epochs, fb_gain, "on")
                r["p1_cos"] = None
                rec = {"task": task, "phase": "test",
                       "config": "frozen", "seed": seed,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs,
                       "fb_gain": fb_gain, **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} frozen seed={seed}: "
                      f"omit_m2={r['omit_acc_m2']:.3f} "
                      f"omit_m1={r['omit_acc_m1']:.3f}", flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
