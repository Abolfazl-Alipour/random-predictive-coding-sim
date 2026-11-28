"""Round 20 driver: aligned-feedback clamp (revised experiment B)
+ multi-step omission rollout (experiment D).

Protocol: PROTOCOL-R20-ALIGNED.md (frozen Nov 27, 2026). Seeds
302..321 (fresh). Tasks: planted1-occ, planted2-occ, sine.
Wire types: random (A26 control), aligned (A29 fixed -W_prox.T),
learned (A29, starts aligned, NLMS update). Winner config from
round 15 (alpha 1.0, smooth_sigma 0.0, eta_d 0.05 planted;
eta_d 0.2 sine).

Occlusion (planted tasks only): 40% of filler positions are fed
as the blank token (A29, token K) in train AND test; motif
positions are never occluded, and the filler probe position
(first non-motif index >= 2) is never occluded, so the round-16
omission measures keep their meaning. Calibration and
prototypes use the CLEAN train data; the training loop uses the
occluded train data. The occlusion mask is deterministic per
(task, seed, split), so every condition sees identical data.

Phases:
  val:  wire_type x fb_gain grid {0.1, 0.3, 1.0} on planted1-occ
        val seeds, one worker per (wire, gain), one file per
        (wire, gain). Reports val normal acc, val omit_m2
        on/cut/shuffle gaps, val p1_cos, val r1_max. The monitor
        picks one gain per wire type (largest val on-cut gap
        among gains passing the p1_cos < 0.90 state gate).
  test: 5 nets per (task, seed): aligned, learned, random
        (at the ALIGNED winner's gain, matched control),
        wire-off, frozen-aligned. Aligned/learned/random are
        tested on/cut/shuffle on one trained net each. One JSON
        line per (task, condition, seed); a worker skips lines
        already present, so a killed worker resumes.

Usage:
  python rpc_round20_aligned.py <task> <val|test> --seeds A-B --out FILE [--winners FILE] [--epochs N] [--wire-type W] [--fb-gain G]
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
                                  detect_motifs, filler_pos)
from rpc_round17_clamp import (net_kw, normal_acc, paired_p1_cos,
                               run_omission_fb, splits3)
from rpc_simple import make_sine_fixed
from rpc_v2 import RPCNetwork

TASKS = ["planted1-occ", "planted2-occ", "sine"]
WIRE_TYPES = ["random", "aligned", "learned"]
FB_GAINS = [0.1, 0.3, 1.0]
EPOCHS = 5
N_TRAIN = 200
K_SINE = 32
OCCL_RATE = 0.4
SINE_CFG = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.2}
# Rollout: sine cue 5 tokens + 15 blanks; planted lag-1 blanks
# the 2 motif steps, lag-2 blanks 4 steps (m1 at +2, m2 at +4).
ROLLOUT_SINE_CUE = 5
ROLLOUT_SINE_H = 15


def occlude(seqs, lag, seed, split_code):
    """Blank 40% of filler positions (token K) in copies of seqs.

    Motif positions are never occluded; the filler probe
    position (first non-motif index >= 2) is never occluded.
    Deterministic per (seed, split_code): identical masks for
    every condition on the same (task, seed)."""
    rng = np.random.default_rng(1_000_003 * seed + split_code)
    out = []
    for seq in seqs:
        _, _, mpos = detect_motifs(seq, lag)
        fp = filler_pos(seq, mpos)
        s = list(seq)
        for i in range(len(s)):
            if i in mpos or i == fp:
                continue
            if rng.random() < OCCL_RATE:
                s[i] = K  # A29 blank token.
        out.append(s)
    return out


def splits(task, seed):
    """(train_clean, val_clean, test_clean,
    train_occ, val_occ, test_occ). For sine the occ copies are
    the clean sequences (no occlusion: sine has no motif/filler
    structure)."""
    if task == "sine":
        tr = make_sine_fixed(50, seed=seed)
        va = make_sine_fixed(20, seed=seed + 2000)
        te = make_sine_fixed(20, seed=seed + 1000)
        return tr, va, te, tr, va, te
    lag = 1 if task == "planted1-occ" else 2
    np.random.seed(seed)
    gen = make_data if lag == 1 else (lambda n: make_data_lag(n, lag=2))
    tr, va, te = gen(200), gen(50), gen(50)
    return (tr, va, te,
            occlude(tr, lag, seed, 0), occlude(va, lag, seed, 1),
            occlude(te, lag, seed, 2))


def cfg_for(task):
    return dict(SINE_CFG) if task == "sine" else dict(WINNER)


def k_for(task):
    return K_SINE if task == "sine" else K


def train_net(train_clean, train_occ, cfg, wire_type, fb_gain, seed,
              epochs, condition, k):
    """Build, calibrate on CLEAN train, prototype on CLEAN train,
    train on the OCCLUDED train (sine: clean == occ). Returns
    (net, train_stats). condition "frozen" trains with
    plasticity off."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(cfg, fb_gain)
    kw["K"] = k
    kw["fb_wire"] = wire_type
    net = RPCNetwork(**{kk: vv for kk, vv in kw.items()
                        if kk not in ("use_confirmation", "proto_n")})
    net.calibrate(train_clean)
    net.build_prototypes(train_clean, proto_n=N_TRAIN)
    skw = dict(use_confirmation=False,
               plasticity=(condition != "frozen"))
    err_by_epoch = []
    r1_max = 0.0
    for _ in range(epochs):
        errs = []
        for seq in train_occ:
            net.reset()
            for tok in seq:
                out = net.step(tok, train=True, **skw)
                errs.append(float((out["p1"] - out["d1"]).norm()))
                r1_max = max(r1_max, float(out["r1"].abs().max()))
        err_by_epoch.append(float(np.mean(errs)))
    return net, {"err_mag_by_epoch": err_by_epoch, "r1_max": r1_max}


def drive_ratio_pass(net, test, fb_mode):
    """One normal pass; mean fb_drive_ratio (A29, total drive)
    and fb_rms_ratio (A26, bottom-up only)."""
    ratios, ratios_bu = [], []
    for seq in test:
        net.reset()
        for tok in seq:
            out = net.step(tok, train=False, use_confirmation=False,
                           fb_mode=fb_mode)
            ratios.append(out["fb_drive_ratio"])
            ratios_bu.append(out["fb_rms_ratio"])
    return (float(np.mean(ratios)) if ratios else 0.0,
            float(np.mean(ratios_bu)) if ratios_bu else 0.0)


def fb_align_stats(net):
    """Frobenius cosine between W_fb and -W_prox.T (1.0 = exactly
    the aligned map), W_fb norm, and the fraction of W_fb entries
    at the w_max clamp."""
    a = net.W_fb.flatten().float()
    b = (-net.mod1.W_prox.T).flatten().float()
    cos = float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-9))
    return {"fb_align": cos,
            "fb_norm": float(net.W_fb.norm()),
            "fb_clamp_frac": float(
                (net.W_fb.abs() >= net.w_max - 1e-6).float().mean())}


def rollout_sine(net, test, fb_mode, cue=ROLLOUT_SINE_CUE,
                 H=ROLLOUT_SINE_H):
    """Closed-loop generation on sine: cue steps fed normally
    (fb on), then H blank steps with fb in fb_mode. The input
    stays blank; only the feedback loop drives generation.
    Scores pred(t) against the true token at each blank step.
    Returns acc over all blank steps, acc over the last 5, and
    MSE on bin centres (32 bins over [-1, 1])."""
    blank = torch.zeros(K_SINE)
    accs, accs_late, mses = [], [], []
    n = 0
    for seq in test:
        if len(seq) < cue + H:
            continue
        net.reset()
        for tok in seq[:cue]:
            net.step(tok, train=False, use_confirmation=False,
                     fb_mode="on")
        for j, tok in enumerate(seq[cue:cue + H]):
            out = net._step_core(blank, train=False,
                                 use_confirmation=False,
                                 fb_mode=fb_mode)
            accs.append(int(out["pred"] == tok))
            if j >= H - 5:
                accs_late.append(int(out["pred"] == tok))
            # Bin centres: bin b covers [-1 + 2b/32, -1 + 2(b+1)/32).
            pc = -1.0 + (out["pred"] + 0.5) * 2.0 / K_SINE
            tc = -1.0 + (tok + 0.5) * 2.0 / K_SINE
            mses.append(float((pc - tc) ** 2))
        n += 1
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"rollout_acc": mean(accs),
            "rollout_acc_late": mean(accs_late),
            "rollout_mse": mean(mses), "rollout_n": n}


def rollout_planted(net, test, fb_mode, lag):
    """Closed-loop motif completion on planted: cue = tokens up to
    and including the motif starter, then blank the motif steps
    (lag-1: t+1, t+2; lag-2: t+1..t+4, scoring m1 at t+2 and m2
    at t+4). One rollout per sequence (first detected motif).
    The cue is always fb-on; fb_mode applies to blank steps."""
    blank = torch.zeros(K)
    comp, m1s, m2s = [], [], []
    n = 0
    for seq in test:
        m1, m2, _ = detect_motifs(seq, lag)
        # First matched motif triple: lag-1 (t, t+1, t+2), lag-2
        # (t, t+2, t+4). m1[0]/m2[0] alone can belong to different
        # motifs when an ambiguous position was dropped, so scan
        # for a triple whose m1 and m2 both survive detection.
        triple = None
        if lag == 1:
            m2set = set(m2)
            for p in m1:
                if p + 1 in m2set:
                    triple = (p - 1, 1, 2)
                    break
        else:
            m1set = set(m1)
            for p in m2:
                if p - 2 in m1set:
                    triple = (p - 4, 2, 4)
                    break
        if triple is None:
            continue
        t, dt1, dt2 = triple
        if t < 0 or t + dt2 >= len(seq):
            continue
        net.reset()
        for tok in seq[:t + 1]:
            net.step(tok, train=False, use_confirmation=False,
                     fb_mode="on")
        preds = {}
        for j in range(1, dt2 + 1):
            out = net._step_core(blank, train=False,
                                 use_confirmation=False,
                                 fb_mode=fb_mode)
            preds[t + j] = out["pred"]
        ok1 = int(preds[t + dt1] == seq[t + dt1])
        ok2 = int(preds[t + dt2] == seq[t + dt2])
        m1s.append(ok1)
        m2s.append(ok2)
        comp.append(int(ok1 and ok2))
        n += 1
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"rollout_comp": mean(comp),
            "rollout_m1": mean(m1s), "rollout_m2": mean(m2s),
            "rollout_n": n}


def eval_condition(net, ts, task, test_clean, test_occ, cfg, seed,
                   fb_gain, fb_mode):
    """Full evaluation of one trained net in one fb_mode:
    round-17 omission measures (planted tasks; the lag is set by
    passing the base task name), drive ratios, fb alignment,
    and the rollout. Returns the metric dict (no task/config/
    seed keys; the caller adds them)."""
    out = {}
    if task == "sine":
        out["normal_acc"] = normal_acc(net, test_occ, fb_mode=fb_mode)
        for f in ("omit_acc_m2", "omit_acc_m1", "omit_acc_filler",
                  "omit_acc_cont_m1", "margin_m2", "omit_acc_shuf_m2",
                  "cos_p1_omit_m2", "p1_norm_omit_m2",
                  "d1_norm_omit_m2", "err_proj_m2"):
            out[f] = None
    else:
        base = "planted1" if task == "planted1-occ" else "planted2"
        # train is unused when net is given; pass the occluded
        # test set for both so no clean sequence leaks in.
        r = run_omission_fb(base, test_occ, test_occ, cfg, "delta",
                            seed, EPOCHS, fb_gain, fb_mode,
                            net=net, train_stats=ts)
        out.update(r)
    dr, dr_bu = drive_ratio_pass(net, test_occ, fb_mode)
    out["fb_drive_ratio"] = dr
    out["fb_rms_ratio_test"] = dr_bu
    out.update(fb_align_stats(net))
    if task == "sine":
        out.update(rollout_sine(net, test_occ, fb_mode))
    else:
        lag = 1 if task == "planted1-occ" else 2
        out.update(rollout_planted(net, test_occ, fb_mode, lag))
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
                          r.get("seed"), r.get("wire_type"),
                          r.get("fb_gain")))
    return done


def write_line(f, rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task = sys.argv[1]
    phase = sys.argv[2]
    assert task in TASKS and phase in ("val", "test"), (task, phase)
    seeds, out_path, winners_path, epochs = None, None, None, EPOCHS
    wire_type, fb_gain = None, None
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
        elif sys.argv[i] == "--wire-type":
            wire_type = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--fb-gain":
            fb_gain = float(sys.argv[i + 1])
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    cfg = cfg_for(task)
    k = k_for(task)
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            tr_c, va_c, te_c, tr_o, va_o, te_o = splits(task, seed)
            if phase == "val":
                assert task == "planted1-occ", "val is planted1-occ only"
                assert wire_type in WIRE_TYPES and fb_gain is not None
                key = (task, f"val-{wire_type}-g{fb_gain}", seed,
                       wire_type, fb_gain)
                if key in done:
                    continue
                net, ts = train_net(tr_c, tr_o, cfg, wire_type,
                                    fb_gain, seed, epochs, "delta", k)
                # Validation measures on the val set: normal acc
                # plus the omission on/cut/shuffle gaps and p1_cos.
                vacc = normal_acc(net, va_o, fb_mode="on")
                p1_cos = paired_p1_cos(net, va_o)
                gaps = {}
                for mode in ("on", "cut", "shuffle"):
                    r = run_omission_fb("planted1", va_o, va_o, cfg,
                                        "delta", seed, epochs, fb_gain,
                                        mode, net=net, train_stats=ts)
                    gaps[mode] = r["omit_acc_m2"]
                rec = {"task": task, "phase": "val",
                       "config": f"val-{wire_type}-g{fb_gain}",
                       "seed": seed, "wire_type": wire_type,
                       "fb_gain": fb_gain,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"], "epochs": epochs,
                       "val_acc": vacc, "val_p1_cos": p1_cos,
                       "val_omit_m2_on": gaps["on"],
                       "val_omit_m2_cut": gaps["cut"],
                       "val_omit_m2_shuffle": gaps["shuffle"],
                       "val_gap_on_cut": gaps["on"] - gaps["cut"],
                       "err_mag_first": ts["err_mag_by_epoch"][0],
                       "err_mag_last": ts["err_mag_by_epoch"][-1],
                       "r1_max": ts["r1_max"]}
                write_line(f, rec)
                print(f"{task} val-{wire_type}-g{fb_gain} seed={seed}: "
                      f"val_acc={vacc:.3f} gap={rec['val_gap_on_cut']:+.3f} "
                      f"p1_cos={p1_cos:.3f} "
                      f"err {rec['err_mag_first']:.3f}"
                      f"->{rec['err_mag_last']:.3f}", flush=True)
                continue
            # Test phase: winners.json carries one gain per wire
            # type; the random wire runs at the ALIGNED gain
            # (matched control).
            assert winners_path, "test needs --winners"
            with open(winners_path) as wf:
                w = json.load(wf)
            gains = {wt: w[wt]["fb_gain"] for wt in WIRE_TYPES}
            # (wire_type, gain, condition, modes): one net each for
            # aligned/learned/random/off/frozen; on/cut/shuffle reuse
            # the trained net.
            plan = [(("aligned", gains["aligned"], "delta"),
                     ("on", "cut", "shuffle")),
                    (("learned", gains["learned"], "delta"),
                     ("on", "cut", "shuffle")),
                    (("random", gains["aligned"], "delta"),
                     ("on", "cut", "shuffle")),
                    (("aligned", 0.0, "delta"), ("on",)),
                    (("aligned", gains["aligned"], "frozen"), ("on",))]
            cond_names = {("aligned", "on"): "aligned-on",
                          ("aligned", "cut"): "aligned-cut",
                          ("aligned", "shuffle"): "aligned-shuffle",
                          ("learned", "on"): "learned-on",
                          ("learned", "cut"): "learned-cut",
                          ("learned", "shuffle"): "learned-shuffle",
                          ("random", "on"): "random-on",
                          ("random", "cut"): "random-cut",
                          ("random", "shuffle"): "random-shuffle"}
            for (wt, g, cond), modes in plan:
                net, ts = None, None
                p1_cos = None
                for mode in modes:
                    if (wt, g) == ("aligned", 0.0):
                        cname = "wire-off"
                    elif cond == "frozen":
                        cname = "frozen-aligned"
                    else:
                        cname = cond_names[(wt, mode)]
                    key = (task, cname, seed, wt, g)
                    if key in done:
                        continue
                    if net is None:
                        net, ts = train_net(tr_c, tr_o, cfg, wt, g,
                                            seed, epochs, cond, k)
                        p1_cos = (paired_p1_cos(net, te_o)
                                  if g > 0 else None)
                    r = eval_condition(net, ts, task, te_c, te_o,
                                       cfg, seed, g, mode)
                    r["p1_cos"] = p1_cos
                    rec = {"task": task, "phase": "test",
                           "config": cname, "seed": seed,
                           "wire_type": wt, "fb_gain": g,
                           "alpha": cfg["alpha"],
                           "smooth_sigma": cfg["smooth_sigma"],
                           "eta_d": cfg["eta_d"], "epochs": epochs,
                           **r}
                    write_line(f, rec)
                    print(f"{task} {cname} seed={seed}: "
                          f"normal_acc={r.get('normal_acc')}"
                          + (f" omit_m2={r.get('omit_acc_m2')}"
                             if r.get("omit_acc_m2") is not None
                             else "")
                          + f" rollout_acc={r.get('rollout_acc')} "
                            f"rollout_comp={r.get('rollout_comp')}",
                          flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
