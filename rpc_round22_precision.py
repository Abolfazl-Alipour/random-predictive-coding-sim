"""Round 22 driver: the PRECISION test (experiment C).

Design (PROTOCOL-R22-PRECISION.md, frozen Nov 27, 2026):
precision is operationalized as test-time input reliability.
At the probe step only, the encoded probe token is corrupted
with additive Gaussian noise, x = normalize(encode(tok) + c*z),
z ~ N(0, I), fed through step_vec. High precision is c = 0;
low precision is c = c_low, picked per task in the noise
validation phase by a decode rule. Training is always clean,
so the learned prediction is identical across precision
conditions; only the sensory evidence at the probe differs.

The core measurement is the expectation-by-precision
interaction on the stimulus-specific error response
e = p1 - d1, decoded per motif, with the wire cut and
shuffled as controls. Wire: aligned at gain 1.0 is the live
wire; random at gain 1.0 is the negative control; wire-off
is a separate gain-0.0 net; frozen-aligned is the learning
control. Four nets per (task, seed); 8 config lines per
(task, seed).

Tasks: planted lag-1, planted lag-2, sine. Round-15 winner
configs, 5 epochs. Val seeds 347-351; test seeds 352-371.

Noise determinism: every noise vector is drawn from
np.random.default_rng with an integer seed computed by
arithmetic from (seed, trial-type code, trial index, c).
Never use Python hash() for a seed: string hashing is
randomized per process (found in the design probe: the
noisy-condition numbers varied between identical runs).
The same noise vector is reused across wire modes for the
same trial.

One BLAS thread per process (round-15+ rule).
Resumable: rerun the same command; finished
(task, config, seed, c_low) keys are skipped.
"""
import argparse
import json
import os
import sys

# One BLAS thread per process, set BEFORE importing numpy/torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import make_data, make_data_lag
from rpc_round16_omission import K, MOTIFS, WINNER
from rpc_round17_clamp import net_kw, normal_acc, paired_p1_cos
from rpc_round20_aligned import fb_align_stats
from rpc_simple import make_sine_fixed
from rpc_v2 import RPCNetwork

TASKS = ["planted1", "planted2", "sine"]
SINE_CFG = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.2}
K_SINE = 32
N_TRAIN = 200
EPOCHS = 5
# Noise levels scanned in the noise validation phase.
NOISE_GRID = [0.10, 0.15, 0.25, 0.35, 0.50]
N_PREFIX = 20
# Non-motif filler tokens (motifs are (1,2,3), (5,6,7), (9,10,11)).
FILLER_TOKENS = [0, 4, 8, 12, 13, 14, 15]
NON_MOTIF = [t for t in range(16) if t not in (1, 2, 3, 5, 6, 7,
                                               9, 10, 11)]
TCODE = {"EXP": 1, "VIO": 2, "UNEXP": 3, "CTL": 4}
TRIALS = ["exp", "vio", "unexp", "ctl"]
METRICS = ["err", "cePres", "cePred", "decE", "decP1", "p1n", "fbdr"]
# Sine VIO probe: true token + 8 bins (mod 32).
SINE_VIO_SHIFT = 8
SINE_CUE = 50


def cfg_for(task):
    return dict(SINE_CFG) if task == "sine" else dict(WINNER)


def k_for(task):
    return K_SINE if task == "sine" else K


def splits(task, seed):
    """(train, val, test). Planted: same draw order as rounds
    15-17 (train 200, val 50, test 50 from one RNG stream).
    Sine: same as round 20."""
    if task == "sine":
        return (make_sine_fixed(50, seed=seed),
                make_sine_fixed(20, seed=seed + 2000),
                make_sine_fixed(20, seed=seed + 1000))
    np.random.seed(seed)
    gen = make_data if task == "planted1" else (
        lambda n: make_data_lag(n, lag=2))
    return gen(200), gen(50), gen(50)


def train_net(train, cfg, wire_type, fb_gain, seed, epochs, condition,
              k):
    """Build, calibrate, prototype, and train on CLEAN train
    (no occlusion in this round). condition "frozen" trains with
    plasticity off. Returns (net, train_stats)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(cfg, fb_gain)
    kw["K"] = k
    kw["fb_wire"] = wire_type
    net = RPCNetwork(**{kk: vv for kk, vv in kw.items()
                        if kk not in ("use_confirmation", "proto_n")})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=N_TRAIN)
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


def build_probes(task, seed, test, n_prefix=N_PREFIX):
    """Matched probe trials. Returns a list of dicts
    {ctx, probe, ttype, idx}. Planted: 4 trial types per prefix
    (same prefix across types); sine: 2 trial types per test
    sequence (EXP true continuation, VIO +8 bins)."""
    if task == "sine":
        trials = []
        for i, seq in enumerate(test[:n_prefix]):
            ctx = list(seq[:SINE_CUE])
            trials.append({"ctx": ctx, "probe": seq[SINE_CUE],
                           "ttype": "EXP", "idx": 2 * i})
            trials.append({"ctx": ctx,
                           "probe": (seq[SINE_CUE] + SINE_VIO_SHIFT)
                           % K_SINE, "ttype": "VIO",
                           "idx": 2 * i + 1})
        return trials
    lag = 1 if task == "planted1" else 2
    rng = np.random.default_rng(seed + 77_000)
    trials = []
    for i in range(n_prefix):
        prefix = rng.choice(FILLER_TOKENS, size=10).tolist()
        m = MOTIFS[int(rng.integers(3))]
        x_tok = int(rng.choice(NON_MOTIF))
        y_tok = int(rng.choice(NON_MOTIF))
        if lag == 1:
            ctx_exp = prefix + [m[0]]
            ctx_unexp = prefix + [int(rng.choice(FILLER_TOKENS))]
            ctx_ctl = prefix + [int(rng.choice(FILLER_TOKENS))]
            probes = [("EXP", ctx_exp, m[1]),
                      ("VIO", ctx_exp, x_tok),
                      ("UNEXP", ctx_unexp, m[1]),
                      ("CTL", ctx_ctl, y_tok)]
        else:
            ctx_exp = prefix + [m[0],
                                int(rng.choice(FILLER_TOKENS)), m[1]]
            f3 = [int(rng.choice(FILLER_TOKENS)) for _ in range(3)]
            probes = [("EXP", ctx_exp, m[2]),
                      ("VIO", ctx_exp, x_tok),
                      ("UNEXP", prefix + f3, m[2]),
                      ("CTL", prefix + f3, y_tok)]
        for j, (ttype, ctx, probe) in enumerate(probes):
            trials.append({"ctx": list(ctx), "probe": int(probe),
                           "ttype": ttype, "idx": 4 * i + j})
    return trials


def noise_seed(seed, ttype, idx, c):
    """Deterministic integer noise seed by arithmetic only.
    Never hash() a string: string hashing is randomized per
    process."""
    return (2_000_003 * seed + 7919 * TCODE[ttype]
            + 104729 * idx + int(round(c * 1000)))


def probe_trial(net, protos, ctx, probe, c, fb_mode, z, k):
    """Feed the context in fb_mode, then the probe (clean, or
    noisy with vector z at level c). Returns the per-trial
    measurements at the probe step."""
    net.reset()
    for tok in ctx:
        net.step(tok, train=False, use_confirmation=False,
                 fb_mode=fb_mode)
    x = net._encode(probe).float()
    if c > 0:
        x = F.normalize(x + c * torch.as_tensor(
            z, dtype=torch.float32), dim=0)
    out = net.step_vec(x, train=False, use_confirmation=False,
                       fb_mode=fb_mode)
    p1, d1 = out["p1"].float(), out["d1"].float()
    e = p1 - d1
    ce = F.cosine_similarity(e.unsqueeze(0), protos, dim=1)
    cp = F.cosine_similarity(p1.unsqueeze(0), protos, dim=1)
    return {
        "err": float(e.norm()),
        "cePres": float(ce[probe]),
        "cePred": float(ce[int(out["pred"])]),
        "decE": int(int(ce.argmax()) == probe),
        "decP1": int(int(cp.argmax()) == probe),
        "p1n": float(p1.norm()),
        "fbdr": float(out["fb_drive_ratio"]),
        "pred": int(out["pred"]),
    }


def probe_pass(net, trials, c_values, fb_modes, seed, k):
    """Run all trials at each (fb_mode, c). Noise vectors are
    drawn once per (trial, c) and reused across modes.
    Returns agg[fb_mode][c][ttype] = mean metrics, plus
    p1scale (mean ||p1|| on clean CTL trials for planted,
    clean EXP trials for sine, fb_mode "on")."""
    protos = F.normalize(net.prototypes.float(), dim=1)
    noise = {}
    for t in trials:
        for c in c_values:
            if c > 0:
                rng = np.random.default_rng(
                    noise_seed(seed, t["ttype"], t["idx"], c))
                noise[(t["idx"], c)] = rng.standard_normal(k)
    agg = {}
    for mode in fb_modes:
        agg[mode] = {}
        for c in c_values:
            per = {}
            for t in trials:
                r = probe_trial(net, protos, t["ctx"], t["probe"],
                                c, mode, noise.get((t["idx"], c)), k)
                per.setdefault(t["ttype"], []).append(r)
            agg[mode][c] = {}
            for ttype, rs in per.items():
                agg[mode][c][ttype] = {
                    m: float(np.mean([r[m] for r in rs]))
                    for m in METRICS}
    cal = "CTL" if "CTL" in agg["on"][0.0] else "EXP"
    p1scale = agg["on"][0.0][cal]["p1n"]
    return agg, p1scale


def flat_fields(agg, c_values, p1scale):
    """Flatten one (wire, mode) probe pass into record fields:
    {trial}_{c0|cl}_{metric}. Sine has no unexp/ctl (null)."""
    fields = {"p1scale": p1scale}
    for ttype in TRIALS:
        for ci, c in enumerate(c_values):
            tag = "c0" if ci == 0 else "cl"
            for m in METRICS:
                key = f"{ttype}_{tag}_{m}"
                try:
                    fields[key] = agg[c][ttype.upper()][m]
                except KeyError:
                    fields[key] = None
    return fields


def eval_wire(net, task, seed, trials, c_values, k, test,
              modes=("on", "cut", "shuffle")):
    """Run the probe pass for one trained net in the given fb
    modes, plus normal_acc, p1_cos, and wire stats. Returns
    {mode: record-fields} for the requested modes."""
    agg, p1scale = probe_pass(net, trials, c_values, list(modes),
                              seed, k)
    nacc = normal_acc(net, test, fb_mode="on")
    p1c = paired_p1_cos(net, test)
    fa = fb_align_stats(net)
    out = {}
    for mode in modes:
        fields = flat_fields(agg[mode], c_values, p1scale)
        fields.update({
            "normal_acc": nacc,
            "p1_cos": p1c,
            "fb_align": fa["fb_align"],
            "fb_norm": fa["fb_norm"],
            "fb_clamp_frac": fa["fb_clamp_frac"],
        })
        out[mode] = fields
    return out


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r.get("task"), r.get("config"), r.get("seed"),
                      r.get("c_low"), r.get("fb_gain")))
    return done


def write_line(f, rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()


def run_noise_val(task, seeds, out_path, epochs):
    """Phase 1: one aligned net per seed; calibration probe at
    each c in NOISE_GRID. One line per (task, c, seed) with
    config noise-c{c:.2f}."""
    cfg = cfg_for(task)
    k = k_for(task)
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            tr, _, te = splits(task, seed)
            trials = build_probes(task, seed, te, N_PREFIX)
            net, ts = train_net(tr, cfg, "aligned", 1.0, seed,
                                epochs, "trained", k)
            # One probe pass over all noise levels; the net is
            # identical for every c (training is always clean).
            agg, p1scale = probe_pass(
                net, trials, [0.0] + NOISE_GRID, ["on"], seed, k)
            cal = "CTL" if task != "sine" else "EXP"
            for c in NOISE_GRID:
                key = (task, f"noise-c{c:.2f}", seed, c, 1.0)
                if key in done:
                    continue
                rec = {
                    "task": task, "config": f"noise-c{c:.2f}",
                    "seed": seed, "wire_type": "aligned",
                    "fb_gain": 1.0, "c_low": c,
                    "p1scale": p1scale,
                    "cal_decP1": agg["on"][c][cal]["decP1"],
                    "cal_decE": agg["on"][c][cal]["decE"],
                    "cal_err": agg["on"][c][cal]["err"],
                    "exp_c0_decP1": agg["on"][0.0]["EXP"]["decP1"],
                    "err_mag_first": ts["err_mag_by_epoch"][0],
                    "err_mag_last": ts["err_mag_by_epoch"][-1],
                    "r1_max": ts["r1_max"],
                }
                write_line(f, rec)
                done.add(key)
    return out_path


def run_full(task, seeds, out_path, epochs, c_low):
    """Phase 2 (interaction val) and test: 4 nets per seed,
    8 config lines per (task, seed), probe at c=0 and c=c_low."""
    cfg = cfg_for(task)
    k = k_for(task)
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            tr, _, te = splits(task, seed)
            trials = build_probes(task, seed, te, N_PREFIX)
            nets = {}
            # (wire_type, fb_gain, condition) per net key.
            for net_key, (wire, gain, cond) in {
                "aligned": ("aligned", 1.0, "trained"),
                "random": ("random", 1.0, "trained"),
                "off": ("aligned", 0.0, "trained"),
                "frozen": ("aligned", 1.0, "frozen"),
            }.items():
                net, ts = train_net(tr, cfg, wire, gain, seed,
                                    epochs, cond, k)
                nets[net_key] = (net, ts)
            # Config lines: (net key, fb_mode, config name).
            lines = [
                ("aligned", "on", "aligned-on"),
                ("aligned", "cut", "aligned-cut"),
                ("aligned", "shuffle", "aligned-shuffle"),
                ("random", "on", "random-on"),
                ("random", "cut", "random-cut"),
                ("random", "shuffle", "random-shuffle"),
                ("off", "on", "wire-off"),
                ("frozen", "on", "frozen-aligned"),
            ]
            evals = {}
            for net_key in ("aligned", "random", "off", "frozen"):
                net, _ = nets[net_key]
                gain = 0.0 if net_key == "off" else 1.0
                modes = ("on", "cut", "shuffle") if net_key in (
                    "aligned", "random") else ("on",)
                evals[net_key] = (eval_wire(
                    net, task, seed, trials, [0.0, c_low], k, te,
                    modes), gain)
            for net_key, mode, config in lines:
                fields, gain = evals[net_key]
                wire = "aligned" if net_key in ("aligned", "off",
                                                 "frozen") \
                    else "random"
                key = (task, config, seed, c_low, gain)
                if key in done:
                    continue
                rec = {
                    "task": task, "config": config, "seed": seed,
                    "wire_type": wire, "fb_gain": gain,
                    "c_low": c_low,
                    "err_mag_first": nets[net_key][1][
                        "err_mag_by_epoch"][0],
                    "err_mag_last": nets[net_key][1][
                        "err_mag_by_epoch"][-1],
                    "r1_max": nets[net_key][1]["r1_max"],
                }
                rec.update(fields[mode])
                write_line(f, rec)
                done.add(key)
    return out_path


def main():
    global N_PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("phase", choices=["val", "test"])
    ap.add_argument("--seeds", required=True,
                    help="A-B inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--winners",
                    help="JSON with per-task c_low "
                         "(and floors, for test)")
    ap.add_argument("--val-kind", choices=["noise", "interaction"],
                    default="noise")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--n-prefix", type=int, default=N_PREFIX)
    args = ap.parse_args()
    N_PREFIX = args.n_prefix
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.phase == "val" and args.val_kind == "noise":
        run_noise_val(args.task, seeds, args.out, args.epochs)
        return
    if not args.winners:
        ap.error("--winners is required for interaction val and test")
    with open(args.winners) as f:
        winners = json.load(f)
    c_low = winners[args.task]["c_low"]
    run_full(args.task, seeds, args.out, args.epochs, c_low)


if __name__ == "__main__":
    main()
