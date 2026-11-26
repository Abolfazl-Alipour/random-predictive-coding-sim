"""Sine fix tests, run by Muse directly.

Diagnostics (rpc_sine_diag_results.txt, 5 seeds, frozen net):
- Markov-1 ceiling 0.459, Markov-2 0.802, Markov-3 0.833. Every test
  step's previous bin is ambiguous (fraction 1.000): the same bin
  occurs rising and falling with different successors.
- Stage probes, next-bin decode / direction decode:
  r2_prev 0.953 / 0.984, y2_prev 0.607 / 0.748, d1 0.602 / 0.740,
  r1 1.000 / 0.998, p1 1.000 / 0.946.
  The reservoir holds the answer (0.953); the top-6 code throws away
  a third of it (0.607); the learned distal drive transmits what the
  code contains (0.602), so learning is NOT the bottleneck, the code is.
- Same-bin rising-vs-falling prototype cosine 0.419 (off-diagonal
  0.177): prototypes DO split by direction, so the readout is not a
  blend; it is the code that mixes rising and falling.
Fixes (sine-fixed, alpha 1.0, coactivity rule, 3 seeds screen, 5
epochs; a 5-seed confirmation follows only for a variant that looks
close to the bars: acc >= 0.231, gain >= 0.10):
1. delay (A10): two-step distal drive, the direct fix for one-step
   ambiguity. Ceiling 0.802.
2. y2_frac 0.25 / 0.5: wider code (16 / 32 of 64 winners), the direct
   fix for the top-k information loss (0.953 -> 0.607).
3. delay + y2_frac 0.25 combined.
4. smooth_sigma 1.5 (A14): neighbouring bins share input.
5. learn_prox1 (A13): clean module-1 code (post side).
6. learn_prox + learn_prox1: both sides clean.
7. ridge readout on TRAINED d1: does learning help or hurt the
   information in d1 (frozen random d1 ridge-decodes 0.602)?
Planted motifs (alpha 1.0, coactivity, 3 seeds): learn_prox1 alone
and learn_prox + learn_prox1, the module-1 projection test Reza
approved. Reference planted coactivity full 0.072, control 0.064.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_planted_alpha import run_planted
from rpc_v2 import RPCNetwork


def fit_ridge(D, Y, lam=1.0):
    return np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ Y).T


def ridge_d1_sine(seed, plasticity=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(n_seq=50, length=200, seed=seed)
    test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
    net = RPCNetwork(K=32, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64, alpha=1.0, learn_rule="coactivity")
    net.calibrate(train)
    net.build_prototypes(train, proto_n=50)
    if plasticity:
        for _ in range(5):
            for seq in train:
                net.reset()
                for tok in seq:
                    net.step(tok, train=True)

    def collect(data):
        feats, toks, preds = [], [], []
        with torch.no_grad():
            for seq in data:
                net.reset()
                for i, tok in enumerate(seq):
                    out = net.step(tok, train=False)
                    if i > 0:
                        feats.append((out["d1"] / net.sd_d1).numpy().copy())
                        toks.append(tok)
                        preds.append(out["pred"])
        return np.stack(feats), np.array(toks), np.array(preds)

    Dtr, ytr, _ = collect(train)
    Dte, yte, pte = collect(test)
    W = fit_ridge(Dtr, np.eye(32)[ytr])
    ridge_acc = float(((Dte @ W.T).argmax(axis=1) == yte).mean())
    proto_acc = float((pte == yte).mean())
    return ridge_acc, proto_acc


if __name__ == "__main__":
    variants = [
        ("delay", {"use_delay": True}),
        ("y2frac0.25", {"y2_frac": 0.25}),
        ("y2frac0.5", {"y2_frac": 0.5}),
        ("delay+y2frac0.25", {"use_delay": True, "y2_frac": 0.25}),
        ("smooth1.5", {"smooth_sigma": 1.5}),
        ("prox1", {"learn_prox1": True}),
        ("prox+prox1", {"learn_prox": True, "learn_prox1": True}),
    ]
    for name, extra in variants:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(3):
                r = run_task("sine-fixed", seed, alpha=1.0, learn_rule="coactivity",
                             **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                  flush=True)
    # ridge readout on trained d1
    for cond, plast in [("full", True), ("no-plasticity", False)]:
        ra, pa = [], []
        for seed in range(3):
            r, p = ridge_d1_sine(seed, plasticity=plast)
            ra.append(r)
            pa.append(p)
        print(f"sine-fixed ridge-d1 {cond}: ridge mean={np.mean(ra):.3f} "
              f"[{min(ra):.3f},{max(ra):.3f}] proto mean={np.mean(pa):.3f} "
              f"[{min(pa):.3f},{max(pa):.3f}] (frozen random d1 ridge = 0.602)",
              flush=True)
    # planted motifs, module-1 projection
    for name, extra in [("prox1", {"learn_prox1": True}),
                        ("prox+prox1", {"learn_prox": True, "learn_prox1": True})]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs = []
            for seed in range(3):
                r = run_planted(seed, 1.0, learn_rule="coactivity", **extra, **ex2)
                accs.append(r["acc"])
            print(f"planted alpha1.0 coactivity {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
