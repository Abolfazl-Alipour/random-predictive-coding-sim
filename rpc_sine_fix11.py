"""Sine fix round 11: A20 sharpened learning pre-term and the
A12+A19 clean-code combination, run by Muse directly.

Round 10 root cause (section 37): the full graded code (A18)
is richer (y2 decode 0.516 to 0.709) but the learned distal
drive is worse (d1 0.509 to 0.457), because each Hebbian update
spreads across about 32 active columns instead of 6 and each
row's fixed mass is diluted about 5x per column. A20 sharpens
the learning pre-term (y2**p, renormalised) so the update
concentrates on the strongest cells while the drive stays
rich. R3 tests the other combination: learned W_prox (A12,
clean p2 code 0.984) plus kNN readout (A19, no averaging).

Sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
coactivity, 5 epochs, 5 seeds, full + no-plasticity control.
References: cosine top-6 0.202, dot 0.217, kNN 0.218, A18
alone 0.105. Bars: accuracy >= 0.231, gain >= 0.10.

Also a code probe for the sharp2 condition: ridge decode of
the current token from y2, d1, r1 on held-out test, trained
network, 3 seeds. If sharpening works, d1 should rise back
toward or above the top-6 0.509 while y2 stays near 0.709.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork

def probe_code(seed, epochs=5, **net_kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(50, seed=seed)
    test = make_sine_fixed(20, seed=seed + 100)
    net = RPCNetwork(K=32, smooth_sigma=3.0, alpha=0.7,
                     graded_conf=True, eta_b=0.15,
                     learn_rule="coactivity", **net_kw)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=20)
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True)
    def collect(data):
        F = {"y2": [], "d1": [], "r1": []}
        toks = []
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    F["y2"].append(out["y2"].numpy().copy())
                    F["d1"].append(out["d1"].numpy().copy())
                    F["r1"].append(out["r1"].numpy().copy())
                    toks.append(tok)
        return {k: np.stack(v) for k, v in F.items()}, np.array(toks)
    Ftr, ytr = collect(train)
    Fte, yte = collect(test)
    Ytr = np.eye(32)[ytr]
    res = {}
    for s in ("y2", "d1", "r1"):
        W = np.linalg.solve(Ftr[s].T @ Ftr[s] + np.eye(Ftr[s].shape[1]),
                            Ftr[s].T @ Ytr).T
        res[s] = float(((Fte[s] @ W.T).argmax(axis=1) == yte).mean())
    return res

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15)
    for name, extra in [
        ("A18+sharp2", {"full_code": True, "pre_sharpen": 2.0}),
        ("A18+sharp3", {"full_code": True, "pre_sharpen": 3.0}),
        ("A12+knn", {"learn_prox": True, "knn_readout": True}),
        ("A18+A19+sharp2", {"full_code": True, "knn_readout": True,
                             "pre_sharpen": 2.0}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, y2s = [], [], []
            for seed in range(5):
                r = run_task("sine-fixed", seed, epochs=5, **base, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                y2s.append(r["mean_y2"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    for name, kw in [("sharp2", {"full_code": True, "pre_sharpen": 2.0})]:
        res = {"y2": [], "d1": [], "r1": []}
        for seed in range(3):
            r = probe_code(seed, **kw)
            for s in res:
                res[s].append(r[s])
        print(f"probe trained sine {name}: " +
              " ".join(f"{s}={np.mean(res[s]):.3f}" for s in ("y2", "d1", "r1")) +
              f" (d1 range [{min(res['d1']):.3f},{max(res['d1']):.3f}])",
              flush=True)
