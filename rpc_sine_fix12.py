"""Sine fix round 12: A21 sharpened distal-drive read, run by Muse
directly.

Round 11 correction (section 38): A20 sharpened the learning
pre-term and changed nothing (sharp2 0.099, sharp3 0.110, both
bimodal), so the dilution explanation is wrong as stated. The
drive stays degraded (d1 decode 0.478 vs top-6 0.509) even with
a concentrated update, so the degradation sits in the READ:
d1 = W_dist1 @ y2_prev sums about 32 rows of the map, and the
extra weak rows add noise at read time, no matter how the map
was learned. A21 sharpens the READ code (y2_prev**p,
renormalised) while the learning keeps the raw graded code:
rich write, concentrated read. All cells still project back
(no hard cutoff).

Sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
coactivity, 5 epochs, 5 seeds, full + no-plasticity control.
References: cosine top-6 0.202, dot 0.217, kNN 0.218, A18
alone 0.105, A18+sharp2 0.099. Bars: accuracy >= 0.231,
gain >= 0.10.

Also a per-seed probe: ridge decode of the NEXT token from d1
on held-out test, trained network, seeds 0-4, for the A18
network (no sharpen) and the top-6 baseline. d1 at step i is
computed from the code of token i-1, so decoding token i from
d1 is the next-token decode. If the collapsed seeds (2, 3, 4)
decode at chance while the working seeds (0, 1) decode well,
the map is degenerate on those seeds; if all seeds decode
alike, the readout (not the map) is the problem.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork


def probe_next(seed, epochs=5, **net_kw):
    """Per-seed ridge decode of the next token from d1 (trained
    sine network). Returns the held-out accuracy."""
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
        D, toks = [], []
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    D.append(out["d1"].numpy().copy())
                    toks.append(tok)
        return np.stack(D), np.array(toks)

    Ftr, ytr = collect(train)
    Fte, yte = collect(test)
    Ytr = np.eye(32)[ytr]
    W = np.linalg.solve(Ftr.T @ Ftr + np.eye(Ftr.shape[1]),
                        Ftr.T @ Ytr).T
    return float(((Fte @ W.T).argmax(axis=1) == yte).mean())


if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15)
    for name, extra in [
        ("A18+read2", {"full_code": True, "read_sharpen": 2.0}),
        ("A18+read3", {"full_code": True, "read_sharpen": 3.0}),
        ("A18+read2+knn", {"full_code": True, "read_sharpen": 2.0,
                            "knn_readout": True}),
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
    for label, kw in [("top-6 baseline", {}),
                      ("A18 full code", {"full_code": True})]:
        dec = [probe_next(seed, **kw) for seed in range(5)]
        print(f"probe next-token from d1, {label}: "
              f"mean={np.mean(dec):.3f} per-seed="
              f"{[round(d, 3) for d in dec]}", flush=True)
