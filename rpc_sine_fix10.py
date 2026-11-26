"""Sine fix round 10: A18 full graded code and A19 kNN readout, run
by Muse directly.

Reza's correction (Nov 26): the top-6 code and the prototype
readout are my assumptions, not the theory. A18: all module-2
cells project back, graded, no rank cutoff (per-step
standardised relu, divisive normalisation). A19: the readout
matches the distal drive against every stored training p1
vector (nearest neighbour) instead of one mean prototype per
token. Record correction: the prototypes were always mean p1,
not mean d1, so the readout already matched distal against
proximal; what is untested is the averaging, not the space.

One change at a time, at the best-known config (graded + eta_b
0.15, alpha 0.7, smooth 3.0, coactivity, 5 epochs, 5 seeds,
full + no-plasticity control). References (cosine, top-6):
graded+eta alpha 0.7 is 0.202 at 5 seeds; binary alpha 1.0 is
0.197-0.199. Bars: accuracy >= 0.231, gain >= 0.10.

Also a code probe: ridge decode of the CURRENT token from y2
(the code), d1 (the distal drive), and r1 (the reservoir) on
held-out test, trained network, 3 seeds, top-6 vs full code.
If the full code carries the token, y2 decode should rise well
above the top-6 0.332.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork

def probe_code(full_code, seed, epochs=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(50, seed=seed)
    test = make_sine_fixed(20, seed=seed + 100)
    net = RPCNetwork(K=32, smooth_sigma=3.0, alpha=0.7,
                     graded_conf=True, eta_b=0.15,
                     learn_rule="coactivity", full_code=full_code)
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
        ("A18-fullcode", {"full_code": True}),
        ("A19-knn", {"knn_readout": True}),
        ("A18+A19", {"full_code": True, "knn_readout": True}),
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
    for name, fc in [("top6", False), ("fullcode", True)]:
        res = {"y2": [], "d1": [], "r1": []}
        for seed in range(3):
            r = probe_code(fc, seed)
            for s in res:
                res[s].append(r[s])
        print(f"probe trained sine {name}: " +
              " ".join(f"{s}={np.mean(res[s]):.3f}" for s in ("y2", "d1", "r1")) +
              f" (y2 range [{min(res['y2']):.3f},{max(res['y2']):.3f}])",
              flush=True)
