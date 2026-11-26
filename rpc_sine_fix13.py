"""Sine fix round 13: A18 under the thesis burst rule, and A22
prototype-span projection readout, run by Muse directly.

Round 12 finding (section 39): the collapse localises to the
readout's space. The per-seed probe shows the drive holds the
next-token information on collapsed seed 2 (ridge 0.495) but
the fixed-space readout extracts none of it (network 0.035);
the learned map does not reliably land in the prototype
space. Two tests: (a) the thesis-faithful burst rule instead
of coactivity, which gates learning to steps where the
prediction already matches and may land the map better;
(b) A22, project d1 onto the span of the prototypes before
the cosine comparison, which tests directly whether the
misalignment is the problem.

NOTE on the rule: rounds 10-12 on sine all used coactivity,
so the burst rule was never tested there. (On planted the
default burst rule was used throughout, so the burst test is
sine-only.)

Sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
5 epochs, 5 seeds, full + no-plasticity control. Base rule
is coactivity unless the condition says burst. References:
cosine top-6 0.202, dot 0.217, kNN 0.218, A18 coactivity
0.105, A18+read2 0.119. Bars: accuracy >= 0.231,
gain >= 0.10.

Also a per-seed probe for the burst condition: ridge decode
of the NEXT token from d1, seeds 0-4. If the burst rule lands
the map, the collapsed seeds' extraction ratio (network score
divided by probe decode) should rise toward the working-seed
level of about 0.55.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork


def probe_next_rule(seed, epochs=5, **net_kw):
    """Per-seed ridge decode of the next token from d1 (trained
    sine network, rule passed through net_kw)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(50, seed=seed)
    test = make_sine_fixed(20, seed=seed + 100)
    net = RPCNetwork(K=32, smooth_sigma=3.0, alpha=0.7,
                     eta_b=0.15, **net_kw)
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
        ("A18+burst", {"full_code": True, "learn_rule": "burst",
                       "graded_conf": False}),
        ("A18+proj", {"full_code": True, "proto_proj": True}),
        ("A18+burst+proj", {"full_code": True, "learn_rule": "burst",
                            "graded_conf": False, "proto_proj": True}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, y2s = [], [], []
            for seed in range(5):
                kw = {**base, **extra, **ex2}
                r = run_task("sine-fixed", seed, epochs=5, **kw)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                y2s.append(r["mean_y2"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    dec = [probe_next_rule(seed, full_code=True, learn_rule="burst",
                           graded_conf=False) for seed in range(5)]
    print(f"probe next-token from d1, A18+burst: "
          f"mean={np.mean(dec):.3f} per-seed="
          f"{[round(d, 3) for d in dec]}", flush=True)
