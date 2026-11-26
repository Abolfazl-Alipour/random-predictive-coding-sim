"""Planted-motif fix round 11: A20 sharpened learning pre-term
and the A12+A19 clean-code combination, run by Muse directly.

Same design as rpc_sine_fix11.py. Planted motifs, alpha 1.0,
coactivity, 5 epochs, 5 seeds, full + no-plasticity control.
References: cosine 0.072-0.073, dot 0.078, learned W_prox
(A12) 0.113, A18 alone 0.065, kNN alone 0.072. Bars:
accuracy >= 0.2625, gain >= 0.10.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [
        ("A18+sharp2", {"full_code": True, "pre_sharpen": 2.0}),
        ("A12+knn", {"learn_prox": True, "knn_readout": True}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_planted(seed, 1.0, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"planted {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
