"""Planted-motif fix round 10: A18 full graded code and A19 kNN
readout, run by Muse directly.

Same design as rpc_sine_fix10.py: one change at a time, at
alpha 1.0, coactivity, 5 epochs, 5 seeds, full + no-plasticity
control. References (cosine, top-6): planted coactivity alpha
1.0 is 0.072-0.073; learned W_prox (A12) is 0.113. Bars:
accuracy >= 0.2625, gain >= 0.10.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [
        ("A18-fullcode", {"full_code": True}),
        ("A19-knn", {"knn_readout": True}),
        ("A18+A19", {"full_code": True, "knn_readout": True}),
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
