"""Sine fix round 4, run by Muse directly.

Round 3 (rpc_sine_fix3_results.txt, 5 seeds): smooth 3.0 is the best
5-seed score at 0.197 (gain +0.175); smooth 2.0 is 0.182 (gain
+0.161); smooth 2.5 is 0.163. Smooth 2.0 with 10 epochs scores
0.220 at 3 seeds (gain +0.201), so more training helps. Mechanism
probe on a frozen smooth-2.0 net confirms the mechanism: the code's
direction decode rises from 0.748 (one-hot) to 0.915, and d1's from
0.740 to 0.892, while the reservoir is unchanged (0.953 -> 0.944).
Round 4: 5-seed confirmations of smooth 2.0 and 3.0 with 10 epochs,
and the direction-split prototype test (A15, three prototypes per
bin: rising, falling, flat), which attacks the readout gap
(prototype 0.078 vs ridge 0.556 on the same d1). Smoke test, 1
epoch, seed 0: split + smooth 2.0 scores 0.204, split + one-hot
scores 0.118.
All sine-fixed, alpha 1.0, coactivity, 5 epochs unless noted.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    for name, extra, seeds, epochs in [
        ("smooth2.0-10ep", {"smooth_sigma": 2.0}, range(5), 10),
        ("smooth3.0-10ep", {"smooth_sigma": 3.0}, range(5), 10),
        ("split-smooth2.0", {"smooth_sigma": 2.0, "split_proto": True}, range(5), 5),
        ("split-smooth2.0-10ep", {"smooth_sigma": 2.0, "split_proto": True}, range(3), 10),
        ("split-onehot", {"split_proto": True}, range(3), 5),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in seeds:
                r = run_task("sine-fixed", seed, alpha=1.0, learn_rule="coactivity",
                             epochs=epochs, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                  flush=True)
