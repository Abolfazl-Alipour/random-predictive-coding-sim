"""Sine fix round 2, run by Muse directly.

Round 1 (rpc_sine_fix_results.txt, 3 seeds): only the smooth input
helped (0.145, gain +0.114, passes the gain bar, fails the accuracy
bar 0.231). Delay 0.076, wider code 0.057 (hurts), prox1 0.028
(hurts), both clean 0.032. Ridge on trained d1 0.556 vs frozen 0.606:
learning adds nothing to d1; the prototype readout (0.078) loses
~0.48 vs the ridge readout on the same d1.
Round 2: sigma sweep for the smooth input (1.0, 2.0, 3.0), smooth +
delay (the two fixes address different stages, they may stack), and
a 5-seed confirmation of smooth 1.5 (3-seed range 0.116-0.168).
All sine-fixed, alpha 1.0, coactivity, 5 epochs.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    variants = [
        ("smooth1.0", {"smooth_sigma": 1.0}),
        ("smooth2.0", {"smooth_sigma": 2.0}),
        ("smooth3.0", {"smooth_sigma": 3.0}),
        ("smooth1.5+delay", {"smooth_sigma": 1.5, "use_delay": True}),
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
    # 5-seed confirmation of smooth 1.5
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_task("sine-fixed", seed, alpha=1.0, learn_rule="coactivity",
                         smooth_sigma=1.5, **ex2)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
        print(f"sine-fixed smooth1.5 5seed {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
              flush=True)
