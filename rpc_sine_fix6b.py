"""Sine fix round 6b: 5-seed confirmation of graded + eta compensation, run by Muse directly.

Round 6 (rpc_sine_fix6_results.txt): graded confirmation at the
same learning rate HURTS everywhere (fixed sine smooth 3.0 alpha
1.0 falls 0.197 to 0.154; one-hot falls 0.091 to 0.057; planted
falls 0.072 to 0.068; variable sine is flat at 0.108). But graded
with eta_b raised 5x to 0.25 scores 0.220 at 3 seeds, the best
3-seed fixed-sine score yet, with one seed at 0.283 above the bar.
3-seed numbers have overestimated before (smooth 2.0: 0.212 at 3
seeds, 0.182 at 5), so this round confirms it at 5 seeds, and
tests the alpha 0.7 variant where mean_g was highest (0.322, so
grading has the most signal there) with eta_b 0.15 (about 3x).
All coactivity, smooth 3.0, 5 epochs. Bars: accuracy >= 0.231,
gain >= 0.10.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    for name, extra, seeds in [
        ("graded-eta0.25-alpha1.0", {"smooth_sigma": 3.0, "alpha": 1.0, "eta_b": 0.25}, range(5)),
        ("graded-eta0.15-alpha0.7", {"smooth_sigma": 3.0, "alpha": 0.7, "eta_b": 0.15}, range(3)),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, gs, gss = [], [], [], []
            for seed in seeds:
                r = run_task("sine-fixed", seed, learn_rule="coactivity",
                             graded_conf=True, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                gs.append(r["mean_g"])
                gss.append(r["std_g"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
