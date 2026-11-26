"""Sine fix round 6c: 5-seed confirmation of graded + eta at alpha 0.7, run by Muse directly.

Round 6b (rpc_sine_fix6b_results.txt): graded + eta_b 0.25 at
alpha 1.0 regressed from 0.220 at 3 seeds to 0.196 at 5 seeds,
identical to the binary reference (0.197-0.199), so that bump was
seed noise. But graded + eta_b 0.15 at alpha 0.7 scores 0.230 at
3 seeds (range 0.144 to 0.275, control 0.027, gain +0.203),
0.001 short of the 0.231 accuracy bar, with two of three seeds
above the bar (0.272, 0.275). Reference binary alpha 0.7 smooth
3.0 is 0.210 at 3 seeds. This round: 5-seed confirmation of the
alpha 0.7 version, plus a 10-epoch version at 3 seeds, since more
training helped the binary version before (smooth 2.0 10 epochs:
0.212 to 0.220 at 3 seeds). All coactivity, smooth 3.0, graded
confirmation on. Bars: accuracy >= 0.231, gain >= 0.10.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    for name, extra, seeds, epochs in [
        ("graded-eta0.15-alpha0.7", {"smooth_sigma": 3.0, "alpha": 0.7, "eta_b": 0.15}, range(5), 5),
        ("graded-eta0.15-alpha0.7-10ep", {"smooth_sigma": 3.0, "alpha": 0.7, "eta_b": 0.15}, range(3), 10),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, gs, gss = [], [], [], []
            for seed in seeds:
                r = run_task("sine-fixed", seed, learn_rule="coactivity",
                             graded_conf=True, epochs=epochs, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                gs.append(r["mean_g"])
                gss.append(r["std_g"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
