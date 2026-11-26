"""Sine fix round 7: finish the A16 story, run by Muse directly.

Round 6c (rpc_sine_fix6c_results.txt): graded + eta_b 0.15 at
alpha 0.7 regressed from 0.230 at 3 seeds to 0.202 at 5 seeds,
and 10 epochs did not help (0.215 at 3 seeds). A16 is dead at 5
seeds; the fixed-sine boundary stands at about 0.20. This round
runs the two remaining cheap, genuinely untested combinations
before reporting the verdict:
1. No-confirmation on the best config. The confirmation rule
   was shown to destroy a learned W_prox (section 23: 0.113 to
   0.076), but it was never ablated on the sine with smooth
   input. Graded + eta_b 0.15, alpha 0.7, smooth 3.0, with
   use_confirmation=False, 3 seeds, plus the binary version
   with use_confirmation=False as its control.
2. Generalisation of the best graded config: variable sine and
   Lorenz with smooth 3.0, alpha 0.7, graded + eta_b 0.15,
   3 seeds. The variable sine graded at alpha 1.0 was flat at
   0.108; alpha 0.7 had the highest mean_g (0.306), so it may
   do better.
All coactivity, smooth 3.0, 5 epochs. Bars: accuracy >= 0.231
(fixed sine), gain >= 0.10.
"""
import numpy as np
from rpc_simple import run_task
from rpc_spatial import run_one

if __name__ == "__main__":
    for name, extra in [
        ("graded-eta0.15-noconf", {"smooth_sigma": 3.0, "alpha": 0.7, "eta_b": 0.15, "graded_conf": True, "use_confirmation": False}),
        ("binary-noconf", {"smooth_sigma": 3.0, "alpha": 0.7, "use_confirmation": False}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, gs, gss = [], [], [], []
            for seed in range(3):
                r = run_task("sine-fixed", seed, learn_rule="coactivity",
                             **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                gs.append(r["mean_g"])
                gss.append(r["std_g"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    for task in ["sine", "lorenz"]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, gs, gss = [], [], [], []
            for seed in range(3):
                r = run_one(task, seed, epochs=5, learn_rule="coactivity",
                            smooth_sigma=3.0, alpha=0.7, graded_conf=True,
                            eta_b=0.15, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                gs.append(r["mean_g"])
                gss.append(r["std_g"])
            print(f"{task} graded-eta0.15-alpha0.7 {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
