"""Sine fix round 6: graded confirmation (A16), run by Muse directly.

Reza chose this after round 5 (Nov 26 00:12 PT). A16: the
confirmation signal is a strength, not a switch. Each step
computes g, the cosine match between the prediction (the distal
drive d1) and the actual input's code (the proximal drive p1),
clipped to [0, 1], and the distal Hebbian update is scaled by g.
The binary burst gate still decides WHERE learning happens; the
confirmation rule and inhibition keep the binary burst. This
differs from A3, which grades the burst per unit and removes the
threshold. Smoke test, 1 epoch, seed 0, fixed sine smooth 3.0:
mean_g 0.192, std_g 0.134, so g varies meaningfully across steps
and grading is not just a smaller learning rate. But mean_g 0.192
also means total learning is about 5x smaller, so block 6 runs
graded with eta_b 0.25 to match total learning.
References: fixed sine smooth 3.0 alpha 1.0 is 0.197-0.199 (5
seeds); smooth 3.0 alpha 0.7 is 0.210 (3 seeds); one-hot alpha
1.0 is 0.091 (5 seeds); planted motifs coactivity alpha 1.0 is
0.072-0.073; variable sine smooth 3.0 is 0.108 (5 seeds).
All coactivity, 5 epochs unless noted. Bars: accuracy >= 0.231
(fixed sine), gain >= 0.10; planted bar 0.2625.
"""
import numpy as np
from rpc_simple import run_task
from rpc_spatial import run_one
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra, seeds in [
        ("smooth3.0-alpha1.0", {"smooth_sigma": 3.0, "alpha": 1.0}, range(5)),
        ("smooth3.0-alpha0.7", {"smooth_sigma": 3.0, "alpha": 0.7}, range(3)),
        ("onehot-alpha1.0", {"alpha": 1.0}, range(3)),
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
            print(f"sine-fixed graded {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs, bursts, gs, gss = [], [], [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, graded_conf=True, **ex2)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
            gs.append(r["mean_g"])
            gss.append(r["std_g"])
        print(f"planted graded alpha1.0 {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
              f"mean_burst={np.mean(bursts):.3f} "
              f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs, bursts, gs, gss = [], [], [], []
        for seed in range(3):
            r = run_one("sine", seed, epochs=5, learn_rule="coactivity",
                        smooth_sigma=3.0, alpha=1.0, graded_conf=True, **ex2)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
            gs.append(r["mean_g"])
            gss.append(r["std_g"])
        print(f"variable-sine graded smooth3.0 {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} "
              f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    accs, bursts, gs, gss = [], [], [], []
    for seed in range(3):
        r = run_task("sine-fixed", seed, learn_rule="coactivity",
                     smooth_sigma=3.0, alpha=1.0, graded_conf=True, eta_b=0.25)
        accs.append(r["acc"])
        bursts.append(r["mean_burst"])
        gs.append(r["mean_g"])
        gss.append(r["std_g"])
    print(f"sine-fixed graded smooth3.0 eta_b0.25 full: acc mean={np.mean(accs):.3f} "
          f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
          f"mean_burst={np.mean(bursts):.3f} "
          f"mean_g={np.mean(gs):.3f} std_g={np.mean(gss):.3f} "
          f"accs={[round(a, 3) for a in accs]}", flush=True)
