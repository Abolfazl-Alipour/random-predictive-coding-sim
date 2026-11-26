"""Sine fix round 8: dot-product readout (A17), run by Muse directly.

Reza was offered three options after round 7 (Nov 26 01:07 PT):
stop, accept the trained readout, or try the dot-product readout.
This round runs the dot-product test so the result is ready
whichever way he decides. A17: the cosine readout normalises d1
and every prototype, so only the angle matters. The dot-product
readout scores each prototype by the raw dot product d1 .
proto_k, which is ||d1|| * ||proto_k|| * cos, so each prototype
is weighted by its stored strength. (A dot product against UNIT
prototypes would be identical to cosine, since ||d1|| is the same
for every k, so that variant is not tested.) Diagnostic on the
fixed sine, smooth 3.0, alpha 0.7: prototype norms range 0.55 to
0.75 (mean 0.635, std 0.061), so the norms do vary and the test
is real, but the variation is modest (max/min 1.36), so the
effect is likely small. Smoke test, 1 epoch, seed 0: dot 0.286
vs cosine 0.282.
References (cosine): fixed sine graded + eta alpha 0.7 is 0.202
(5 seeds); fixed sine binary smooth 3.0 alpha 1.0 is 0.197-0.199;
one-hot alpha 1.0 is 0.091; planted motifs coactivity alpha 1.0
is 0.072-0.073. All coactivity, 5 epochs. Bars: accuracy >=
0.231 (fixed sine) and >= 0.2625 (planted), gain >= 0.10.
"""
import numpy as np
from rpc_simple import run_task
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra, seeds in [
        ("graded-eta0.15-alpha0.7", {"smooth_sigma": 3.0, "alpha": 0.7, "graded_conf": True, "eta_b": 0.15}, range(5)),
        ("binary-alpha1.0", {"smooth_sigma": 3.0, "alpha": 1.0}, range(5)),
        ("onehot-alpha1.0", {"alpha": 1.0}, range(3)),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in seeds:
                r = run_task("sine-fixed", seed, learn_rule="coactivity",
                             dot_proto=True, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed dot {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, dot_proto=True, **ex2)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
        print(f"planted dot alpha1.0 {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
              f"mean_burst={np.mean(bursts):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
