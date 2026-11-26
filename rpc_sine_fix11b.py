"""Sine fix round 11, resume after the runtime restart killed
the first run mid-way (Nov 26 05:20 PT).

Already done and kept (rpc_sine_fix11_results.txt): A18+sharp2
full 0.099 and A18+sharp3 full 0.110, both still bimodal
(seeds 0,1 work, seeds 2,3,4 collapse to chance), so
sharpening the pre-term does NOT fix the A18 collapse. This
resume runs only the missing parts: A12+knn, A18+A19+sharp2,
and the sharp2 code probe. Appends to the same results file.
Same config: graded + eta_b 0.15, alpha 0.7, smooth 3.0,
coactivity, 5 epochs, 5 seeds, full + no-plasticity control.
"""
import numpy as np
from rpc_simple import run_task
from rpc_sine_fix11 import probe_code

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15)
    for name, extra in [
        ("A12+knn", {"learn_prox": True, "knn_readout": True}),
        ("A18+A19+sharp2", {"full_code": True, "knn_readout": True,
                             "pre_sharpen": 2.0}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts, y2s = [], [], []
            for seed in range(5):
                r = run_task("sine-fixed", seed, epochs=5, **base, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
                y2s.append(r["mean_y2"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
    res = {"y2": [], "d1": [], "r1": []}
    for seed in range(3):
        r = probe_code(seed, full_code=True, pre_sharpen=2.0)
        for s in res:
            res[s].append(r[s])
    print(f"probe trained sine sharp2: " +
          " ".join(f"{s}={np.mean(res[s]):.3f}" for s in ("y2", "d1", "r1")) +
          f" (d1 range [{min(res['d1']):.3f},{max(res['d1']):.3f}])",
          flush=True)
