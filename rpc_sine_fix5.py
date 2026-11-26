"""Sine fix round 5, run by Muse directly.

Round 4 (rpc_sine_fix4_results.txt): the fixed sine has stalled at
~0.20 at 5 seeds. Smooth 3.0 with 10 epochs is 0.199 (gain +0.177),
same as 5 epochs (0.197); smooth 2.0 with 10 epochs is 0.193
(gain +0.172), +0.011 over its 5-epoch 5-seed 0.182. Split
prototypes add nothing: split + smooth 2.0 is 0.183 vs 0.182
plain, split + smooth 2.0 with 10 epochs is 0.205 vs 0.220 plain,
split + one-hot is 0.074 vs 0.071 plain. A15 is dead.
Round 5 tests the remaining levers on the fixed sine and then
generalisation, which is the actual question (variable sine and
Lorenz):
1. smooth 3.0 with alpha 0.7 and 0.5 (the alpha sweep was done
   with one-hot input only; smooth input plus a slower reservoir
   may combine history with metric structure), 3 seeds.
2. smooth 3.0 with n_res 400 (more reservoir capacity for phase),
   3 seeds.
3. Variable-frequency sine (each sequence its own frequency
   0.02-0.08 and phase; reference one-hot alpha 0.3 is 0.040,
   5 seeds) with smooth 3.0 10 epochs, plus the one-hot alpha 1.0
   reference in the same settings, 5 seeds.
4. Lorenz x (reference one-hot alpha 0.3 is 0.045) with smooth 3.0
   10 epochs, plus the one-hot alpha 1.0 reference, 5 seeds.
All coactivity, 5 epochs unless noted. Variable sine and Lorenz use
rpc_spatial defaults (proto_n 20), matching their references.
"""
import numpy as np
from rpc_simple import run_task
from rpc_spatial import run_one

if __name__ == "__main__":
    for name, extra in [
        ("smooth3.0-alpha0.7", {"smooth_sigma": 3.0, "alpha": 0.7}),
        ("smooth3.0-alpha0.5", {"smooth_sigma": 3.0, "alpha": 0.5}),
        ("smooth3.0-nres400", {"smooth_sigma": 3.0, "n_res": 400}),
    ]:
        kw = {"alpha": 1.0}
        kw.update(extra)
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(3):
                r = run_task("sine-fixed", seed, learn_rule="coactivity",
                             **kw, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                  flush=True)
    for task in ["sine", "lorenz"]:
        for name, extra in [
            ("smooth3.0-10ep", {"smooth_sigma": 3.0, "alpha": 1.0}),
            ("onehot-alpha1.0", {"alpha": 1.0}),
        ]:
            for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
                accs, bursts = [], []
                for seed in range(5):
                    r = run_one(task, seed, epochs=5, learn_rule="coactivity",
                                **extra, **ex2)
                    accs.append(r["acc"])
                    bursts.append(r["mean_burst"])
                print(f"{task} {name} {cond}: acc mean={np.mean(accs):.3f} "
                      f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                      f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                      flush=True)
