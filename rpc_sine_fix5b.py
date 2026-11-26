"""Sine fix round 5, part B (resume), run by Muse directly.

The round-5 driver (rpc_sine_fix5.py) was killed by a runtime
restart at about 23:28 PT after completing the fixed-sine block
(6 lines in rpc_sine_fix5_results.txt, verified): smooth 3.0 with
alpha 0.7 scores 0.210 (3 seeds), alpha 0.5 scores 0.169, n_res 400
scores 0.200. This resume driver runs only the remaining blocks:
variable-frequency sine and Lorenz, each with smooth 3.0 (10
epochs) and its own one-hot alpha 1.0 reference in the same
settings, 5 seeds, full + no-plasticity control.
Output: rpc_sine_fix5b_results.txt.
"""
import numpy as np
from rpc_spatial import run_one

if __name__ == "__main__":
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
