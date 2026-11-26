"""Next step after rpc_simple.py (3 seeds): firm up the motif result with
5 seeds, and complete the missing trackF no-confirmation condition.
Run by Muse directly. Pass bars: acc >= chance + 0.20 and gain over
no-plasticity >= 0.10, mean over 5 seeds, held-out test.
"""
import numpy as np
import torch
from rpc_simple import run_task
from rpc_track_F import run_one as run_trackF

if __name__ == "__main__":
    # 1. motif (1,2,3 repeating, K=4, chance 0.25), 5 seeds 0-4
    for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                        ("no-confirmation", {"use_confirmation": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_task("motif", seed, **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"motif5 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.250 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
    # 2. trackF no-confirmation, 5 seeds 0-4, 3 epochs
    accs, bursts = [], []
    for seed in range(5):
        r = run_trackF(seed, epochs=3, use_confirmation=False)
        accs.append(r["acc"]); bursts.append(r["mean_burst"])
    print(f"trackF no-confirmation: acc mean={np.mean(accs):.3f} "
          f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
          f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
          flush=True)
