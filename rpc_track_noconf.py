"""Finish the 1D track run: no-confirmation condition only, 5 seeds.

The full and no-plasticity lines are already in rpc_track_results.txt
(run 2026-11-25 04:22 PST, deterministic, same seeds/code). This driver
reuses rpc_track.run_one and prints the missing line in the same format.
"""
import numpy as np
from rpc_track import run_one

if __name__ == "__main__":
    accs, bursts = [], []
    for seed in [0, 1, 2, 3, 4]:
        r = run_one(seed, epochs=3, use_confirmation=False)
        accs.append(r["acc"])
        bursts.append(r["mean_burst"])
    print(f"track no-confirmation: acc mean={np.mean(accs):.3f} "
          f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
          f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
          flush=True)
