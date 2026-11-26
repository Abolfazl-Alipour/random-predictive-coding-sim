"""Motif and fixed sine at alpha 1.0, run by Muse directly.

Follow-up to the cycle proof (section 14): raising the reservoir leak
rate alpha from 0.3 to 1.0 separated the prototypes (cosine 0.890 to
0.147) and the cycle went from 0.126 to 0.843. Question: does the same
fix work on the other simple tasks, or was the cycle special?

Tasks (same as rpc_simple.py, 5 seeds 0-4, 5 epochs, prototypes from
all train sequences, alpha=1.0):
1. motif: tokens 1,2,3 repeating, K=4, chance 0.250. At alpha 0.3:
   full 0.334, control 0.160 (5 seeds). Pass bars: acc >= 0.45 and
   gain >= 0.10. Markov-1 scores 1.000.
2. sine-fixed: one sine, fixed frequency 0.05, random phase, K=32,
   chance 0.031. At alpha 0.3: full 0.071, control 0.036 (3 seeds).
   Pass bars: acc >= 0.231 and gain >= 0.10.

Conditions per task: full, no-plasticity, no-confirmation.
"""
import numpy as np
import torch
from rpc_simple import run_task

if __name__ == "__main__":
    chances = {"motif": 0.250, "sine-fixed": 0.031}
    for task in ["motif", "sine-fixed"]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                            ("no-confirmation", {"use_confirmation": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_task(task, seed, alpha=1.0, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"{task} alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] "
                  f"chance={chances[task]:.3f} "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
