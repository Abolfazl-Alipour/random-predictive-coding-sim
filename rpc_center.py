"""Centered-prototype test, run by Muse directly.

Diagnostic finding: raw cycle prototypes are near-parallel
(off-diag cosine mean 0.890, min 0.827), so no distal vector can
discriminate them by raw cosine. The distal drive DOES learn
(correct-prototype cosine 0.343 after training vs 0.089 control),
but the wrong prototype scores higher (0.418).

Fix tested here: subtract the mean prototype (train classes only)
from every prototype and from d1 before the cosine prediction.
No new learned weights, train-data means only.

Tasks: cycle (K=8, chance 0.125) and motif (K=4, chance 0.250),
3 seeds, 5 epochs, full vs no-plasticity. Compare with the
uncentered numbers in rpc_simple_results.txt.
"""
import numpy as np
import torch
from rpc_simple import TASKS, run_task

if __name__ == "__main__":
    for task in ["cycle", "motif"]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(3):
                r = run_task(task, seed, center_proto=True, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            chance = 0.125 if task == "cycle" else 0.250
            print(f"{task} centered {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance={chance:.3f} "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
