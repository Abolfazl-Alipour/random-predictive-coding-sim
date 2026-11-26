"""Sine-fixed at intermediate alpha, run by Muse directly.

Alpha 1.0 fixes deterministic token sequences (cycle 0.843, motif
0.934) but not the fixed sine (0.091, gain +0.069, bars not met).
Hypothesis: the sine needs phase history. The same bin occurs twice
per period, once rising and once falling, so the next bin cannot be
read from the current bin alone. Alpha 1.0 tracks the current token
and weakens history; alpha 0.3 keeps history but prototypes are
confusable (cosine 0.890). A middle alpha may do both: prototypes
separated enough (0.461 at 0.7) with history still present.

Test: sine-fixed, alpha 0.7 and 0.5, full vs no-plasticity, 3 seeds
0-2 (search; a 5-seed confirmation follows only if a value looks
close to the bars: acc >= 0.231, gain >= 0.10).
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    for alpha in [0.7, 0.5]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(3):
                r = run_task("sine-fixed", seed, alpha=alpha, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"sine-fixed alpha{alpha} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
