"""Two-step distal drive (delay line), run by Muse directly.

A10 in rpc_v2.py: d1 = W_dist1 @ y2_prev + W_dist1b @ y2_prev2.
CORRECTED expectation (simulation-results.md 19b): the planted task
does NOT require two-step memory. One-step ceiling is about 0.36
(Markov-1 proves 0.278); second-order ceiling is about 0.39. So
this is a check with an expected gain of about 0.03, not the
bottleneck. Task: planted motifs, alpha 1.0, learn_rule="coactivity",
prototype readout, 5 seeds 0-4, 5 epochs, proto_n=200.
Conditions: delay-full, delay no-plasticity control.
Pass bars: acc >= 0.2625 and gain >= 0.10 over the control.
Reference without delay: coactivity-full 0.072, control 0.064.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [("delay-full", {"use_delay": True}),
                        ("delay-no-plasticity", {"use_delay": True,
                                                "plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
