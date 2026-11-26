"""Fix variants for the code-degradation bottleneck, run by Muse directly.

Section 20 probe: r1 decodes the current token at 1.000, y2_prev
at 0.361. Two candidate fixes, each a single change, planted
motifs, alpha 1.0, learn_rule="coactivity", prototype readout,
5 seeds 0-4, 5 epochs, proto_n=200, each with its own no-plasticity
control. Pass bars: acc >= 0.2625 and gain >= 0.10.
1. rho0: rho2=0.0, module 2 feedforward, no recurrence. If the
   recurrence mixes history into the code, killing it should make
   y2 a clean function of the current token and lift end-to-end
   accuracy.
2. y2k16: y2_frac=0.25, 16 winners of 64 instead of 6. If the top-k
   sparsification throws the information away, more winners should
   keep more of it.
Reference: coactivity-full 0.072, control 0.064, Markov-1 0.278,
direct-full 0.088.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [("rho0-full", {"rho2": 0.0}),
                        ("rho0-no-plasticity", {"rho2": 0.0,
                                               "plasticity": False}),
                        ("y2k16-full", {"y2_frac": 0.25}),
                        ("y2k16-no-plasticity", {"y2_frac": 0.25,
                                                 "plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
