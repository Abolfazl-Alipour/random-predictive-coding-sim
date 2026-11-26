"""mod2-direct ablation, run by Muse directly.

A11 in rpc_v2.py: module 2 reads the CLEAN token straight in
(u2 = x) instead of the reservoir state r1, so y2_prev becomes a
clean code of the previous token and the distal mapping becomes a
clean token-to-token map learned by RPC's own rules. If mod2-direct
passes the planted bar while the reservoir-fed version fails at
0.072, the reservoir is the bottleneck and the distal learning rule
is fine; if it still fails, the learning rule itself cannot
concentrate the signal. Task: planted motifs, alpha 1.0,
learn_rule="coactivity", prototype readout, 5 seeds 0-4, 5 epochs,
proto_n=200. Conditions: direct-full, direct no-plasticity control.
Pass bars: acc >= 0.2625 and gain >= 0.10 over the control.
Reference: coactivity-full 0.072, control 0.064, Markov-1 0.278.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [("direct-full", {"mod2_direct": True}),
                        ("direct-no-plasticity", {"mod2_direct": True,
                                                 "plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
