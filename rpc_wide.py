"""Wider proximal layer, run by Muse directly.

Section 21 hypothesis: the stage probe shows the token code drops
at W_prox (r2 0.984 -> p2 0.711) and at the top-k y2 (0.711 ->
0.332). The 64-dim proximal layer may be the bottleneck: a random
projection to 64 dims mixes the weak token signal with strong
history noise, and the ridge probe cannot isolate it. Widening
n_out to 256 gives the projection 4x more room. Single change:
n_out=256 (y2_k stays 10%, now 25 winners). Task: planted motifs,
alpha 1.0, learn_rule="coactivity", prototype readout, 5 seeds 0-4,
5 epochs, proto_n=200, full plus no-plasticity control.
Pass bars: acc >= 0.2625 and gain >= 0.10.
Reference: coactivity-full 0.072, control 0.064, rho0-full 0.090.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [("wide256-full", {"n_out": 256}),
                        ("wide256-no-plasticity", {"n_out": 256,
                                                  "plasticity": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
