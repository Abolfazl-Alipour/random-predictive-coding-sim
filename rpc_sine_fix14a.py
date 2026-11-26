"""Round 14a: A23 drive prototypes as a diagnostic, sine, top-6 code.

Kimi's framing: A23 discriminates the comparison-space hypothesis
from the centroid hypothesis. AGY's 1-epoch check (seed 0) says
A23 hands the frozen control 0.444 against trained 0.313, so it
measures readout leakage, not learning. This is the 5-seed
5-epoch verdict at the best-known sine config.

Config (explicit, matching the 0.202 reference): learn_rule=
"coactivity", smooth_sigma=3.0, alpha=0.7, graded_conf=True,
eta_b=0.15, use_confirmation=True, norm_rows=True, s_inh=0.5
(defaults), 5 epochs, 5 seeds, full + no-plasticity control.
Conditions: p1-prototype cosine (drive_proto off) vs
drive-prototype cosine (drive_proto on), trained and frozen.
The frozen control builds drive prototypes from the FROZEN
drive (plasticity off), so it is the correct leakage control.
Bars: the diagnostic passes if drive_proto raises the TRAINED
score without raising the FROZEN score by as much; it fails as
a fix if the frozen control gains as much or more.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15, use_confirmation=True)
    for name, extra in [
        ("top6-p1proto", {}),
        ("top6-driveproto", {"drive_proto": True}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_task("sine-fixed", seed, epochs=5, **base, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed A23 {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
