"""Round 14a (planted): A23 drive prototypes as a diagnostic, top-6.

Same design as the sine A23 diagnostic, on planted motifs.
Config (explicit, matching the A12-alone config minus learn_prox):
learn_rule="coactivity", use_confirmation=False, eta_b=0.05,
alpha=1.0, one-hot, norm_rows=True, s_inh=0.5 (defaults),
5 epochs, 5 seeds, full + no-plasticity control. Conditions:
p1-prototype cosine vs drive-prototype cosine, trained and
frozen. The frozen control builds drive prototypes from the
frozen drive, so it is the leakage control.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", use_confirmation=False, eta_b=0.05)
    for name, extra in [
        ("top6-p1proto", {}),
        ("top6-driveproto", {"drive_proto": True}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_planted(seed, 1.0, **base, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"planted A23 {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
