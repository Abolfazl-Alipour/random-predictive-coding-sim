"""Round 14b: A23 drive prototypes as a diagnostic, sine, A18 full code.

Same design as rpc_sine_fix14a.py, but the full graded code
(full_code=True). AGY's 1-epoch check (seed 0) gave A18 frozen
0.363 against trained 0.198 under drive prototypes (p1-proto
frozen 0.031). This is the 5-seed 5-epoch verdict.

Config (explicit): learn_rule="coactivity", smooth_sigma=3.0,
alpha=0.7, graded_conf=True, eta_b=0.15, use_confirmation=True,
full_code=True, norm_rows=True, s_inh=0.5 (defaults), 5 epochs,
5 seeds, full + no-plasticity control.
"""
import numpy as np
from rpc_simple import run_task

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15, use_confirmation=True,
                full_code=True)
    for name, extra in [
        ("A18-p1proto", {}),
        ("A18-driveproto", {"drive_proto": True}),
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
