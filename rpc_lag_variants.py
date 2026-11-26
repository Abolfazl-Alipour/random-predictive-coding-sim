"""Lag-5 variants, run by Muse directly.

Variants on the lag-5 motif task (5 seeds, 200 train / 50 test, 5 epochs):
1. long-memory reservoir: rho 0.99, alpha 0.1 (default rho 0.9, alpha 0.3).
   The reservoir must hold the cue token across 5 filler steps, so a
   slower reservoir is the most plausible fix for a lag failure.
2. prototypes from all 200 train sequences instead of 20 (proto_n=200).
   Prototypes are train-only, so this is fair; it tests whether the
   readout, not the learning, is the bottleneck.
3. both together.
Baseline lag-5 full: acc 0.073, gain +0.015 over no-plasticity.
"""
import numpy as np
import torch
from baselines import make_data_lag
from rpc_v2 import run

def run_lag(lag, seed, epochs=5, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data_lag(n_seq=200, lag=lag)
    test = make_data_lag(n_seq=50, lag=lag)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    variants = [
        ("longmem", dict(rho=0.99, alpha=0.1)),
        ("proto200", dict(proto_n=200)),
        ("longmem+proto200", dict(rho=0.99, alpha=0.1, proto_n=200)),
    ]
    for vname, vkw in variants:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_lag(5, seed, **{**vkw, **extra})
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"lag5 {vname} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
