"""RPC v2 on motif lag 2 and lag 5, run by Muse directly.

Motif tokens sit at positions pos, pos+lag, pos+2*lag, with random
filler between them (make_data_lag from baselines.py). This is the
thesis claim in its hardest form: a cue token must predict a token
2 or 5 steps later, across random filler.

Baselines (baselines_B.md, 5 seeds, held-out test): lag 2 Markov-2
0.266, ESN 0.263, LSTM 0.285; lag 5 Markov-2 0.073, ESN 0.230,
LSTM 0.219. Chance 0.0625.
"""
import sys
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
    for lag in [2, 5]:
        for name, extra in [("full", {}), ("no-confirmation", {"use_confirmation": False}),
                            ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_lag(lag, seed, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"lag{lag} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
