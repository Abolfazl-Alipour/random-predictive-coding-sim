"""Scaled lag-5 test, run by Muse directly.

Designated next test (simulation-results.md section 6.12): the best
lag-5 variant so far (prototypes from 200 train sequences, gain +0.019
at 200 train / 5 epochs) scaled up to 500 train sequences and 10 epochs,
with the no-confirmation condition (confirmation hurt Lorenz and added
nothing on motifs, so the no-confirmation run is the one that can move
the verdict). Conditions: full, no-confirmation, no-plasticity, 5 seeds.
Lag 5, K=16, chance 0.0625. Pass bars: acc >= 0.2625 (chance + 0.20)
and gain over no-plasticity >= 0.10, mean over 5 seeds, held-out test.
"""
import numpy as np
import torch
from baselines import make_data_lag
from rpc_v2 import run

N_TRAIN = 500
N_TEST = 50
EPOCHS = 10

def run_lag_scale(seed, epochs=EPOCHS, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data_lag(n_seq=N_TRAIN, lag=5)
    test = make_data_lag(n_seq=N_TEST, lag=5)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, proto_n=200)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    import sys
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    for name, extra in [("full", {}),
                        ("no-confirmation", {"use_confirmation": False}),
                        ("no-plasticity", {"plasticity": False})]:
        accs, bursts = [], []
        for seed in seeds:
            r = run_lag_scale(seed, epochs=epochs, **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"lag5 scale500x10 proto200 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
