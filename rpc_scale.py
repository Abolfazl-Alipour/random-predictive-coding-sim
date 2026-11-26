"""Scale test, run by Muse directly.

Question: does the small RPC effect grow with more data and more epochs?
Best variant so far is proto200 (prototypes from all train sequences).
Test on lag 5, 500 train / 50 test sequences, 10 epochs, 3 seeds,
full vs no-confirmation vs no-plasticity.

Baseline lag-5 (200 train, 5 epochs, proto20): full 0.073, gain +0.015.
Proto200 at that scale: full 0.081, gain +0.019.
"""
import numpy as np
import torch
from baselines import make_data_lag
from rpc_v2 import run

def run_lag(lag, seed, n_train=500, epochs=10, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data_lag(n_seq=n_train, lag=lag)
    test = make_data_lag(n_seq=50, lag=lag)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, proto_n=n_train)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    for name, extra in [("full", {}), ("no-confirmation", {"use_confirmation": False}),
                        ("no-plasticity", {"plasticity": False})]:
        accs, bursts = [], []
        for seed in range(3):
            r = run_lag(5, seed, **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"lag5 scale500x10 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
