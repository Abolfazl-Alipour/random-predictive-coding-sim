"""RPC v2 on the 1D track task (tasks.py), run by Muse directly.

Adaptation, same as rpc_spatial.py for sine/Lorenz: the network is the
token network unchanged; the data are position-bin tokens. Position on
the 180 cm track is binned into K=32 bins (5.625 cm per bin), sequences
are the first 100 steps of each trial from make_track_data (AR(1)
velocity, rho 0.85, vmax 6 cm/step). Prediction is next bin, scored on
held-out test trials, 5 seeds.

Chance = 1/32 = 0.03125. Pass bars (set before the overnight run): beat
chance by >=0.20 points, i.e. acc >= 0.23, and beat the no-plasticity
control by >=0.10. Persistence (predict current bin) is reported as a
reference: with vmax 6 cm/step and 5.625 cm bins it is far above chance,
so beating chance alone is not the bar; the bar is 0.23 and +0.10 over
the control. Ridge control from tasks_C.md (continuous next-F task):
position RMSE 1.04 cm vs persistence 1.89 cm, chance 52.3 cm.
"""
import numpy as np
import torch
from rpc_v2 import run
from tasks import make_track_data

K = 32
TRACK_LEN = 180.0
SEQ_LEN = 100

def make_track_tokens(n_trials=100, seed=0):
    d = make_track_data(track_len=TRACK_LEN, n_trials=n_trials,
                        n_steps=1800, seed=seed)
    pos = d["positions"][:, :SEQ_LEN]  # first 100 steps of each trial
    bins = np.clip((pos / TRACK_LEN * K).astype(int), 0, K - 1)
    return [row.tolist() for row in bins]

def persistence_acc(test):
    correct, total = 0, 0
    for seq in test:
        for i in range(1, len(seq)):
            total += 1
            correct += int(seq[i] == seq[i - 1])
    return correct / max(total, 1)

def run_one(seed, epochs=3, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_track_tokens(n_trials=100, seed=seed)
    test = make_track_tokens(n_trials=20, seed=seed + 1000)
    defaults = dict(K=K, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    import sys
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    test0 = make_track_tokens(n_trials=20, seed=1000)
    print(f"track persistence (predict current bin): {persistence_acc(test0):.3f}",
          flush=True)
    for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                        ("no-confirmation", {"use_confirmation": False})]:
        accs, bursts = [], []
        for seed in seeds:
            r = run_one(seed, epochs=epochs, **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"track {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
