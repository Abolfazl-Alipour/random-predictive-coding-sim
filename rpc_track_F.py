"""RPC v2 on the 1D track with thesis-faithful leaky-integrator inputs
(tasks.py Task 1), run by Muse directly.

This is the test simulation-results.md section 1 names as next: the
position-bin version (rpc_track.py) feeds one-hot bin tokens, but the
thesis Chapter 4 input is the leaky-integrator vector F itself. Here
each step feeds x_t = [F(t), F(t-1)] (200 numbers, F(s, t) = exp(-s*x(t)),
s log-spaced 1e-3 to 1e-1, 100 units) into module 1 through the new
vector-input path in rpc_v2.py (step_vec / run_vec, same step core,
calibration, prototypes, and controls; only the input encoding differs).
The target is the next position bin (K=32 bins over 180 cm), same
scoring as rpc_track.py, so the two track versions are directly
comparable. Same data otherwise: first 100 steps of each trial from
make_track_data (AR(1) velocity, rho 0.85, vmax 6 cm/step), 100 train /
20 test trials, 3 epochs, 5 seeds 0-4.

ASSUMPTION A7 (input encoding for integrator vectors): the 200-number
F vector is fed straight into module 1's reservoir (W_exp @ x, same
fixed random expansion as the one-hot path). The thesis feeds integrator
vectors into a sparse autoencoder, not into the RPC reservoir, so this
encoding is new; it is the natural generalisation of the one-hot input
(the reservoir takes a vector either way) and the A1 calibration absorbs
the scale difference. No input standardisation is applied: F is already
in [0, 1].

Chance = 1/32 = 0.03125. Pass bars (set before the overnight run): beat
chance by >=0.20 points, i.e. acc >= 0.23, and beat the no-plasticity
control by >=0.10. Persistence (predict current bin) on the same
trajectories is 0.776 (rpc_track_results.txt), so beating chance alone is
not the bar. Ridge control from tasks_C.md (continuous next-F task):
position RMSE 1.04 cm vs persistence 1.89 cm, chance 52.3 cm.
"""
import numpy as np
import torch
from rpc_v2 import run_vec
from tasks import make_track_data, leaky_integrators

K = 32
TRACK_LEN = 180.0
SEQ_LEN = 100


def make_track_F_data(n_trials=100, seed=0):
    """Sequences of (x, y) pairs: x = [F(t), F(t-1)] (200 floats),
    y = bin of position at t+1. t runs 1..98, so 98 pairs per trial."""
    d = make_track_data(track_len=TRACK_LEN, n_trials=n_trials,
                        n_steps=1800, seed=seed)
    pos = d["positions"][:, :SEQ_LEN]  # first 100 steps of each trial
    F = leaky_integrators(pos)  # (n_trials, 100, 100), float32
    bins = np.clip((pos / TRACK_LEN * K).astype(int), 0, K - 1)
    seqs = []
    for i in range(n_trials):
        seq = []
        for t in range(1, SEQ_LEN - 1):
            x = np.concatenate([F[i, t], F[i, t - 1]]).astype(np.float32)
            seq.append((x, int(bins[i, t + 1])))
        seqs.append(seq)
    return seqs


def run_one(seed, epochs=3, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_track_F_data(n_trials=100, seed=seed)
    test = make_track_F_data(n_trials=20, seed=seed + 1000)
    defaults = dict(K=K, d_in=200, theta_p=1.0, theta_d=1.0,
                    eta_b=0.05, eta_c=0.05, n_res=200, n_out=64)
    return run_vec(train, test, epochs=epochs, **{**defaults, **kw})


if __name__ == "__main__":
    import sys
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                        ("no-confirmation", {"use_confirmation": False})]:
        accs, bursts = [], []
        for seed in seeds:
            r = run_one(seed, epochs=epochs, **extra)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
        print(f"trackF {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
