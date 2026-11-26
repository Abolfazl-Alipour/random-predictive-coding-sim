"""RPC v2 on real temporal signals, run by Muse directly (no Kimi, no AGY).

Reuses RPCNetwork from rpc_v2.py unchanged. Input is a binned token
sequence, K=32 bins, so the network is exactly the motif network, only
the data change: sine waves (each sequence its own frequency and phase)
and Lorenz x (chaotic). Prediction is next token, scored on held-out
test sequences, 5 seeds when run in full, 1 seed for the pilot.

Chance = 1/32 = 0.03125. Baselines from baselines_B.md (same binning,
5 seeds): sine Markov-2 0.769, ESN 0.900, LSTM 0.894; Lorenz Markov-2
0.559, ESN 0.388, LSTM 0.391. RPC pass bar (set by the coordinator
brief): beat chance by >=0.20 points, i.e. sine/Lorenz acc >= 0.23,
and beat the no-plasticity control by >=0.10.
"""
import numpy as np
import torch
from rpc_v2 import RPCNetwork, run

K = 32

def make_sine(n_seq=100, length=100, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_seq):
        f = rng.uniform(0.02, 0.08)
        ph = rng.uniform(0, 2 * np.pi)
        t = np.arange(length)
        x = np.sin(2 * np.pi * f * t + ph)
        bins = np.clip(((x + 1) / 2 * K).astype(int), 0, K - 1)
        data.append(bins.tolist())
    return data

def make_lorenz(n_seq=20, length=100, seed=0):
    # Lorenz x only, sigma=10, rho=28, beta=8/3, dt=0.02, RK4, 500 burn-in.
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_seq):
        x, y, z = rng.uniform(-10, 10, 3)
        dt = 0.02
        def f(s):
            return np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1],
                             s[0] * s[1] - (8 / 3) * s[2]])
        s = np.array([x, y, z], dtype=float)
        for _ in range(500):
            k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs = []
        for _ in range(length):
            k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xs.append(s[0])
        xs = np.array(xs)
        bins = np.clip(((xs + 20) / 40 * K).astype(int), 0, K - 1)
        data.append(bins.tolist())
    return data

def run_one(task, seed, epochs=3, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    maker = make_sine if task == "sine" else make_lorenz
    n_train = 100 if task == "sine" else 20
    n_test = 20 if task == "sine" else 10
    train = maker(n_seq=n_train, seed=seed)
    test = maker(n_seq=n_test, seed=seed + 1000)
    defaults = dict(K=K, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    import sys, json
    task = sys.argv[1] if len(sys.argv) > 1 else "sine"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                        ("no-confirmation", {"use_confirmation": False})]:
        accs, bursts = [], []
        for seed in seeds:
            r = run_one(task, seed, epochs=epochs, **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"{task} {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
