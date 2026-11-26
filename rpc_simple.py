"""Simplest time-series tests for RPC, run by Muse directly.

Reza's redirect: start simple, just learning a time series first.
No random background noise, no random frequency per sequence, no
filler tokens. If RPC cannot learn these, it cannot learn anything
harder. If it can, the next step is to add noise back, one knob at a time.

Tasks (all next-token prediction on held-out test sequences):
1. cycle: tokens 0..7 repeating, random start offset per sequence.
   K=8, chance 0.125. A Markov-1 model scores 1.000 here.
2. motif: tokens 1,2,3 repeating, random start offset per sequence.
   K=4, chance 0.250. Markov-1 scores 1.000 here.
3. sine-fixed: one sine wave, fixed frequency 0.05 cycles per step,
   random phase per sequence, binned to 32. K=32, chance 0.031.
   Markov-2 scored 0.769 on random-frequency sine (baselines_B.md);
   fixed frequency is strictly easier.

Network is RPC v2 unchanged, best variant so far: prototypes built
from all train sequences (proto_n = n_train), 5 epochs, defaults
theta_p=1.0, theta_d=1.0, eta=0.05, reservoir 200, output 64.
"""
import numpy as np
import torch
from rpc_v2 import run

def make_cycle(n_seq, length=96, K=8, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_seq):
        off = int(rng.integers(0, K))
        data.append([(off + i) % K for i in range(length)])
    return data

def make_motif(n_seq, length=96, K=4, seed=0):
    rng = np.random.default_rng(seed)
    pat = [1, 2, 3]
    data = []
    for _ in range(n_seq):
        off = int(rng.integers(0, 3))
        data.append([pat[(off + i) % 3] for i in range(length)])
    return data

def make_sine_fixed(n_seq, length=200, K=32, seed=0):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_seq):
        ph = rng.uniform(0, 2 * np.pi)
        t = np.arange(length)
        x = np.sin(2 * np.pi * 0.05 * t + ph)
        bins = np.clip(((x + 1) / 2 * K).astype(int), 0, K - 1)
        data.append(bins.tolist())
    return data

TASKS = {
    "cycle": (make_cycle, 8, 100, 20, 96),
    "motif": (make_motif, 4, 100, 20, 96),
    "sine-fixed": (make_sine_fixed, 32, 50, 20, 200),
}

def run_task(task, seed, epochs=5, **kw):
    maker, K, n_train, n_test, length = TASKS[task]
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = maker(n_seq=n_train, length=length, seed=seed)
    test = maker(n_seq=n_test, length=length, seed=seed + 1000)
    defaults = dict(K=K, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, proto_n=n_train)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    chances = {"cycle": 0.125, "motif": 0.250, "sine-fixed": 0.031}
    for task in ["cycle", "motif", "sine-fixed"]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                            ("no-confirmation", {"use_confirmation": False})]:
            accs, bursts = [], []
            for seed in range(3):
                r = run_task(task, seed, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"{task} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] "
                  f"chance={chances[task]:.3f} "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
