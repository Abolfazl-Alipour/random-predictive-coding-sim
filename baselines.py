"""Baselines for the RPC toy tasks (worker B).

Answers: are the tasks learnable at all, and by how much?
Models: chance, Markov order-1 / order-2 (counts from TRAIN only),
plain echo state network (ESN, fixed random reservoir + ridge readout,
trained on TRAIN only), small LSTM trained by backprop on TRAIN only.
All evaluated on the same held-out 50-sequence test set, 5 seeds (0-4).
Train accuracy is never reported as test accuracy.

Data format mirrors rpc_toy.py: make_data(n_seq, length=24, K=16,
motifs=((1,2,3),(5,6,7),(9,10,11))), 200 train / 50 test sequences.

Run:
    ~/workspace/thesis-rpc/venv/bin/python ~/workspace/thesis-rpc/baselines.py
Output: prints the table and writes baselines_B.md next to this file.
CPU only, torch/numpy only. Runtime is a few minutes at most.
"""

import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "baselines_B.md")

N_TRAIN = 200
N_TEST = 50
LENGTH = 24
K_MOTIF = 16
MOTIFS = ((1, 2, 3), (5, 6, 7), (9, 10, 11))
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------- data
# Data are drawn with the global numpy RNG, exactly as in rpc_toy.py
# (np.random.randint / np.random.rand), after np.random.seed(1000 + s).
# Train is drawn first, then test, so both are fixed given the seed.


def make_data(n_seq=200, length=24, K=16, motifs=((1, 2, 3), (5, 6, 7), (9, 10, 11))):
    """Verbatim copy of rpc_toy.make_data: random tokens, motifs planted
    contiguously (lag 1). Later motifs overwrite earlier ones on overlap."""
    data = []
    for _ in range(n_seq):
        seq = np.random.randint(0, K, size=length).tolist()
        for motif in motifs:
            pos = np.random.randint(0, length - len(motif))
            seq[pos:pos + len(motif)] = list(motif)
        data.append(seq)
    return data


def make_data_lag(n_seq=200, length=24, K=16,
                  motifs=((1, 2, 3), (5, 6, 7), (9, 10, 11)), lag=2):
    """Motif tokens separated by (lag-1) random filler tokens.

    lag=2: 1, X, 2, X, 3   (predictive token arrives 2 steps after the cue)
    lag=5: 1, X, X, X, X, 2, X, X, X, X, 3
    Fillers are the background random tokens, left untouched. The motif
    occupies positions pos, pos+lag, pos+2*lag, so pos is drawn from
    [0, length - 2*lag - 1]. Later motifs overwrite earlier ones.
    """
    assert lag >= 1 and 2 * lag < length
    data = []
    for _ in range(n_seq):
        seq = np.random.randint(0, K, size=length).tolist()
        for motif in motifs:
            pos = np.random.randint(0, length - 2 * lag)
            for j, tok in enumerate(motif):
                seq[pos + j * lag] = tok
        data.append(seq)
    return data


def make_sine(n_seq, length=24, K=32, period=12.0):
    """Sine wave, one random phase per sequence.

    x(t) = sin(2*pi*t/period + phi), phi ~ Uniform(0, 2*pi).
    Quantised to K bins over [-1, 1]: bin = floor((x+1)/2 * K), clipped.
    Returns (bins, continuous) as lists of lists of equal shape.
    Bin centres decode as v = -1 + (bin + 0.5) / K * 2.
    """
    bins, cont = [], []
    for _ in range(n_seq):
        phi = np.random.rand() * 2 * math.pi
        t = np.arange(length, dtype=float)
        x = np.sin(2 * math.pi * t / period + phi)
        b = np.clip(np.floor((x + 1.0) / 2.0 * K).astype(int), 0, K - 1)
        bins.append(b.tolist())
        cont.append(x.tolist())
    return bins, cont


def _lorenz_step(state, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """One RK4 step of the Lorenz system."""
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_lorenz(n_seq, length=24, K=32, dt=0.01, stride=10,
                lo=-25.0, hi=25.0):
    """Lorenz x coordinate, quantised to K bins over fixed [lo, hi].

    sigma=10, rho=28, beta=8/3, RK4 with dt=0.01. Per sequence a random
    initial state (x,y in [-10,10], z in [20,30]) is integrated, the first
    10 steps are discarded as transient, then every `stride`-th step is
    sampled, so one length-24 sequence spans 24*stride*dt = 2.4 time
    units (about 2 Lorenz oscillations). Sampling every step would span
    only 0.24 time units, in which x barely moves and next-step
    prediction would be trivially solved by persistence.
    Returns (bins, continuous x) as lists of lists.
    """
    bins, cont = [], []
    for _ in range(n_seq):
        state = np.array([np.random.uniform(-10, 10),
                          np.random.uniform(-10, 10),
                          np.random.uniform(20, 30)])
        n_steps = 10 + length * stride
        xs = []
        for i in range(n_steps):
            state = _lorenz_step(state, dt)
            if i >= 10 and (i - 10) % stride == 0:
                xs.append(state[0])
        xs = np.array(xs[:length])
        b = np.clip(np.floor((xs - lo) / (hi - lo) * K).astype(int), 0, K - 1)
        bins.append(b.tolist())
        cont.append(xs.tolist())
    return bins, cont


def bin_centre_sine(b, K=32):
    return -1.0 + (b + 0.5) / K * 2.0


def bin_centre_lorenz(b, K=32, lo=-25.0, hi=25.0):
    return lo + (b + 0.5) / K * (hi - lo)


def bin_centre_range(b, K=32, lo=0.0, hi=1.0):
    """Bin centre for any continuous task binned over [lo, hi]."""
    return lo + (b + 0.5) / K * (hi - lo)


def bin_centre_narma10(b, K=32):
    return bin_centre_range(b, K, 0.0, 1.0)


def bin_centre_mackeyglass(b, K=32):
    return bin_centre_range(b, K, 0.0, 1.5)


def _narma10_seq(length, burn, lo, hi):
    """One NARMA-10 sequence (burn-in + sampled steps), or None if
    the trajectory leaves the stable basin or the bin range.

    The recurrence has an unstable fixed point near y = 1.02
    (mean-field roots 0.38 and 1.02): a run of large inputs can
    push y past it, after which the quadratic term explodes
    (measured: about 1 in 1000 sequences diverges). A divergent
    draw is not a NARMA-10 trajectory, so the caller resamples
    it; the retained sequences are the canonical system
    conditioned on staying in its stable basin, which is where
    every reported literature run lives. Draws that leave
    [lo, hi] are also resampled, so the returned values never
    clip at the bin edges.
    """
    hist = [0.0] * 10        # y(t-9) .. y(t), most recent last
    u_hist = [0.0] * 9       # u(t-9) .. u(t-1), most recent last
    for _ in range(burn):
        u = float(np.random.rand()) * 0.5
        y_new = (0.3 * hist[-1] + 0.05 * hist[-1] * sum(hist)
                 + 1.5 * u_hist[0] * u + 0.1)
        if (not np.isfinite(y_new) or abs(y_new) > 5.0
                or y_new < lo or y_new > hi):
            return None
        hist.append(y_new)
        hist.pop(0)
        u_hist.append(u)
        u_hist.pop(0)
    xs = []
    for _ in range(length):
        u = float(np.random.rand()) * 0.5
        y_new = (0.3 * hist[-1] + 0.05 * hist[-1] * sum(hist)
                 + 1.5 * u_hist[0] * u + 0.1)
        if (not np.isfinite(y_new) or abs(y_new) > 5.0
                or y_new < lo or y_new > hi):
            return None
        hist.append(y_new)
        hist.pop(0)
        u_hist.append(u)
        u_hist.pop(0)
        xs.append(y_new)
    return xs


def make_narma10(n_seq, length=100, K=32, lo=0.0, hi=1.0, burn=200):
    """NARMA-10 (Atiya and Parlos 2000): a nonlinear autoregressive
    moving-average system of order 10. The input u(t) is iid
    Uniform(0, 0.5); the output follows
        y(t+1) = 0.3*y(t) + 0.05*y(t)*sum_{i=0}^{9} y(t-i)
                 + 1.5*u(t-9)*u(t) + 0.1.
    Predicting y(t+1) needs the last 10 outputs and the driving
    input 9 steps back, so a 1- or 2-step bin lookup cannot solve
    it: the task needs genuine multi-step memory. The history
    starts at zero and 200 steps are discarded as burn-in, so
    each sequence starts from the stationary distribution.
    Returns (bins, continuous y) as lists of lists. Draws from
    the global numpy RNG (np.random.seed first), exactly like
    make_lorenz, so train/val/test draws in sequence are distinct.
    y stays inside roughly [0.15, 0.95] in the stable basin; bins
    cover [0, 1]. Divergent draws (about 1 in 1000 sequences)
    and draws that leave [lo, hi] are resampled, see
    _narma10_seq, so the returned values never clip at the bin
    edges; if rejections ever exceed the requested count the
    generator raises instead of looping forever.
    """
    bins, cont = [], []
    rejected = 0
    while len(bins) < n_seq:
        xs = _narma10_seq(length, burn, lo, hi)
        if xs is None:
            rejected += 1
            if rejected > n_seq:
                raise RuntimeError(
                    "make_narma10: too many divergent draws; "
                    "the recurrence is unstable, not just unlucky")
            continue
        xs = np.array(xs)
        b = np.clip(np.floor((xs - lo) / (hi - lo) * K).astype(int),
                    0, K - 1)
        bins.append(b.tolist())
        cont.append(xs.tolist())
    return bins, cont


def make_mackeyglass(n_seq, length=100, K=32, lo=0.0, hi=1.5,
                     delay=17, burn=1000):
    """Mackey-Glass, discrete Euler map with delay D = 17 steps,
    the standard reservoir-computing benchmark form (NOT the
    continuous delay-differential equation with tau = 17 time
    units sampled at dt = 0.1; that form has a 170-step delay).
        x(t+1) = x(t) + 0.2*x(t-D) / (1 + x(t-D)^10) - 0.1*x(t).
    The delay makes the next value depend on the state 17 steps
    back, so a 2-bin lookup cannot solve it. History starts at
    0.9 plus small noise and 1000 steps are discarded as burn-in.
    Returns (bins, continuous x). Draws from the global numpy RNG,
    like make_lorenz. x stays inside roughly [0.3, 1.4] (the map
    is globally bounded: the 0.2*x/(1+x^10) term vanishes for
    large x and -0.1*x pulls it back); bins cover [0, 1.5].
    """
    bins, cont = [], []
    for _ in range(n_seq):
        hist = list(0.9 + 0.05 * np.random.randn(delay + 1))
        for _ in range(burn):
            x_new = (hist[-1] + 0.2 * hist[0] / (1.0 + hist[0] ** 10)
                     - 0.1 * hist[-1])
            hist.append(x_new)
            hist.pop(0)
        xs = []
        for _ in range(length):
            x_new = (hist[-1] + 0.2 * hist[0] / (1.0 + hist[0] ** 10)
                     - 0.1 * hist[-1])
            hist.append(x_new)
            hist.pop(0)
            xs.append(x_new)
        xs = np.array(xs)
        b = np.clip(np.floor((xs - lo) / (hi - lo) * K).astype(int),
                    0, K - 1)
        bins.append(b.tolist())
        cont.append(xs.tolist())
    return bins, cont


# ---------------------------------------------------------------- models
def chance_acc(K):
    return 1.0 / K


def markov_counts(train, K, order):
    """Count tables from TRAIN only. Order 1: counts[prev, next].
    Order 2: counts[prev2, prev1, next]."""
    if order == 1:
        c = np.zeros((K, K))
        for seq in train:
            for a, b in zip(seq[:-1], seq[1:]):
                c[a, b] += 1
        return c
    c = np.zeros((K, K, K))
    for seq in train:
        for a, b, d in zip(seq[:-2], seq[1:-1], seq[2:]):
            c[a, b, d] += 1
    return c


def markov_predict(train, test, K, order, alpha=0.1):
    """Next-token prediction from TRAIN counts only (add-alpha smoothing).

    Order 1: P(next|prev) with smoothing alpha.
    Order 2: P(next|prev2,prev1) with smoothing alpha; an unseen 2-token
    context backs off to the order-1 distribution P(next|prev1).
    Returns list of predicted tokens per test sequence (positions 1..)."""
    c1 = markov_counts(train, K, 1)
    p1 = (c1 + alpha) / (c1.sum(axis=1, keepdims=True) + alpha * K)
    if order == 1:
        return [[int(np.argmax(p1[seq[t]])) for t in range(len(seq) - 1)]
                for seq in test]
    c2 = markov_counts(train, K, 2)
    seen2 = c2.sum(axis=2)  # (K, K) counts of each 2-token context
    p2 = (c2 + alpha) / (seen2[:, :, None] + alpha * K)
    out = []
    for seq in test:
        preds = []
        for t in range(len(seq) - 1):
            if t == 0:
                preds.append(int(np.argmax(p1[seq[0]])))
            elif seen2[seq[t - 1], seq[t]] > 0:
                preds.append(int(np.argmax(p2[seq[t - 1], seq[t]])))
            else:
                preds.append(int(np.argmax(p1[seq[t]])))
        out.append(preds)
    return out


class ESN:
    """Plain echo state network: fixed random reservoir, ridge readout.

    r_t = (1-a) r_{t-1} + a tanh(W_in x_t + W_res r_{t-1}), x one-hot.
    Readout W_out (ridge, lambda) maps reservoir state after token t to
    a distribution over token t+1. State is reset per sequence.
    Hyperparameters: N_r=300, spectral radius rho=0.9, leak a=0.3,
    input scaling 1.0, ridge lambda=1.0, washout=2 steps per sequence.
    Reservoir weights drawn from a numpy RandomState seeded 2000 + seed,
    so the reservoir differs per seed but is fixed within a seed.
    """

    def __init__(self, K, seed, n_res=300, rho=0.9, leak=0.3,
                 input_scale=1.0, ridge=1.0, washout=2):
        self.K, self.n_res = K, n_res
        self.leak, self.ridge, self.washout = leak, ridge, washout
        rng = np.random.RandomState(2000 + seed)
        self.W_in = rng.randn(n_res, K) * input_scale / math.sqrt(K)
        W = rng.randn(n_res, n_res) * 0.1
        ev = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_res = W / ev * rho

    def _states(self, seqs):
        """Reservoir states after each token; returns (X, Y) with one row
        per (sequence, position t >= washout), Y = one-hot of token t+1."""
        X, Y = [], []
        for seq in seqs:
            r = np.zeros(self.n_res)
            for t, tok in enumerate(seq[:-1]):
                x = np.zeros(self.K)
                x[tok] = 1.0
                r = ((1 - self.leak) * r
                     + self.leak * np.tanh(self.W_in @ x + self.W_res @ r))
                if t >= self.washout:
                    X.append(r.copy())
                    y = np.zeros(self.K)
                    y[seq[t + 1]] = 1.0
                    Y.append(y)
        return np.array(X), np.array(Y)

    def fit(self, train):
        X, Y = self._states(train)
        A = X.T @ X + self.ridge * np.eye(self.n_res)
        self.W_out = np.linalg.solve(A, X.T @ Y)  # (n_res, K)
        return self

    def predict(self, test):
        """Predicted next token for positions 1..23 of each test sequence
        (state after token t predicts token t+1)."""
        out = []
        for seq in test:
            r = np.zeros(self.n_res)
            preds = []
            for t, tok in enumerate(seq[:-1]):
                x = np.zeros(self.K)
                x[tok] = 1.0
                r = ((1 - self.leak) * r
                     + self.leak * np.tanh(self.W_in @ x + self.W_res @ r))
                preds.append(int(np.argmax(r @ self.W_out)))
            out.append(preds)
        return out


class LSTMModel(torch.nn.Module):
    def __init__(self, K, hidden=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(K, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden, K)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


def lstm_predict(train, test, K, seed, hidden=64, epochs=20, lr=0.005,
                 batch=32):
    """Small LSTM trained by backprop (Adam) on TRAIN only.

    Input: one-hot of token t; target: token t+1 (positions 0..22).
    Loss: cross-entropy summed over positions. Hidden state zeroed per
    batch (sequences are independent). torch.manual_seed(seed) fixes
    init and batch shuffling. Returns predicted tokens per test seq."""
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = LSTMModel(K, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xtr = torch.nn.functional.one_hot(
        torch.tensor(train, dtype=torch.long), K).float()
    Ytr = torch.tensor(train, dtype=torch.long)
    n = len(train)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(Xtr[idx])[:, :-1, :]          # predict t+1
            loss = loss_fn(logits.reshape(-1, K), Ytr[idx][:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        Xte = torch.nn.functional.one_hot(
            torch.tensor(test, dtype=torch.long), K).float()
        logits = model(Xte)[:, :-1, :]
        return torch.argmax(logits, dim=-1).tolist()


# ---------------------------------------------------------------- scoring
def accuracy(preds, test):
    """Fraction correct over positions 1..23 (preds[t] vs test[t+1])."""
    correct = total = 0
    for p, seq in zip(preds, test):
        for t, pr in enumerate(p):
            total += 1
            correct += int(pr == seq[t + 1])
    return correct / total


def persistence_preds(test):
    """Predict the last token (repeat token t as the prediction for t+1)."""
    return [[seq[t] for t in range(len(seq) - 1)] for seq in test]


def mse_from_bin_preds(preds, cont, decode):
    """MSE between decoded predicted bin centre and true continuous
    next value. preds[t] predicts token t+1, cont[t+1] is its true value."""
    se, n = 0.0, 0
    for p, seq in zip(preds, cont):
        for t, pr in enumerate(p):
            se += (decode(pr) - seq[t + 1]) ** 2
            n += 1
    return se / n


def persistence_mse(cont):
    """MSE of predicting the last continuous value."""
    se, n = 0.0, 0
    for seq in cont:
        for t in range(len(seq) - 1):
            se += (seq[t] - seq[t + 1]) ** 2
            n += 1
    return se / n


def mean_predictor_mse(train_cont, test_cont):
    """MSE of predicting the TRAIN mean continuous value (constant)."""
    mu = float(np.mean([v for seq in train_cont for v in seq]))
    se, n = 0.0, 0
    for seq in test_cont:
        for t in range(1, len(seq)):
            se += (mu - seq[t]) ** 2
            n += 1
    return se / n


def summarise(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.min()), float(v.max())


# ---------------------------------------------------------------- tasks
def load_task(name, seed):
    """Returns (train_bins, test_bins, train_cont, test_cont, K).
    cont is None for motif tasks. RNG state is set here per seed."""
    np.random.seed(1000 + seed)
    if name == "motif lag 1":
        return (make_data(N_TRAIN, LENGTH, K_MOTIF, MOTIFS),
                make_data(N_TEST, LENGTH, K_MOTIF, MOTIFS), None, None, K_MOTIF)
    if name == "motif lag 2":
        return (make_data_lag(N_TRAIN, LENGTH, K_MOTIF, MOTIFS, lag=2),
                make_data_lag(N_TEST, LENGTH, K_MOTIF, MOTIFS, lag=2),
                None, None, K_MOTIF)
    if name == "motif lag 5":
        return (make_data_lag(N_TRAIN, LENGTH, K_MOTIF, MOTIFS, lag=5),
                make_data_lag(N_TEST, LENGTH, K_MOTIF, MOTIFS, lag=5),
                None, None, K_MOTIF)
    if name == "sine (32 bins)":
        tr_b, tr_c = make_sine(N_TRAIN, LENGTH, 32)
        te_b, te_c = make_sine(N_TEST, LENGTH, 32)
        return tr_b, te_b, tr_c, te_c, 32
    if name == "Lorenz x (32 bins)":
        tr_b, tr_c = make_lorenz(N_TRAIN, LENGTH, 32)
        te_b, te_c = make_lorenz(N_TEST, LENGTH, 32)
        return tr_b, te_b, tr_c, te_c, 32
    raise ValueError(name)


TASKS = ["motif lag 1", "motif lag 2", "motif lag 5",
         "sine (32 bins)", "Lorenz x (32 bins)"]


def run_seed(task, seed):
    train, test, train_c, test_c, K = load_task(task, seed)
    res = {"chance": chance_acc(K)}
    res["markov1"] = accuracy(markov_predict(train, test, K, 1), test)
    res["markov2"] = accuracy(markov_predict(train, test, K, 2), test)
    esn = ESN(K, seed).fit(train)
    res["esn"] = accuracy(esn.predict(test), test)
    lstm_p = lstm_predict(train, test, K, seed)
    res["lstm"] = accuracy(lstm_p, test)
    if test_c is not None:
        decode = bin_centre_sine if "sine" in task else bin_centre_lorenz
        pers_p = persistence_preds(test)
        res["persist_acc"] = accuracy(pers_p, test)
        res["persist_mse"] = persistence_mse(test_c)
        res["mean_mse"] = mean_predictor_mse(train_c, test_c)
        res["esn_mse"] = mse_from_bin_preds(esn.predict(test), test_c, decode)
        res["lstm_mse"] = mse_from_bin_preds(lstm_p, test_c, decode)
        res["markov2_mse"] = mse_from_bin_preds(
            markov_predict(train, test, K, 2), test_c, decode)
        res["markov1_mse"] = mse_from_bin_preds(
            markov_predict(train, test, K, 1), test_c, decode)
    return res


def main():
    all_res = {task: [] for task in TASKS}
    for task in TASKS:
        for seed in SEEDS:
            r = run_seed(task, seed)
            all_res[task].append(r)
            print(f"[{task}] seed {seed}: " +
                  " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

    lines = []
    lines.append("# Baselines for the RPC toy tasks (worker B)")
    lines.append("")
    lines.append("Question: are the tasks learnable at all, and by how much?")
    lines.append("Every number below is TEST accuracy on the same held-out "
                 "50-sequence test set (200 train / 50 test, length 24), "
                 "mean / min / max across 5 seeds (0-4). Train accuracy is "
                 "never reported as test accuracy. No claim is made here "
                 "about RPC itself; this establishes the ceiling and the "
                 "controls only.")
    lines.append("")
    lines.append("## Accuracy (next-token, fraction correct)")
    lines.append("")
    lines.append("| task | K | chance | Markov-1 | Markov-2 | ESN | LSTM | persistence |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        rs = all_res[task]
        K = 16 if "motif" in task else 32

        def fmt(key):
            m, lo, hi = summarise([r[key] for r in rs])
            return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

        pers = fmt("persist_acc") if "persist_acc" in rs[0] else "n/a"
        lines.append(f"| {task} | {K} | {fmt('chance')} | {fmt('markov1')} | "
                     f"{fmt('markov2')} | {fmt('esn')} | {fmt('lstm')} | {pers} |")
    lines.append("")
    lines.append("Chance = 1/K (0.0625 for K=16 motifs, 0.03125 for K=32). "
                 "Persistence = repeat the last token/bin.")
    lines.append("")
    lines.append("## Continuous MSE (sine / Lorenz only)")
    lines.append("")
    lines.append("MSE between the predicted value and the true continuous "
                 "next value. Model predictions are decoded from the "
                 "predicted bin centre (sine: bins over [-1, 1]; Lorenz x: "
                 "bins over [-25, 25]). Persistence = predict the last "
                 "continuous value. Mean = predict the TRAIN mean "
                 "(constant). Lower is better.")
    lines.append("")
    lines.append("| task | persistence | TRAIN-mean | Markov-1 | Markov-2 | ESN | LSTM |")
    lines.append("|---|---|---|---|---|---|---|")
    for task in ["sine (32 bins)", "Lorenz x (32 bins)"]:
        rs = all_res[task]

        def fmt(key):
            m, lo, hi = summarise([r[key] for r in rs])
            return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"

        lines.append(f"| {task} | {fmt('persist_mse')} | {fmt('mean_mse')} | "
                     f"{fmt('markov1_mse')} | {fmt('markov2_mse')} | "
                     f"{fmt('esn_mse')} | {fmt('lstm_mse')} |")
    lines.append("")
    lines.append("## Per-seed test accuracy")
    lines.append("")
    for task in TASKS:
        rs = all_res[task]
        keys = ["markov1", "markov2", "esn", "lstm"]
        if "persist_acc" in rs[0]:
            keys.append("persist_acc")
        lines.append(f"### {task} (chance {1 / (16 if 'motif' in task else 32):.4f})")
        lines.append("")
        lines.append("| seed | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for seed, r in zip(SEEDS, rs):
            lines.append("| " + str(seed) + " | " +
                         " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
        lines.append("")
    lines.append("## Per-seed continuous MSE (sine / Lorenz)")
    lines.append("")
    for task in ["sine (32 bins)", "Lorenz x (32 bins)"]:
        rs = all_res[task]
        keys = ["persist_mse", "mean_mse", "markov1_mse", "markov2_mse",
                "esn_mse", "lstm_mse"]
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| seed | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---|" * len(keys))
        for seed, r in zip(SEEDS, rs):
            lines.append("| " + str(seed) + " | " +
                         " | ".join(f"{r[k]:.4f}" for k in keys) + " |")
        lines.append("")
    lines.append("## Exact hyperparameters and data")
    lines.append("")
    lines.append("1. Data: 200 train / 50 test sequences, length 24, drawn "
                 "per seed with `np.random.seed(1000 + seed)`, train first "
                 "then test, using the global numpy RNG exactly as in "
                 "`rpc_toy.make_data`.")
    lines.append("2. Motif lag 1: `make_data` verbatim from `rpc_toy.py` "
                 "(K=16, motifs (1,2,3), (5,6,7), (9,10,11), planted "
                 "contiguously; later motifs overwrite earlier ones).")
    lines.append("3. Motif lag 2 / lag 5: same background and motifs, but "
                 "motif tokens sit at positions pos, pos+lag, pos+2*lag, "
                 "with the (lag-1) positions between them left as random "
                 "filler tokens; pos drawn from [0, length-2*lag). "
                 "Lag 2: `1, X, 2, X, 3`. Lag 5: `1, X, X, X, X, 2, X, X, "
                 "X, X, 3`.")
    lines.append("4. Sine: x(t) = sin(2*pi*t/12 + phi), phi ~ Uniform(0, "
                 "2*pi) per sequence, t = 0..23 (period 12 steps, 2 cycles "
                 "per sequence), quantised to 32 bins over [-1, 1], bin "
                 "centre -1 + (bin+0.5)/32*2.")
    lines.append("5. Lorenz: sigma=10, rho=28, beta=8/3, RK4, dt=0.01, x "
                 "coordinate only, quantised to 32 bins over fixed "
                 "[-25, 25], bin centre -25 + (bin+0.5)/32*50. Per sequence "
                 "a random initial state (x,y in [-10,10], z in [20,30]) "
                 "is integrated; the first 10 steps are discarded, then "
                 "every 10th step is sampled, so a length-24 sequence "
                 "spans 2.4 time units. Sampling every step would span "
                 "only 0.24 time units, in which x barely moves and "
                 "persistence would solve the task trivially.")
    lines.append("6. Chance: 1/K. Markov-1 / Markov-2: counts from TRAIN "
                 "only, add-alpha smoothing alpha=0.1; order-2 backs off "
                 "to the order-1 distribution for unseen 2-token contexts.")
    lines.append("7. ESN: N_r=300, spectral radius 0.9, leak 0.3, input "
                 "scaling 1.0, tanh reservoir, ridge lambda=1.0, washout "
                 "2 steps per sequence, state reset per sequence, readout "
                 "trained by ridge regression on TRAIN states only. "
                 "Reservoir weights from `np.random.RandomState(2000 + "
                 "seed)`.")
    lines.append("8. LSTM: 1 layer, hidden 64, one-hot input (size K), "
                 "linear head to K logits, Adam lr=0.005, batch 32, "
                 "epochs 20, cross-entropy over positions 0..22 predicting "
                 "token t+1, hidden state zeroed per batch, "
                 "`torch.manual_seed(seed)` (fixes init and shuffling).")
    lines.append("9. Scoring: accuracy counts positions 1..23 of each test "
                 "sequence (prediction of token t uses tokens 0..t-1 "
                 "only). MSE uses the true continuous next value.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("~/workspace/thesis-rpc/venv/bin/python "
                 "~/workspace/thesis-rpc/baselines.py")
    lines.append("```")
    lines.append("")
    lines.append("CPU only (torch 2.15.0+cpu, numpy 2.5.4). This file was "
                 "written by `baselines.py`; re-running it regenerates it.")
    lines.append("")

    text = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(text)
    print("\nWrote " + OUT_MD)


if __name__ == "__main__":
    main()
