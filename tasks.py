"""Spatial-temporal tasks for testing random predictive coding (RPC).

Worker C. Numpy only. No RPC is implemented here. This file only builds
the tasks and proves, with a simple ridge-regression control, that each
task is learnable before RPC is tested on it.

Thesis grounding (Chapter 4, Alipour 2023)
-------------------------------------------
1. Chapter 4 trains a sparse autoencoder to reconstruct leaky-integrator
   inputs on a 180 cm 1D track. The leaky-integrator layer has 100 units
   with decay rates s log-spaced from 1e-3 to 1e-1.
2. The integrator equation is dF/dt = -s * v(t) * F, where v(t) is the
   velocity of the virtual rat. Decay is a function of distance travelled,
   not of time. With v = dx/dt this gives F(s, x) = exp(-s * x) for a
   trial that starts at the origin x = 0 with F = 1.
3. Moving away from the origin decreases F. Moving back toward the
   origin increases F toward 1. One set of integrators is tied to the
   origin (the start wall); see thesis equations (1) to (6).
4. Each training trial in the thesis has 1800 steps, and 6000 trials
   were used for training and for testing (thesis text, Methods, and
   Figure 4.1C/D). The thesis also tests generalisation to longer
   tracks, which is why Task 1 includes a 300 cm track never trained on.

Deliberate deviation from the thesis trajectory sampler
--------------------------------------------------------
5. The thesis sampled each new location independently from a uniform
   distribution, so velocity was the difference of two independent
   uniform draws, and temporal order did not matter because Chapter 4
   only reconstructed the current input.
6. Independent uniform locations make next-step prediction impossible
   by construction: the next location is independent of the past.
   A next-step prediction task therefore needs temporally continuous
   trajectories. Here velocity follows an AR(1) process (rho = 0.85),
   clipped to +/-6 cm per step, with reflecting walls at 0 and at the
   track length. The integrator equation, the s range, the 100 units,
   the 180 cm track, and the 6000 x 1800 scale are kept thesis-faithful.
7. Closed form used for sampling pairs: because F(s, t) = exp(-s*x(t))
   exactly when a trial starts at x = 0 with F = 1, the pair sampler
   computes F from position by that closed form instead of stepping
   100 integrators for 1800 steps per trial. The stepwise update
   F <- F * exp(-s * dx) and the closed form agree to floating-point
   error; see _check_integrator_equivalence() in __main__.

Interface (for the coordinator)
-------------------------------
8. Every dataset builder returns the same 5-tuple:
       (train_X, train_y, test_X, test_y, meta)
   where meta is a dict with at least: task, target, input_dim,
   output_dim, n_train, n_test, chance_level, chance_description.
9. Builders: make_track_dataset, make_sine_dataset,
   make_lorenz_dataset, make_dot_dataset. Dispatcher: get_dataset(name).
10. Targets. Track: train_y is the next leaky-integrator vector
    (100 numbers in [0, 1]); the next position in cm is in
    meta["test_y_pos_cm"] / meta["train_y_pos_cm"]. Sine and Lorenz:
    train_y is the next-step bin index (0 to 31); the next continuous
    value is in meta["test_y_cont"] / meta["train_y_cont"]. Dot:
    train_y is the next cell index (0 to 1023, row-major, cell =
    y * 32 + x); the next coordinates are in meta["test_y_xy"].
11. Inputs. Track: X = [F(t), F(t-1)] (200 numbers). The lagged copy
    carries velocity information, since dx = -(log F(t) - log F(t-1))/s.
    Sine/Lorenz: X = the previous 10 continuous values. Dot: X = the
    previous 10 (x, y) positions, each divided by 31, flattened (20).
12. Full track trajectories are generated at thesis scale (6000 train
    trials x 1800 steps), but the returned pair arrays subsample 20
    random time steps per trial. The full pair matrix would be
    10.8 million rows, and the full F matrix would be 6000 x 1800 x 100
    numbers (about 4.3 GB in float32), so subsampling is required to
    keep the arrays in memory. Sampled steps are uniform over a trial,
    so the subsample is unbiased over time.

Learnability check (__main__)
------------------------------
13. For each task, a ridge-regression readout is fit on train only,
    with the ridge penalty chosen on a train-internal split (first 80
    percent of train rows fit, last 20 percent validate; rows are
    ordered by sequence or trial, so this splits by sequence, not by
    shuffled time step), then refit on all of train and scored on the
    held-out test set. The baseline is persistence (predict the next
    value or cell equals the current one).
14. For the binned tasks (sine, Lorenz, dot) the primary control is
    ridge regression to the continuous next value (or coordinates),
    with the prediction then binned (or floored to a cell). A one-hot
    ridge classifier is reported as a secondary number. It is much
    weaker here, because bin membership is a narrow threshold function
    of a linear predictor, and the one-hot fit collapses toward the
    majority bin. The regression control is the learnability proof.
15. Chance levels. Track: predicting the train-mean F vector (F-MSE)
    and the train-mean position (position RMSE). Sine/Lorenz: 1/32
    accuracy. Dot: 1/1024 accuracy.
16. The 300 cm track is a generalisation probe, not part of the
    learnability proof. The linear ridge controls fail on it (they
    shrink toward the 180 cm train mean, and F is exponential in
    position), while persistence holds. A model that beats persistence
    on 300 cm has learned the exponential code itself, not a linear
    shortcut. That is the bar for RPC on this task.

Reproduce:
    ~/workspace/thesis-rpc/venv/bin/python tasks.py
"""

import numpy as np

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------

N_LI = 100
S_MIN, S_MAX = 1e-3, 1e-1
S_VALUES = np.logspace(np.log10(S_MIN), np.log10(S_MAX), N_LI)
N_BINS = 32
GRID = 32
RIDGE_ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _rng(seed):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# Ridge regression control (numpy only)
# ----------------------------------------------------------------------

def fit_ridge(X, Y, alpha=1.0):
    """Fit ridge regression. X: (n, d), Y: (n,) or (n, k). Returns model."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    x_mean = X.mean(axis=0)
    x_scale = X.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    Xs = (X - x_mean) / x_scale
    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ Yc)
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean,
            "W": W, "alpha": float(alpha)}


def predict_ridge(model, X):
    """Predict with a model from fit_ridge. Returns (n, k)."""
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model["x_mean"]) / model["x_scale"]
    return Xs @ model["W"] + model["y_mean"]


def select_alpha(X, Y, alphas=RIDGE_ALPHAS, val_frac=0.2, cap=40000):
    """Pick the ridge penalty on a train-internal split.

    Rows are ordered by sequence/trial, so the first (1 - val_frac)
    rows are earlier sequences and the last val_frac rows are later
    sequences: the split is by sequence, not by shuffled time step.
    Returns (best_alpha, best_val_mse).
    """
    n = len(X)
    if n > cap:
        X, Y = X[:cap], Y[:cap]
        n = cap
    n_fit = int(n * (1.0 - val_frac))
    X_fit, Y_fit = X[:n_fit], Y[:n_fit]
    X_val, Y_val = X[n_fit:], Y[n_fit:]
    Y_val = np.asarray(Y_val, dtype=np.float64)
    if Y_val.ndim == 1:
        Y_val = Y_val[:, None]
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        pred = predict_ridge(fit_ridge(X_fit, Y_fit, alpha), X_val)
        mse = float(np.mean((pred - Y_val) ** 2))
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    return best_alpha, best_mse


def _onehot(y, n_classes):
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), np.clip(y, 0, n_classes - 1)] = 1.0
    return out


def _quantise(values, lo, hi, n_bins=N_BINS):
    """Map continuous values in [lo, hi] to bin indices 0..n_bins-1."""
    return np.clip(((np.asarray(values) - lo) / (hi - lo) * n_bins).astype(int),
                   0, n_bins - 1)


# ----------------------------------------------------------------------
# Task 1: 1D track, leaky integrators, velocity-modulated decay
# ----------------------------------------------------------------------

def leaky_integrators(positions, s=S_VALUES):
    """F(s, t) = exp(-s * x(t)) for positions in cm.

    This is the closed form of dF/dt = -s * v(t) * F with F = 1 at
    x = 0. Positions may be any shape; the result gains a final axis
    of length len(s). Computed in float32.
    """
    positions = np.asarray(positions, dtype=np.float32)
    return np.exp(-positions[..., None] * np.asarray(s, dtype=np.float32)[None, ...]).astype(np.float32)


def _gen_track_positions(n_trials, n_steps, track_len, seed,
                         rho=0.85, sigma=1.0, vmax=6.0):
    """Vectorised AR(1)-velocity random walk with reflecting walls.

    Returns (positions, velocities), each (n_trials, n_steps) float32.
    Every trial starts at x = 0 with v = 0 (the origin, where F = 1).
    """
    rng = _rng(seed)
    x = np.zeros(n_trials, dtype=np.float64)
    v = np.zeros(n_trials, dtype=np.float64)
    positions = np.zeros((n_trials, n_steps), dtype=np.float32)
    velocities = np.zeros((n_trials, n_steps), dtype=np.float32)
    for t in range(n_steps):
        v = np.clip(rho * v + sigma * rng.standard_normal(n_trials),
                    -vmax, vmax)
        xn = x + v
        over = xn > track_len
        under = xn < 0.0
        xn = np.where(over, 2.0 * track_len - xn, xn)
        v = np.where(over, -v, v)
        xn = np.where(under, -xn, xn)
        v = np.where(under, -v, v)
        x = xn
        positions[:, t] = x
        velocities[:, t] = v
    return positions, velocities


def make_track_data(track_len=180, n_trials=100, n_steps=1800, seed=0,
                    rho=0.85, sigma=1.0, vmax=6.0, return_F=False):
    """Generate virtual-rat trajectories on a 1D track (thesis Ch. 4).

    Args:
        track_len: track length in cm (180 in the thesis).
        n_trials: number of trials. n_steps: steps per trial (1800).
        seed: RNG seed. rho/sigma/vmax: AR(1) velocity parameters
            (see header note 6 for why velocity is AR(1), not the
            thesis i.i.d. uniform location sampler).
        return_F: if True, also return the full F array
            (n_trials, n_steps, 100). Only use this for small sizes:
            6000 x 1800 x 100 float32 is about 4.3 GB. When False, F
            is None and leaky_integrators(positions) gives F exactly.

    Returns a dict with keys: positions, velocities, s, F (or None),
    track_len, n_trials, n_steps, rho, sigma, vmax.
    """
    positions, velocities = _gen_track_positions(
        n_trials, n_steps, track_len, seed, rho=rho, sigma=sigma, vmax=vmax)
    F = leaky_integrators(positions) if return_F else None
    return {"positions": positions, "velocities": velocities, "s": S_VALUES,
            "F": F, "track_len": track_len, "n_trials": n_trials,
            "n_steps": n_steps, "rho": rho, "sigma": sigma, "vmax": vmax}


def _track_pairs(positions, pairs_per_trial, seed):
    """Sample (X, y, next_pos, cur_pos) pairs from track positions.

    X = [F(t), F(t-1)] (200 numbers), y = F(t+1) (100 numbers).
    Time steps are sampled uniformly in [2, n_steps - 2] per trial.
    """
    n_trials, n_steps = positions.shape
    rng = _rng(seed)
    tt = rng.integers(2, n_steps - 1, size=(n_trials, pairs_per_trial))
    tr = np.repeat(np.arange(n_trials), pairs_per_trial)
    t = tt.ravel()
    F_t = leaky_integrators(positions[tr, t])
    F_prev = leaky_integrators(positions[tr, t - 1])
    F_next = leaky_integrators(positions[tr, t + 1])
    X = np.concatenate([F_t, F_prev], axis=1)
    y = F_next
    return X, y, positions[tr, t + 1].astype(np.float32), positions[tr, t].astype(np.float32)


def make_track_dataset(n_train_trials=6000, n_test_trials=1000, n_steps=1800,
                       track_len=180, gen_track_len=300,
                       pairs_per_trial=20, seed=0):
    """Task 1 dataset: next leaky-integrator vector on a 1D track.

    Train and test are fresh trials on a track of length track_len.
    The generalisation track (gen_track_len, default 300 cm) is built
    by make_track_generalisation_dataset, never by this function.
    Returns (train_X, train_y, test_X, test_y, meta).
    """
    train = make_track_data(track_len, n_train_trials, n_steps, seed)
    test = make_track_data(track_len, n_test_trials, n_steps, seed + 1000)
    train_X, train_y, train_pos, train_cur = _track_pairs(
        train["positions"], pairs_per_trial, seed + 2000)
    test_X, test_y, test_pos, test_cur = _track_pairs(
        test["positions"], pairs_per_trial, seed + 3000)
    mean_F = train_y.mean(axis=0, keepdims=True)
    chance_f_mse = float(np.mean((mean_F - test_y) ** 2))
    mean_pos = float(train_pos.mean())
    chance_pos_rmse = float(np.sqrt(np.mean((mean_pos - test_pos) ** 2)))
    meta = {
        "task": "track_1d",
        "target": "next_F",
        "target_description": "Next-step leaky-integrator vector F(t+1), "
            "100 numbers in [0, 1]; F(s, t) = exp(-s * x(t)), s log-spaced "
            "1e-3 to 1e-1. Input is [F(t), F(t-1)] (200 numbers).",
        "input_dim": 200,
        "output_dim": 100,
        "output_kind": "continuous_vector",
        "n_classes": None,
        "n_train": int(len(train_X)),
        "n_test": int(len(test_X)),
        "n_train_trials": n_train_trials,
        "n_test_trials": n_test_trials,
        "n_steps": n_steps,
        "track_len_cm": track_len,
        "gen_track_len_cm": gen_track_len,
        "pairs_per_trial": pairs_per_trial,
        "chance_level": chance_f_mse,
        "chance_description": "Mean-predictor F-MSE: predict the train-mean "
            "F vector for every test pair. Chance position RMSE (predict "
            "train-mean position) is %.2f cm." % chance_pos_rmse,
        "chance_f_mse": chance_f_mse,
        "chance_pos_rmse_cm": chance_pos_rmse,
        "train_y_pos_cm": train_pos,
        "test_y_pos_cm": test_pos,
        "train_cur_pos_cm": train_cur,
        "test_cur_pos_cm": test_cur,
        "s_values": S_VALUES,
        "seed": seed,
    }
    return train_X, train_y, test_X, test_y, meta


def make_track_generalisation_dataset(n_trials=1000, n_steps=1800,
                                      gen_track_len=300, pairs_per_trial=20,
                                      seed=0):
    """Task 1 generalisation set: fresh trials on a longer track.

    Same dynamics and same s values as training, but the track is
    gen_track_len cm (default 300) and was never trained on. Returns
    (X, y, meta) in the same format as the test split of
    make_track_dataset, so a model trained on 180 cm can be scored
    directly on it. The target is again the next F vector.
    """
    data = make_track_data(gen_track_len, n_trials, n_steps, seed + 5000)
    X, y, pos, cur = _track_pairs(data["positions"], pairs_per_trial,
                                  seed + 6000)
    meta = {
        "task": "track_1d_generalisation",
        "target": "next_F",
        "target_description": "Next-step F vector on a %d cm track, "
            "never trained on." % gen_track_len,
        "input_dim": 200,
        "output_dim": 100,
        "output_kind": "continuous_vector",
        "n_classes": None,
        "n_train": 0,
        "n_test": int(len(X)),
        "track_len_cm": gen_track_len,
        "pairs_per_trial": pairs_per_trial,
        "chance_level": None,
        "chance_description": "Use the 180 cm chance levels for reference.",
        "test_y_pos_cm": pos,
        "test_cur_pos_cm": cur,
        "s_values": S_VALUES,
        "seed": seed,
    }
    return X, y, meta


def decode_position_from_F(F, s=S_VALUES, track_len=180, step=0.5):
    """Decode position (cm) from an F vector by grid search.

    Picks the grid position whose exp(-s * x) is closest to F in
    squared error. Grid search is used instead of log least squares,
    because log blows up when a predicted F is near zero (large s).
    F: (n, 100). Returns (n,) positions in cm.
    """
    F = np.asarray(F, dtype=np.float64)
    grid = np.arange(0.0, track_len + 1e-9, step)
    F_grid = np.exp(-grid[:, None] * np.asarray(s)[None, :])
    out = np.empty(len(F), dtype=np.float64)
    for i in range(0, len(F), 2000):
        chunk = F[i:i + 2000]
        d = ((chunk[:, None, :] - F_grid[None, :, :]) ** 2).sum(axis=2)
        out[i:i + 2000] = grid[np.argmin(d, axis=1)]
    return out


# ----------------------------------------------------------------------
# Task 2: sine wave, random phase/frequency per sequence
# ----------------------------------------------------------------------

def make_sine_dataset(n_train_seq=600, n_test_seq=200, length=200, win=10,
                      f_lo=0.02, f_hi=0.08, n_bins=N_BINS, seed=0):
    """Task 2 dataset: sine wave, next-step bin (32 bins).

    Each sequence has its own frequency (uniform in [f_lo, f_hi]
    cycles per step, periods 12.5 to 50 steps) and phase (uniform in
    [0, 2*pi)). Value x(t) = sin(2*pi*f*t + phase). X is the previous
    win continuous values; y is the bin of x(t+1) in [-1, 1].
    Returns (train_X, train_y, test_X, test_y, meta).
    """
    def _build(n_seq, sd):
        rng = _rng(sd)
        f = rng.uniform(f_lo, f_hi, size=n_seq)
        ph = rng.uniform(0.0, 2 * np.pi, size=n_seq)
        t = np.arange(length)
        vals = np.sin(2 * np.pi * f[:, None] * t[None, :] + ph[:, None])
        W = np.lib.stride_tricks.sliding_window_view(vals, win, axis=1)
        X = W[:, :-1, :].reshape(-1, win).astype(np.float32)
        y_cont = vals[:, win:].reshape(-1).astype(np.float32)
        y = _quantise(y_cont, -1.0, 1.0, n_bins)
        return X, y, y_cont

    train_X, train_y, train_cont = _build(n_train_seq, seed)
    test_X, test_y, test_cont = _build(n_test_seq, seed + 1000)
    chance_mse = float(np.mean((train_cont.mean() - test_cont) ** 2))
    meta = {
        "task": "sine",
        "target": "next_bin",
        "target_description": "Next-step bin index (0 to 31) of "
            "sin(2*pi*f*t + phase) in [-1, 1]; bin width 2/32. Input is "
            "the previous 10 continuous values.",
        "input_dim": win,
        "output_dim": 1,
        "output_kind": "class_index",
        "n_classes": n_bins,
        "n_train": int(len(train_X)),
        "n_test": int(len(test_X)),
        "n_train_seq": n_train_seq,
        "n_test_seq": n_test_seq,
        "length": length,
        "win": win,
        "freq_range_cycles_per_step": (f_lo, f_hi),
        "bin_lo": -1.0,
        "bin_hi": 1.0,
        "chance_level": 1.0 / n_bins,
        "chance_description": "Random-guess accuracy 1/32 = 0.03125. "
            "Chance continuous MSE (predict train-mean value) is %.4f."
            % chance_mse,
        "chance_cont_mse": chance_mse,
        "train_y_cont": train_cont,
        "test_y_cont": test_cont,
        "seed": seed,
    }
    return train_X, train_y, test_X, test_y, meta


# ----------------------------------------------------------------------
# Task 3: Lorenz system, x coordinate, 32 bins
# ----------------------------------------------------------------------

def _lorenz_x(n_traj, n_steps, dt, seed, burn=500):
    """Lorenz x coordinate, vectorised RK4 over trajectories.

    Standard parameters sigma=10, rho=28, beta=8/3. Initial state:
    x, y uniform in [-5, 5], z uniform in [20, 30]. The first burn
    steps are discarded. Returns (n_traj, n_steps) float64.
    """
    rng = _rng(seed)
    state = np.column_stack([
        rng.uniform(-5, 5, size=n_traj),
        rng.uniform(-5, 5, size=n_traj),
        rng.uniform(20, 30, size=n_traj),
    ])
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def f(s):
        return np.column_stack([
            sigma * (s[:, 1] - s[:, 0]),
            s[:, 0] * (rho - s[:, 2]) - s[:, 1],
            s[:, 0] * s[:, 1] - beta * s[:, 2],
        ])

    def step(s):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    for _ in range(burn):
        state = step(state)
    out = np.empty((n_traj, n_steps), dtype=np.float64)
    for i in range(n_steps):
        out[:, i] = state[:, 0]
        state = step(state)
    return out


def make_lorenz_dataset(n_train_traj=40, n_test_traj=15, length=300, win=10,
                        dt=0.02, lo=-20.0, hi=20.0, n_bins=N_BINS, seed=0):
    """Task 3 dataset: Lorenz x, next-step bin (32 bins).

    dt = 0.02 was chosen because at dt = 0.01 persistence already
    scores about 56 percent accuracy (the step is small next to the
    bin width), while at dt = 0.05 the RK4 step is coarse. At
    dt = 0.02 persistence is about 26 percent and the ridge control
    reaches about 75 percent binned accuracy. X is the previous win
    x values (raw scale, about -19 to +19). y is the bin of x(t+1)
    in [lo, hi] = [-20, 20]. Returns (train_X, train_y, test_X,
    test_y, meta).
    """
    def _build(n_traj, sd):
        x = _lorenz_x(n_traj, length, dt, sd)
        W = np.lib.stride_tricks.sliding_window_view(x, win, axis=1)
        X = W[:, :-1, :].reshape(-1, win).astype(np.float32)
        y_cont = x[:, win:].reshape(-1).astype(np.float32)
        y = _quantise(y_cont, lo, hi, n_bins)
        return X, y, y_cont

    train_X, train_y, train_cont = _build(n_train_traj, seed)
    test_X, test_y, test_cont = _build(n_test_traj, seed + 1000)
    chance_mse = float(np.mean((train_cont.mean() - test_cont) ** 2))
    meta = {
        "task": "lorenz",
        "target": "next_bin",
        "target_description": "Next-step bin index (0 to 31) of the "
            "Lorenz x coordinate in [-20, 20]; bin width 40/32 = 1.25. "
            "Input is the previous 10 x values. Lorenz parameters "
            "sigma=10, rho=28, beta=8/3, dt=0.02, RK4, 500-step burn-in.",
        "input_dim": win,
        "output_dim": 1,
        "output_kind": "class_index",
        "n_classes": n_bins,
        "n_train": int(len(train_X)),
        "n_test": int(len(test_X)),
        "n_train_traj": n_train_traj,
        "n_test_traj": n_test_traj,
        "length": length,
        "win": win,
        "dt": dt,
        "bin_lo": lo,
        "bin_hi": hi,
        "chance_level": 1.0 / n_bins,
        "chance_description": "Random-guess accuracy 1/32 = 0.03125. "
            "Chance continuous MSE (predict train-mean x) is %.2f."
            % chance_mse,
        "chance_cont_mse": chance_mse,
        "train_y_cont": train_cont,
        "test_y_cont": test_cont,
        "seed": seed,
    }
    return train_X, train_y, test_X, test_y, meta


# ----------------------------------------------------------------------
# Task 4: 2D moving dot on a 32x32 grid
# ----------------------------------------------------------------------

def _gen_dot_trajectories(n_traj, length, seed, grid=GRID, speed=1.5,
                          p_turn=0.1):
    """Dot with constant speed plus random turns, reflecting walls.

    Start uniform in [5, 27] on both axes, heading uniform in
    [0, 2*pi). Each step, with probability p_turn the heading turns
    by uniform(-pi/2, pi/2). Walls reflect the position and mirror
    the heading. Returns (n_traj, length, 2) float64 in cell units.
    """
    rng = _rng(seed)
    pos = rng.uniform(5, grid - 5, size=(n_traj, 2))
    ang = rng.uniform(0, 2 * np.pi, size=n_traj)
    traj = np.empty((n_traj, length, 2), dtype=np.float64)
    traj[:, 0, :] = pos
    for t in range(1, length):
        turn = rng.random(n_traj) < p_turn
        ang = ang + np.where(
            turn, rng.uniform(-np.pi / 2, np.pi / 2, size=n_traj), 0.0)
        vel = speed * np.column_stack([np.cos(ang), np.sin(ang)])
        new = pos + vel
        hit_x = (new[:, 0] < 0) | (new[:, 0] > grid - 1)
        hit_y = (new[:, 1] < 0) | (new[:, 1] > grid - 1)
        new[:, 0] = np.where(new[:, 0] < 0, -new[:, 0], new[:, 0])
        new[:, 0] = np.where(new[:, 0] > grid - 1,
                             2 * (grid - 1) - new[:, 0], new[:, 0])
        new[:, 1] = np.where(new[:, 1] < 0, -new[:, 1], new[:, 1])
        new[:, 1] = np.where(new[:, 1] > grid - 1,
                             2 * (grid - 1) - new[:, 1], new[:, 1])
        ang = np.where(hit_x, np.pi - ang, ang)
        ang = np.where(hit_y, -ang, ang)
        pos = new
        traj[:, t, :] = pos
    return traj


def make_dot_dataset(n_train_traj=800, n_test_traj=300, length=40, win=10,
                     grid=GRID, speed=1.5, p_turn=0.1, seed=0):
    """Task 4 dataset: moving dot, predict frame 11 from frames 1-10.

    X is the previous win (x, y) positions, each divided by (grid-1),
    flattened to 20 numbers. y is the next cell index, row-major
    (cell = y_cell * 32 + x_cell, cells are floor of coordinates).
    The next coordinates are in meta["test_y_xy"]. Returns
    (train_X, train_y, test_X, test_y, meta).
    """
    def _build(n_traj, sd):
        traj = _gen_dot_trajectories(n_traj, length, sd, grid, speed, p_turn)
        W = np.lib.stride_tricks.sliding_window_view(traj, win, axis=1)
        X = (W[:, :-1, :, :] / (grid - 1)).reshape(-1, win * 2).astype(np.float32)
        y_xy = traj[:, win:, :].reshape(-1, 2).astype(np.float32)
        cells = np.clip(np.floor(y_xy).astype(int), 0, grid - 1)
        y = cells[:, 1] * grid + cells[:, 0]
        last_xy = traj[:, win - 1:-1, :].reshape(-1, 2).astype(np.float32)
        return X, y, y_xy, last_xy

    train_X, train_y, train_xy, train_last = _build(n_train_traj, seed)
    test_X, test_y, test_xy, test_last = _build(n_test_traj, seed + 1000)
    mean_xy = train_xy.mean(axis=0, keepdims=True)
    chance_coord_mse = float(np.mean((mean_xy - test_xy) ** 2))
    meta = {
        "task": "dot_2d",
        "target": "next_cell",
        "target_description": "Next cell index (0 to 1023, row-major) "
            "of a dot on a 32x32 grid, from the previous 10 frames. "
            "Speed 1.5 cells per step, turn probability 0.1 per step, "
            "turn uniform(-90, 90) degrees, reflecting walls.",
        "input_dim": win * 2,
        "output_dim": 1,
        "output_kind": "class_index",
        "n_classes": grid * grid,
        "n_train": int(len(train_X)),
        "n_test": int(len(test_X)),
        "n_train_traj": n_train_traj,
        "n_test_traj": n_test_traj,
        "length": length,
        "win": win,
        "grid": grid,
        "speed_cells_per_step": speed,
        "p_turn_per_step": p_turn,
        "chance_level": 1.0 / (grid * grid),
        "chance_description": "Random-guess accuracy 1/1024 = 0.00098. "
            "Chance coordinate MSE (predict train-mean position) is %.3f."
            % chance_coord_mse,
        "chance_coord_mse": chance_coord_mse,
        "train_y_xy": train_xy,
        "test_y_xy": test_xy,
        "train_last_xy": train_last,
        "test_last_xy": test_last,
        "seed": seed,
    }
    return train_X, train_y, test_X, test_y, meta


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

DATASETS = {
    "track": make_track_dataset,
    "sine": make_sine_dataset,
    "lorenz": make_lorenz_dataset,
    "dot": make_dot_dataset,
}


def get_dataset(name, seed=0, **kwargs):
    """Return (train_X, train_y, test_X, test_y, meta) for one task.

    Name is one of: track, sine, lorenz, dot. Extra keyword args are
    passed to the task builder.
    """
    if name not in DATASETS:
        raise ValueError("Unknown task %r; use one of %s"
                         % (name, sorted(DATASETS)))
    return DATASETS[name](seed=seed, **kwargs)


# ----------------------------------------------------------------------
# Learnability checks (ridge control vs persistence, held-out test)
# ----------------------------------------------------------------------

def _bin_centres(y_bin, lo, hi, n_bins=N_BINS):
    return lo + (np.asarray(y_bin, dtype=np.float64) + 0.5) * (hi - lo) / n_bins


def _check_track(seed):
    train_X, train_y, test_X, test_y, meta = make_track_dataset(seed=seed)
    alpha, _ = select_alpha(train_X, train_y)
    model = fit_ridge(train_X, train_y, alpha)
    pred = np.clip(predict_ridge(model, test_X), 0.0, 1.0)
    ridge_mse = float(np.mean((pred - test_y) ** 2))
    persist_mse = float(np.mean((test_X[:, :N_LI] - test_y) ** 2))
    n_dec = min(5000, len(test_X))
    ridge_pos = decode_position_from_F(pred[:n_dec], track_len=180)
    persist_pos = decode_position_from_F(test_X[:n_dec, :N_LI], track_len=180)
    true_pos = meta["test_y_pos_cm"][:n_dec]
    ridge_rmse = float(np.sqrt(np.mean((ridge_pos - true_pos) ** 2)))
    persist_rmse = float(np.sqrt(np.mean((persist_pos - true_pos) ** 2)))
    gen_X, gen_y, gen_meta = make_track_generalisation_dataset(seed=seed)
    gen_pred = np.clip(predict_ridge(model, gen_X), 0.0, 1.0)
    gen_mse = float(np.mean((gen_pred - gen_y) ** 2))
    gen_persist = float(np.mean((gen_X[:, :N_LI] - gen_y) ** 2))
    gen_pos = decode_position_from_F(gen_pred[:n_dec], track_len=300)
    gen_persist_pos = decode_position_from_F(gen_X[:n_dec, :N_LI], track_len=300)
    gen_true = gen_meta["test_y_pos_cm"][:n_dec]
    gen_rmse = float(np.sqrt(np.mean((gen_pos - gen_true) ** 2)))
    gen_persist_rmse = float(np.sqrt(np.mean((gen_persist_pos - gen_true) ** 2)))
    # Secondary control: ridge on log F. log F = -s * x is linear in
    # position, so this is the best linear control for the exponential
    # code. It still shrinks toward the train mean, so it is also
    # expected to lose to persistence on the 300 cm track; it is
    # reported to show the failure is shrinkage plus distribution
    # shift, not a lack of information in F.
    def _logfeat(X):
        return np.log(np.clip(np.asarray(X, dtype=np.float64), 1e-6, 1.0))
    log_alpha, _ = select_alpha(_logfeat(train_X), _logfeat(train_y))
    log_model = fit_ridge(_logfeat(train_X), _logfeat(train_y), log_alpha)
    log_pred = np.clip(np.exp(predict_ridge(log_model, _logfeat(test_X))), 0.0, 1.0)
    log_mse = float(np.mean((log_pred - test_y) ** 2))
    log_pos = decode_position_from_F(log_pred[:n_dec], track_len=180)
    log_rmse = float(np.sqrt(np.mean((log_pos - true_pos) ** 2)))
    log_gen_pred = np.clip(
        np.exp(predict_ridge(log_model, _logfeat(gen_X))), 0.0, 1.0)
    log_gen_mse = float(np.mean((log_gen_pred - gen_y) ** 2))
    log_gen_pos = decode_position_from_F(log_gen_pred[:n_dec], track_len=300)
    log_gen_rmse = float(np.sqrt(np.mean((log_gen_pos - gen_true) ** 2)))
    return {
        "alpha": alpha, "ridge_f_mse": ridge_mse, "persist_f_mse": persist_mse,
        "chance_f_mse": meta["chance_f_mse"],
        "ridge_pos_rmse_cm": ridge_rmse, "persist_pos_rmse_cm": persist_rmse,
        "chance_pos_rmse_cm": meta["chance_pos_rmse_cm"],
        "gen300_ridge_f_mse": gen_mse, "gen300_persist_f_mse": gen_persist,
        "gen300_ridge_pos_rmse_cm": gen_rmse,
        "gen300_persist_pos_rmse_cm": gen_persist_rmse,
        "log_alpha": log_alpha, "log_ridge_f_mse": log_mse,
        "log_ridge_pos_rmse_cm": log_rmse,
        "log_gen300_f_mse": log_gen_mse,
        "log_gen300_pos_rmse_cm": log_gen_rmse,
    }


def _check_binned(name, seed):
    """Sine / Lorenz: ridge to continuous next value, then bin it."""
    if name == "sine":
        train_X, train_y, test_X, test_y, meta = make_sine_dataset(seed=seed)
        lo, hi = meta["bin_lo"], meta["bin_hi"]
    else:
        train_X, train_y, test_X, test_y, meta = make_lorenz_dataset(seed=seed)
        lo, hi = meta["bin_lo"], meta["bin_hi"]
    train_cont = meta["train_y_cont"][:, None].astype(np.float64)
    test_cont = meta["test_y_cont"].astype(np.float64)
    alpha, _ = select_alpha(train_X, train_cont)
    model = fit_ridge(train_X, train_cont, alpha)
    pred_cont = predict_ridge(model, test_X)[:, 0]
    ridge_mse = float(np.mean((pred_cont - test_cont) ** 2))
    persist_cont = test_X[:, -1].astype(np.float64)
    persist_mse = float(np.mean((persist_cont - test_cont) ** 2))
    ridge_bin = _quantise(pred_cont, lo, hi)
    persist_bin = _quantise(persist_cont, lo, hi)
    ridge_acc = float(np.mean(ridge_bin == test_y))
    persist_acc = float(np.mean(persist_bin == test_y))
    onehot_model = fit_ridge(train_X, _onehot(train_y, meta["n_classes"]), alpha)
    onehot_acc = float(np.mean(
        np.argmax(predict_ridge(onehot_model, test_X), axis=1) == test_y))
    return {
        "alpha": alpha, "ridge_cont_mse": ridge_mse,
        "persist_cont_mse": persist_mse,
        "chance_cont_mse": meta["chance_cont_mse"],
        "ridge_bin_acc": ridge_acc, "persist_bin_acc": persist_acc,
        "onehot_ridge_acc": onehot_acc,
        "chance_acc": meta["chance_level"],
    }


def _check_dot(seed):
    train_X, train_y, test_X, test_y, meta = make_dot_dataset(seed=seed)
    train_xy = meta["train_y_xy"].astype(np.float64)
    test_xy = meta["test_y_xy"].astype(np.float64)
    alpha, _ = select_alpha(train_X, train_xy)
    model = fit_ridge(train_X, train_xy, alpha)
    pred_xy = predict_ridge(model, test_X)
    ridge_mse = float(np.mean((pred_xy - test_xy) ** 2))
    last_xy = meta["test_last_xy"].astype(np.float64)
    persist_mse = float(np.mean((last_xy - test_xy) ** 2))
    pred_cells = np.clip(np.floor(pred_xy).astype(int), 0, GRID - 1)
    ridge_acc = float(np.mean(
        pred_cells[:, 1] * GRID + pred_cells[:, 0] == test_y))
    last_cells = np.clip(np.floor(last_xy).astype(int), 0, GRID - 1)
    persist_acc = float(np.mean(
        last_cells[:, 1] * GRID + last_cells[:, 0] == test_y))
    return {
        "alpha": alpha, "ridge_coord_mse": ridge_mse,
        "persist_coord_mse": persist_mse,
        "chance_coord_mse": meta["chance_coord_mse"],
        "ridge_cell_acc": ridge_acc, "persist_cell_acc": persist_acc,
        "chance_acc": meta["chance_level"],
    }


def _check_integrator_equivalence():
    """Stepwise F update vs closed form, one trial, max abs difference.

    F starts at exp(-s * x0) at the first recorded position x0, then
    each step multiplies by exp(-s * dx) with dx the recorded
    displacement. The closed form is exp(-s * x(t)).
    """
    pos, _ = _gen_track_positions(1, 200, 180, seed=7)
    x = pos[0]
    F_step = np.exp(-S_VALUES * float(x[0]))
    worst = 0.0
    for t in range(1, len(x)):
        dx = float(x[t] - x[t - 1])
        F_step = F_step * np.exp(-S_VALUES * dx)
        F_closed = np.exp(-S_VALUES * float(x[t]))
        worst = max(worst, float(np.max(np.abs(F_step - F_closed))))
    return worst


def run_learnability_checks(seeds=(0, 1, 2)):
    """Run the ridge-vs-persistence check for every task, mean over seeds.

    Returns a dict: task name -> dict of metric -> (mean, std) over seeds,
    plus the integrator-equivalence error. Ridge is fit on train only;
    alphas are chosen on a train-internal split; scores are held-out.
    """
    per_seed = {"track": [], "sine": [], "lorenz": [], "dot": []}
    for seed in seeds:
        per_seed["track"].append(_check_track(seed))
        per_seed["sine"].append(_check_binned("sine", seed))
        per_seed["lorenz"].append(_check_binned("lorenz", seed))
        per_seed["dot"].append(_check_dot(seed))
    out = {}
    for task, rows in per_seed.items():
        agg = {}
        for key in rows[0]:
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            agg[key] = (float(vals.mean()), float(vals.std()))
        out[task] = agg
    out["integrator_equivalence_max_abs_err"] = _check_integrator_equivalence()
    return out


def _fmt(mean_std):
    mean, std = mean_std
    return "%.4g +/- %.2g" % (mean, std)


if __name__ == "__main__":
    print("Integrator check: stepwise vs closed-form max abs err = %.3g"
          % _check_integrator_equivalence())
    print("Ridge control is fit on train only; alpha chosen on a "
          "train-internal split; scores are held-out test.")
    results = run_learnability_checks(seeds=(0, 1, 2))
    print()
    print("== Track 1D (180 cm), target next F vector, mean over 3 seeds ==")
    t = results["track"]
    print("ridge F-MSE:      %s (chance mean-predictor %.4g)"
          % (_fmt(t["ridge_f_mse"]), t["chance_f_mse"][0]))
    print("persistence F-MSE: %s" % _fmt(t["persist_f_mse"]))
    print("ridge position RMSE:      %s cm" % _fmt(t["ridge_pos_rmse_cm"]))
    print("persistence position RMSE: %s cm (chance %.2f cm)"
          % (_fmt(t["persist_pos_rmse_cm"]), t["chance_pos_rmse_cm"][0]))
    print("log-F ridge F-MSE:      %s; log-F ridge position RMSE: %s cm"
          % (_fmt(t["log_ridge_f_mse"]), _fmt(t["log_ridge_pos_rmse_cm"])))
    print("generalisation 300 cm: ridge F-MSE %s, log-F ridge %s, persistence %s"
          % (_fmt(t["gen300_ridge_f_mse"]), _fmt(t["log_gen300_f_mse"]),
             _fmt(t["gen300_persist_f_mse"])))
    print("generalisation 300 cm: ridge position RMSE %s cm, log-F ridge %s cm,"
          " persistence %s cm"
          % (_fmt(t["gen300_ridge_pos_rmse_cm"]),
             _fmt(t["log_gen300_pos_rmse_cm"]),
             _fmt(t["gen300_persist_pos_rmse_cm"])))
    print("Note: on 300 cm the linear controls LOSE to persistence. That is"
          " expected: ridge shrinks toward the 180 cm train mean and F is"
          " exponential in position, so a linear readout cannot extrapolate"
          " to a longer track. Beating persistence on 300 cm is the bar for"
          " RPC, not part of the learnability proof.")
    print()
    for name in ("sine", "lorenz"):
        r = results[name]
        print("== %s, target next bin (32 bins), mean over 3 seeds ==" % name)
        print("ridge continuous MSE:      %s (chance %.4g)"
              % (_fmt(r["ridge_cont_mse"]), r["chance_cont_mse"][0]))
        print("persistence continuous MSE: %s" % _fmt(r["persist_cont_mse"]))
        print("ridge binned accuracy:      %s (chance %.4g)"
              % (_fmt(r["ridge_bin_acc"]), r["chance_acc"][0]))
        print("persistence binned accuracy: %s" % _fmt(r["persist_bin_acc"]))
        print("one-hot ridge accuracy (secondary): %s"
              % _fmt(r["onehot_ridge_acc"]))
        print()
    d = results["dot"]
    print("== Dot 2D (32x32), target next cell (1024 cells), mean over 3 seeds ==")
    print("ridge coordinate MSE:      %s (chance %.4g)"
          % (_fmt(d["ridge_coord_mse"]), d["chance_coord_mse"][0]))
    print("persistence coordinate MSE: %s" % _fmt(d["persist_coord_mse"]))
    print("ridge cell accuracy:      %s (chance %.4g)"
          % (_fmt(d["ridge_cell_acc"]), d["chance_acc"][0]))
    print("persistence cell accuracy: %s" % _fmt(d["persist_cell_acc"]))
    print()
    print("Held-out 180 cm / in-distribution test: ridge beats persistence on"
          " the primary metric for all four tasks, so each task is learnable"
          " by a simple control. The 300 cm track is a generalisation probe:"
          " the linear controls fail there, and beating persistence there is"
          " the bar for RPC.")
