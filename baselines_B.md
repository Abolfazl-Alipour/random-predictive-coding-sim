# Baselines for the RPC toy tasks (worker B)

Question: are the tasks learnable at all, and by how much?
Every number below is TEST accuracy on the same held-out 50-sequence test set (200 train / 50 test, length 24), mean / min / max across 5 seeds (0-4). Train accuracy is never reported as test accuracy. No claim is made here about RPC itself; this establishes the ceiling and the controls only.

## Accuracy (next-token, fraction correct)

| task | K | chance | Markov-1 | Markov-2 | ESN | LSTM | persistence |
|---|---|---|---|---|---|---|---|
| motif lag 1 | 16 | 0.062 [0.062, 0.062] | 0.278 [0.263, 0.291] | 0.270 [0.251, 0.282] | 0.274 [0.262, 0.288] | 0.283 [0.266, 0.290] | n/a |
| motif lag 2 | 16 | 0.062 [0.062, 0.062] | 0.081 [0.072, 0.087] | 0.266 [0.252, 0.287] | 0.263 [0.254, 0.283] | 0.285 [0.267, 0.304] | n/a |
| motif lag 5 | 16 | 0.062 [0.062, 0.062] | 0.097 [0.083, 0.106] | 0.073 [0.064, 0.088] | 0.230 [0.224, 0.235] | 0.219 [0.210, 0.223] | n/a |
| sine (32 bins) | 32 | 0.031 [0.031, 0.031] | 0.410 [0.392, 0.450] | 0.769 [0.752, 0.774] | 0.900 [0.886, 0.910] | 0.894 [0.864, 0.908] | 0.059 [0.043, 0.070] |
| Lorenz x (32 bins) | 32 | 0.031 [0.031, 0.031] | 0.194 [0.182, 0.208] | 0.559 [0.540, 0.570] | 0.388 [0.374, 0.398] | 0.391 [0.349, 0.450] | 0.170 [0.152, 0.197] |

Chance = 1/K (0.0625 for K=16 motifs, 0.03125 for K=32). Persistence = repeat the last token/bin.

## Continuous MSE (sine / Lorenz only)

MSE between the predicted value and the true continuous next value. Model predictions are decoded from the predicted bin centre (sine: bins over [-1, 1]; Lorenz x: bins over [-25, 25]). Persistence = predict the last continuous value. Mean = predict the TRAIN mean (constant). Lower is better.

| task | persistence | TRAIN-mean | Markov-1 | Markov-2 | ESN | LSTM |
|---|---|---|---|---|---|---|
| sine (32 bins) | 0.1345 [0.1339, 0.1351] | 0.4986 [0.4968, 0.5007] | 0.2498 [0.2278, 0.2653] | 0.0114 [0.0089, 0.0140] | 0.0219 [0.0175, 0.0307] | 0.0199 [0.0161, 0.0283] |
| Lorenz x (32 bins) | 12.6042 [11.9361, 13.5027] | 67.1044 [65.6239, 71.1576] | 11.9331 [11.1783, 12.6654] | 1.3629 [1.2516, 1.5211] | 4.4307 [4.1900, 4.5089] | 3.6786 [2.1485, 6.4540] |

## Per-seed test accuracy

### motif lag 1 (chance 0.0625)

| seed | markov1 | markov2 | esn | lstm |
|---|---|---|---|---|
| 0 | 0.291 | 0.282 | 0.288 | 0.290 |
| 1 | 0.277 | 0.273 | 0.264 | 0.283 |
| 2 | 0.275 | 0.272 | 0.280 | 0.289 |
| 3 | 0.263 | 0.251 | 0.262 | 0.266 |
| 4 | 0.284 | 0.270 | 0.276 | 0.284 |

### motif lag 2 (chance 0.0625)

| seed | markov1 | markov2 | esn | lstm |
|---|---|---|---|---|
| 0 | 0.083 | 0.252 | 0.257 | 0.267 |
| 1 | 0.087 | 0.265 | 0.254 | 0.287 |
| 2 | 0.077 | 0.287 | 0.283 | 0.304 |
| 3 | 0.084 | 0.264 | 0.263 | 0.294 |
| 4 | 0.072 | 0.260 | 0.257 | 0.274 |

### motif lag 5 (chance 0.0625)

| seed | markov1 | markov2 | esn | lstm |
|---|---|---|---|---|
| 0 | 0.106 | 0.069 | 0.235 | 0.221 |
| 1 | 0.100 | 0.068 | 0.232 | 0.223 |
| 2 | 0.093 | 0.088 | 0.225 | 0.223 |
| 3 | 0.103 | 0.064 | 0.224 | 0.219 |
| 4 | 0.083 | 0.074 | 0.233 | 0.210 |

### sine (32 bins) (chance 0.0312)

| seed | markov1 | markov2 | esn | lstm | persist_acc |
|---|---|---|---|---|---|
| 0 | 0.392 | 0.773 | 0.900 | 0.895 | 0.057 |
| 1 | 0.403 | 0.752 | 0.910 | 0.908 | 0.057 |
| 2 | 0.450 | 0.774 | 0.886 | 0.897 | 0.070 |
| 3 | 0.405 | 0.774 | 0.895 | 0.903 | 0.067 |
| 4 | 0.401 | 0.774 | 0.908 | 0.864 | 0.043 |

### Lorenz x (32 bins) (chance 0.0312)

| seed | markov1 | markov2 | esn | lstm | persist_acc |
|---|---|---|---|---|---|
| 0 | 0.197 | 0.564 | 0.398 | 0.412 | 0.163 |
| 1 | 0.182 | 0.570 | 0.382 | 0.349 | 0.182 |
| 2 | 0.208 | 0.565 | 0.398 | 0.377 | 0.152 |
| 3 | 0.190 | 0.540 | 0.374 | 0.365 | 0.157 |
| 4 | 0.197 | 0.556 | 0.386 | 0.450 | 0.197 |

## Per-seed continuous MSE (sine / Lorenz)

### sine (32 bins)

| seed | persist_mse | mean_mse | markov1_mse | markov2_mse | esn_mse | lstm_mse |
|---|---|---|---|---|---|---|
| 0 | 0.1339 | 0.5007 | 0.2514 | 0.0140 | 0.0185 | 0.0186 |
| 1 | 0.1340 | 0.4985 | 0.2448 | 0.0089 | 0.0198 | 0.0161 |
| 2 | 0.1351 | 0.4976 | 0.2278 | 0.0104 | 0.0307 | 0.0283 |
| 3 | 0.1343 | 0.4995 | 0.2598 | 0.0115 | 0.0175 | 0.0200 |
| 4 | 0.1350 | 0.4968 | 0.2653 | 0.0125 | 0.0228 | 0.0163 |

### Lorenz x (32 bins)

| seed | persist_mse | mean_mse | markov1_mse | markov2_mse | esn_mse | lstm_mse |
|---|---|---|---|---|---|---|
| 0 | 12.3142 | 65.6239 | 11.1783 | 1.5211 | 4.5084 | 3.4200 |
| 1 | 12.4305 | 66.7552 | 11.9359 | 1.2972 | 4.5089 | 3.4906 |
| 2 | 12.8376 | 71.1576 | 12.1440 | 1.4558 | 4.4742 | 2.8798 |
| 3 | 13.5027 | 66.0334 | 11.7417 | 1.2890 | 4.1900 | 6.4540 |
| 4 | 11.9361 | 65.9520 | 12.6654 | 1.2516 | 4.4722 | 2.1485 |

## Exact hyperparameters and data

1. Data: 200 train / 50 test sequences, length 24, drawn per seed with `np.random.seed(1000 + seed)`, train first then test, using the global numpy RNG exactly as in `rpc_toy.make_data`.
2. Motif lag 1: `make_data` verbatim from `rpc_toy.py` (K=16, motifs (1,2,3), (5,6,7), (9,10,11), planted contiguously; later motifs overwrite earlier ones).
3. Motif lag 2 / lag 5: same background and motifs, but motif tokens sit at positions pos, pos+lag, pos+2*lag, with the (lag-1) positions between them left as random filler tokens; pos drawn from [0, length-2*lag). Lag 2: `1, X, 2, X, 3`. Lag 5: `1, X, X, X, X, 2, X, X, X, X, 3`.
4. Sine: x(t) = sin(2*pi*t/12 + phi), phi ~ Uniform(0, 2*pi) per sequence, t = 0..23 (period 12 steps, 2 cycles per sequence), quantised to 32 bins over [-1, 1], bin centre -1 + (bin+0.5)/32*2.
5. Lorenz: sigma=10, rho=28, beta=8/3, RK4, dt=0.01, x coordinate only, quantised to 32 bins over fixed [-25, 25], bin centre -25 + (bin+0.5)/32*50. Per sequence a random initial state (x,y in [-10,10], z in [20,30]) is integrated; the first 10 steps are discarded, then every 10th step is sampled, so a length-24 sequence spans 2.4 time units. Sampling every step would span only 0.24 time units, in which x barely moves and persistence would solve the task trivially.
6. Chance: 1/K. Markov-1 / Markov-2: counts from TRAIN only, add-alpha smoothing alpha=0.1; order-2 backs off to the order-1 distribution for unseen 2-token contexts.
7. ESN: N_r=300, spectral radius 0.9, leak 0.3, input scaling 1.0, tanh reservoir, ridge lambda=1.0, washout 2 steps per sequence, state reset per sequence, readout trained by ridge regression on TRAIN states only. Reservoir weights from `np.random.RandomState(2000 + seed)`.
8. LSTM: 1 layer, hidden 64, one-hot input (size K), linear head to K logits, Adam lr=0.005, batch 32, epochs 20, cross-entropy over positions 0..22 predicting token t+1, hidden state zeroed per batch, `torch.manual_seed(seed)` (fixes init and shuffling).
9. Scoring: accuracy counts positions 1..23 of each test sequence (prediction of token t uses tokens 0..t-1 only). MSE uses the true continuous next value.

## Reproduce

```bash
~/workspace/thesis-rpc/venv/bin/python ~/workspace/thesis-rpc/baselines.py
```

CPU only (torch 2.15.0+cpu, numpy 2.5.4). This file was written by `baselines.py`; re-running it regenerates it.
