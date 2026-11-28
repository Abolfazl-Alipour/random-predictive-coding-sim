# Round 18 analysis: the continuous-task wall

Seeds: 262..281 (n = 20). All metrics convention A (pred(t) vs token t; MSE decode(pred(t)) vs cont[t]). Bars are paired per-seed differences, 95% CI = mean +/- 1.96 * std / sqrt(20).

Input files: /home/reza/zfs-1tb/rpc-sim/thesis-rpc/r18/r18_lorenz_test_262-281.jsonl, /home/reza/zfs-1tb/rpc-sim/thesis-rpc/r18/r18_varsine_test_262-281.jsonl
Coverage complete: True (360 test lines read).

## 1. Per-condition summary (mean [95% CI], median, IQR)

### lorenz

| condition | acc | within1 | mse |
|---|---|---|---|
| C1_onehot_delta_proto | 0.333 [0.320, 0.347] med 0.324 IQR [0.312, 0.346] | 0.654 | 22.390 [20.030, 24.749] med 20.926 IQR [17.994, 25.673] |
| C2_onehot_delta_ridge | 0.195 [0.188, 0.203] med 0.194 IQR [0.182, 0.202] | 0.537 | 11.403 [10.718, 12.087] med 11.305 IQR [10.647, 12.188] |
| C3_bump_delta_proto | 0.348 [0.334, 0.363] med 0.344 IQR [0.319, 0.382] | 0.781 | 7.292 [6.108, 8.477] med 6.907 IQR [5.737, 8.356] |
| C4_bump_delta_ridge | 0.345 [0.334, 0.357] med 0.345 IQR [0.335, 0.365] | 0.798 | 3.795 [3.509, 4.080] med 3.749 IQR [3.274, 4.060] |
| C5_onehot_frozen_proto | 0.038 [0.033, 0.042] med 0.038 IQR [0.033, 0.043] | 0.112 | 201.252 [176.758, 225.746] med 209.981 IQR [167.061, 231.555] |
| C6_bump_frozen_proto | 0.031 [0.023, 0.040] med 0.037 IQR [0.013, 0.044] | 0.092 | 257.403 [216.201, 298.605] med 251.626 IQR [190.282, 337.081] |
| C2f_onehot_frozen_ridge | 0.200 [0.192, 0.208] med 0.198 IQR [0.192, 0.213] | 0.548 | 10.818 [10.127, 11.510] med 10.726 IQR [9.728, 11.789] |
| C4f_bump_frozen_ridge | 0.350 [0.337, 0.363] med 0.360 IQR [0.323, 0.369] | 0.805 | 3.615 [3.327, 3.902] med 3.598 IQR [3.117, 3.805] |
| baseline Markov-1 | 0.208 | | |
| baseline persistence | 0.171 | | |
| baseline ESN | 0.397 | | |
| baseline persistence MSE | | | 12.605 [12.256, 12.955] |
| baseline Markov-1 MSE | | | 11.960 |
| baseline ESN MSE | | | 4.308 [4.112, 4.505] |

### varsine

| condition | acc | within1 | mse |
|---|---|---|---|
| C1_onehot_delta_proto | 0.295 [0.284, 0.307] med 0.295 IQR [0.277, 0.315] | 0.583 | 0.064 [0.055, 0.074] med 0.059 IQR [0.051, 0.073] |
| C2_onehot_delta_ridge | 0.158 [0.152, 0.163] med 0.159 IQR [0.148, 0.169] | 0.348 | 0.066 [0.061, 0.071] med 0.064 IQR [0.058, 0.073] |
| C3_bump_delta_proto | 0.278 [0.266, 0.290] med 0.274 IQR [0.260, 0.297] | 0.566 | 0.039 [0.033, 0.046] med 0.036 IQR [0.031, 0.045] |
| C4_bump_delta_ridge | 0.183 [0.178, 0.189] med 0.185 IQR [0.174, 0.190] | 0.412 | 0.044 [0.042, 0.047] med 0.045 IQR [0.042, 0.050] |
| C5_onehot_frozen_proto | 0.041 [0.032, 0.049] med 0.036 IQR [0.027, 0.049] | 0.096 | 0.862 [0.786, 0.938] med 0.861 IQR [0.721, 0.989] |
| C6_bump_frozen_proto | 0.038 [0.032, 0.044] med 0.036 IQR [0.029, 0.043] | 0.091 | 0.879 [0.781, 0.976] med 0.923 IQR [0.693, 1.063] |
| C2f_onehot_frozen_ridge | 0.159 [0.153, 0.166] med 0.159 IQR [0.148, 0.170] | 0.349 | 0.066 [0.061, 0.071] med 0.064 IQR [0.058, 0.073] |
| C4f_bump_frozen_ridge | 0.183 [0.178, 0.188] med 0.182 IQR [0.173, 0.191] | 0.414 | 0.045 [0.042, 0.047] med 0.045 IQR [0.042, 0.049] |
| baseline Markov-1 | 0.284 | | |
| baseline persistence | 0.126 | | |
| baseline ESN | 0.393 | | |
| baseline persistence MSE | | | 0.056 [0.053, 0.059] |
| baseline Markov-1 MSE | | | 0.110 |
| baseline ESN MSE | | | 0.046 [0.042, 0.050] |

## 2. Bars D1-D8 (paired per-seed differences)

### D1 Lorenz C1 acc beats persistence AND Markov-1: PASS
- D1a Lorenz C1 acc - persistence acc: mean paired diff 0.162, 95% CI [0.149, 0.175], PASS
- D1b Lorenz C1 acc - Markov-1 acc: mean paired diff 0.125, 95% CI [0.112, 0.137], PASS

### D2 var-sine C1 acc beats persistence AND Markov-1: PASS
- D2a var-sine C1 acc - persistence acc: mean paired diff 0.169, 95% CI [0.157, 0.181], PASS
- D2b var-sine C1 acc - Markov-1 acc: mean paired diff 0.011, 95% CI [0.000, 0.021], PASS

### D3 var-sine C1 mse < persistence mse: FAIL
- D3 var-sine persistence mse - C1 mse: mean paired diff -0.008, 95% CI [-0.018, 0.002], FAIL

### D4 Lorenz C1 mse < persistence mse: FAIL
- D4 Lorenz persistence mse - C1 mse: mean paired diff -9.785, 95% CI [-12.029, -7.540], FAIL

### D5 Lorenz C2 ridge mse < C1 mse: PASS
- D5 Lorenz C1 mse - C2 ridge mse: mean paired diff 10.987, 95% CI [8.852, 13.122], PASS

### D6 Lorenz C3 bump mse < C1 mse: PASS
- D6 Lorenz C1 mse - C3 bump mse: mean paired diff 15.097, 95% CI [12.959, 17.235], PASS

### D7 Lorenz C4 mse < C2 mse AND C4 mse < C3 mse (judged only if D5 or D6 passes): PASS
- D7a Lorenz C2 mse - C4 mse: mean paired diff 7.608, 95% CI [7.046, 8.170], PASS
- D7b Lorenz C3 mse - C4 mse: mean paired diff 3.498, 95% CI [2.446, 4.550], PASS

### D8 Lorenz C2 ridge mse < C2f frozen-ridge mse: FAIL
- D8 Lorenz C2f frozen-ridge mse - C2 mse: mean paired diff -0.584, 95% CI [-0.735, -0.434], FAIL

## 3. Var-sine secondary analogues (no bars)

- D5-var: C2 ridge mse < C1 mse: mean paired diff -0.002, 95% CI [-0.010, 0.007] (n=20)
- D6-var: C3 bump mse < C1 mse: mean paired diff 0.025, 95% CI [0.021, 0.029] (n=20)
- D7-var: C4 mse < C2 mse and C4 mse < C3 mse: mean paired diff 0.022, 95% CI [0.018, 0.025] (n=20)
- D8-var: C2 ridge mse < C2f frozen-ridge mse: mean paired diff -0.000, 95% CI [-0.001, 0.001] (n=20)

## 4. Verdict (protocol section 9)

- H0 (D1, D2, D3 pass): FAIL
- D4 (Lorenz C1 MSE beats persistence): FAIL
- Verdict: H0 fails. The continuous wall survives correct scoring; the pilot was not the whole story. Read the per-condition table and the paired differences before designing the next round.
- D8 (delta adds continuous information): FAIL

## 5. Frozen controls (sanity)

- lorenz: C1 acc 0.333 vs C5 frozen acc 0.038 (learning gap).
- varsine: C1 acc 0.295 vs C5 frozen acc 0.041 (learning gap).
