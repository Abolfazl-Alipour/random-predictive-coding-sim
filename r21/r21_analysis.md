# Round 21 analysis: continuous re-validation

Protocol: PROTOCOL-R21-CONTINUOUS.md. Seeds 327..346 (20 fresh seeds). CI = mean +/- exact Student-t * sample-std / sqrt(n): t(19) = 2.0930 for test bars, t(4) = 2.7760 for val.

Test rows: 640 (unique keys 640, duplicates 0).
  lorenz token_delta: 20 seeds
  lorenz token_frozen: 20 seeds
  lorenz token_w_delta: 20 seeds
  lorenz token_w_frozen: 20 seeds
  lorenz bump_delta: 20 seeds
  lorenz bump_frozen: 20 seeds
  lorenz baseline: 20 seeds
  lorenz token_primary_delta: 20 seeds
  lorenz token_primary_frozen: 20 seeds
  varsine token_delta: 20 seeds
  varsine token_frozen: 20 seeds
  varsine token_w_delta: 20 seeds
  varsine token_w_frozen: 20 seeds
  varsine bump_delta: 20 seeds
  varsine bump_frozen: 20 seeds
  varsine baseline: 20 seeds
  varsine token_primary_delta: 20 seeds
  varsine token_primary_frozen: 20 seeds
  narma10 token_delta: 20 seeds
  narma10 token_frozen: 20 seeds
  narma10 token_w_delta: 20 seeds
  narma10 token_w_frozen: 20 seeds
  narma10 bump_delta: 20 seeds
  narma10 bump_frozen: 20 seeds
  narma10 baseline: 20 seeds
  mackeyglass token_delta: 20 seeds
  mackeyglass token_frozen: 20 seeds
  mackeyglass token_w_delta: 20 seeds
  mackeyglass token_w_frozen: 20 seeds
  mackeyglass bump_delta: 20 seeds
  mackeyglass bump_frozen: 20 seeds
  mackeyglass baseline: 20 seeds

## Per-task means (prototype MSE, 20 seeds)

| task | token_delta proto | token_frozen proto | token_w_delta proto | bump_delta proto | persistence | Markov-2 | ESN argmax | ESN scalar-std |
|---|---|---|---|---|---|---|---|---|
| Lorenz x (K=32, len 24) | 4.2875 | 236.8511 | 4.5180 | 4.7361 | 12.5914 | 1.3961 | 4.5633 | 1.7799 |
| Variable-frequency sine (K=32, len 100) | 0.0270 | 0.9303 | 0.0448 | 0.0274 | 0.0560 | 0.0052 | 0.0450 | 0.0082 |
| NARMA-10 (K=32, len 100) | 0.0117 | 0.0915 | 0.0120 | 0.0104 | 0.0091 | 0.0128 | 0.0124 | 0.0070 |
| Mackey-Glass D=17 (K=32, len 100) | 0.0020 | 0.1911 | 0.0035 | 0.0019 | 0.0012 | 0.0009 | 0.0013 | 0.0004 |

## Bar table

Pass/fail bars: 26 pass, 24 fail, 4 n/a or measurement-only, of 54 rows.

1. [PASS] W1 (lorenz primary proto MSE beats persistence): paired diff mean=+8.3039 95% CI=[7.5853,9.0225] n=20
1. [FAIL] W3 (lorenz primary proto MSE beats ESN argmax): paired diff mean=+0.2758 95% CI=[-0.3146,0.8663] n=20
1. [PASS] W5 (lorenz primary proto acc beats Markov-1): paired diff mean=+0.1920 95% CI=[0.1754,0.2086] n=20
1. [PASS] W7 (lorenz delta proto MSE beats frozen proto MSE): paired diff mean=+232.5637 95% CI=[188.6496,276.4778] n=20
1. [PASS] W9 (lorenz preregistered primary proto MSE beats persistence): paired diff mean=+7.6045 95% CI=[6.8184,8.3906] n=20
1. [PASS] W2 (varsine primary proto MSE beats persistence): paired diff mean=+0.0290 95% CI=[0.0219,0.0360] n=20
1. [PASS] W4 (varsine primary proto MSE beats ESN argmax): paired diff mean=+0.0180 95% CI=[0.0113,0.0247] n=20
1. [PASS] W6 (varsine primary proto acc beats Markov-1): paired diff mean=+0.0442 95% CI=[0.0330,0.0555] n=20
1. [PASS] W8 (varsine delta proto MSE beats frozen proto MSE): paired diff mean=+0.9033 95% CI=[0.8092,0.9974] n=20
1. [PASS] W10 (varsine preregistered primary proto MSE beats persistence): paired diff mean=+0.0187 95% CI=[0.0131,0.0244] n=20
1. [FAIL] N1 (narma10 proto MSE beats persistence): paired diff mean=-0.0026 95% CI=[-0.0030,-0.0022] n=20
1. [PASS] N3 (narma10 proto MSE beats ESN argmax): paired diff mean=+0.0007 95% CI=[0.0003,0.0011] n=20
1. [PASS] N5 (narma10 delta proto MSE beats frozen proto MSE): paired diff mean=+0.0798 95% CI=[0.0612,0.0984] n=20
1. [PASS] N7 (narma10 proto MSE beats Markov-2): paired diff mean=+0.0011 95% CI=[0.0007,0.0015] n=20
1. [FAIL] N2 (mackeyglass proto MSE beats persistence): paired diff mean=-0.0008 95% CI=[-0.0009,-0.0006] n=20
1. [FAIL] N4 (mackeyglass proto MSE beats ESN argmax): paired diff mean=-0.0006 95% CI=[-0.0008,-0.0004] n=20
1. [PASS] N6 (mackeyglass delta proto MSE beats frozen proto MSE): paired diff mean=+0.1892 95% CI=[0.1418,0.2365] n=20
1. [FAIL] N8 (mackeyglass proto MSE beats Markov-2): paired diff mean=-0.0011 95% CI=[-0.0013,-0.0009] n=20
1. [FAIL] R1 (lorenz standardized trained ridge beats standardized frozen ridge): paired diff mean=+0.0185 95% CI=[-0.0122,0.0493] n=20
1. [PASS] R3 (lorenz standardized trained ridge beats persistence): paired diff mean=+8.0380 95% CI=[7.5405,8.5355] n=20
1. [FAIL] R4 (lorenz standardized trained ridge beats ESN scalar-std ridge): paired diff mean=-2.7735 95% CI=[-2.9972,-2.5498] n=20
1. [FAIL] C1 (lorenz centroid beta=10 beats prototype): paired diff mean=-0.0038 95% CI=[-0.4054,0.3979] n=20
1. [PASS] C2 (lorenz centroid beta=10 beats persistence): paired diff mean=+8.3001 95% CI=[7.6400,8.9603] n=20
1. [n/a] F1 (lorenz wire effect on proto MSE: off - on): mean=-0.2306 95% CI=[-0.9934,0.5322] n=20 -> NULL (CI covers 0)
1. [PASS] F2 (lorenz wire-live gate: mean fb_drive_ratio > 0.05): mean fb_drive_ratio=0.3161 95% CI=[0.3101,0.3221] n=20
1. [FAIL] B1 (lorenz bump proto MSE beats token proto MSE): paired diff mean=-0.4487 95% CI=[-1.0270,0.1297] n=20
1. [PASS] B2 (lorenz bump proto MSE beats persistence): paired diff mean=+7.8553 95% CI=[7.0127,8.6978] n=20
1. [FAIL] R1 (varsine standardized trained ridge beats standardized frozen ridge): paired diff mean=+0.0001 95% CI=[-0.0001,0.0004] n=20
1. [PASS] R3 (varsine standardized trained ridge beats persistence): paired diff mean=+0.0137 95% CI=[0.0088,0.0185] n=20
1. [FAIL] R4 (varsine standardized trained ridge beats ESN scalar-std ridge): paired diff mean=-0.0341 95% CI=[-0.0376,-0.0306] n=20
1. [FAIL] C1 (varsine centroid beta=10 beats prototype): paired diff mean=-0.0121 95% CI=[-0.0183,-0.0059] n=20
1. [PASS] C2 (varsine centroid beta=10 beats persistence): paired diff mean=+0.0168 95% CI=[0.0097,0.0240] n=20
1. [n/a] F1 (varsine wire effect on proto MSE: off - on): mean=-0.0178 95% CI=[-0.0337,-0.0020] n=20 -> HURTS (wire-on MSE higher, CI below 0)
1. [PASS] F2 (varsine wire-live gate: mean fb_drive_ratio > 0.05): mean fb_drive_ratio=0.3127 95% CI=[0.3063,0.3191] n=20
1. [FAIL] B1 (varsine bump proto MSE beats token proto MSE): paired diff mean=-0.0005 95% CI=[-0.0037,0.0027] n=20
1. [PASS] B2 (varsine bump proto MSE beats persistence): paired diff mean=+0.0285 95% CI=[0.0211,0.0359] n=20
1. [FAIL] R1 (narma10 standardized trained ridge beats standardized frozen ridge): paired diff mean=-0.0000 95% CI=[-0.0000,0.0000] n=20
1. [PASS] R3 (narma10 standardized trained ridge beats persistence): paired diff mean=+0.0011 95% CI=[0.0009,0.0014] n=20
1. [FAIL] R4 (narma10 standardized trained ridge beats ESN scalar-std ridge): paired diff mean=-0.0009 95% CI=[-0.0011,-0.0008] n=20
1. [PASS] C1 (narma10 centroid beta=10 beats prototype): paired diff mean=+0.0024 95% CI=[0.0019,0.0029] n=20
1. [FAIL] C2 (narma10 centroid beta=10 beats persistence): paired diff mean=-0.0002 95% CI=[-0.0007,0.0003] n=20
1. [n/a] F1 (narma10 wire effect on proto MSE: off - on): mean=-0.0003 95% CI=[-0.0008,0.0002] n=20 -> NULL (CI covers 0)
1. [PASS] F2 (narma10 wire-live gate: mean fb_drive_ratio > 0.05): mean fb_drive_ratio=0.3424 95% CI=[0.3379,0.3470] n=20
1. [PASS] B1 (narma10 bump proto MSE beats token proto MSE): paired diff mean=+0.0013 95% CI=[0.0011,0.0016] n=20
1. [FAIL] B2 (narma10 bump proto MSE beats persistence): paired diff mean=-0.0013 95% CI=[-0.0016,-0.0010] n=20
1. [FAIL] R1 (mackeyglass standardized trained ridge beats standardized frozen ridge): paired diff mean=+0.0000 95% CI=[-0.0000,0.0000] n=20
1. [FAIL] R3 (mackeyglass standardized trained ridge beats persistence): paired diff mean=-0.0020 95% CI=[-0.0023,-0.0018] n=20
1. [FAIL] R4 (mackeyglass standardized trained ridge beats ESN scalar-std ridge): paired diff mean=-0.0029 95% CI=[-0.0031,-0.0026] n=20
1. [FAIL] C1 (mackeyglass centroid beta=10 beats prototype): paired diff mean=-0.0002 95% CI=[-0.0004,-0.0001] n=20
1. [FAIL] C2 (mackeyglass centroid beta=10 beats persistence): paired diff mean=-0.0010 95% CI=[-0.0012,-0.0008] n=20
1. [n/a] F1 (mackeyglass wire effect on proto MSE: off - on): mean=-0.0015 95% CI=[-0.0028,-0.0003] n=20 -> HURTS (wire-on MSE higher, CI below 0)
1. [PASS] F2 (mackeyglass wire-live gate: mean fb_drive_ratio > 0.05): mean fb_drive_ratio=0.3103 95% CI=[0.3038,0.3169] n=20
1. [FAIL] B1 (mackeyglass bump proto MSE beats token proto MSE): paired diff mean=+0.0000 95% CI=[-0.0001,0.0002] n=20
1. [FAIL] B2 (mackeyglass bump proto MSE beats persistence): paired diff mean=-0.0007 95% CI=[-0.0010,-0.0005] n=20

## Verdict

1. WALL FALLS for the re-validation tasks: the Convention-A prototype readout beats persistence on fresh seeds AND the win is due to learning (delta beats frozen).
1. BEATS MARKOV-2 ON ONE TASK ONLY: narma10 passes N7/N8; mackeyglass does not. The tasks are not equally discriminating.

Round 21 does NOT establish predictive coding even if the wall falls: round 20 failed the revised clamp (B') and the generative rollout (D). A wall-fall here establishes only that the signed-error rule learns continuous trajectories. See protocol section 8.

## Diagnostics (not bars)

1. lorenz centroid MSE beta 5/10/20: 8.4300 / 4.2912 / 3.5158 (beta=10 is primary; a >20% gap between the best beta and beta=10 flags beta-sensitivity).
1. lorenz ridge MSE trained std/raw: 4.5533 / 4.7122; frozen std/raw: 4.5719 / 4.7118 (D8: if raw frozen beats raw trained while std trained beats std frozen, the round-18 frozen-ridge win was a regularization artifact).
1. lorenz wire effect on the FROZEN net (frozen-off minus frozen-on proto MSE): -22.2907 (positive means the wire helps even without learning).
1. varsine centroid MSE beta 5/10/20: 0.1121 / 0.0391 / 0.0247 (beta=10 is primary; a >20% gap between the best beta and beta=10 flags beta-sensitivity).
1. varsine ridge MSE trained std/raw: 0.0423 / 0.0439; frozen std/raw: 0.0424 / 0.0435 (D8: if raw frozen beats raw trained while std trained beats std frozen, the round-18 frozen-ridge win was a regularization artifact).
1. varsine wire effect on the FROZEN net (frozen-off minus frozen-on proto MSE): -0.0282 (positive means the wire helps even without learning).
1. narma10 centroid MSE beta 5/10/20: 0.0124 / 0.0093 / 0.0091 (beta=10 is primary; a >20% gap between the best beta and beta=10 flags beta-sensitivity).
1. narma10 ridge MSE trained std/raw: 0.0079 / 0.0080; frozen std/raw: 0.0079 / 0.0079 (D8: if raw frozen beats raw trained while std trained beats std frozen, the round-18 frozen-ridge win was a regularization artifact).
1. narma10 wire effect on the FROZEN net (frozen-off minus frozen-on proto MSE): -0.0134 (positive means the wire helps even without learning).
1. mackeyglass centroid MSE beta 5/10/20: 0.0071 / 0.0022 / 0.0018 (beta=10 is primary; a >20% gap between the best beta and beta=10 flags beta-sensitivity).
1. mackeyglass ridge MSE trained std/raw: 0.0032 / 0.0034; frozen std/raw: 0.0033 / 0.0034 (D8: if raw frozen beats raw trained while std trained beats std frozen, the round-18 frozen-ridge win was a regularization artifact).
1. mackeyglass wire effect on the FROZEN net (frozen-off minus frozen-on proto MSE): -0.0066 (positive means the wire helps even without learning).

## Val summary (5 seeds 322-326)

1. lorenz grid|G1: val proto MSE mean=4.9682 median=5.3005 IQR=[2.5832,6.3213] 95% CI=[1.9330,8.0034] n=5
1. lorenz grid|G2: val proto MSE mean=6.1658 median=4.8742 IQR=[4.4873,8.4206] 95% CI=[3.3972,8.9343] n=5
1. lorenz grid|G3: val proto MSE mean=6.7281 median=6.7514 IQR=[6.2137,7.0923] 95% CI=[4.0759,9.3802] n=5
1. lorenz grid|G4: val proto MSE mean=4.6268 median=4.4440 IQR=[3.1066,5.4378] 95% CI=[2.0933,7.1603] n=5
1. lorenz grid|G5: val proto MSE mean=8.1088 median=7.7852 IQR=[5.2678,9.8040] 95% CI=[4.0644,12.1531] n=5
1. lorenz bumpsigma|2.0: val proto MSE mean=4.3965 median=4.2875 IQR=[2.8967,5.8291] 95% CI=[2.1459,6.6471] n=5
1. lorenz bumpsigma|3.0: val proto MSE mean=4.6281 median=4.5171 IQR=[2.8192,5.0521] 95% CI=[1.9818,7.2744] n=5
1. varsine grid|G1: val proto MSE mean=0.0373 median=0.0370 IQR=[0.0306,0.0458] 95% CI=[0.0259,0.0487] n=5
1. varsine grid|G2: val proto MSE mean=0.0391 median=0.0373 IQR=[0.0281,0.0481] 95% CI=[0.0213,0.0569] n=5
1. varsine grid|G3: val proto MSE mean=0.0320 median=0.0297 IQR=[0.0223,0.0374] 95% CI=[0.0143,0.0497] n=5
1. varsine grid|G4: val proto MSE mean=0.0240 median=0.0238 IQR=[0.0217,0.0305] 95% CI=[0.0150,0.0330] n=5
1. varsine grid|G5: val proto MSE mean=0.0539 median=0.0522 IQR=[0.0521,0.0629] 95% CI=[0.0418,0.0660] n=5
1. varsine bumpsigma|2.0: val proto MSE mean=0.0390 median=0.0377 IQR=[0.0275,0.0512] 95% CI=[0.0209,0.0570] n=5
1. varsine bumpsigma|3.0: val proto MSE mean=0.0236 median=0.0212 IQR=[0.0195,0.0306] 95% CI=[0.0126,0.0346] n=5
1. narma10 grid|G1: val proto MSE mean=0.0138 median=0.0144 IQR=[0.0132,0.0146] 95% CI=[0.0121,0.0155] n=5
1. narma10 grid|G2: val proto MSE mean=0.0133 median=0.0133 IQR=[0.0131,0.0150] 95% CI=[0.0109,0.0157] n=5
1. narma10 grid|G3: val proto MSE mean=0.0127 median=0.0135 IQR=[0.0117,0.0135] 95% CI=[0.0100,0.0154] n=5
1. narma10 grid|G4: val proto MSE mean=0.0129 median=0.0125 IQR=[0.0122,0.0138] 95% CI=[0.0107,0.0152] n=5
1. narma10 grid|G5: val proto MSE mean=0.0109 median=0.0111 IQR=[0.0109,0.0111] 95% CI=[0.0100,0.0117] n=5
1. narma10 bumpsigma|2.0: val proto MSE mean=0.0108 median=0.0110 IQR=[0.0105,0.0111] 95% CI=[0.0100,0.0116] n=5
1. narma10 bumpsigma|3.0: val proto MSE mean=0.0103 median=0.0105 IQR=[0.0098,0.0105] 95% CI=[0.0093,0.0114] n=5
1. mackeyglass grid|G1: val proto MSE mean=0.0021 median=0.0018 IQR=[0.0016,0.0024] 95% CI=[0.0012,0.0029] n=5
1. mackeyglass grid|G2: val proto MSE mean=0.0030 median=0.0031 IQR=[0.0026,0.0034] 95% CI=[0.0023,0.0036] n=5
1. mackeyglass grid|G3: val proto MSE mean=0.0035 median=0.0033 IQR=[0.0028,0.0036] 95% CI=[0.0021,0.0049] n=5
1. mackeyglass grid|G4: val proto MSE mean=0.0025 median=0.0021 IQR=[0.0019,0.0028] 95% CI=[0.0012,0.0039] n=5
1. mackeyglass grid|G5: val proto MSE mean=0.0031 median=0.0034 IQR=[0.0025,0.0034] 95% CI=[0.0019,0.0043] n=5
1. mackeyglass bumpsigma|2.0: val proto MSE mean=0.0022 median=0.0020 IQR=[0.0019,0.0024] 95% CI=[0.0014,0.0030] n=5
1. mackeyglass bumpsigma|3.0: val proto MSE mean=0.0025 median=0.0022 IQR=[0.0017,0.0027] 95% CI=[0.0011,0.0039] n=5
