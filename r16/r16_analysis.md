# Round 16 analysis (omission test, frozen weights)

Protocol: PROTOCOL-R16-OMISSION.md. Seeds 221..240 (20 fresh seeds). Chance = 1/16 = 0.0625.
CI = mean +/- 1.96 * sample-std / sqrt(n).

## Planted lag-1

- delta omit_m2: mean=0.966 median=0.971 IQR=[0.963,0.983] 95% CI=[0.955,0.977] n=20
- delta omit_m1: mean=0.667 median=0.699 IQR=[0.609,0.736] 95% CI=[0.622,0.711] n=20
- delta omit_filler: mean=0.059 median=0.060 IQR=[0.040,0.080] 95% CI=[0.046,0.072] n=20
- delta cont_m1: mean=0.610 median=0.617 IQR=[0.552,0.676] 95% CI=[0.569,0.651] n=20
- delta margin_m2: mean=0.345 median=0.344 IQR=[0.329,0.371] 95% CI=[0.332,0.359] n=20
- delta shuf_m2: mean=0.001 median=0.000 IQR=[0.000,0.000] 95% CI=[-0.001,0.003] n=20
- delta normal_m2: mean=0.966 median=0.971 IQR=[0.963,0.983] 95% CI=[0.955,0.977] n=20
- frozen omit_m2: mean=0.048 median=0.041 IQR=[0.017,0.081] 95% CI=[0.031,0.066] n=20
- frozen omit_m1: mean=0.058 median=0.043 IQR=[0.015,0.083] 95% CI=[0.032,0.084] n=20
- frozen omit_filler: mean=0.064 median=0.080 IQR=[0.040,0.080] 95% CI=[0.049,0.079] n=20
- frozen cont_m1: mean=0.048 median=0.032 IQR=[0.014,0.067] 95% CI=[0.027,0.069] n=20
- frozen margin_m2: mean=-0.004 median=0.001 IQR=[-0.011,0.006] 95% CI=[-0.012,0.004] n=20
- frozen shuf_m2: mean=0.087 median=0.048 IQR=[0.023,0.145] 95% CI=[0.048,0.125] n=20
- frozen normal_m2: mean=0.048 median=0.041 IQR=[0.017,0.081] 95% CI=[0.031,0.066] n=20
- paired delta omit_m2 - omit_filler: mean diff=+0.907 95% CI=[+0.888,+0.927] n=20
- paired delta cont_m1 - t1_filler: mean diff=+0.560 95% CI=[+0.518,+0.602] n=20
- paired delta - frozen cont_m1: mean diff=+0.562 95% CI=[+0.512,+0.612] n=20

- BAR A1 omit_m2 mean >= 0.80, lower CI > 0.50: PASS
- BAR A2 omit_m2 - filler paired mean > 0.30, CI above 0: PASS
- BAR A3 frozen omit_m2 mean < 0.15, margin < 0.05: PASS
- BAR A4 shuf_m2 mean < 0.15: PASS
- BAR A5 omit_m1 mean >= 0.50, lower CI > 0.25: PASS
- BAR A6 cont_m1 lower CI > chance (0.0625): PASS
- BAR A7 cont_m1 - t1_filler paired CI above 0: PASS

## Planted lag-2

- delta omit_m2: mean=0.943 median=0.942 IQR=[0.925,0.968] 95% CI=[0.928,0.957] n=20
- delta omit_m1: mean=0.678 median=0.688 IQR=[0.619,0.711] 95% CI=[0.647,0.709] n=20
- delta omit_filler: mean=0.089 median=0.080 IQR=[0.060,0.105] 95% CI=[0.071,0.107] n=20
- delta cont_m1: mean=0.467 median=0.487 IQR=[0.393,0.556] 95% CI=[0.424,0.510] n=20
- delta margin_m2: mean=0.293 median=0.299 IQR=[0.278,0.313] 95% CI=[0.277,0.309] n=20
- delta shuf_m2: mean=0.006 median=0.000 IQR=[0.000,0.009] 95% CI=[0.001,0.010] n=20
- delta normal_m2: mean=0.943 median=0.942 IQR=[0.925,0.968] 95% CI=[0.928,0.957] n=20
- frozen omit_m2: mean=0.058 median=0.048 IQR=[0.006,0.084] 95% CI=[0.031,0.084] n=20
- frozen omit_m1: mean=0.075 median=0.061 IQR=[0.025,0.114] 95% CI=[0.049,0.101] n=20
- frozen omit_filler: mean=0.075 median=0.080 IQR=[0.040,0.100] 95% CI=[0.060,0.090] n=20
- frozen cont_m1: mean=0.049 median=0.032 IQR=[0.006,0.081] 95% CI=[0.027,0.070] n=20
- frozen margin_m2: mean=-0.000 median=-0.002 IQR=[-0.012,0.010] 95% CI=[-0.009,0.008] n=20
- frozen shuf_m2: mean=0.121 median=0.062 IQR=[0.024,0.170] 95% CI=[0.057,0.184] n=20
- frozen normal_m2: mean=0.058 median=0.048 IQR=[0.006,0.084] 95% CI=[0.031,0.084] n=20
- paired delta omit_m2 - omit_filler: mean diff=+0.854 95% CI=[+0.827,+0.880] n=20
- paired delta cont_m1 - t1_filler: mean diff=+0.397 95% CI=[+0.346,+0.448] n=20
- paired delta - frozen cont_m1: mean diff=+0.418 95% CI=[+0.372,+0.465] n=20

- BAR B1 omit_m2 lower CI > chance (0.0625): PASS
- BAR B2 omit_m2 - filler paired CI above 0: PASS
- BAR B3 frozen omit_m2 mean < 0.15: PASS
- BAR B4 cont_m1 lower CI > chance (0.0625): PASS

## Summary: 11/11 bars pass

Experiment A passes if A1-A5 pass on lag-1. A6, A7, and the lag-2 bars are reported separately.