# Round 17 analysis (top-down clamp, experiment B)

Protocol: PROTOCOL-R17-CLAMP.md. Seeds 241..260 (20 fresh seeds). Chance = 1/16 = 0.0625.
CI = mean +/- 1.96 * sample-std / sqrt(n).

## Validation (planted lag-1, val seeds 241..245)

- fb_gain=0.0: val_acc mean=0.227 median=0.231 IQR=[0.221,0.232] 95% CI=[0.220,0.233] n=5
- fb_gain=0.1: val_acc mean=0.226 median=0.224 IQR=[0.223,0.227] 95% CI=[0.221,0.232] n=5
- fb_gain=0.3: val_acc mean=0.221 median=0.220 IQR=[0.217,0.223] 95% CI=[0.213,0.229] n=5
- fb_gain=1.0: val_acc mean=0.182 median=0.177 IQR=[0.174,0.193] 95% CI=[0.168,0.195] n=5
- winner fb_gain=0.1 (val_acc=0.226); fb_gain=0.0 val_acc=0.227; guard (>= 0.80x): PASS

## Planted lag-1

- wire-on omit_m2: mean=0.953 median=0.963 IQR=[0.929,0.975] 95% CI=[0.939,0.967] n=20
- wire-on omit_m1: mean=0.684 median=0.649 IQR=[0.616,0.743] 95% CI=[0.647,0.722] n=20
- wire-on cont_m1: mean=0.581 median=0.574 IQR=[0.550,0.612] 95% CI=[0.549,0.613] n=20
- wire-on normal_acc: mean=0.228 median=0.227 IQR=[0.220,0.237] 95% CI=[0.223,0.233] n=20
- wire-on fb_rms_ratio: mean=0.065 median=0.064 IQR=[0.063,0.066] 95% CI=[0.063,0.066] n=20
- wire-cut omit_m2: mean=0.948 median=0.956 IQR=[0.926,0.969] 95% CI=[0.934,0.962] n=20
- wire-cut omit_m1: mean=0.696 median=0.684 IQR=[0.645,0.735] 95% CI=[0.662,0.731] n=20
- wire-cut cont_m1: mean=0.587 median=0.585 IQR=[0.550,0.620] 95% CI=[0.555,0.620] n=20
- wire-cut normal_acc: mean=0.229 median=0.230 IQR=[0.223,0.233] 95% CI=[0.224,0.233] n=20
- wire-cut fb_rms_ratio: mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- wire-shuffle omit_m2: mean=0.953 median=0.958 IQR=[0.933,0.969] 95% CI=[0.941,0.965] n=20
- wire-shuffle omit_m1: mean=0.698 median=0.681 IQR=[0.631,0.765] 95% CI=[0.659,0.736] n=20
- wire-shuffle cont_m1: mean=0.587 median=0.584 IQR=[0.532,0.626] 95% CI=[0.551,0.623] n=20
- wire-shuffle normal_acc: mean=0.229 median=0.228 IQR=[0.223,0.237] 95% CI=[0.224,0.234] n=20
- wire-shuffle fb_rms_ratio: mean=0.064 median=0.064 IQR=[0.063,0.066] 95% CI=[0.063,0.066] n=20
- wire-off omit_m2: mean=0.948 median=0.949 IQR=[0.927,0.968] 95% CI=[0.935,0.962] n=20
- wire-off omit_m1: mean=0.700 median=0.684 IQR=[0.644,0.759] 95% CI=[0.665,0.735] n=20
- wire-off cont_m1: mean=0.584 median=0.586 IQR=[0.555,0.629] 95% CI=[0.554,0.615] n=20
- wire-off normal_acc: mean=0.229 median=0.228 IQR=[0.222,0.235] 95% CI=[0.224,0.234] n=20
- wire-off fb_rms_ratio: mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- frozen omit_m2: mean=0.083 median=0.069 IQR=[0.035,0.114] 95% CI=[0.049,0.116] n=20
- frozen omit_m1: mean=0.057 median=0.053 IQR=[0.017,0.085] 95% CI=[0.036,0.077] n=20
- frozen cont_m1: mean=0.078 median=0.049 IQR=[0.029,0.136] 95% CI=[0.046,0.110] n=20
- frozen normal_acc: mean=0.062 median=0.060 IQR=[0.056,0.067] 95% CI=[0.059,0.066] n=20
- frozen fb_rms_ratio: mean=0.229 median=0.227 IQR=[0.225,0.233] 95% CI=[0.226,0.233] n=20
- wire-on p1_cos (on vs cut): mean=0.998 median=0.998 IQR=[0.998,0.998] 95% CI=[0.998,0.998] n=20
- wire-on err_mag first epoch: mean=3.032 median=3.042 IQR=[2.995,3.069] 95% CI=[3.011,3.053] n=20
- wire-on err_mag last epoch: mean=1.619 median=1.625 IQR=[1.585,1.638] 95% CI=[1.602,1.636] n=20
- wire-off err_mag last epoch: mean=1.617 median=1.622 IQR=[1.583,1.636] 95% CI=[1.600,1.634] n=20
- paired wire-on - wire-cut omit_m2: mean diff=+0.005 95% CI=[+0.000,+0.010] n=20
- paired wire-on - wire-shuffle omit_m2: mean diff=-0.001 95% CI=[-0.006,+0.005] n=20
- paired wire-on - wire-off omit_m2: mean diff=+0.004 95% CI=[-0.000,+0.009] n=20

- BAR C1 cut degrades omit_m2 by >= 0.10, CI above 0: FAIL
- BAR C2 shuffle degrades omit_m2 by >= 0.10, CI above 0: FAIL
- BAR C3 p1_cos mean < 0.90: FAIL
- BAR C4 wire-off omit_m2 differs (paired CI excludes 0): FAIL
- BAR C5 guard: wire-on normal_acc >= 0.80x wire-off: PASS

## Planted lag-2

- wire-on omit_m2: mean=0.924 median=0.927 IQR=[0.904,0.953] 95% CI=[0.905,0.943] n=20
- wire-on omit_m1: mean=0.666 median=0.677 IQR=[0.634,0.705] 95% CI=[0.631,0.701] n=20
- wire-on cont_m1: mean=0.490 median=0.509 IQR=[0.422,0.574] 95% CI=[0.445,0.535] n=20
- wire-on normal_acc: mean=0.221 median=0.219 IQR=[0.215,0.222] 95% CI=[0.217,0.225] n=20
- wire-on fb_rms_ratio: mean=0.064 median=0.064 IQR=[0.062,0.066] 95% CI=[0.063,0.066] n=20
- wire-cut omit_m2: mean=0.920 median=0.917 IQR=[0.892,0.953] 95% CI=[0.899,0.940] n=20
- wire-cut omit_m1: mean=0.669 median=0.685 IQR=[0.634,0.703] 95% CI=[0.637,0.701] n=20
- wire-cut cont_m1: mean=0.495 median=0.515 IQR=[0.407,0.561] 95% CI=[0.449,0.541] n=20
- wire-cut normal_acc: mean=0.220 median=0.218 IQR=[0.216,0.226] 95% CI=[0.216,0.225] n=20
- wire-cut fb_rms_ratio: mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- wire-shuffle omit_m2: mean=0.922 median=0.926 IQR=[0.891,0.960] 95% CI=[0.903,0.942] n=20
- wire-shuffle omit_m1: mean=0.667 median=0.677 IQR=[0.625,0.716] 95% CI=[0.631,0.703] n=20
- wire-shuffle cont_m1: mean=0.497 median=0.504 IQR=[0.420,0.578] 95% CI=[0.448,0.546] n=20
- wire-shuffle normal_acc: mean=0.221 median=0.220 IQR=[0.215,0.224] 95% CI=[0.216,0.226] n=20
- wire-shuffle fb_rms_ratio: mean=0.064 median=0.064 IQR=[0.062,0.066] 95% CI=[0.063,0.066] n=20
- wire-off omit_m2: mean=0.922 median=0.924 IQR=[0.898,0.958] 95% CI=[0.899,0.944] n=20
- wire-off omit_m1: mean=0.663 median=0.670 IQR=[0.634,0.693] 95% CI=[0.631,0.695] n=20
- wire-off cont_m1: mean=0.501 median=0.504 IQR=[0.391,0.578] 95% CI=[0.452,0.550] n=20
- wire-off normal_acc: mean=0.220 median=0.219 IQR=[0.215,0.225] 95% CI=[0.216,0.224] n=20
- wire-off fb_rms_ratio: mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- frozen omit_m2: mean=0.092 median=0.082 IQR=[0.057,0.130] 95% CI=[0.066,0.118] n=20
- frozen omit_m1: mean=0.056 median=0.044 IQR=[0.017,0.062] 95% CI=[0.029,0.084] n=20
- frozen cont_m1: mean=0.094 median=0.074 IQR=[0.055,0.133] 95% CI=[0.067,0.122] n=20
- frozen normal_acc: mean=0.066 median=0.065 IQR=[0.062,0.070] 95% CI=[0.063,0.069] n=20
- frozen fb_rms_ratio: mean=0.230 median=0.228 IQR=[0.225,0.233] 95% CI=[0.226,0.233] n=20
- wire-on p1_cos (on vs cut): mean=0.998 median=0.998 IQR=[0.998,0.998] 95% CI=[0.998,0.998] n=20
- wire-on err_mag first epoch: mean=3.043 median=3.055 IQR=[3.006,3.088] 95% CI=[3.021,3.066] n=20
- wire-on err_mag last epoch: mean=1.628 median=1.632 IQR=[1.593,1.655] 95% CI=[1.611,1.645] n=20
- wire-off err_mag last epoch: mean=1.626 median=1.629 IQR=[1.590,1.653] 95% CI=[1.609,1.643] n=20
- paired wire-on - wire-cut omit_m2: mean diff=+0.004 95% CI=[+0.000,+0.008] n=20
- paired wire-on - wire-shuffle omit_m2: mean diff=+0.002 95% CI=[-0.005,+0.008] n=20
- paired wire-on - wire-off omit_m2: mean diff=+0.002 95% CI=[-0.005,+0.010] n=20

- BAR C1-lag2 cut degrades omit_m2, CI above 0: PASS
- BAR C2-lag2 shuffle degrades omit_m2, CI above 0: FAIL
- BAR C3-lag2 p1_cos mean < 0.90: FAIL
- BAR C4-lag2 wire-off omit_m2 differs (CI excludes 0): FAIL

## Summary: 2/9 bars pass

Experiment B passes if C1-C4 pass on lag-1. C5 is a guard, not a pass bar. Lag-2 bars are reported the same way.