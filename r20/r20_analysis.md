# Round 20 analysis: aligned-feedback clamp + rollout

Protocol: PROTOCOL-R20-ALIGNED.md. Seeds 302..321 (20 fresh seeds). Chance: planted 1/16 = 0.0625, sine 1/32 = 0.03125.
CI = mean +/- exact Student-t * sample-std / sqrt(n): t(19) = 2.0930 for test bars, t(4) = 2.7760 for val.
Test lines: 660 (expected 660), unique keys: 660.

## Validation (planted1-occ, val seeds 302..306)

Winner rule (protocol section 7): largest val on-cut omit_m2 gap among gains passing the state gate (val p1_cos < 0.90) and stable (r1_max < 10, no NaN); tie-break smallest gain. If no gain passes the gate, the largest gap among stable gains wins and the gate failure is flagged. The val-phase guard (val normal_acc >= 0.80x wire-off) is computed from the test phase (B7); the val phase has no wire-off condition.
### Wire aligned
- fb_gain=0.1: val_acc mean=0.229 median=0.228 IQR=[0.223,0.230] 95% CI=[0.212,0.246] n=5; val on-cut gap mean=0.005 median=0.008 IQR=[0.000,0.008] 95% CI=[-0.007,0.017] n=5; val p1_cos mean=0.986 median=0.986 IQR=[0.986,0.987] 95% CI=[0.985,0.987] n=5
- fb_gain=0.3: val_acc mean=0.233 median=0.235 IQR=[0.223,0.237] 95% CI=[0.219,0.247] n=5; val on-cut gap mean=0.007 median=0.008 IQR=[0.000,0.009] 95% CI=[-0.002,0.016] n=5; val p1_cos mean=0.949 median=0.948 IQR=[0.946,0.951] 95% CI=[0.945,0.952] n=5
- fb_gain=1.0: val_acc mean=0.236 median=0.232 IQR=[0.231,0.242] 95% CI=[0.227,0.246] n=5; val on-cut gap mean=0.017 median=0.009 IQR=[0.008,0.026] 95% CI=[-0.004,0.039] n=5; val p1_cos mean=0.805 median=0.806 IQR=[0.797,0.812] 95% CI=[0.795,0.815] n=5
- winner aligned: fb_gain=1.0 (val gap +0.017, p1_cos 0.805, val_acc 0.236, gate PASS)

### Wire learned
- fb_gain=0.1: val_acc mean=0.231 median=0.230 IQR=[0.226,0.232] 95% CI=[0.220,0.241] n=5; val on-cut gap mean=0.005 median=0.000 IQR=[0.000,0.009] 95% CI=[-0.011,0.021] n=5; val p1_cos mean=0.983 median=0.984 IQR=[0.983,0.984] 95% CI=[0.981,0.985] n=5
- fb_gain=0.3: val_acc mean=0.235 median=0.233 IQR=[0.231,0.236] 95% CI=[0.224,0.245] n=5; val on-cut gap mean=0.014 median=0.017 IQR=[0.008,0.017] 95% CI=[0.001,0.026] n=5; val p1_cos mean=0.931 median=0.930 IQR=[0.930,0.932] 95% CI=[0.928,0.934] n=5
- fb_gain=1.0: val_acc mean=0.232 median=0.232 IQR=[0.222,0.236] 95% CI=[0.217,0.247] n=5; val on-cut gap mean=-0.002 median=0.000 IQR=[-0.008,0.017] 95% CI=[-0.028,0.025] n=5; val p1_cos mean=0.751 median=0.753 IQR=[0.740,0.762] 95% CI=[0.736,0.766] n=5
- winner learned: fb_gain=1.0 (val gap -0.002, p1_cos 0.751, val_acc 0.232, gate PASS)

### Wire random
- fb_gain=0.1: val_acc mean=0.223 median=0.221 IQR=[0.219,0.223] 95% CI=[0.213,0.232] n=5; val on-cut gap mean=-0.007 median=-0.008 IQR=[-0.009,0.000] 95% CI=[-0.016,0.002] n=5; val p1_cos mean=0.990 median=0.991 IQR=[0.989,0.991] 95% CI=[0.988,0.992] n=5
- fb_gain=0.3: val_acc mean=0.223 median=0.221 IQR=[0.219,0.223] 95% CI=[0.213,0.234] n=5; val on-cut gap mean=0.004 median=0.000 IQR=[0.000,0.009] 95% CI=[-0.009,0.016] n=5; val p1_cos mean=0.968 median=0.967 IQR=[0.966,0.970] 95% CI=[0.965,0.972] n=5
- fb_gain=1.0: val_acc mean=0.190 median=0.188 IQR=[0.180,0.194] 95% CI=[0.175,0.206] n=5; val on-cut gap mean=-0.088 median=-0.103 IQR=[-0.121,-0.043] 95% CI=[-0.154,-0.023] n=5; val p1_cos mean=0.754 median=0.749 IQR=[0.746,0.764] 95% CI=[0.736,0.771] n=5
- winner random: fb_gain=1.0 (val gap -0.088, p1_cos 0.754, val_acc 0.190, gate PASS)

## Test: condition means

### Planted lag-1, 40% occluded

- aligned-on: normal_acc mean=0.232 median=0.232 IQR=[0.227,0.238] 95% CI=[0.227,0.237] n=20; omit_m2 mean=0.993 median=0.992 IQR=[0.990,1.000] 95% CI=[0.990,0.996] n=20; omit_m1 mean=0.878 median=0.881 IQR=[0.839,0.913] 95% CI=[0.856,0.900] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.613 median=0.610 IQR=[0.540,0.680] 95% CI=[0.551,0.675] n=20; p1_cos mean=0.804 median=0.806 IQR=[0.799,0.815] 95% CI=[0.797,0.811] n=20; fb_drive_ratio mean=0.276 median=0.277 IQR=[0.270,0.280] 95% CI=[0.272,0.279] n=20
- aligned-cut: normal_acc mean=0.223 median=0.223 IQR=[0.218,0.228] 95% CI=[0.218,0.228] n=20; omit_m2 mean=0.979 median=0.983 IQR=[0.972,0.992] 95% CI=[0.971,0.988] n=20; omit_m1 mean=0.792 median=0.782 IQR=[0.742,0.850] 95% CI=[0.759,0.824] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.759 median=0.760 IQR=[0.710,0.800] 95% CI=[0.720,0.798] n=20; p1_cos mean=0.804 median=0.806 IQR=[0.799,0.815] 95% CI=[0.797,0.811] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- aligned-shuffle: normal_acc mean=0.214 median=0.217 IQR=[0.207,0.220] 95% CI=[0.208,0.219] n=20; omit_m2 mean=0.960 median=0.967 IQR=[0.948,0.975] 95% CI=[0.948,0.972] n=20; omit_m1 mean=0.741 median=0.744 IQR=[0.692,0.789] 95% CI=[0.701,0.781] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.722 median=0.710 IQR=[0.655,0.765] 95% CI=[0.678,0.766] n=20; p1_cos mean=0.804 median=0.806 IQR=[0.799,0.815] 95% CI=[0.797,0.811] n=20; fb_drive_ratio mean=0.236 median=0.236 IQR=[0.227,0.241] 95% CI=[0.231,0.241] n=20
- learned-on: normal_acc mean=0.227 median=0.227 IQR=[0.223,0.231] 95% CI=[0.222,0.232] n=20; omit_m2 mean=0.980 median=0.983 IQR=[0.967,0.992] 95% CI=[0.973,0.988] n=20; omit_m1 mean=0.851 median=0.845 IQR=[0.816,0.891] 95% CI=[0.825,0.877] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.567 median=0.560 IQR=[0.515,0.605] 95% CI=[0.517,0.617] n=20; p1_cos mean=0.744 median=0.749 IQR=[0.725,0.764] 95% CI=[0.734,0.755] n=20; fb_drive_ratio mean=0.360 median=0.361 IQR=[0.352,0.367] 95% CI=[0.355,0.364] n=20
- learned-cut: normal_acc mean=0.219 median=0.216 IQR=[0.212,0.225] 95% CI=[0.214,0.224] n=20; omit_m2 mean=0.976 median=0.983 IQR=[0.966,0.991] 95% CI=[0.968,0.985] n=20; omit_m1 mean=0.778 median=0.771 IQR=[0.734,0.829] 95% CI=[0.747,0.809] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.763 median=0.770 IQR=[0.720,0.805] 95% CI=[0.720,0.806] n=20; p1_cos mean=0.744 median=0.749 IQR=[0.725,0.764] 95% CI=[0.734,0.755] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- learned-shuffle: normal_acc mean=0.210 median=0.210 IQR=[0.202,0.219] 95% CI=[0.205,0.216] n=20; omit_m2 mean=0.952 median=0.953 IQR=[0.935,0.965] 95% CI=[0.938,0.965] n=20; omit_m1 mean=0.723 median=0.716 IQR=[0.683,0.776] 95% CI=[0.689,0.757] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.722 median=0.730 IQR=[0.675,0.765] 95% CI=[0.683,0.761] n=20; p1_cos mean=0.744 median=0.749 IQR=[0.725,0.764] 95% CI=[0.734,0.755] n=20; fb_drive_ratio mean=0.262 median=0.262 IQR=[0.253,0.268] 95% CI=[0.257,0.267] n=20
- random-on: normal_acc mean=0.183 median=0.184 IQR=[0.172,0.193] 95% CI=[0.177,0.189] n=20; omit_m2 mean=0.873 median=0.882 IQR=[0.838,0.921] 95% CI=[0.845,0.902] n=20; omit_m1 mean=0.556 median=0.564 IQR=[0.490,0.611] 95% CI=[0.514,0.599] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.472 median=0.460 IQR=[0.435,0.550] 95% CI=[0.418,0.526] n=20; p1_cos mean=0.763 median=0.764 IQR=[0.749,0.776] 95% CI=[0.755,0.771] n=20; fb_drive_ratio mean=0.403 median=0.401 IQR=[0.396,0.413] 95% CI=[0.395,0.410] n=20
- random-cut: normal_acc mean=0.201 median=0.199 IQR=[0.185,0.215] 95% CI=[0.193,0.208] n=20; omit_m2 mean=0.931 median=0.941 IQR=[0.895,0.977] 95% CI=[0.906,0.955] n=20; omit_m1 mean=0.658 median=0.648 IQR=[0.618,0.740] 95% CI=[0.612,0.704] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.482 median=0.480 IQR=[0.420,0.540] 95% CI=[0.438,0.526] n=20; p1_cos mean=0.763 median=0.764 IQR=[0.749,0.776] 95% CI=[0.755,0.771] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- random-shuffle: normal_acc mean=0.178 median=0.183 IQR=[0.171,0.190] 95% CI=[0.171,0.186] n=20; omit_m2 mean=0.831 median=0.854 IQR=[0.820,0.886] 95% CI=[0.791,0.870] n=20; omit_m1 mean=0.540 median=0.569 IQR=[0.453,0.605] 95% CI=[0.494,0.585] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.429 median=0.440 IQR=[0.355,0.500] 95% CI=[0.383,0.475] n=20; p1_cos mean=0.763 median=0.764 IQR=[0.749,0.776] 95% CI=[0.755,0.771] n=20; fb_drive_ratio mean=0.405 median=0.404 IQR=[0.388,0.420] 95% CI=[0.396,0.414] n=20
- wire-off: normal_acc mean=0.221 median=0.217 IQR=[0.212,0.231] 95% CI=[0.215,0.227] n=20; omit_m2 mean=0.984 median=0.983 IQR=[0.975,0.994] 95% CI=[0.978,0.990] n=20; omit_m1 mean=0.783 median=0.779 IQR=[0.734,0.836] 95% CI=[0.746,0.820] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.659 median=0.640 IQR=[0.600,0.740] 95% CI=[0.613,0.705] n=20; p1_cos n/a (structurally empty); fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- frozen-aligned: normal_acc mean=0.046 median=0.043 IQR=[0.038,0.053] 95% CI=[0.041,0.051] n=20; omit_m2 mean=0.099 median=0.076 IQR=[0.033,0.137] 95% CI=[0.057,0.140] n=20; omit_m1 mean=0.045 median=0.050 IQR=[0.015,0.057] 95% CI=[0.028,0.062] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.001 median=0.000 IQR=[0.000,0.000] 95% CI=[-0.001,0.003] n=20; p1_cos mean=0.309 median=0.308 IQR=[0.292,0.346] 95% CI=[0.287,0.330] n=20; fb_drive_ratio mean=0.880 median=0.892 IQR=[0.848,0.907] 95% CI=[0.860,0.900] n=20

### Planted lag-2, 40% occluded

- aligned-on: normal_acc mean=0.224 median=0.223 IQR=[0.216,0.232] 95% CI=[0.218,0.229] n=20; omit_m2 mean=0.961 median=0.966 IQR=[0.958,0.975] 95% CI=[0.948,0.975] n=20; omit_m1 mean=0.839 median=0.841 IQR=[0.801,0.874] 95% CI=[0.814,0.865] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.536 median=0.560 IQR=[0.435,0.590] 95% CI=[0.476,0.596] n=20; p1_cos mean=0.810 median=0.809 IQR=[0.795,0.819] 95% CI=[0.802,0.817] n=20; fb_drive_ratio mean=0.273 median=0.273 IQR=[0.267,0.277] 95% CI=[0.270,0.276] n=20
- aligned-cut: normal_acc mean=0.221 median=0.223 IQR=[0.210,0.230] 95% CI=[0.215,0.226] n=20; omit_m2 mean=0.967 median=0.975 IQR=[0.956,0.984] 95% CI=[0.952,0.982] n=20; omit_m1 mean=0.805 median=0.802 IQR=[0.770,0.830] 95% CI=[0.782,0.828] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.761 median=0.780 IQR=[0.695,0.825] 95% CI=[0.716,0.806] n=20; p1_cos mean=0.810 median=0.809 IQR=[0.795,0.819] 95% CI=[0.802,0.817] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- aligned-shuffle: normal_acc mean=0.212 median=0.212 IQR=[0.203,0.221] 95% CI=[0.207,0.218] n=20; omit_m2 mean=0.946 median=0.957 IQR=[0.923,0.969] 95% CI=[0.928,0.964] n=20; omit_m1 mean=0.759 median=0.771 IQR=[0.725,0.793] 95% CI=[0.730,0.789] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.691 median=0.700 IQR=[0.635,0.750] 95% CI=[0.638,0.744] n=20; p1_cos mean=0.810 median=0.809 IQR=[0.795,0.819] 95% CI=[0.802,0.817] n=20; fb_drive_ratio mean=0.236 median=0.236 IQR=[0.230,0.242] 95% CI=[0.232,0.239] n=20
- learned-on: normal_acc mean=0.216 median=0.214 IQR=[0.209,0.225] 95% CI=[0.210,0.222] n=20; omit_m2 mean=0.929 median=0.929 IQR=[0.918,0.951] 95% CI=[0.912,0.946] n=20; omit_m1 mean=0.811 median=0.798 IQR=[0.770,0.865] 95% CI=[0.787,0.836] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.373 median=0.410 IQR=[0.295,0.485] 95% CI=[0.314,0.432] n=20; p1_cos mean=0.749 median=0.749 IQR=[0.732,0.767] 95% CI=[0.737,0.761] n=20; fb_drive_ratio mean=0.357 median=0.357 IQR=[0.349,0.361] 95% CI=[0.353,0.361] n=20
- learned-cut: normal_acc mean=0.217 median=0.218 IQR=[0.206,0.227] 95% CI=[0.211,0.223] n=20; omit_m2 mean=0.956 median=0.959 IQR=[0.950,0.975] 95% CI=[0.942,0.970] n=20; omit_m1 mean=0.789 median=0.786 IQR=[0.750,0.821] 95% CI=[0.761,0.817] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.739 median=0.760 IQR=[0.680,0.800] 95% CI=[0.694,0.784] n=20; p1_cos mean=0.749 median=0.749 IQR=[0.732,0.767] 95% CI=[0.737,0.761] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- learned-shuffle: normal_acc mean=0.208 median=0.207 IQR=[0.196,0.217] 95% CI=[0.201,0.214] n=20; omit_m2 mean=0.920 median=0.931 IQR=[0.898,0.950] 95% CI=[0.900,0.941] n=20; omit_m1 mean=0.724 median=0.721 IQR=[0.660,0.779] 95% CI=[0.689,0.759] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.662 median=0.660 IQR=[0.615,0.715] 95% CI=[0.614,0.710] n=20; p1_cos mean=0.749 median=0.749 IQR=[0.732,0.767] 95% CI=[0.737,0.761] n=20; fb_drive_ratio mean=0.263 median=0.262 IQR=[0.259,0.270] 95% CI=[0.259,0.268] n=20
- random-on: normal_acc mean=0.181 median=0.183 IQR=[0.175,0.191] 95% CI=[0.175,0.187] n=20; omit_m2 mean=0.813 median=0.819 IQR=[0.759,0.865] 95% CI=[0.782,0.844] n=20; omit_m1 mean=0.601 median=0.620 IQR=[0.581,0.651] 95% CI=[0.560,0.643] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.444 median=0.440 IQR=[0.375,0.540] 95% CI=[0.392,0.496] n=20; p1_cos mean=0.766 median=0.766 IQR=[0.753,0.783] 95% CI=[0.757,0.775] n=20; fb_drive_ratio mean=0.404 median=0.402 IQR=[0.394,0.413] 95% CI=[0.396,0.411] n=20
- random-cut: normal_acc mean=0.197 median=0.200 IQR=[0.186,0.213] 95% CI=[0.190,0.205] n=20; omit_m2 mean=0.868 median=0.870 IQR=[0.834,0.920] 95% CI=[0.836,0.900] n=20; omit_m1 mean=0.684 median=0.697 IQR=[0.624,0.748] 95% CI=[0.636,0.731] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.449 median=0.400 IQR=[0.375,0.540] 95% CI=[0.401,0.497] n=20; p1_cos mean=0.766 median=0.766 IQR=[0.753,0.783] 95% CI=[0.757,0.775] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- random-shuffle: normal_acc mean=0.165 median=0.164 IQR=[0.160,0.176] 95% CI=[0.158,0.173] n=20; omit_m2 mean=0.710 median=0.732 IQR=[0.634,0.790] 95% CI=[0.654,0.766] n=20; omit_m1 mean=0.560 median=0.580 IQR=[0.516,0.631] 95% CI=[0.507,0.614] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.307 median=0.250 IQR=[0.190,0.410] 95% CI=[0.238,0.376] n=20; p1_cos mean=0.766 median=0.766 IQR=[0.753,0.783] 95% CI=[0.757,0.775] n=20; fb_drive_ratio mean=0.405 median=0.402 IQR=[0.390,0.418] 95% CI=[0.396,0.413] n=20
- wire-off: normal_acc mean=0.220 median=0.223 IQR=[0.208,0.232] 95% CI=[0.214,0.226] n=20; omit_m2 mean=0.971 median=0.971 IQR=[0.959,0.984] 95% CI=[0.960,0.981] n=20; omit_m1 mean=0.802 median=0.805 IQR=[0.776,0.837] 95% CI=[0.774,0.831] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.694 median=0.680 IQR=[0.655,0.720] 95% CI=[0.657,0.731] n=20; p1_cos n/a (structurally empty); fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- frozen-aligned: normal_acc mean=0.045 median=0.044 IQR=[0.036,0.052] 95% CI=[0.040,0.050] n=20; omit_m2 mean=0.087 median=0.049 IQR=[0.015,0.160] 95% CI=[0.042,0.131] n=20; omit_m1 mean=0.039 median=0.033 IQR=[0.014,0.058] 95% CI=[0.024,0.054] n=20; rollout_acc n/a (structurally empty); rollout_comp mean=0.002 median=0.000 IQR=[0.000,0.000] 95% CI=[-0.001,0.005] n=20; p1_cos mean=0.304 median=0.314 IQR=[0.284,0.344] 95% CI=[0.282,0.327] n=20; fb_drive_ratio mean=0.884 median=0.894 IQR=[0.848,0.917] 95% CI=[0.864,0.905] n=20

### Sine-fixed

- aligned-on: normal_acc mean=0.666 median=0.686 IQR=[0.638,0.703] 95% CI=[0.640,0.691] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.147 median=0.148 IQR=[0.134,0.160] 95% CI=[0.136,0.158] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.818 median=0.826 IQR=[0.801,0.832] 95% CI=[0.804,0.831] n=20; fb_drive_ratio mean=0.272 median=0.269 IQR=[0.265,0.279] 95% CI=[0.267,0.277] n=20
- aligned-cut: normal_acc mean=0.672 median=0.669 IQR=[0.650,0.700] 95% CI=[0.650,0.693] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.184 median=0.190 IQR=[0.157,0.205] 95% CI=[0.167,0.200] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.818 median=0.826 IQR=[0.801,0.832] 95% CI=[0.804,0.831] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- aligned-shuffle: normal_acc mean=0.626 median=0.620 IQR=[0.602,0.672] 95% CI=[0.600,0.653] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.172 median=0.170 IQR=[0.153,0.191] 95% CI=[0.158,0.186] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.818 median=0.826 IQR=[0.801,0.832] 95% CI=[0.804,0.831] n=20; fb_drive_ratio mean=0.168 median=0.166 IQR=[0.163,0.173] 95% CI=[0.165,0.171] n=20
- learned-on: normal_acc mean=0.504 median=0.502 IQR=[0.484,0.537] 95% CI=[0.473,0.534] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.110 median=0.107 IQR=[0.097,0.122] 95% CI=[0.100,0.119] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.773 median=0.778 IQR=[0.751,0.801] 95% CI=[0.756,0.791] n=20; fb_drive_ratio mean=0.334 median=0.334 IQR=[0.326,0.345] 95% CI=[0.327,0.341] n=20
- learned-cut: normal_acc mean=0.575 median=0.564 IQR=[0.544,0.595] 95% CI=[0.545,0.606] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.144 median=0.148 IQR=[0.130,0.164] 95% CI=[0.132,0.156] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.773 median=0.778 IQR=[0.751,0.801] 95% CI=[0.756,0.791] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- learned-shuffle: normal_acc mean=0.513 median=0.508 IQR=[0.482,0.555] 95% CI=[0.484,0.542] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.132 median=0.130 IQR=[0.117,0.154] 95% CI=[0.120,0.144] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.773 median=0.778 IQR=[0.751,0.801] 95% CI=[0.756,0.791] n=20; fb_drive_ratio mean=0.199 median=0.197 IQR=[0.194,0.204] 95% CI=[0.195,0.202] n=20
- random-on: normal_acc mean=0.591 median=0.598 IQR=[0.548,0.651] 95% CI=[0.551,0.631] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.143 median=0.150 IQR=[0.118,0.172] 95% CI=[0.126,0.160] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.691 median=0.710 IQR=[0.675,0.731] 95% CI=[0.664,0.718] n=20; fb_drive_ratio mean=0.465 median=0.462 IQR=[0.447,0.478] 95% CI=[0.455,0.476] n=20
- random-cut: normal_acc mean=0.488 median=0.517 IQR=[0.412,0.568] 95% CI=[0.443,0.532] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.137 median=0.138 IQR=[0.120,0.158] 95% CI=[0.124,0.150] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.691 median=0.710 IQR=[0.675,0.731] 95% CI=[0.664,0.718] n=20; fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- random-shuffle: normal_acc mean=0.267 median=0.257 IQR=[0.219,0.307] 95% CI=[0.232,0.302] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.114 median=0.113 IQR=[0.103,0.127] 95% CI=[0.102,0.126] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.691 median=0.710 IQR=[0.675,0.731] 95% CI=[0.664,0.718] n=20; fb_drive_ratio mean=0.428 median=0.429 IQR=[0.402,0.454] 95% CI=[0.411,0.445] n=20
- wire-off: normal_acc mean=0.892 median=0.889 IQR=[0.876,0.912] 95% CI=[0.881,0.904] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.223 median=0.217 IQR=[0.200,0.250] 95% CI=[0.208,0.238] n=20; rollout_comp n/a (structurally empty); p1_cos n/a (structurally empty); fb_drive_ratio mean=0.000 median=0.000 IQR=[0.000,0.000] 95% CI=[0.000,0.000] n=20
- frozen-aligned: normal_acc mean=0.032 median=0.028 IQR=[0.026,0.036] 95% CI=[0.028,0.036] n=20; omit_m2 n/a (structurally empty); omit_m1 n/a (structurally empty); rollout_acc mean=0.032 median=0.028 IQR=[0.022,0.041] 95% CI=[0.025,0.040] n=20; rollout_comp n/a (structurally empty); p1_cos mean=0.230 median=0.228 IQR=[0.212,0.262] 95% CI=[0.206,0.253] n=20; fb_drive_ratio mean=0.928 median=0.927 IQR=[0.909,0.960] 95% CI=[0.910,0.945] n=20

## Bars: experiment B' (planted, occluded)

### lag-1 (planted1-occ)
- B10-lag-1 (diagnostic, not a bar): learned gap minus aligned gap -0.009 CI [-0.018,-0.001]

### lag-2 (planted2-occ)
- B10-lag-2 (diagnostic, not a bar): learned gap minus aligned gap -0.021 CI [-0.032,-0.010]

## Bars: experiment D (rollout)

## Bar verdicts

- **B1-lag-1 FAIL**: aligned on-cut omit_m2 +0.014 CI [0.005,0.022] (bar >= +0.05, lower CI > 0)
- **B2-lag-1 FAIL**: aligned on-shuffle omit_m2 +0.033 CI [0.022,0.044] (bar >= +0.05, lower CI > 0)
- **B3-lag-1 PASS**: aligned p1_cos mean=0.804 median=0.806 IQR=[0.799,0.815] 95% CI=[0.797,0.811] n=20 (bar mean < 0.90)
- **B4-lag-1 PASS**: aligned on-off omit_m2 +0.009 CI [0.003,0.015] (bar CI excludes 0)
- **B5-lag-1 FAIL**: aligned err_mag_last on-off +0.0138 CI [0.008,0.020] (bar mean < 0, upper CI < 0: explaining away)
- **B6-lag-1 PASS**: aligned cos_p1_omit_m2 on-cut -0.391 CI [-0.408,-0.374]; on mean mean=0.033 median=0.034 IQR=[-0.013,0.078] 95% CI=[0.002,0.063] n=20 (bar mean < 0 with upper CI < 0, on mean < 0.20)
- **B7-lag-1 (guard) PASS**: aligned normal_acc mean=0.232 median=0.232 IQR=[0.227,0.238] 95% CI=[0.227,0.237] n=20 vs wire-off mean=0.221 median=0.217 IQR=[0.212,0.231] 95% CI=[0.215,0.227] n=20 (guard >= 0.80x)
- **B8-lag-1 PASS**: aligned gap minus random gap +0.071 CI [0.048,0.093] (bar >= +0.05, lower CI > 0)
- **B9-lag-1 FAIL**: learned on-cut omit_m2 +0.004 CI [-0.000,0.009] (bar >= +0.05, lower CI > 0)
- **B1-lag-2 FAIL**: aligned on-cut omit_m2 -0.006 CI [-0.016,0.004] (bar >= +0.05, lower CI > 0)
- **B2-lag-2 FAIL**: aligned on-shuffle omit_m2 +0.015 CI [0.000,0.030] (bar >= +0.05, lower CI > 0)
- **B3-lag-2 PASS**: aligned p1_cos mean=0.810 median=0.809 IQR=[0.795,0.819] 95% CI=[0.802,0.817] n=20 (bar mean < 0.90)
- **B4-lag-2 FAIL**: aligned on-off omit_m2 -0.009 CI [-0.019,0.000] (bar CI excludes 0)
- **B5-lag-2 FAIL**: aligned err_mag_last on-off +0.0181 CI [0.013,0.023] (bar mean < 0, upper CI < 0: explaining away)
- **B6-lag-2 PASS**: aligned cos_p1_omit_m2 on-cut -0.381 CI [-0.402,-0.360]; on mean mean=-0.038 median=-0.037 IQR=[-0.112,0.017] 95% CI=[-0.075,-0.001] n=20 (bar mean < 0 with upper CI < 0, on mean < 0.20)
- **B7-lag-2 (guard) PASS**: aligned normal_acc mean=0.224 median=0.223 IQR=[0.216,0.232] 95% CI=[0.218,0.229] n=20 vs wire-off mean=0.220 median=0.223 IQR=[0.208,0.232] 95% CI=[0.214,0.226] n=20 (guard >= 0.80x)
- **B8-lag-2 FAIL**: aligned gap minus random gap +0.049 CI [0.028,0.071] (bar >= +0.05, lower CI > 0)
- **B9-lag-2 FAIL**: learned on-cut omit_m2 -0.027 CI [-0.042,-0.013] (bar >= +0.05, lower CI > 0)
- **D1 FAIL**: sine rollout acc aligned on-cut -0.037 CI [-0.051,-0.022] (bar >= +0.10, lower CI > 0)
- **D2 FAIL**: sine rollout acc aligned on-shuffle -0.025 CI [-0.037,-0.014] (bar >= +0.10, lower CI > 0)
- **D3 FAIL**: sine rollout acc aligned on-off -0.076 CI [-0.092,-0.061] (bar >= +0.10, lower CI > 0)
- **D4 FAIL**: sine rollout late acc (on) mean=0.051 median=0.050 IQR=[0.030,0.062] 95% CI=[0.038,0.064] n=20 (bar mean >= 0.30, lower CI > 0.20; chance 0.031)
- **D5-lag-1 FAIL**: lag-1 rollout completion aligned on-cut -0.146 CI [-0.190,-0.102] (bar >= +0.20, lower CI > 0)
- **D6-lag-1 FAIL**: lag-1 rollout completion aligned on-shuffle -0.109 CI [-0.157,-0.061] (bar >= +0.20, lower CI > 0)
- **D5-lag-2 FAIL**: lag-2 rollout completion aligned on-cut -0.225 CI [-0.265,-0.185] (bar >= +0.20, lower CI > 0)
- **D6-lag-2 FAIL**: lag-2 rollout completion aligned on-shuffle -0.155 CI [-0.204,-0.106] (bar >= +0.20, lower CI > 0)
- **D7 FAIL**: sine rollout acc learned on-cut -0.035 CI [-0.047,-0.022] (bar >= +0.10, lower CI > 0)
- **D8 (guard) PASS**: sine rollout acc frozen (on) mean=0.032 median=0.028 IQR=[0.022,0.041] 95% CI=[0.025,0.040] n=20 (guard mean < 0.10: generation needs learning)

Bars: 9 / 28 pass.

## Verdict

- Experiment B' (aligned wire is causal): FAILS (B1-B6 + B7 guard on lag-1).
- Experiment D (multi-step generation through the top-down pathway): FAILS (D1-D6 + D8 guard).
- Neither half passes: the aligned wire does not establish predictive-coding causality on this architecture.

## Diagnostics

- planted1-occ aligned-on: fb_align mean=1.000 median=1.000 IQR=[1.000,1.000] 95% CI=[1.000,1.000] n=20; fb_norm mean=14.135 median=14.141 IQR=[14.118,14.158] 95% CI=[14.117,14.153] n=20
- planted1-occ learned-on: fb_align mean=0.913 median=0.913 IQR=[0.912,0.913] 95% CI=[0.912,0.913] n=20; fb_norm mean=17.211 median=17.218 IQR=[17.190,17.237] 95% CI=[17.193,17.230] n=20
- planted1-occ random-on: fb_align mean=-0.001 median=-0.002 IQR=[-0.003,0.001] 95% CI=[-0.003,0.001] n=20; fb_norm mean=14.135 median=14.141 IQR=[14.118,14.158] 95% CI=[14.117,14.153] n=20
- planted2-occ aligned-on: fb_align mean=1.000 median=1.000 IQR=[1.000,1.000] 95% CI=[1.000,1.000] n=20; fb_norm mean=14.135 median=14.141 IQR=[14.118,14.158] 95% CI=[14.117,14.153] n=20
- planted2-occ learned-on: fb_align mean=0.912 median=0.912 IQR=[0.912,0.913] 95% CI=[0.912,0.913] n=20; fb_norm mean=17.212 median=17.219 IQR=[17.191,17.238] 95% CI=[17.195,17.229] n=20
- planted2-occ random-on: fb_align mean=-0.001 median=-0.002 IQR=[-0.003,0.001] 95% CI=[-0.003,0.001] n=20; fb_norm mean=14.135 median=14.141 IQR=[14.118,14.158] 95% CI=[14.117,14.153] n=20
- sine aligned-on: fb_align mean=1.000 median=1.000 IQR=[1.000,1.000] 95% CI=[1.000,1.000] n=20; fb_norm mean=14.136 median=14.138 IQR=[14.109,14.169] 95% CI=[14.118,14.155] n=20
- sine learned-on: fb_align mean=0.844 median=0.844 IQR=[0.842,0.846] 95% CI=[0.842,0.845] n=20; fb_norm mean=19.932 median=19.926 IQR=[19.871,19.984] 95% CI=[19.878,19.986] n=20
- sine random-on: fb_align mean=0.001 median=0.001 IQR=[-0.002,0.002] 95% CI=[-0.001,0.002] n=20; fb_norm mean=14.136 median=14.138 IQR=[14.109,14.169] 95% CI=[14.118,14.155] n=20
