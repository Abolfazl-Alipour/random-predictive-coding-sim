# Round 19 analysis: hierarchy (experiment E)

Coverage: 420 / 480 test lines.
Val winner: n_levels=2, fb_gain1=0.1, fb_gain2=None, val_acc=0.268.

## planted1

| condition | acc | m1 | late_m1 | top_acc | mean_burst | clamp |
| --- | --- | --- | --- | --- | --- | --- |
| cand_on | 0.210 (0.202, 0.218) | 0.561 (0.512, 0.609) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_cut | 0.210 (0.203, 0.217) | 0.562 (0.514, 0.610) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_shuffle | 0.209 (0.203, 0.216) | 0.554 (0.510, 0.599) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| hier2_nofb | 0.211 (0.205, 0.218) | 0.569 (0.523, 0.614) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| flat | 0.226 (0.220, 0.233) | 0.682 (0.633, 0.730) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_frozen | 0.059 (0.054, 0.064) | 0.040 (0.012, 0.068) | 0.000 (0.000, 0.000) | - | 0.026 | 0.000 |
| baseline (markov1) | 0.280 | - | - | - | - | - |
| baseline (markov2) | 0.272 | - | - | - | - | - |
| baseline (persist) | 0.046 | - | - | - | - | - |
| baseline (esn) | 0.279 | - | - | - | - | - |

## sine

| condition | acc | m1 | late_m1 | top_acc | mean_burst | clamp |
| --- | --- | --- | --- | --- | --- | --- |
| cand_on | 0.679 (0.648, 0.710) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_cut | 0.680 (0.655, 0.706) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_shuffle | 0.661 (0.636, 0.686) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| hier2_nofb | 0.698 (0.671, 0.725) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| flat | 0.874 (0.860, 0.887) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.000 | 0.000 |
| cand_frozen | 0.036 (0.028, 0.045) | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | - | 0.027 | 0.000 |
| baseline (markov1) | 0.466 | - | - | - | - | - |
| baseline (markov2) | 0.787 | - | - | - | - | - |
| baseline (persist) | 0.126 | - | - | - | - | - |
| baseline (esn) | 0.946 | - | - | - | - | - |

## hier

| condition | acc | m1 | late_m1 | top_acc | mean_burst | clamp |
| --- | --- | --- | --- | --- | --- | --- |
| cand_on | 0.248 (0.238, 0.258) | 0.263 (0.237, 0.288) | 0.278 (0.241, 0.315) | - | 0.000 | 0.000 |
| cand_cut | 0.246 (0.237, 0.256) | 0.266 (0.241, 0.290) | 0.277 (0.245, 0.309) | - | 0.000 | 0.000 |
| cand_shuffle | 0.247 (0.237, 0.257) | 0.262 (0.236, 0.288) | 0.277 (0.242, 0.312) | - | 0.000 | 0.000 |
| hier2_nofb | 0.249 (0.239, 0.258) | 0.268 (0.245, 0.291) | 0.278 (0.247, 0.308) | - | 0.000 | 0.000 |
| flat | 0.255 (0.247, 0.262) | 0.258 (0.239, 0.277) | 0.263 (0.241, 0.286) | - | 0.000 | 0.000 |
| cand_frozen | 0.061 (0.052, 0.070) | 0.058 (0.029, 0.087) | 0.059 (0.028, 0.089) | - | 0.026 | 0.000 |
| baseline (markov1) | 0.301 | 0.255 | - | - | - | - |
| baseline (markov2) | 0.345 | - | - | - | - | - |
| baseline (persist) | 0.040 | - | - | - | - | - |
| baseline (esn) | 0.406 | - | - | - | - | - |

## Bars

1. **H1 FAIL**: hier acc cand_on - flat = -0.007 (-0.014, 0.000) (bar +0.10, lower CI > 0)
1. **H2 FAIL**: hier late m1 cand_on - flat = 0.015 (-0.015, 0.044) (bar +0.20, lower CI > 0)
1. **H3 FAIL**: hier late m1 cand_on = 0.278 (0.241, 0.315) (bar mean >= 0.45, lower CI > 0.35)
1. **H4 FAIL**: not judged: val winner is 2 levels, so hier2_ladder duplicates cand_on (protocol section 9)
1. **H5 FAIL**: planted1 m1 cand_on - flat = -0.121 (-0.148, -0.094) (bar +0.05, lower CI > 0)
1. **H6 FAIL**: planted1 m1 cand_on = 0.561 (0.512, 0.609) (bar mean >= 0.80)
1. **H7 FAIL**: sine acc cand_on - flat = -0.195 (-0.226, -0.163) (bar +0.03, lower CI > 0)
1. **H8 FAIL**: hier acc cand_on - cand_cut = 0.002 (-0.001, 0.004) (bar +0.05, lower CI > 0)
1. **H9 PASS**: hier acc cand_on - cand_frozen = 0.187 (0.177, 0.198) (bar +0.10, lower CI > 0)

## Shuffle diagnostic (not a bar)

1. hier acc cand_on - cand_shuffle = 0.001 (-0.001, 0.004). If H8 passes but the shuffle gap is near zero, the effect is drive magnitude, not feedback content (protocol section 9).

## Verdict

1. H0 FAILS: at least one of H1, H2, H3 fails, so the hierarchy does not win the hierarchical task.
1. H4 fails: the third level does not beat the two-level ladder, so the win (if any) comes from a slower level 2, not from depth.
1. Old tasks: H5 fails, H6 fails, H7 fails.
1. Mechanism: H8 fails (feedback used at test time), H9 passes (learning matters).

## Frozen sanity

1. hier frozen acc = 0.061 (0.052, 0.070) (must stay below 0.25): PASS.

## Diagnostics

1. hier flat acc = 0.255 (0.247, 0.262); cand_on acc = 0.248 (0.238, 0.258).

Bars: 1 / 9 pass (0 N/A).
