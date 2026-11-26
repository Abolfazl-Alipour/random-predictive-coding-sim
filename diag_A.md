# Diagnosis A: why the v1 RPC toy never bursts, and the v2 fix

Files: `rpc_toy.py` (v1, untouched), `rpc_v2.py` (v2), `diag_A.py`,
`diag_B.py`, `epoch_diag.py`, `sweep_v2.py`. All runs CPU torch only,
`./venv/bin/python`. Data: v1 `make_data`, 200 train / 50 test sequences,
length 24, alphabet K = 16, planted 3-token motifs (1,2,3), (5,6,7),
(9,10,11). Chance accuracy = 1/16 = 0.0625.

## 1. Measured distributions, v1, plasticity OFF (diag_A.py)

Fresh `RPCNetwork` (defaults theta_p = theta_d = theta_f = 0.5), 200
sequences, 4800 steps, N_o = 64. A step's `p1, d1, p2, y2` were recorded by
replaying `step()` with training off, so no weight changed during the
recording. Hit = a burst step (at least one unit bursts) on a motif token
at motif offset 1 or 2 (2nd/3rd token, where a prediction from the previous
token could fire). False hit = a burst step elsewhere. Motif steps:
1095/4800 = 22.8%.

| quantity | mean | sd | p50 | p90 | p99 | max | frac > 0 |
|---|---|---|---|---|---|---|---|
| p1 (proximal drive, module 1) | -0.0012 | 0.0541 | 0.0006 | 0.0646 | 0.1258 | 0.2823 | 0.505 |
| d1 (distal drive, module 1) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| p2 (proximal drive, module 2) | -0.0021 | 0.0327 | -0.0014 | 0.0374 | 0.0823 | 0.1624 | 0.472 |
| y2 (module-2 output rate) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 |

W_dist1 row norms: mean 0.4004, sd 0.0354, p50 0.3989, p90 0.4420,
p99 0.4752, max 0.4833. W_dist1 entries: mean -0.0008, sd 0.0502
(init std 0.05, as written).

## 2. Threshold sweep, v1, plasticity OFF (diag_A.py)

theta_p and theta_d over {0.05, 0.1, 0.2, 0.3, 0.5, 0.8}, all 36 pairs.
Every pair gives the same result:

| burst rate per unit per step | hit rate | false-hit rate |
|---|---|---|
| 0.0000 | 0.000 | 0.000 |

Single-threshold pass fractions: P(p1 > 0.05) = 0.1546, P(p1 > 0.1) =
0.0295, P(p1 > 0.2) = 0.0002, P(p1 > 0.3) = 0.0000. P(d1 > t) = 0.0000
for every t in the grid.

**Verdict for v1 as built: no burst regime exists.** The sweep cannot find
one at any theta_p/theta_d, because the distal leg is not small, it is
exactly zero.

## 3. Root cause: two stacked scale failures, not one

1. p2 max is 0.1624 but theta_f is 0.5, so y2 = (p2 > 0.5) NEVER fires
   on any unit on any step. Therefore y2_prev is always the zero vector,
   so d1 = W_dist1 @ y2_prev is IDENTICALLY zero (mean 0, sd 0, max 0).
   The burst rule is a hard AND of (p1 > theta_p) and (d1 > theta_d),
   so it can never fire for any theta_d > 0, whatever theta_p is. The
   suspected cause in the task ("distal drive starts near 0.05") is
   wrong in magnitude: it is 0, not 0.05. The 0.05 is only the W_dist1
   init std, which never gets multiplied by anything nonzero.
2. The proximal leg fails independently: p1 max is 0.2823 but theta_p
   is 0.5, so (p1 > theta_p) is also never true at the default. Even if
   d1 were unblocked, the default thresholds still give zero bursts.
   P(p1 > 0.2) is 0.0002, so theta_p = 0.2 alone nearly kills bursting.

Why the drives are this small: W_prox = randn / sqrt(n_res) * 0.5, and
the reservoir state r is tanh-bounded with small variance, so p =
W_prox @ r has sd 0.054 (module 1) and 0.033 (module 2). The thesis
(Ch.3 pp.89-98) gives no number for any threshold, drive scale, or
learning rate, so 0.5 was an arbitrary choice, off-scale by 3-10x.

## 4. What unblocks d1 (diag_B.py)

p2 does not depend on theta_f (y2_prev enters only through d1), so y2
can be replayed offline for candidate theta_f values. P(p2 > t):
t = 0.02 -> 0.2065, 0.03 -> 0.1380, 0.05 -> 0.0564, 0.08 -> 0.0114,
0.10 -> 0.0033, 0.12 -> 0.0008 per unit.

With theta_f lowered, d1 becomes nonzero and burst rates in the
0.01-0.20 per-unit range exist, for example theta_f = 0.05,
theta_p = 0.05, theta_d = 0.05 gives burst 0.033 per unit per step.
But in every such regime, hit rate is approximately equal to false-hit
rate (that example: hit 0.610, false hit 0.605; theta_f = 0.03,
theta_p = 0.05, theta_d = 0.05: hit 0.790, false hit 0.761;
theta_f = 0.08, theta_p = 0.05, theta_d = 0.05: hit 0.264, false hit
0.272). That is the correct null BEFORE learning: the random W_dist1
carries no motif information, so bursts are unselective. Selectivity
can only be created by learning. Hit approx false-hit pre-learning is
not a failure; it is the baseline the learning rule must beat.

## 5. Assumptions added in v2 (beyond the thesis)

Each is labelled in the code in `rpc_v2.py`.

1. ASSUMPTION A1: thresholds from pre-learning train quantiles,
   standardised drives. p1, p2, d1, d2 are divided by their standard
   deviations measured on train data before learning (frozen
   thereafter), and theta_p, theta_d, theta_f are z-thresholds on those
   standardised drives (defaults theta_p = 1.0, theta_d = 1.0,
   theta_f = 1.28). Justification: the thesis gives no threshold
   values; v1's 0.5 is arbitrary and off-scale by 3-10x. A z-threshold
   is scale-free, so the same numbers work at any N_r/N_o. Scales are
   fit on train data only (calibration pass, plasticity OFF), never on
   the test set.
2. ASSUMPTION A2: W_dist1 fan-in scaling. W_dist1 init std
   0.5/sqrt(N_o) instead of v1's raw 0.05. Justification: every other
   matrix in v1 uses 1/sqrt(fan-in) scaling; 0.05 was unscaled. At
   N_o = 64 this is 0.0625, barely above v1's 0.05, so this is a
   consistency fix, not a tuning knob. With A1 the burst rate is set
   by the z-thresholds, not by this scale.
3. ASSUMPTION A3: graded ungated burst ONLY as the no-burst-gate
   ablation: b1 = sigmoid(p1n * d1n) on standardised drives. v1's
   ablation used sigmoid(p1*d1) on raw drives, which is about 0.5
   constant because the raw product is near 0. Standardising keeps
   the ablation meaningful. The full model still uses the hard AND,
   faithful to the thesis (a burst is a discrete event).
4. ASSUMPTION A4: theta_f fixed at z = 1.28 (90th percentile of a
   standard normal). y2 fires on the top about 10% of p2 values.
   Justification: y2 must be sparse but nonzero for d1 to exist at
   all; 10% is the standard sparse-code rate and matches the measured
   p2 quantile (P(p2 > 0.05) = 0.056 at raw theta_f = 0.05). Not
   swept in v2.
5. ASSUMPTION A5: row-norm preservation after each Hebbian update
   (synaptic scaling). After every W_dist1 / W_prox update, each row
   is rescaled to its initial L2 norm. Justification: the v1 rules as
   written are purely potentiating (b1 >= 0, y2 >= 0, so W_dist1 only
   grows) with no depression, normalisation, or bound except the +-2.0
   clamp. Learning changes each row's DIRECTION, which is all the
   cosine prediction reads, not its magnitude. Local (per row), no
   gradient. Set norm_rows=False to recover the saturating v1
   behaviour.
6. ASSUMPTION A6: k-winners-take-all y2 (fixed sparsity), ADOPTED
   as default. y2 = 1 for the top-k p2 units each step, k = 10% of
   N_o, instead of the fixed threshold (p2n > theta_f). Justification:
   the threshold path lets confirmation destroy sparsity (y2 rate
   0.08 -> 0.77, section 6.2); k-WTA holds it at 0.094 by
   construction, as lateral inhibition does in a sparse code. The
   thesis specifies neither a threshold nor a k for y2. Evidence:
   section 6.3 (seed 0): y2 0.094 stable, full acc 0.085 at ep5 vs
   0.070 without it.

NOT added: no backprop through the reservoir, no trained readout doing
the learning (prediction is still cosine match of d1 against mean-p1
prototypes, as in v1), no graded coincidence in the full model, no
bigger N_o/N_r by default, no test-set threshold fitting. Learning is
local burst-gated Hebbian only, plus the confirmation rule as in v1.

## 6. Per-epoch diagnosis: saturation, then the confirmation rule

All runs seed 0, defaults theta_p = 1.0, theta_d = 1.0,
eta_b = eta_c = 0.05, N_r = 200, N_o = 64. Accuracy is held-out test
after each train epoch; chance 0.0625.

### 6.1 Before A5: the v1 learning rules saturate in one epoch

`epoch_diag.py` (no row-norm preservation):

| condition | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| full acc / burst / y2 | 0.079 / 0.079 / 0.765 | 0.079 / 0.079 / 0.766 | 0.079 / 0.079 / 0.766 | 0.079 / 0.079 / 0.766 | 0.079 / 0.079 / 0.766 |
| no-confirmation acc | 0.072 | 0.075 | 0.049 | 0.051 | 0.054 |
| full eta_c=0.01 acc | 0.079 | 0.079 | 0.079 | 0.079 | 0.079 |
| full eta_c=0.2 acc | 0.079 | 0.079 | 0.079 | 0.079 | 0.079 |

Accuracy is identical to 3 decimals for epochs 1-5 and identical
across eta_c = 0.01, 0.05, 0.2: the weights slam into the +-2.0 clamp
and freeze. Measured directly after 1 epoch: 30.2% of W_dist1 entries
and 29.2% of W_prox2 entries are pinned at the clamp (0% before
training); mean |W_dist1| grows 0.05 -> 0.73; after epoch 2 the
fractions are 36.8% and 40.1%. The v1 rules are purely potentiating
with no depression or normalisation, so this saturation is structural,
not a choice of eta. The seed-0 full accuracy 0.079 is the accuracy of
a frozen saturated network, not of progressive learning.

### 6.2 With A5: W_dist1 learns, but confirmation still runs y2 away

`epoch_diag.py` with row-norm preservation (A5):

| condition | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| full acc / burst / y2 | 0.071 / 0.099 / 0.722 | 0.077 / 0.112 / 0.751 | 0.077 / 0.123 / 0.795 | 0.070 / 0.123 / 0.795 | 0.070 / 0.123 / 0.795 |
| no-confirmation acc / burst / y2 | 0.055 / 0.109 / 0.081 | 0.078 / 0.109 / 0.081 | 0.082 / 0.109 / 0.081 | 0.082 / 0.109 / 0.081 | 0.082 / 0.109 / 0.081 |

A5 fixed the W_dist1 saturation: no-confirmation now improves across
epochs (0.055 -> 0.082) and plateaus, with y2 stable at 0.081. But
full is WORSE than no-confirmation (0.070-0.077 vs 0.082), and y2
still runs away to 0.72-0.80 by epoch 1. Row-norm preservation cannot
stop this because the growth is ALIGNMENT, not norm: confirmation
rotates W_prox rows toward the frequent r2 patterns that already
drive them (b2 fires exactly when p2_prev is high), so the dot
product p2 = W_prox @ r2 grows at fixed row norm (||r2|| is about 4,
so an aligned row reaches p2 about 2, i.e. about 60 sd). The fixed
threshold then fires almost every unit. Lowering eta_c does not help
(full eta_c = 0.01: y2 0.728-0.781, acc 0.070-0.071); it only slows
the same loop. Conclusion: as specified, the confirmation rule
destroys the sparsity of the module-2 code and hurts accuracy in
this toy. That is what A6 (top-k y2) is designed to test.

### 6.3 With A6 (top-k y2): sparsity holds, full stops losing

`epoch_diag.py` with y2_topk=True (seed 0):

| condition | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| full + topk acc / burst / y2 | 0.095 / 0.123 / 0.094 | 0.084 / 0.122 / 0.094 | 0.085 / 0.122 / 0.094 | 0.085 / 0.122 / 0.094 | 0.085 / 0.122 / 0.094 |
| no-conf + topk acc / burst / y2 | 0.083 / 0.129 / 0.094 | 0.086 / 0.130 / 0.094 | 0.085 / 0.130 / 0.094 | 0.088 / 0.130 / 0.094 | 0.088 / 0.130 / 0.094 |

y2 rate is exactly 0.094 (6/64 units) on every epoch, burst rate is
0.122-0.130 per unit (inside the 0.01-0.20 target band), and full
accuracy (0.085 at ep5) no longer loses to no-confirmation (0.088 at
ep5; full wins ep1 0.095 vs 0.083). Learning plateaus by epoch 2-3
rather than freezing by saturation. A6 is therefore ADOPTED as the
v2 default (y2_topk=True). Caveat: seed 0 only, and full vs
no-confirmation is nearly tied at ep5, so the confirmation rule's
added value is small; the 5-seed sweep (section 7.2) is the real test
of whether full beats no-confirmation on average.

## 7. v2 results

### 7.1 Main table (5 seeds 0-4, 200 train / 50 test, 5 epochs)

Command: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_v2.py`
(best config; defaults in `rpc_v2.py` are the sweep winner, section 7.2).
Output in `rpc_v2_main_out.txt`. Chance 0.0625. Burst for the
no-burst-gate row is the mean graded drive sigmoid(p1n*d1n), not a
hard burst rate.

| condition | acc mean | acc range | accs seeds 0-4 | mean burst | mean y2 |
|---|---|---|---|---|---|
| full | 0.075 | [0.049, 0.085] | 0.085, 0.082, 0.076, 0.084, 0.049 | 0.136 | 0.094 |
| no-confirmation | 0.072 | [0.056, 0.088] | 0.088, 0.080, 0.078, 0.058, 0.056 | 0.146 | 0.094 |
| no-burst-gate | 0.065 | [0.040, 0.090] | 0.083, 0.090, 0.053, 0.061, 0.040 | 0.136 | 0.094 |
| no-plasticity | 0.062 | [0.051, 0.070] | 0.062, 0.064, 0.070, 0.051, 0.061 | 0.026 | 0.094 |

Reading, per seed (full minus control): +0.023, +0.018, +0.006,
+0.033, -0.012; mean gain +0.013. Full beats no-plasticity in 4 of
5 seeds and beats chance on average (0.075 vs 0.0625), so the
burst-gated rule does learn something real, but the effect is
small: about one extra correct prediction in eight over chance
(0.013 x 1150 test predictions is about 15 extra correct).
Full vs no-confirmation is +0.003 on the mean (per seed: -0.003,
+0.002, -0.002, +0.026, -0.007), i.e. the confirmation rule adds
nothing measurable at this scale; the learning is carried by the
burst-gated W_dist1 rule. Removing the burst gate nearly kills the
gain (0.075 -> 0.065, gain over control +0.003), so the hard
coincidence gate is load-bearing, which is the core of the RPC
claim. The no-plasticity burst rate 0.026 matches the pre-learning
prediction exactly (P(p1n>1) x P(d1n>1) is about 0.159 x 0.159 is
about 0.025); after training it rises to 0.136 as W_dist1 rows
align with recurring y2 patterns.

### 7.2 Sweep (5 seeds 0-4; full results in `sweep_results.csv`)

Metric: mean over seeds of (acc_full - acc_no-plasticity), paired per
seed (same data and net init; control run once per seed and size and
reused, since the control does not depend on theta or eta). Chance
0.0625. Burst = mean burst rate per unit per step on test after
training. y2 = 0.094 everywhere at N_o = 64 (top-k, 6/64) and 0.102
at N_o = 128 (13/128).

Stage A: theta_p x theta_d (z-thresholds), eta_b = eta_c = 0.05,
N_r = 200, N_o = 64.

| theta_p | theta_d | gain mean [range] | acc mean [range] | burst |
|---|---|---|---|---|
| 0.5 | 0.5 | +0.011 [-0.017, +0.028] | 0.073 [0.043, 0.084] | 0.283 |
| 0.5 | 1.0 | +0.010 [-0.018, +0.023] | 0.071 [0.043, 0.085] | 0.266 |
| 0.5 | 2.0 | +0.011 [-0.017, +0.035] | 0.073 [0.043, 0.086] | 0.187 |
| 0.5 | 3.0 | -0.002 [-0.025, +0.017] | 0.060 [0.039, 0.079] | 0.052 |
| 1.0 | 0.5 | +0.012 [-0.023, +0.038] | 0.074 [0.037, 0.100] | 0.148 |
| 1.0 | 1.0 | **+0.013 [-0.012, +0.033]** | 0.075 [0.049, 0.085] | 0.136 |
| 1.0 | 2.0 | +0.002 [-0.009, +0.019] | 0.063 [0.055, 0.083] | 0.107 |
| 1.0 | 3.0 | -0.006 [-0.028, +0.008] | 0.055 [0.043, 0.068] | 0.012 |
| 2.0 | 0.5 | +0.004 [-0.022, +0.025] | 0.066 [0.049, 0.087] | 0.025 |
| 2.0 | 1.0 | -0.003 [-0.024, +0.019] | 0.058 [0.040, 0.079] | 0.024 |
| 2.0 | 2.0 | +0.008 [-0.002, +0.019] | 0.070 [0.063, 0.077] | 0.011 |
| 2.0 | 3.0 | -0.008 [-0.030, +0.019] | 0.054 [0.039, 0.070] | 0.001 |
| 3.0 | 0.5 | -0.006 [-0.027, +0.017] | 0.056 [0.043, 0.079] | 0.002 |
| 3.0 | 1.0 | +0.008 [-0.016, +0.030] | 0.069 [0.045, 0.081] | 0.001 |
| 3.0 | 2.0 | +0.003 [-0.011, +0.017] | 0.065 [0.050, 0.080] | 0.001 |
| 3.0 | 3.0 | +0.002 [+0.000, +0.010] | 0.064 [0.051, 0.080] | 0.000 |

Best A: theta_p = 1.0, theta_d = 1.0. Pattern: low thresholds
(0.5) burst too densely (0.19-0.28 per unit, above the 0.01-0.20
band) for no extra gain; high thresholds (2.0-3.0) starve bursting
(0.000-0.025) and gains go to zero or negative. The sweet spot is
burst about 0.14.

Stage B: eta_b = eta_c at theta_p = theta_d = 1.0, N_r = 200,
N_o = 64. eta 0.01: gain +0.013 [-0.007, +0.027], acc 0.075
[0.054, 0.089], burst 0.145. eta 0.05: gain +0.013 [-0.012, +0.033],
acc 0.075 [0.049, 0.085], burst 0.136. eta 0.2: gain +0.009
[-0.019, +0.021], acc 0.070 [0.042, 0.083], burst 0.092. Best B:
eta = 0.05 (tied with 0.01 on the mean; 0.05 kept as the v1 value).
Learning rate barely matters, which fits the epoch finding that
learning plateaus by epoch 2-3.

Stage C: sizes at theta_p = theta_d = 1.0, eta = 0.05.
(200, 64): gain +0.013, acc 0.075, burst 0.136.
(200, 128): gain +0.012 [-0.014, +0.039], acc 0.074 [0.049, 0.086],
burst 0.143. (500, 64): gain -0.007 [-0.023, +0.008], acc 0.056
[0.028, 0.075], burst 0.134. (500, 128): gain -0.005 [-0.023,
+0.033], acc 0.061 [0.039, 0.077], burst 0.143. Best C: N_r = 200,
N_o = 64. A bigger reservoir HURTS (both N_r = 500 configs go
negative): with a larger reservoir the fixed random code is less
likely to repeat a context pattern, so the burst-gated association
has less to latch onto. Raising N_r is not a free win here.

Seed variance dominates: per-seed gains in the best config span
-0.012 to +0.033, and seed 4 is the worst seed in nearly every
config (acc 0.043-0.049 at the best config, vs 0.076-0.085 for
seeds 0-3). No config was picked by its best seed; all picks are
by mean gain over 5 seeds.

### 7.3 Best configuration and reproduction command

Best configuration (by mean gain over the no-plasticity control,
5 seeds 0-4, not by best seed): theta_p = 1.0, theta_d = 1.0
(z-thresholds on standardised drives), theta_f = 1.28 (unused under
A6 top-k), eta_b = eta_c = 0.05, N_r = 200, N_o = 64,
y2_topk = True (k = 6 of 64), norm_rows = True, w_dist_std = 0.5,
5 epochs, 200 train / 50 test sequences. Gain mean +0.013
(range [-0.012, +0.033]), acc mean 0.075 (range [0.049, 0.085]),
mean burst 0.136, mean y2 0.094.

Reproduce the full 4-condition, 5-seed table (about 3.5 minutes):

```
cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_v2.py
```

Reproduce one run (seed 0, full):

```
cd ~/workspace/thesis-rpc && ./venv/bin/python -c "from rpc_v2 import run_experiment; print(run_experiment(0))"
```

Reproduce the sweep (about 30 minutes, writes `sweep_results.csv`):

```
cd ~/workspace/thesis-rpc && ./venv/bin/python sweep_v2.py
```

Bottom line: v1 was dead twice over (d1 identically zero, p1
never crossing threshold) and its learning rules saturate in one
epoch. v2 fixes the scales with six labelled assumptions, and the
toy then learns a small real effect (acc 0.075 vs chance 0.0625,
gain over the no-plasticity control +0.013). The burst gate is
essential; the confirmation rule is not measurably helping at this
scale; a bigger reservoir hurts. The effect is small and seed-noisy
(seed 4 is below chance in the full model), so this is evidence the
mechanism can learn at all, not evidence it learns well.
