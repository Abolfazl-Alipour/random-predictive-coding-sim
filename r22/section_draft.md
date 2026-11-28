# 62. Round 22 (design): the precision test (experiment C)

Status: DESIGN ONLY. Protocol frozen Nov 27, 2026
(`PROTOCOL-R22-PRECISION.md`). Driver
`rpc_round22_precision.py`, analyzer `r22/r22_analyze.py`,
monitor `r22/monitor.py`, smoke `rpc_round22_smoke.py`.
Nothing launched on Beast. The val and test phases are
not run. This section records the design, the design-time
probe, and the VM smoke test.

## 62.1 What C is

Experiment C is the last unrun discriminator. Omission (A)
passed (round 16); the top-down clamp with the random wire
(B) failed (round 17); the aligned-wire clamp (B') failed
with a "causal but small" lag-1 effect (round 20); the
generative rollout (D) failed cleanly (round 20). What
remains is precision: in Rao-Ballard predictive coding,
prediction errors are weighted by precision (inverse
variance), so the expectation effect on the error is larger
when the sensory input is reliable than when it is noisy,
and attention reverses prediction-induced silencing by
up-weighting precision (Kok, Rahnev et al. 2012). The
model has no precision parameter, so the question is
whether it shows the expectation-by-precision interaction
anyway, and whether the live aligned feedback wire carries
it.

## 62.2 The precision operationalization

There is NO precision parameter in this model, and the
protocol does not pretend otherwise. Precision is
operationalized as test-time input reliability: at the
probe step only, the encoded probe token is corrupted
with additive Gaussian noise, x = normalize(encode(tok)
+ c*z), z ~ N(0, I), fed through `step_vec`. High
precision is c = 0; low precision is c = c_low, picked
per task in validation by the decode rule (smallest c
with mean decode(p1) < 0.75 on calibration trials: CTL
for planted, EXP for sine). Training is always clean,
so the learned prediction is identical across precision
conditions; only the sensory evidence at the probe
differs. Two alternatives were considered and rejected:
an error gain during training (scaling eta_d) changes
what is learned, not the test-time error response, and
confounds learning rate with precision; bump width
(smooth_sigma) as precision changes the calibration
basis and interacts with the delta rule in its own right
(round 15). The NLMS denominator ||y2||^2 + 1 is
explicitly NOT precision: it is input-norm
normalization of the learning rate, acting during
training only.

The core measurement is the expectation-by-precision
interaction on the stimulus-specific error response
e = p1 - d1: gap(c) = rel_e(VIO, c) - rel_e(EXP, c),
interaction = gap(0) - gap(c_low), with rel_e = ||e||
divided by the net's own mean ||p1|| on clean
calibration trials. Specificity is spec(c) = cos(e,
proto_presented) - cos(e, proto_predicted) on VIO
trials. Probe trials are matched within a seed (same
filler prefix across EXP/VIO/UNEXP/CTL) and the noise
vector is identical across wire modes for the same
trial. 20 prefixes per (task, seed). Nets per (task,
seed): aligned (gain 1.0), random (gain 1.0), off
(gain 0.0), frozen-aligned (gain 1.0, plasticity off);
8 config lines per (task, seed). Tasks: planted lag-1,
planted lag-2, sine. Round-15 winner configs, 5 epochs.
Val seeds 347-351; test seeds 352-371.

## 62.3 Design probe (seed 347, 20 prefixes)

Aligned wire, gain 1.0. Planted lag-1: VIO-EXP gap at
c = 0 is +0.157 raw; interaction (c = 0 minus c = 0.5)
is +0.066 to +0.098 raw (two noise draws), rel_e about
+0.048. VIO specificity spec(0) +0.54, spec(0.5) +0.20,
interaction +0.34. decode(e) on VIO at c = 0 is 1.00.
decode(p1) on CTL: 1.00 at c = 0, 0.85 at 0.25, 0.50
at 0.50. p1 decode EXP vs UNEXP 1.00 vs 1.00; ||p1||
1.64 vs 1.64 (no p1 suppression). Wire-cut interaction
+0.021 to +0.034; shuffle +0.002 to +0.023. Frozen:
gap0 -0.007, interaction about 0.00. Random wire:
interaction +0.072 raw, rel_e +0.030 (aligned +0.096 /
+0.056 in the same run); the random wire inflates
||p1|| (p1scale 2.37 vs 1.72 aligned).

Sine: gap0 +0.322 raw; interaction (c = 0 minus
c = 0.5) +0.238 to +0.277 raw, rel_e about 0.22.
spec(0) +0.64, spec(0.5) +0.26, interaction +0.38.
decode(e) on VIO at c = 0 is 1.00. decode(p1) on EXP:
1.00 at c = 0, 0.60 at 0.25, 0.20 at 0.50. Wire-cut
interaction +0.114; shuffle +0.102. Frozen: gap0
+0.017, interaction -0.010. Random wire: interaction
+0.317 raw (LARGER than aligned +0.285 in raw units)
but rel_e +0.164 (smaller than aligned +0.282)
because p1scale is 1.94 vs 1.01.

Headline of the probe: the interaction EXISTS wire-on
on both tasks, but it PERSISTS wire-cut at 30-45% of
the wire-on size, and the random wire carries 75% of
the aligned wire's interaction on planted. The wire
AMPLIFIES the interaction roughly 2-3x but does not
CREATE it. Kimi's strict bar (cut removes the effect)
is predicted to FAIL; the amplification (W1/W2) is the
testable wire claim. The interaction is mostly a
bottom-up sensory-noise effect.

A design-time bug: the probe used Python hash() on
trial-type strings for noise seeds; string hashing is
randomized per process, so noisy-condition numbers
varied between identical runs (planted gap at c = 0.5
was +0.091 in one run, +0.059 in another). The driver
uses integer-arithmetic seeds only (section 9.3 of the
protocol) and the smoke confirms determinism (S5).

## 62.4 Bars (frozen)

P-family (per task; sine primary, planted1
generalization, planted2 replication): P1 interaction
(aligned-on) >= floor_P1, lower CI > 0; P2 specificity
interaction >= floor_P2, lower CI > 0; P3 gap0 trained
minus frozen >= floor_P3, lower CI > 0; P4 guard:
spec(0) mean >= 0.25 with lower CI > 0.15 AND decode(e)
on VIO at c = 0 mean >= 0.50 with lower CI > 0.30.
Floors are set by the interaction validation phase as
50% of the val-measured value, minima 0.02 rel_e (P1,
P3) and 0.05 cosine (P2). W-family (per task): W1
interaction on minus cut >= floor_W1; W2 on minus
shuffle >= floor_W2; W3 aligned minus random >=
floor_W3; W4 on minus off >= floor_W4; W1s diagnostic
(strict Kimi: |interaction_cut| < floor_W1 and CI
covers 0); W5 gate (mean fb_drive_ratio > 0.05 AND
mean p1_cos < 0.90; if it fails the W-family is void
for that task); W6 guard (normal_acc aligned-on >=
0.80 x wire-off). S-family (planted only, wire-off,
equivalence): S1 |decP1 EXP-UNEXP| < 0.03 with CI
covering 0; S2 |p1n EXP-UNEXP| < 0.05 with CI covering
0; S3 measurement only wire-on. 34 pass/fail bars;
the verdict uses the preregistered subset (sine P1-P4,
planted1 P1, sine W1/W2).

Verdict: C STRONG PASS iff sine P1-P4 pass AND
planted1 P1 passes AND W1+W2 pass on sine. C PASS iff
sine P1-P4 pass AND planted1 P1 passes but W1/W2 fail
(interaction exists, wire does not carry it: a
bottom-up sensory-noise effect, not top-down
precision). C PARTIAL iff sine P1+P2 pass but planted1
P1 fails. C FAIL iff sine P1 or P2 fails, or sine P4
fails. Thesis-level: no outcome of round 22
establishes predictive coding. Round 20 failed B' and
D; a C pass establishes only that the model produces
precision-like behaviour from input noise plus error
selection.

## 62.5 VM smoke test (rpc_round22_smoke.py, seed 347)

All checks pass (run Nov 27, 2026, about 10 min on the
VM). Noise val: decP1 falls with c (planted1 CTL 1.00
at 0.10 -> 0.40 at 0.50; sine EXP 1.00 at 0.10 -> 0.20
at 0.25). c_low picked by the protocol rule: planted1
0.35, planted2 0.35, sine 0.25. Wire live: fbdr 0.251
planted1 / 0.302 sine, p1_cos 0.811 / 0.819 (round-20
values 0.31-0.38 / 0.804-0.810). Interaction on sine
0.362 rel_e, frozen -0.009 (null). Deterministic noise
confirmed (probe_pass twice on the same net: identical
aggregates). CLI resumable (noise val run twice: 5 lines
both times, 5 unique keys). Lag-2 machinery: 8 config
lines, key fields complete. Analyzer writes
r22_analysis.md with a bar tally (n = 1, so bars fail
on the n != 20 check, as designed; P4, W5, S1 pass on
the single seed).

Smoke floors (1 seed, 5 prefixes; the 5-seed val will
replace them): sine P1 0.181, P2 0.178, P3 0.235, W1
0.143, W2 0.154, W3 0.124, W4 0.020 (minimum); planted1
P1 0.031, P2 0.190, P3 0.042, W1-W4 0.020 (minimum);
planted2 P1 0.020 (minimum; lag-2 interaction -0.015
on the smoke seed), P2 0.149, W2 0.025.

Two findings from the smoke, recorded here because
they shape the verdict reading:
1. The shuffle is NOT magnitude-matched for the
   aligned wire: fb_drive_ratio on vs shuffle is 0.251
   vs 0.203 on planted1 (19% drop) and 0.302 vs 0.159
   on sine (47% drop). The aligned wire is a structured
   map (-W_prox.T); permuting d1 moves it out of the
   subspace where the map has large gain. A W2 pass
   therefore confounds feedback direction with feedback
   magnitude; the cut (W1) is the primary wire control.
   A rescaled shuffle would need an rpc_v2.py change
   and would break comparability with rounds 17/20; it
   is a follow-up, not part of this round.
2. Lag-2 is marginal at smoke scale: the magnitude
   interaction is -0.015 while the specificity
   interaction is +0.297, and the S2 p1n difference is
   -0.058, outside the 0.05 bound (||p1|| slightly
   SMALLER on expected trials, the direction the
   mapping report says the model should not show). The
   20-seed test decides whether either is real.

## 62.6 Sharpest risks (from protocol section 10)

1. The precision definition is a stipulation: the round
   tests sensory reliability, not precision weighting. A
   pass means the interaction exists WITHOUT precision
   weighting, which is the informative direction, but a
   critic can fairly say the round does not test
   precision at all.
2. The interaction is mostly bottom-up by construction
   (persists cut at 30-45%; random wire carries 75% on
   planted). If W1/W2 fail, the headline is "the
   interaction does not need the wire" -- a real result
   that does not advance the thesis claim.
3. The planted interaction is small (rel_e about 0.048
   vs 0.22 sine); the planted P1 floor (about 0.024-0.031)
   is close to measurement noise across seeds.
4. The floors are relative (50% of val); a weak val
   interaction produces a weak floor. The absolute minima
   are the only guard against a trivial bar.
5. rel_e normalization confounds wire comparisons: the
   random wire inflates ||p1||, so W3 is partly a test
   of the p1scale difference. The analyzer reports raw
   ||e|| interactions alongside rel_e.
6. The wire's effect on ||e|| is measurement-dependent:
   round 20 B5 found err_mag_last HIGHER with the wire
   on during training, while the design probe finds
   probe-step ||e|| LOWER with the wire on at test.
7. Sine has no UNEXP/CTL trials, so the S-family cannot
   run on sine and P3 (frozen) is the only sine control
   for "no learning".

## 62.7 Launch (not run)

Copy the driver, analyzer, monitor, and protocol to
Beast; Beast smoke (one noise val + one interaction val
+ one test invocation); monitor val launch; create the
r22-monitor cron (every 30 min, same pattern as
r21-monitor). Monitor phases: noise val (75 lines) ->
c_low -> interaction val (120 lines) -> floors ->
winner.json -> test (480 lines) -> concat + analysis.
Test phase is 240 nets, about 60-90 min on a loaded
Beast; total round about 3 h; give it a full day. Do NOT
edit simulation-results.md from this subagent; merge
this draft as section 62 when the round runs.
