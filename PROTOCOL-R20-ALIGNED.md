# Round 20 protocol: the aligned-feedback clamp (revised experiment B) + multi-step omission rollout (experiment D)

Frozen Nov 27, 2026, before any round-20 test result is seen.

Question: does the top-down prediction causally shape the lower
module, when the feedback has the geometry Rao and Ballard
specify? Round 17 tested the wrong wire (a fixed random
additive wire, A26) and found it inert: cutting, shuffling,
or never building it changed omission accuracy by at most
half a point. Both external reviewers of rounds 17/18 (Kimi
and AGY Gemini, section 57) agree the failure stands only
for that wire. Rao and Ballard (1999) and thesis Chapter 3
specify ALIGNED SUBTRACTIVE feedback: minus the readout
transpose times the prediction. AGY's one-seed control
(reviewer-reported, not re-run by us at the time) reached
omit_m2 1.000 with that wire at gain 1.0. This round fixes
the wire first, then re-runs the clamp (revised experiment
B) and adds the multi-step omission rollout (generative
experiment D, first half). Precision (experiment C) is NOT
in this round: both reviewers agree it is premature without
a causal wire, and it follows after this round.

## 0. What round 17 got wrong, and what this round changes

Round 17 added a fixed random additive wire (+W_fb @ d1,
A26) and found it inert. The wire was live in code (the
feedback term was computed and added to the drive), the
state changed a little (p1_cos 0.998), but the prediction
did no work: the omission accuracy was the same wire-on,
wire-cut, and wire-shuffled. The reviewers' verdict
(section 57): that failure stands only for the tested wire.
The tested wire has the wrong geometry in two ways. First,
it is random, not aligned: it injects the prediction into
module 1 through an arbitrary direction, so most of the
feedback is noise relative to the code the readout decodes.
Second, it is additive, not subtractive: Rao-Ballard
feedback subtracts the prediction from the lower level, so
the lower level represents the error; an additive wire adds
the prediction to the drive, which is a different circuit.
This round tests the aligned subtractive wire (A29). If
that wire is also inert, the failure is architectural, not
geometric, and the round says so plainly.

## 1. Assumption A29: the aligned subtractive wire, the learned back-projection wire, and the blank token

A29 has three parts, all in rpc_v2.py, following the A26
pattern (constructor option, default off, previous results
reproduce exactly).

### 1a. Aligned subtractive feedback

Constructor option fb_wire in {"random", "aligned",
"learned"} (default "random"). "aligned" replaces W_fb at
init with -W_prox.T (module 1's readout transpose, negated),
rescaled to the Frobenius norm of the random W_fb it
replaces, so at the same fb_gain the drive magnitude
matches the random wire and only the direction differs.
The random W_fb is still drawn first (same RNG stream),
then overwritten. The feedback term in the drive is
unchanged in form: fb = fb_gain * (W_fb @ d1), and module
1's reservoir update is r_new = (1 - alpha) * r + alpha *
tanh(W_in @ e_sparse + W_res @ r + fb - inhib). The wire
is fixed (never trained). The sign is subtractive: the
feedback, passed through the readout, subtracts the
prediction from the code, which is the Rao-Ballard form.

Scale (measured, Nov 27, 2026, seed 241, planted lag-1,
round-15 winner config, 5 epochs): the aligned wire at
gain 1.0 has fb_drive_ratio 0.383 (the feedback is 38% of
the total drive, A29 corrected ratio), p1_cos 0.8565
(passes the 0.90 gate), omit_m2 on 1.000 vs cut 0.966
(gap +0.034), normal_acc 0.2330 vs wire-off 0.2209,
err_last 1.6162 vs wire-off 1.6186. These numbers
reproduce AGY's one-seed control EXACTLY (all four
decimals: normal_acc 0.2330, omit_m2 1.000, p1_cos 0.8565,
err 1.6162, cos_p1_omit_m2 on 0.036 vs cut 0.401), which
validates the implementation against an independent
implementation of the same wire. At gain 0.3 the aligned
wire has p1_cos 0.9667 (fails the gate) and gap 0.000, so
the gate is doing real work: a gain too small to move the
state is caught, not passed.

### 1b. Learned back-projection feedback

"learned": W_fb starts as the aligned map and is trained
by a local composite delta rule, once per training step
after mod1.forward, only when train and plasticity are on.
The rule learns the map whose back-projection through
W_prox subtracts the prediction from the code, i.e.
W_prox @ (W_fb @ d1) is approximately -d1. The update is
gradient descent on (1/2)||-d1 - W_prox @ W_fb @ d1||^2,
in NLMS form: delta W_fb = eta_fb / (||d1||^2 + 1) *
outer(W_prox.T @ e, d1) with e = -d1 - W_prox @ W_fb @ d1.
It uses only d1, W_prox, and W_fb (all local to module 1),
no backprop through time, no global information. fb_eta
defaults to eta_d. Entries are clamped at w_max (a safety
rail only; no row-norm preservation, because the map's
scale is the feedback magnitude). The driver records
fb_align (Frobenius cosine between W_fb and -W_prox.T)
and fb_norm after training, so the round documents whether
learning kept the subtractive geometry.

This is NOT AGY's literal formula (delta W_fb
proportional to (r1 - W_fb d1) d1^T). That rule was
implemented first and measured, and it is unusable: on
seed 241 it collapses at gain 1.0 (normal_acc 0.061,
chance 0.0625, feedback 4.8x the total drive) and rotates
AWAY from the aligned direction at every gain (fb_align
-0.02 at gain 0.1, 0.11 at 0.3, 0.63 at 1.0), because d1
barely predicts the reservoir state r1 (r1 is mostly
driven by the current input) and the target has state
scale while the drive has drive scale. The composite rule
fixes both defects: its target is the code (drive scale),
and its fixed point is the back-projection (the aligned
direction). Measured (seed 241, 5 epochs): the composite
learned wire is stable at every gain (normal_acc 0.232 to
0.240, all above wire-off 0.221), keeps its subtractive
direction (fb_align 0.91 at all gains), and grows its norm
from 14.20 to about 17.2 toward the back-projection
optimum. On seed 302 (occluded, 5 epochs, gain 1.0):
normal_acc 0.2296, fb_align 0.912, fb_norm 17.20, r1_max
0.97, err_last 1.6213. The deviation from AGY's literal
formula is deliberate, measured, and documented here; the
learned wire is a secondary condition, and its bars (B9,
D7) are secondary.

### 1c. Blank token

_encode returns the zero vector for tok == K (one past the
last class). No existing code passes token K, so no
previous result changes. The blank token is used for
occlusion (section 2) and for rollout blanks (section 5).

### 1d. Corrected drive ratio

ReservoirModule.forward records last_drive_norm =
||W_in @ e_sparse + W_res @ r - inhib||, and _step_core
returns fb_drive_ratio = ||fb|| / ||total drive||. Round
17's fb_rms_ratio divided by the bottom-up norm only and
overstated the feedback fraction by about 1.7x. The bars
use fb_drive_ratio; the A26 ratio is kept for
comparability.

### 1e. Timing subtlety (carried over from round 17)

d1(t) and pred(t) are computed before fb(t) enters mod1,
so a one-step omission decode at t cannot test fb(t). The
rollout (experiment D) is where fb(t) does work, because
the input is blank and fb is the only external drive.

## 2. Tasks, occlusion, and config

Tasks: planted1-occ (lag-1, occluded), planted2-occ
(lag-2, occluded), sine (fixed, no occlusion). Seeds
302..321 (20 fresh seeds; round 19 used 282..301, round 17
used 241..260). Config: the round-15 winner for planted
(alpha 1.0, smooth_sigma 0.0, eta_d 0.05) and for sine
(alpha 1.0, smooth_sigma 0.0, eta_d 0.2), K = 16 planted /
32 sine, length 24 planted / 200 sine, train 200 planted /
50 sine, 5 epochs, delta rule, graded_conf off,
use_confirmation off, unmodified p1-prototype cosine
readout. Validation seeds 302..306, test seeds 302..321.

Occlusion (planted tasks only): 40% of filler positions
are fed as the blank token (A29, token K) in train AND
test. Motif positions are never occluded. The filler probe
position (first non-motif index >= 2) is never occluded,
so the round-16 omission measures keep their meaning.
Calibration and prototypes use the CLEAN train data; the
training loop uses the OCCLUDED train data. The mask is
deterministic per (task, seed, split) (default_rng with a
seeded stream), so every condition sees identical data.
Smoke-measured (seed 302): 1235 blanked positions in the
200-sequence train split, 0 at motif positions, 0 at the
probe position, and the mask is identical across repeated
calls. Normal_acc on occluded data counts blanked
positions as errors (a consistent, small depression of
about 0.015); this is a caveat, not a confound, because
every condition sees the same data.

Why occlusion: on clean data the one-step omission is
near ceiling (omit_m2 0.95 to 1.00), so the wire has
almost no headroom to show a gap (measured clean gap
+0.034 on seed 241). Occlusion degrades the context the
net relies on, which should give the feedback more work
to do. Whether it does is an empirical question the round
answers; the self-critique (point 4) records that the
measured occluded single-seed gap (+0.017 on seed 302)
is SMALLER than the clean gap, so occlusion may not
amplify the effect at all.

## 3. Validation: pick fb_gain per wire type

Grid fb_gain in {0.1, 0.3, 1.0}, planted1-occ only, val
seeds 302..306, wire-on training, wire-on testing. One
worker per (wire type, gain), 9 workers, 45 val lines.
Per (wire type, gain): val normal acc, val on-cut omit_m2
gap, val p1_cos, val r1_max.

Selection rule per wire type (frozen here, not chosen
after seeing test results): among gains passing the state
gate (val p1_cos < 0.90) AND stable (r1_max < 10, no NaN),
pick the gain with the largest val on-cut omit_m2 gap;
tie-break smallest gain. If no gain passes the gate, pick
the gain with the largest val gap among stable gains and
report "gate failed" in winner.json (the test still runs;
the gate failure is a finding, not a stop). The guard
(selected gain's val normal_acc >= 0.80x wire-off val
normal_acc) is reported, not a stop, because the val
phase has no wire-off condition; the test-phase guard
(B7) is the binding one. Selection is on the omission
gap (late-m1 / omission effect), NOT on overall accuracy,
per both reviewers' demand. The random wire's test gain
is the ALIGNED winner's gain (matched control), not its
own selected gain, so bar B8 compares geometry at matched
gain.

## 4. Test conditions

All trained from scratch per condition, 5 epochs, same
seeds. Conditions on/cut/shuffle reuse one trained net
(train once, test three ways), so the test phase is 5 nets
per (task, seed), 11 lines per (task, seed):

1. aligned-on (fb_gain = aligned winner), tested wire-on.
2. aligned-cut (same net, fb_mode=cut at test).
3. aligned-shuffle (same net, fb_mode=shuffle at test).
4. learned-on (fb_gain = learned winner), tested wire-on.
5. learned-cut (same net, fb_mode=cut).
6. learned-shuffle (same net, fb_mode=shuffle).
7. random-on (fb_gain = ALIGNED winner's gain, matched
   control), tested wire-on.
8. random-cut (same net, fb_mode=cut).
9. random-shuffle (same net, fb_mode=shuffle).
10. wire-off (fb_gain = 0.0), tested wire-off. The
    round-16 config retrained on the new seeds and the
    occluded data: the control that the wire changed
    anything at all.
11. frozen-aligned (no plasticity, wire on at the aligned
    winner's gain). The round-16 frozen control, with the
    aligned wire: generation must need learning, not just
    the wire.

3 tasks x 20 seeds x 11 lines = 660 test lines.

## 5. Measures

Per run, the round-16/17 measures plus the new rollout
measures (all from the same test passes, one JSON line
per (task, condition, seed)):

- normal_acc: standard next-token accuracy, wire as
  tested. On occluded data it counts blanked positions as
  errors (section 2 caveat).
- omit_m2, omit_m1, cont_m1, margin_m2, shuf_m2: the
  round-16 omission measures (same code path, same motif
  detection by pattern, same frozen weights).
- p1_cos: mean cosine between p1 with the wire on and p1
  with the wire cut, on identical test sequences (the
  round-17 paired state pass). The state-effect gate:
  mean below 0.90 means the wire moves the state.
- err_mag_last: mean ||p1 - d1|| over the final training
  epoch. Rao-Ballard predicts the top-down prediction
  explains away input, so the error should fall with the
  wire on relative to wire-off (bar B5).
- cos_p1_omit_m2: mean cosine between p1 at the omitted
  m2 position and the omitted token's prototype. If the
  wire carries the prediction, p1 at the blank should
  match the omitted token's code when the wire is on and
  not when it is cut (bar B6).
- fb_drive_ratio: ||fb|| / ||total drive|| over the test
  passes at the tested mode (A29 corrected ratio).
- fb_align: Frobenius cosine between W_fb and -W_prox.T
  after training (1.0 = exactly the aligned map). For the
  learned wire this documents whether learning kept the
  subtractive geometry.
- fb_norm: Frobenius norm of W_fb after training.
- fb_clamp_frac: fraction of W_fb entries at the w_max
  clamp (a safety diagnostic; expected 0.000).
- r1_max: max |r1| over training and test (stability).
- clamp_frac and mean_burst: as in rounds 15-17.
- rollout_acc (sine): fraction of blank-step predictions
  correct over 15 blank steps after a 5-token cue, per
  test sequence. rollout_acc_late: the same over the last
  5 blank steps (the sustained-generation measure).
  rollout_mse: mean squared error on bin centres (a
  diagnostic, not a bar).
- rollout_comp (planted): fraction of first-detected motif
  triples completed correctly (both m1 and m2 predicted
  right) after blanking the motif steps. Lag-1 blanks t+1
  and t+2 (m1 at t+1, m2 at t+2); lag-2 blanks t+1..t+4
  (m1 at t+2, m2 at t+4). One rollout per sequence. The
  cue is always wire-on; the fb mode applies to the blank
  steps only. m1 is cue-determined for lag-1 (the starter
  determines m1), so m2 is the discriminating rollout
  step; the completion requires both, which is the
  stricter measure.

## 6. Bars (lag-1 is the primary planted task; lag-2 reported with the same floors)

CI convention (correcting rounds 17/18): EXACT Student-t,
not z=1.96. t(19) = 2.0930 for the 20-seed test bars;
t(4) = 2.7760 for 5-seed val summaries. scipy is absent
from the venv, so the constants are hardcoded from the
standard t table. Paired means are per-seed differences.
None values are structurally empty measures (e.g.
omission fields on sine) and are skipped, never treated
as zero.

### Experiment B' (the aligned wire is causal)

- B1: aligned on-cut omit_m2 mean >= +0.05 AND lower
  t-CI > 0 (lag-1 primary; lag-2 reported).
- B2: aligned on-shuffle omit_m2 mean >= +0.05 AND lower
  t-CI > 0.
- B3: aligned p1_cos mean < 0.90 (the state gate).
- B4: aligned on-off omit_m2 paired t-CI excludes 0.
- B5: aligned err_mag_last on < off: mean < 0 AND upper
  t-CI < 0 (explaining away, no practical floor).
- B6: aligned cos_p1_omit_m2 on < cut: mean < 0 AND upper
  t-CI < 0, AND the on mean < 0.20 (the prediction is in
  the state when the wire is on).
- B7 (guard, not a pass bar): aligned normal_acc >= 0.80x
  wire-off normal_acc on the same seeds. If B7 fails, the
  wire destabilizes learning; that is the finding and the
  round reports it instead of a pass.
- B8 (secondary): aligned on-cut gap exceeds random
  on-cut gap by >= +0.05 AND lower t-CI > 0 (geometry
  matters, not just having a wire, at matched gain).
- B9 (secondary): learned on-cut omit_m2 mean >= +0.05
  AND lower t-CI > 0.
- B10 (diagnostic, not a bar): learned gap minus aligned
  gap, reported with CI.

Experiment B' passes iff B1-B6 pass AND the B7 guard
passes. B8 and B9 are secondary (they are reported and
discussed, but they do not decide the pass).

The B1 floor is 0.05, not the round-17 0.10, because
AGY's one-seed control shows a +0.034 gap on clean data
and the round-17 0.10 floor would hide a real effect of
that size. The floor is still above the measured
single-seed gaps (+0.034 clean on seed 241, +0.017
occluded on seed 302), so the bar depends on the 20-seed
mean exceeding the single-seed values; if it does not,
the round reports the "causal but small" outcome below
instead of claiming a pass.

"Causal but small" (a distinct outcome, pre-registered):
if B1 (or B2, or B9) has a mean > 0 with lower t-CI > 0
but the mean is below the floor, the round reports
"causal but small" for that bar: the wire does real work,
but the effect is below the practical threshold. This is
neither a pass nor a fail; it is a named result the
verdict discusses. It exists so a small real effect is
neither hidden behind a failed bar nor inflated into a
pass.

### Experiment D (multi-step generation through the top-down pathway)

- D1: sine rollout_acc aligned on-cut mean >= +0.10 AND
  lower t-CI > 0.
- D2: sine rollout_acc aligned on-shuffle mean >= +0.10
  AND lower t-CI > 0.
- D3: sine rollout_acc aligned on-off mean >= +0.10 AND
  lower t-CI > 0.
- D4: sine rollout_acc_late (aligned on) mean >= 0.30
  AND lower t-CI > 0.20 (chance 1/32 = 0.031; the last 5
  steps are the sustained-generation measure).
- D5: planted1-occ rollout_comp aligned on-cut mean >=
  +0.20 AND lower t-CI > 0 (lag-1 primary; lag-2
  reported with the same floor).
- D6: planted1-occ rollout_comp aligned on-shuffle mean
  >= +0.20 AND lower t-CI > 0.
- D7 (secondary): sine rollout_acc learned on-cut mean
  >= +0.10 AND lower t-CI > 0.
- D8 (guard, not a pass bar): sine rollout_acc
  frozen-aligned (on) mean < 0.10. If D8 fails,
  generation does not need learning; that is the finding.

Experiment D passes iff D1-D6 pass AND the D8 guard
passes. D7 is secondary.

### The thesis-level decision rule (AGY)

Predictive-coding causality is genuinely established on
this architecture iff B' AND D both pass: the aligned
wire is causal for one-step prediction AND sustains
multi-step generation. If B' passes but D fails, the wire
is causal for one-step prediction but does not sustain
generation. If D passes but B' fails, generation works
but the one-step clamp does not: report both plainly;
that combination was not predicted. If neither passes,
the aligned wire does not establish predictive-coding
causality on this architecture.

## 7. Smoke test (VM only, before any Beast run)

Seed 302, 1 epoch for the fast checks, one 5-epoch
aligned run for the gate. Measured Nov 27, 2026
(rpc_round20_smoke.py, all 8 checks PASS):

- S1 occlusion: deterministic=True, 1235 blanks in the
  200-sequence train split, 0 blanked at motif positions,
  0 blanked at the probe position.
- S2 repro gain-0: driver normal_acc 0.195652,
  omit_m2 0.875000, identical to the round-17 code path
  to 6 decimals (A29 default changes nothing).
- S3 aligned gate: p1_cos 0.8033 (< 0.90), on-cut gap
  +0.0167, normal_acc 0.2383 (5 epochs, gain 1.0,
  occluded).
- S4 shuffle RMS: on 0.2781, shuffle 0.2756, relative
  difference 0.0090 (within 5%).
- S5 clamp_frac: 0.0000.
- S6 learned stable: normal_acc 0.2296, fb_align 0.912,
  fb_norm 17.20, r1_max 0.97, err_last 1.6213 (5 epochs,
  gain 1.0, occluded).
- S7 rollout: sine rollout_acc 0.160, rollout_acc_late 0.030
  (n=20 sequences), planted rollout_comp 0.680 (n=50), frozen
  sine rollout_acc 0.043. The sine rollout decays to chance
  by the late steps (see self-critique 11); the frozen
  rollout is near chance (0.031), so the D8 guard should
  hold.
- S8 CLI resumable: val lines 1 (expect 1), test lines 11
  (expect 11) after two runs each. No duplicate lines.

## 8. Execution

Driver: rpc_round20_aligned.py. CLI:
python rpc_round20_aligned.py <task> <val|test> --seeds
A-B --out FILE [--winners FILE] [--epochs N]
[--wire-type W] [--fb-gain G]
Tasks: planted1-occ, planted2-occ, sine. Val runs
planted1-occ only and needs --wire-type and --fb-gain.
Test needs --winners. One JSON line per (task, condition,
seed); a worker skips lines already present, so a killed
worker resumes cleanly. One BLAS thread per process
(OMP/MKL/OpenBLAS/NumExpr = 1, torch.set_num_threads(1)).

Monitor: r20/monitor.py, following the r19 pattern
(detached workers, marker at launch time, idempotent,
exits 0 always so a cron can run it repeatedly). Val: 9
workers (3 wires x 3 gains), 45 lines. Winner pick: the
selection rule of section 3, applied per wire type.
Test: 6 workers (3 tasks x 2 shards of 10 seeds), 660
lines. Analysis at 660 lines: concatenates the test
shards into r20_{task}_test_302-321.jsonl and runs
r20/r20_analyze.py, which writes r20/r20_analysis.md.

Budget (measured basis: one 5-epoch planted training run
plus two omission evals takes about 42 seconds on the VM;
the sine run is about twice that): val 45 runs, test 300
training runs (3 tasks x 20 seeds x 5 nets), about 6
CPU-hours total, about 1 hour wall with 6 workers. The
round runs on Beast (the monitor uses the Beast venv,
/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python, as in
round 19).

Analyzer: r20/r20_analyze.py. It reads
r20/r20_{task}_test_302-321.jsonl and the val files,
re-derives the winner per wire type with the same rule
as the monitor, computes per-condition means, medians,
IQRs, exact-t 95% CIs, paired differences, checks every
bar, and writes r20/r20_analysis.md.

## 9. Round 21 (split out of this round)

Continuous re-validation does not fit this round cleanly:
it needs a different driver (continuous readouts, ridge
probes, new tasks) and a different bar set. It is split
into round 21, sketched here, not implemented here:
Convention-A re-validation of the Lorenz/var-sine
prototype configs (alpha 1.0, smooth_sigma 2.0, eta_d 0.2,
per the section 57 spot check, which scored Lorenz
prototype MSE 5.07 vs persistence 13.05 on 5 seeds)
plus NARMA-10 and Mackey-Glass (tau 17, horizon h=5),
with COLUMN-STANDARDIZED ridge probes (the D8 lesson:
never an unstandardized fixed-lambda ridge again) and a
local parameter-free continuous readout (softmax-weighted
prototype centroid). Round 21 runs after round 20.

## 10. Self-critique (frozen before any test result)

1. Wire-geometry risk: this round bets that the geometry
   was the problem. If the aligned wire is also inert
   (B1/B2 fail, B3 passes but the gap is zero), the
   failure is architectural: this model cannot do
   predictive coding in the Rao-Ballard sense at any
   geometry, and the next step is a different
   architecture (e.g. an explicit error unit, or a
   plastic lower level), not another wire. The round is
   designed to distinguish these: B3 (the state gate)
   separates "the wire does not move the state" from
   "the state moves but the prediction does no work".
2. One-seed AGY control limitation: the aligned wire's
   promise rests on ONE seed (241) from a reviewer's
   control run, not on our own multi-seed data. The
   seed-241 numbers reproduced exactly in our code, which
   validates the implementation, but a single seed cannot
   establish the effect size. The 20-seed test is the
   first real estimate; the B1 floor (0.05) is above the
   single-seed gap (+0.034), so the round may well report
   "causal but small" rather than a pass. That is an
   honest outcome, not a failure of the design.
3. Learned-wire instability (found and fixed): AGY's
   literal r1-target learning rule collapses at gain 1.0
   (normal_acc 0.061, feedback 4.8x the drive) and
   rotates away from alignment at every gain. The
   composite back-projection rule replaces it, and it is
   stable and aligned in smoke (fb_align 0.91). But the
   composite rule is OUR fix, not AGY's formula, and its
   fixed point (a map about 1.2x the aligned norm after 5
   epochs) is not yet shown to beat the fixed aligned
   wire on any bar. B9 and D7 are secondary for exactly
   this reason: the learned wire is an exploration, not
   the claim.
4. Occlusion may shrink the gap: the design assumes
   occlusion gives the feedback more work to do. The
   measured single-seed occluded gap (+0.017 on seed 302)
   is SMALLER than the clean gap (+0.034 on seed 241).
   If the 20-seed mean follows the single-seed pattern,
   B1 fails at the 0.05 floor and the round reports
   "causal but small". The occlusion also depresses
   normal_acc by counting blanks as errors, which makes
   the B7 guard easier to pass (both sides are depressed
   equally) but makes normal_acc uninterpretable as an
   absolute number.
5. B' is the weak half: the one-step omission is near
   ceiling (omit_m2 0.95 to 1.00), so the on-cut gap has
   almost no headroom. D (the rollout) is the more
   sensitive test, because the input is blank for many
   steps and the feedback is the only external drive. If
   B' fails and D passes, the verdict must say the wire
   sustains generation but the one-step clamp cannot
   resolve it, not that the wire is inert.
6. The gate is a mean, not a per-seed check: B3 requires
   the MEAN p1_cos below 0.90. A wire could pass the mean
   while failing on a third of the seeds. The analyzer
   reports the per-seed distribution (median, IQR) so a
   mean driven by a few seeds is visible, but the bar
   itself is on the mean. A per-seed gate would be
   stricter and is a possible round-21 refinement.
7. Rollout completion conflates m1 and m2: rollout_comp
   requires BOTH m1 and m2 correct. For lag-1, m1 is
   cue-determined (the starter determines m1), so the
   completion is dominated by m2, which is the
   discriminating step. For lag-2, m1 (at t+2) is also a
   genuine prediction (two steps of blank). The analyzer
   reports rollout_m1 and rollout_m2 separately so the
   two steps can be disentangled if the completion is
   ambiguous.
8. The frozen control's prototypes are fitted on the
   clean train split (like every other condition), so
   the frozen net differs from the trained nets only in
   that its weights never move. That is the intended
   control. But the frozen net's W_dist1 is the random
   init, so its d1 is noise; the frozen rollout measures
   whether a random prediction plus the wire generates
   anything, which is a weak control. A stronger control
   (a trained net with the wire cut during the rollout)
   is already present as the "cut" condition (D1/D5).
9. Val/test seed overlap: the validation seeds (302..306)
   are a subset of the test seeds, as in rounds 15-19.
   The gain is picked on validation omission gaps and the
   bars are judged on the full 20-seed test, which
   includes those 5 seeds. The paired clamp contrasts
   (B1, B2, B8) compare conditions on identical seeds, so
   the overlap cannot manufacture a clamp gap, but it can
   inflate the winner's apparent gap slightly. The
   analyzer reports the val and test gaps separately so
   the inflation is visible.
10. The sine task has no occlusion (it has no motif/filler
    structure), so the clamp bars (B1-B10) are planted-only
    and the sine task contributes only the rollout bars
    (D1-D4, D7, D8). If the planted clamp fails but the
    sine rollout passes, the round cannot say whether the
    clamp would have passed on sine; that is a limit of
    the design, and it is stated here rather than
    discovered later.
11. The sine rollout decays (smoke S7, seed 302, aligned
    gain 1.0, 5 epochs): rollout_acc 0.160 overall but
    rollout_acc_late 0.030 (the last 5 of 15 blank steps),
    which is chance (1/32 = 0.031). The net generates the
    early blank steps well above chance but does not
    sustain generation to step 15. D4 (late mean >= 0.30)
    is therefore the hardest bar in D and is likely to
    fail even if D1-D3 pass. This is pre-registered here,
    not discovered later: if D1-D3 pass but D4 fails, the
    verdict is "the wire helps generation but generation
    decays to chance by step 11-15", not "the wire is
    inert". The D4 bar is NOT lowered to match the smoke:
    lowering it would move the goalposts. A relative late
    bar (on-cut on the late steps) is a possible round-21
    refinement; the analyzer already reports
    rollout_acc_late per condition, so the data will be
    visible even without a bar.
