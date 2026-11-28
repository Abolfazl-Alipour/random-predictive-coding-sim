# PROTOCOL-R22-PRECISION.md — Round 22: the precision test (experiment C)

Frozen Nov 27, 2026. Protocol for the precision test of the
random predictive coding (RPC) model: an explicit precision
manipulation with an expectation-by-precision interaction on
stimulus-specific error responses, with a live state-effective
aligned wire whose cut and shuffle are the controls.

Related records: `simulation-results.md` sections 57 (round-17/18
critiques, Kimi's bar for C), 58-59 (round-20 design and verdict:
B' fails, D fails, wire causal but small on lag-1 only), 61
(round-21 verdict: wall falls on re-validation, wire a net tax
on continuous tasks). Round-20 protocol:
`PROTOCOL-R20-ALIGNED.md`. Driver: `rpc_round22_precision.py`.
Analyzer: `r22/r22_analyze.py`. Monitor: `r22/monitor.py`.
Smoke test: `rpc_round22_smoke.py`.

## 1. Question and hypothesis

Experiment C is the last unrun discriminator. Omission (A)
passed (round 16). The top-down clamp with the random wire (B)
failed (round 17); the revised clamp with the aligned wire (B')
failed with a "causal but small" lag-1 effect (round 20); the
generative rollout (D) failed cleanly (round 20). What remains
is precision: in Rao-Ballard predictive coding, prediction
errors are weighted by their precision (inverse variance), so
an expected stimulus produces a small error, an unexpected
stimulus a large error, and the expectation effect is LARGER
when the sensory input is reliable (high precision) than when
it is noisy (low precision). Attention reverses
prediction-induced silencing by up-weighting precision
(Kok, Rahnev et al. 2012). The model has no precision
parameter, so the question is whether it shows the
expectation-by-precision interaction anyway, and whether the
live aligned feedback wire carries it.

Hypothesis H22: the signed-error model shows an
expectation-by-precision interaction on its error response
without any precision parameter, because input noise degrades
the sensory code p1 and therefore shrinks the error's
expectation dependence. The wire AMPLIFIES the interaction
(it subtracts the prediction from p1, which grows the error
more on violations than on matches) but does not CREATE it:
the interaction persists with the wire cut, at a reduced
size. The null for each bar is that the paired per-seed
difference is zero or favours the control.

What round 22 does NOT test: predictive coding itself. Even
a strong pass here does not establish it: an interaction that
survives the wire cut is a bottom-up sensory-noise effect,
not top-down precision weighting, and round 20 already failed
B' and D. A pass establishes that the model produces
precision-like behaviour from input noise plus error
selection, which is itself a result: it means a
precision-like interaction is NOT diagnostic of precision
weighting, and any claim that it is must rule this
mechanism out. The verdict section states this explicitly.

## 2. Background: what the design rests on

1. The error is e = p1 - d1, available from `_step_core`
   (it returns p1 and d1; the driver forms e). d1 at step t
   is computed from y2_prev before the current token is fed
   in, so d1 carries no leakage of the current token.
2. Round 16: the trained net holds a stimulus-specific
   prediction through a missing input (omit_m2 0.966 vs
   frozen 0.048). The prediction d1 is real and specific.
3. Round 20: the aligned wire at gain 1.0 is live
   (p1_cos 0.804-0.810, fb_drive_ratio about 0.31-0.38)
   and causal but small on lag-1 omission (B1 +0.014,
   B2 +0.033). It rotates module-1 state away from the
   prediction when module 2 is omitted (B6 passes on
   both lags, about 0.39). But it does NOT reduce the
   error magnitude (B5 fails on both lags with the
   wrong sign: err_mag_last is HIGHER with the wire on,
   +0.0138 lag-1, +0.0181 lag-2). The learned wire fails
   everywhere and is not used here.
4. Round 21: the wire is a net tax on continuous one-step
   prediction (F1 NULL twice, HURTS twice) while live
   (F2 0.31-0.34). The ridge probe shows no learning
   signal (R1 ties); the prototype probe shows a large
   one. The wire's only positive evidence remains the
   lag-1 clamp effect.
5. Mapping research (report section 2.3 and section 4,
   tests 4 and 1): the model predicts NO suppression of
   the sensory code p1 itself under expectation (Kok et
   al. 2012: expected gratings reduce V1 BOLD but
   IMPROVE decoding). This is a falsifiable divergence
   from Rao-Ballard, tested here as the S-family. The
   report also notes the omission confound (on omission
   trials e is about -d1, so prediction vs negative error
   are confounded) and the local-global test (the
   model predicts a local MMN-like error and NO global
   P3); the local-global test itself needs EEG data and
   is out of scope for this round, which tests the
   precision mechanism in silico.
6. Kimi's bar for C (section 57.4): an explicit precision
   manipulation with an expectation-by-precision
   interaction on stimulus-specific error responses, with
   a live state-effective wire whose cut and shuffle
   remove the effect. This protocol implements the bar
   literally and preregisters the strict version (W1s)
   separately from the amplification version (W1/W2),
   because the design-time probe (section 11) shows the
   strict version is unlikely to pass.

## 3. The precision operationalization (the honest core)

There is NO precision parameter in this model, and the
protocol does not pretend otherwise. The three candidate
operationalizations were considered and two were rejected:

1. ADOPTED: sensory reliability. Precision is the
   reliability of the probe input, manipulated at TEST
   time only as additive Gaussian noise on the encoded
   probe vector: x = normalize(encode(tok) + c * z),
   z ~ N(0, I), fed through `step_vec`. High precision
   is c = 0 (clean); low precision is c = c_low, picked
   per task in validation (section 6). Training is always
   clean, so the learned prediction is identical across
   precision conditions; only the sensory evidence at the
   probe step differs. This is the Kok-Rahnev analogue:
   attention changes the gain/reliability of the incoming
   sensory signal at test time, not what was learned.
   The noise vectors are deterministic per (seed, trial,
   c) by integer arithmetic (section 9); a design-time
   bug used Python `hash()` on trial-type strings, which
   is randomized per process and made noisy-condition
   numbers vary between runs -- the driver must never
   use `hash()` for seeds.
2. REJECTED: an error gain during training (scaling eta_d
   by a "precision" factor). It changes what is learned,
   not the test-time error response, and confounds the
   learning rate with precision. The Kok-Rahnev effect
   is a test-time attentional gain, not a learning-rate
   change.
3. REJECTED: bump width (smooth_sigma) as precision. A
   wider bump is a less precise sensory input, but
   changing smooth_sigma at test time also changes the
   input norm and the calibration basis (prototypes are
   built at the training sigma), and round 15 showed
   smooth_sigma interacts with the delta rule in its own
   right (one-hot 0.847 vs smooth 0.467). The noise
   manipulation keeps the same mean code and only adds
   variance, which is the cleaner reliability manipulation.

The NLMS denominator ||y2_prev||^2 + 1 is explicitly NOT
precision: it is input-norm normalization of the learning
rate, it acts during training only, and it has no
test-time gain on the error. Any reading of round 22 as
"testing the denominator" is wrong; the test is of the
sensory-reliability interaction defined above.

## 4. Tasks, data, and probe design

Tasks: planted lag-1 (`planted1`), planted lag-2
(`planted2`), sine (`sine`). Round-15 winner configs
(planted alpha 1.0, smooth 0.0, eta_d 0.05; sine alpha 1.0,
smooth 0.0, eta_d 0.2), 5 epochs, K = 16 planted, K = 32
sine. Splits exactly as round 20 (`rpc_round20_aligned.
splits`): planted 200 train / 50 val / 50 test per seed,
sine 50 / 20 / 20. No occlusion in this round: training
and probes are clean except for the probe-step noise.

Probe design (matched, within-net). A probe trial is a
short context sequence followed by ONE probe token, fed
with train=False. The context and the probe are matched
across trial types within a seed (same filler prefix),
and the noise vector is identical across wire modes for
the same trial. 20 prefixes per (task, seed).

Planted lag-1 (motifs (1,2,3), (5,6,7), (9,10,11); m0/m1/
m2 are the motif tokens; F is a non-motif filler token):
1. EXP (expected): context = prefix + [m0], probe = m1.
   The network predicts m1 after m0.
2. VIO (violation): context = prefix + [m0], probe = X, a
   random non-motif token. The network predicts m1 but
   gets X.
3. UNEXP (same token, unexpected): context = prefix + [F],
   probe = m1. The token m1 arrives with no motif context.
4. CTL (control): context = prefix + [F], probe = Y, a
   random non-motif token. No expectation either way.
Planted lag-2 (motif at (t, t+2, t+4)): EXP context =
prefix + [m0, F, m1], probe = m2; VIO same context,
probe = X; UNEXP context = prefix + [F, F, F], probe =
m2; CTL context = prefix + [F, F, F], probe = Y.
Sine: context = a 50-token sine prefix from the test
split; EXP probe = the true next token; VIO probe = the
true token + 8 bins (mod 32). Sine has no UNEXP/CTL:
any token is either the continuation or a violation.

Per probe trial the driver records, at the probe step:
e = p1 - d1, ||e||, ||p1||, cos(e, proto_presented),
cos(e, proto_predicted) where predicted = out["pred"],
decode(e) = argmax_k cos(e, proto_k), decode(p1),
pred, p1_cos = cos(p1, d1), fb_drive_ratio. Prototypes
are the trained net's own (net.prototypes, normalized).
Per-net scale: p1scale = mean ||p1|| on clean CTL trials
(planted) or clean EXP trials (sine) of the same net.
Relative error: rel_e = ||e|| / p1scale. The bars use
rel_e (scale-free) for magnitudes and cosines (already
scale-free) for specificity. Raw ||e|| is reported for
backward comparability only.

Wire modes: on / cut / shuffle are probe modes on the
same trained net (fb_mode in `_step_core`); off is a
separate net built with fb_gain = 0.0. Nets per
(task, seed): aligned (gain 1.0), random (gain 1.0),
off (gain 0.0), frozen-aligned (gain 1.0, plasticity
off). 8 config lines per (task, seed): aligned-on,
aligned-cut, aligned-shuffle, random-on, random-cut,
random-shuffle, wire-off, frozen-aligned.

## 5. Conditions, validation, and test arms

Validation has two phases (same monitor pattern as round
21: token val, then bump val).

Phase 1 (noise calibration): one aligned net per (task,
val seed). For each c in {0.10, 0.15, 0.25, 0.35, 0.50},
measure mean decode(p1) on calibration trials (CTL for
planted, EXP for sine) at the probe step. Pick c_low per
task as the SMALLEST c with mean decP1 < 0.75. Target
range is 0.50-0.75 (the input is degraded but still
decodable). If no c in the grid gives decP1 < 0.75, pick
0.50 and flag it. If the smallest grid c already gives
decP1 < 0.50, pick it and flag "over-corrupted". The
selection rule is frozen here; the value comes from val.
Expected from the design probe (seed 347): planted
c_low 0.35-0.50 (decP1 CTL: 0.85 at 0.25, 0.50 at 0.50),
sine c_low 0.15-0.25 (decP1 EXP: 0.60 at 0.25).

Phase 2 (interaction val): at c_low, run the full
8-config probe on val seeds 347-351. Compute per task
the val interactions (wire-on aligned, rel_e) and the
wire differences, and set the floors by the rule in
section 7. Floors go into winner.json before the test
launches. The val interaction must be reported in full
in the section draft, including the per-config values,
so the floor rule can be audited.

Test: 8 config lines per (task, seed) x 3 tasks x 20
test seeds 352-371 = 480 lines. Workers: 6 (3 tasks x
2 shards of 10 seeds). The monitor drives noise val (75
lines) -> c_low -> interaction val (120 lines) ->
floors -> winner.json -> test (480 lines) -> analysis.

## 6. Definitions (frozen)

Per (task, seed, config), from the probe aggregates:
- gap(c) = rel_e(VIO, c) - rel_e(EXP, c), the expectation
  effect on error magnitude at noise level c. rel_e uses
  the config's own p1scale.
- interaction = gap(0) - gap(c_low), the
  expectation-by-precision interaction on error magnitude.
- spec(c) = cos(e, proto_presented) - cos(e,
  proto_predicted) on VIO trials at noise level c, the
  stimulus-specificity of the error (positive means the
  error points at what arrived, not what was predicted).
- spec_interaction = spec(0) - spec(c_low).
- decE_VIO = decode(e) accuracy on VIO trials at c = 0
  (fraction where decode(e) == presented token).
- decP1_EXP / decP1_UNEXP = decode(p1) accuracy on EXP /
  UNEXP trials at c = 0 (planted only; same token m1).
- p1n_EXP / p1n_UNEXP = mean ||p1|| on EXP / UNEXP trials
  at c = 0, divided by p1scale (planted only).

## 7. Bars

All test bars use paired per-seed differences over the 20
test seeds with EXACT Student-t 95% CIs (t(19) = 2.0930).
A bar passes when the paired difference favours the
first-named side with its lower 95% CI above 0, unless the
bar says otherwise. Floors for P1, P2, P3, W1-W4 are set
by val (section 5, phase 2) as 50% of the val-measured
value, with absolute minima stated below; the val values
and the resulting floors are recorded in winner.json and
in the section draft before the test launches. n = 20 for
every bar; a bar with fewer paired seeds fails loudly in
the analyzer.

P bars: the interaction (per task: sine, planted1,
planted2). Sine is primary; planted1 is the
generalization to probabilistic expectation; planted2
is the replication on 2-step memory.
1. P1: interaction (wire-on, aligned) >= floor_P1[task]
   with lower CI > 0. floor_P1 = max(0.50 * val
   interaction, 0.02 rel_e). Smoke expectation (seed 347,
   one noise draw): sine about 0.11, planted1 about
   0.024.
2. P2: spec_interaction (wire-on, aligned) >=
   floor_P2[task] with lower CI > 0. floor_P2 =
   max(0.50 * val spec_interaction, 0.05). Smoke
   expectation: sine about 0.19, planted1 about 0.17.
3. P3: gap(0) trained minus gap(0) frozen (rel_e,
   wire-on aligned vs frozen-aligned) >= floor_P3[task]
   with lower CI > 0. floor_P3 = max(0.50 * val gap
   difference, 0.02 rel_e). Smoke expectation: sine
   about 0.15, planted1 about 0.047. This is the
   learning control: the interaction must be learned,
   not a reservoir artifact.
4. P4 (guard): spec(0) (wire-on, aligned) mean >= 0.25
   with lower CI > 0.15, AND decE_VIO (wire-on, aligned)
   mean >= 0.50 with lower CI > 0.30. The error must be
   stimulus-specific before any interaction on it is
   interpretable. Smoke: spec(0) sine 0.64, planted1
   0.54; decE_VIO 1.00 on both.

W bars: the wire (per task; sine primary).
1. W1: interaction_on - interaction_cut >= floor_W1[task]
   with lower CI > 0. floor_W1 = max(0.50 * val
   (on-cut) difference, 0.02 rel_e). Smoke expectation:
   sine about 0.055, planted1 about 0.019. If the mean
   is > 0 with lower CI > 0 but below the floor, the
   pre-registered "causal but small" outcome applies
   (as in round 20), and the verdict records it.
2. W2: interaction_on - interaction_shuffle >=
   floor_W2[task] with lower CI > 0. floor_W2 = max(0.50
   * val (on-shuffle) difference, 0.02 rel_e). Smoke
   expectation: sine about 0.06, planted1 about 0.028.
   CAVEAT (self-critique point 13): the shuffle is NOT
   magnitude-matched for the aligned wire (fbdr drops
   19% planted, 47% sine), so a W2 pass confounds
   feedback direction with feedback magnitude. The cut
   (W1) is the primary wire control; W2 is secondary.
3. W3: interaction_aligned - interaction_random >=
   floor_W3[task] with lower CI > 0. floor_W3 = max(0.50
   * val (aligned-random) difference, 0.02 rel_e).
   Smoke expectation: sine about 0.06, planted1 about
   0.013 (the random wire's interaction is 75% of the
   aligned wire's on planted, larger than aligned in raw
   units on sine but smaller in rel_e because the random
   wire inflates ||p1||).
4. W4: interaction_on - interaction_off >= floor_W4[task]
   with lower CI > 0. floor_W4 = max(0.50 * val (on-off)
   difference, 0.02 rel_e). The off net was not probed
   at design time; its floor comes from val alone.
5. W1s (diagnostic, NOT a pass/fail bar): the strict
   Kimi version -- the interaction is REMOVED by cutting
   the wire. It passes iff |interaction_cut| <
   floor_W1[task] AND its CI covers 0. The design probe
   predicts it FAILS: the interaction persists cut at
   30-45% of the wire-on size (planted +0.034 vs +0.098,
   sine +0.114 vs +0.238 on seed 347). Report the value;
   do not let a W1s fail veto a W1 pass.
6. W5 (gate): mean fb_drive_ratio on the aligned probe
   pass > 0.05 AND mean p1_cos < 0.90. If W5 fails, the
   W-family is void for that task (the arm did not carry
   a live wire) and the analyzer says so. Round-20
   values for reference: p1_cos 0.804-0.810,
   fb_drive_ratio about 0.31-0.38.
7. W6 (guard): normal_acc aligned-on >= 0.80 x
   normal_acc wire-off, per task. If the guard fails,
   the wire's cost invalidates the W-family comparison.

S bars: the p1-suppression divergence (planted tasks
only; sine has no UNEXP trials). Wire-off net, clean
(c = 0). Equivalence bars: they pass when the difference
is SMALL and the CI covers 0.
1. S1: |decP1_EXP - decP1_UNEXP| < 0.03 with the CI
   covering 0. The model predicts NO suppression of the
   sensory code: the same token decodes identically
   whether expected or not. Smoke: 1.00 vs 1.00 on seed
   347 in every wire mode.
2. S2: |p1n_EXP - p1n_UNEXP| < 0.05 (normalized units)
   with the CI covering 0. Same prediction for the code
   magnitude. Smoke: 1.64 vs 1.64 wire-on, 2.07 vs 2.01
   wire-cut (normalized diff about 0.03).
3. S3 (measurement only, no bar): S1/S2 quantities on
   the aligned wire-on net, reported for comparison. If
   S1/S2 pass wire-off but fail wire-on, the wire (not
   the input code) carries an expectation effect on p1.

## 8. Verdict rules

1. C STRONG PASS: sine P1, P2, P3, P4 pass AND planted1
   P1 passes AND W1 and W2 pass on sine. The interaction
   exists, is stimulus-specific and learned, generalizes
   to probabilistic expectation, and the live wire carries
   it.
2. C PASS: sine P1, P2, P3, P4 pass AND planted1 P1
   passes, but W1/W2 fail on sine. The interaction exists
   but the wire does not carry it: it is a bottom-up
   sensory-noise effect, not top-down precision.
3. C PARTIAL: sine P1 and P2 pass but planted1 P1 fails
   (or is "real but small": CI excludes 0, mean below
   the floor). The interaction exists under deterministic
   expectation only.
4. C FAIL: sine P1 or P2 fails, or sine P4 fails. No
   interaction, or the error is not stimulus-specific.
5. The lag-2 (planted2) bars are a replication: they do
   not change the verdict, but the verdict names whether
   the interaction survives 2-step memory.
6. Thesis-level: no outcome of round 22 establishes
   predictive coding. Round 20 failed B' and D; a C pass
   establishes only that the model produces
   precision-like behaviour from input noise plus error
   selection. A C PASS (interaction without wire control)
   is evidence AGAINST the precision-weighting reading:
   the interaction does not need a precision parameter
   or a top-down wire.

What would change the verdict:
1. If W1s passes (the interaction is removed by cutting
   the wire) on sine: the wire DOES carry the interaction,
   and the top-down precision reading strengthens
   considerably; a follow-up should test whether the
   wire's effect is specific to the aligned direction
   (W3) rather than any feedback.
2. If S1 or S2 fails (p1 IS suppressed by expectation):
   the mapping report's falsifiable divergence is
   falsified in the model's favour -- the reservoir's
   history dependence creates expectation suppression of
   the sensory code after all. That would be a positive
   result for the model and must be reported as such,
   not buried.
3. If P3 fails (frozen gap about equal to trained gap):
   the interaction is a reservoir artifact, not learning,
   and the P1/P2 passes are void as evidence about the
   delta rule.
4. If the val interaction is near zero on a task (lower
   t(4) CI <= 0 on 5 seeds): the floors fall to their
   minima and the test will likely fail P1 on that task.
   That is the correct outcome, not a design failure:
   it means the interaction is not measurable there.

## 9. Seeds, statistics, and launch

1. Seeds: val 347-351 (5), test 352-371 (20). Fresh: no
   overlap with rounds 15-18 (262-281), round 20
   (302-321), or round 21 (322-346). The val seeds pick
   c_low and set floors; the test seeds decide bars.
2. Statistics: exact Student-t CIs throughout, t(19) =
   2.0930 for the 20-seed test bars, t(4) = 2.7760 for
   5-seed val summaries. scipy is absent from the venv,
   so the constants are hardcoded from the standard t
   table. Paired per-seed differences, keyed by seed
   (never by list position). Means, medians, and IQRs
   are reported alongside every bar.
3. Noise determinism: every noise vector is drawn from
   np.random.default_rng with an integer seed computed
   by arithmetic from (seed, trial index, trial-type
   code, c): seed = 2_000_003 * seed + 7919 * tcode +
   104729 * trial_index + int(c * 1000), tcode in
   {EXP: 1, VIO: 2, UNEXP: 3, CTL: 4}. The same vector
   is reused across wire modes for the same trial. Never
   use Python hash() for a seed: string hashing is
   randomized per process (found in the design probe:
   noisy-condition numbers varied between identical runs).
4. Launch: copy the driver, analyzer, monitor, and
   protocol to Beast; Beast smoke (one val-noise + one
   val-interaction + one test invocation); monitor val
   launch; create the r22-monitor cron (every 30 min,
   same pattern as r21-monitor). Monitor phases: noise
   val (75 lines) -> pick c_low per task -> interaction
   val (120 lines) -> pick floors -> winner.json -> test
   (480 lines) -> concat + analysis. Detached workers,
   marker written at launch time, idempotent via marker
   files. The monitor stops (no new launches) if any
   JSONL contains a traceback.
5. Cost estimate (measured in the design probe on the VM:
   training one net is about 10-15 s planted, about 25 s
   sine; the probe pass is about 1 s per (net, mode)).
   Test phase: 240 nets, about 60-90 min on a loaded
   Beast. Val phases add about 135 nets. Total round
   about 3 h on Beast; give it a full day.

## 10. Self-critique (frozen before any test result)

1. The precision definition is a stipulation, not a
   discovery. "Precision = input noise" is adopted
   because the model has no precision parameter; a
   critic can fairly say the round tests sensory
   reliability, not precision weighting. The protocol
   says so in section 3 and the verdict turns on it:
   a pass means the interaction exists WITHOUT precision
   weighting, which is the informative direction.
2. The interaction is mostly bottom-up by construction.
   The design probe shows the interaction persists cut
   at 30-45% and the random wire carries 75% of the
   aligned wire's interaction on planted. If W1/W2 fail
   and W1s fails, the round's headline is "the
   interaction does not need the wire" -- a real result,
   but it does not advance the thesis claim that the
   model implements predictive coding. It advances the
   opposite: precision-like behaviour is cheap.
3. The planted interaction is small. Smoke rel_e
   interaction about 0.048 (planted) vs 0.22 (sine).
   The planted P1 floor (about 0.024) is close to the
   measurement noise across seeds; a planted P1 fail
   may reflect power, not absence. The verdict rule
   handles this by making sine primary and planted1 a
   generalization test, with "real but small" as a
   recorded outcome.
4. The floors are relative (50% of val), so a weak val
   interaction produces a weak floor. The absolute minima
   (0.02 rel_e, 0.05 cosine) are the only guard against
   a trivial bar. If the val interaction is small, the
   section draft must say so plainly and the test's P1
   pass must be reported with the val value alongside,
   not as a standalone win.
5. The noise level c_low is picked by a decode rule that
   can over-corrupt. If val picks a c where decP1 is
   0.50, the low-precision condition is heavily degraded
   and the interaction is close to the full gap0 -- the
   bar is then easy in one direction (the gap at low
   precision is near zero) but the measurement is
   uninformative about graded precision. The val must
   report decP1 at the picked c; if it is below 0.50,
   the "over-corrupted" flag applies and the verdict
   notes the interaction was measured against a near-
   destroyed input.
6. rel_e normalization confounds wire comparisons. The
   random wire inflates ||p1|| (p1scale 2.37 planted,
   1.94 sine vs 1.72 / 1.01 aligned), so raw ||e||
   comparisons across wire types are meaningless and
   rel_e is used throughout. But rel_e divides by a
   quantity the wire itself changes, so a wire that
   inflates p1 shrinks rel_e without changing the error.
   W3 is therefore partly a test of the p1scale
   difference, not only of the interaction. The analyzer
   reports raw ||e|| interactions alongside rel_e so the
   confound is visible.
7. The wire's effect on ||e|| is measurement-dependent.
   Round 20 B5 found err_mag_last HIGHER with the wire
   on during training (+0.014); the design probe finds
   probe-step ||e|| LOWER with the wire on at test
   (planted EXP 1.70 vs 1.87 cut). These are different
   quantities (training mean vs probe positions, occluded
   vs clean training) and do not contradict each other,
   but if the test reproduces the probe direction, the
   protocol's B5 discussion needs revisiting: the wire
   reduces the error at the probe while raising it
   during training, which would mean the wire's error
   effect depends on the learning state.
8. The sine task has no UNEXP/CTL trials, so the S-family
   cannot run on sine and the sine P1 interaction has no
   no-expectation baseline within the task. The frozen
   net (P3) is the only sine control for "no learning".
   If P3 fails on sine, the sine interaction is
   uninterpretable as learning, and the round rests on
   the planted tasks.
9. Twenty prefixes per (task, seed) is a small probe
   sample: the per-seed aggregates average 20 trials,
   and the 20-seed CI averages over seeds, but a single
   unlucky prefix set could bias one seed. The prefixes
   are fixed per (task, seed) by integer RNG, identical
   across all configs on that seed, so prefix effects
   cancel in the paired differences. The residual risk
   is a seed whose prefixes are unrepresentative; the
   analyzer reports per-seed values so it can be seen.
10. The lag-2 probe context ([m0, F, m1] then m2) assumes
    the network's lag-2 prediction is carried by the
    reservoir across the intervening filler. Round 15
    showed lag-2 planted accuracy 0.222 (vs 0.287 Bayes),
    so the prediction exists but is weak; the lag-2
    interaction may be too small to measure, and its
    floor will fall to the minimum. That is acceptable
    for a replication arm, but a lag-2 fail must not be
    read as a fail of the lag-1/sine result.
11. The S-family equivalence bars (S1/S2) can pass by
    low power: with 20 seeds, a CI covering 0 with a
    0.03 bound is a weak equivalence claim if the SE is
    large. The analyzer reports the CI width alongside
    the pass/fail; a pass with a wide CI is reported as
    "no detected suppression (CI wide)", not as proof of
    absence.
12. Multiple comparisons: 34 pass/fail bars. The verdict
    rule uses only a preregistered subset (sine P1-P4,
    planted1 P1, sine W1/W2), so the family-wise risk is
    contained to those 7 bars; the rest are reported
    without verdict weight. No correction is applied,
    because the bars are not independent claims but
    facets of one interaction; the protocol states this
    and the verdict does not rest on the full table.
13. The shuffle is NOT a magnitude-matched control for
    the aligned wire. Smoke (seed 347, 5 prefixes):
    fb_drive_ratio on vs shuffle is 0.251 vs 0.203 on
    planted1 (19% drop) and 0.302 vs 0.159 on sine (47%
    drop). The aligned wire is a structured map
    (-W_prox.T); permuting d1 moves it out of the
    subspace where the map has large gain, so the
    shuffled feedback is smaller. A W2 pass therefore
    confounds "the aligned direction matters" with "more
    feedback matters". The cut (W1) is the primary wire
    control because it sets the feedback to zero; the
    shuffle (W2) is secondary and its magnitude confound
    must be disclosed in the verdict. A rescaled shuffle
    (normalizing the shuffled feedback to the on-mode
    norm) would fix this but requires an rpc_v2.py change
    (a new A-numbered assumption) and would break
    comparability with the round-17/20 shuffle controls;
    it is a follow-up, not part of this round.

## 11. Appendix: numbers the design rests on

Design probe, seed 347, one noise draw per noisy
condition (deterministic clean numbers; noisy numbers
vary by about +/-0.02 raw across noise draws -- see
section 9.3 on the hash() bug). Aligned wire, gain 1.0,
round-15 winner configs, 20 prefixes.

Planted lag-1, aligned fb=on: gap0 (VIO-EXP) +0.1573
raw; gap at c=0.25 +0.1148, at c=0.50 +0.0592 to +0.0912
(two runs), at c=0.75 -0.0079. Interaction (c=0 minus
c=0.50) +0.066 to +0.098 raw, rel_e about +0.048
(p1scale 1.72). VIO specificity spec(0) +0.54,
spec(0.50) +0.20, interaction +0.34. decE VIO c=0 1.00,
c=0.50 0.40. decP1 CTL: 1.00 at c=0, 0.85 at 0.25, 0.50
at 0.50, 0.35 at 0.75. p1 decode EXP vs UNEXP 1.00 vs
1.00; ||p1|| 1.64 vs 1.64. Wire-cut interaction +0.021
to +0.034; shuffle +0.002 to +0.023. Frozen: gap0
-0.007, interaction about 0.00, err about 8.4 (frozen
d1 is far from p1).
Random wire, planted1: interaction +0.0719 raw, rel_e
+0.0303 (aligned +0.0963 / +0.0560 in the same run);
spec interaction +0.311 (aligned +0.372). p1scale 2.37
vs 1.72 aligned: the random wire inflates ||p1||.

Sine, aligned fb=on: gap0 +0.3220 raw; gap at c=0.25
+0.0899, at c=0.50 +0.0374 to +0.0841, at c=0.75
+0.0714. Interaction (c=0 minus c=0.50) +0.238 to
+0.277 raw, rel_e about 0.22 (p1scale 1.01). VIO
specificity spec(0) +0.64, spec(0.50) +0.26,
interaction +0.38. decE VIO c=0 1.00, c=0.50 0.15.
decP1 EXP: 1.00 at c=0, 0.60 at 0.25, 0.20 at 0.50.
Wire-cut interaction +0.114; shuffle +0.102. Frozen:
gap0 +0.017, interaction -0.010.
Random wire, sine: interaction +0.3174 raw (LARGER
than aligned +0.2846 in raw units) but rel_e +0.1635
(smaller than aligned +0.2820) because p1scale is 1.94
vs 1.01. Spec interaction +0.345 (aligned +0.365).

Round-20 values reused: aligned p1_cos 0.804-0.810;
fb_drive_ratio about 0.31-0.38; B1 +0.014, B2 +0.033
(lag-1, "causal but small"); B5 wrong sign (+0.0138,
+0.0181); sine normal_acc aligned-on 0.666 vs wire-off
0.892.

Smoke test (rpc_round22_smoke.py, seed 347, 5 prefixes,
1 seed -- NOT the val, which uses 5 seeds and 20 prefixes):
c_low picked by the protocol rule is planted1 0.35,
planted2 0.35, sine 0.25. Floors from the smoke (50% rule):
sine P1 0.181, P2 0.178, P3 0.235, W1 0.143, W2 0.154,
W3 0.124, W4 0.020 (minimum); planted1 P1 0.031, P2 0.190,
P3 0.042, W1-W4 0.020 (minimum); planted2 P1 0.020
(minimum; the lag-2 interaction is NEGATIVE, -0.015, on the
smoke seed), P2 0.149, W2 0.025. The val (5 seeds) will
replace these numbers; the smoke values are a sanity check
that the floor machinery produces sane magnitudes.
decP1 on the smoke's 5-prefix calibration is non-monotonic
(planted1 CTL: 1.00 at 0.10, 0.80 at 0.15, 1.00 at 0.25,
0.40 at 0.35, 0.40 at 0.50) because each value is a mean
over 5 trials; the val averages 5 seeds x 20 prefixes =
100 trials per c, so the selection is stable there.
Lag-2 on the smoke seed: the magnitude interaction is
absent (-0.015) while the specificity interaction is strong
(+0.297), and the S2 p1n difference is -0.058, outside the
0.05 bound -- the lag-2 arm and the S2 bound are both
marginal at smoke scale and the 20-seed test decides.
