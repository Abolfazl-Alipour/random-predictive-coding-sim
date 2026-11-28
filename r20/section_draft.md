## 58. Round 20: aligned-feedback clamp (revised experiment B) + multi-step omission rollout (experiment D) -- DESIGN, NOT LAUNCHED

(Nov 27, 2026. Design subagent report; protocol frozen in
PROTOCOL-R20-ALIGNED.md. No Beast launch. Smoke tests run on
the VM only.)

### Why this round exists

Round 17 tested the wrong wire. It used a fixed random
additive wire (+W_fb @ d1, A26) and found it inert: cutting,
shuffling, or never building it changed omission accuracy by
at most half a point. Both external reviewers of rounds 17/18
(Kimi and AGY Gemini, section 57) agree the failure stands
only for that wire. Rao and Ballard (1999) and Chapter 3
specify ALIGNED SUBTRACTIVE feedback: minus the readout
transpose times the prediction. AGY's one-seed control
(reviewer-reported) reached omit_m2 1.000 with that wire at
gain 1.0. This round fixes the wire first (A29), then re-runs
the clamp (revised experiment B) and adds the multi-step
omission rollout (generative experiment D, first half).
Precision (experiment C) is NOT in this round: both reviewers
agree it is premature without a causal wire.

### Assumption A29 (rpc_v2.py)

Three parts, all default-off, all previous results reproduce
exactly (smoke S2: gain 0.0 reproduces the round-17 code path
to 6 decimals).

1. Aligned subtractive feedback: fb_wire="aligned" replaces
   W_fb at init with -W_prox.T, rescaled to the Frobenius norm
   of the random W_fb it replaces, so at the same fb_gain the
   drive magnitude matches the random wire and only the
   direction differs. Reproduces AGY's seed-241 control
   EXACTLY (normal_acc 0.2330, omit_m2 on 1.000 / cut 0.966,
   omit_m1 on 0.8034, p1_cos 0.8565, err_last 1.6162,
   cos_p1_omit_m2 on 0.036 / cut 0.401, fb_drive_ratio 0.383).
   At gain 0.3 the gate fails (p1_cos 0.9667, gap 0.000).
2. Learned back-projection feedback: fb_wire="learned" starts
   as the aligned map and is trained by a local composite
   delta rule (gradient descent on
   (1/2)||-d1 - W_prox @ W_fb @ d1||^2, NLMS form), so the
   feedback, passed through the readout, subtracts the
   prediction from the code. This is NOT AGY's literal
   formula (delta W_fb proportional to (r1 - W_fb d1) d1^T):
   that rule was implemented first and measured, and it is
   unusable. On seed 241 it collapses at gain 1.0
   (normal_acc 0.061, chance 0.0625, feedback 4.8x the total
   drive) and rotates AWAY from the aligned direction at
   every gain (fb_align -0.02 at 0.1, 0.11 at 0.3, 0.63 at
   1.0), because d1 barely predicts the reservoir state r1
   and the target has state scale while the drive has drive
   scale. The composite rule is stable at every gain
   (normal_acc 0.232 to 0.240), keeps its subtractive
   direction (fb_align 0.91), and grows its norm from 14.20
   to about 17.2. The deviation from AGY's literal formula
   is deliberate, measured, and documented in the protocol.
3. Blank token: _encode returns the zero vector for tok == K.
   No existing code passes token K, so no previous result
   changes. Used for occlusion and rollout blanks.
4. Corrected drive ratio: fb_drive_ratio = ||fb|| / ||total
   drive|| (round 17's fb_rms_ratio divided by the bottom-up
   norm only and overstated the feedback fraction by about
   1.7x).

### Design

Tasks: planted1-occ, planted2-occ (40% of filler positions
blanked in train AND test; motif positions and the filler
probe position never occluded; mask deterministic per
(task, seed, split); calibration and prototypes on CLEAN
train), sine (no occlusion). Seeds 302..321 (20 fresh).
Round-15 winner configs (planted alpha 1.0, smooth 0.0,
eta_d 0.05; sine eta_d 0.2), 5 epochs.

Validation: 3 wire types (aligned, learned, random) x 3
gains (0.1, 0.3, 1.0) on planted1-occ, val seeds 302..306.
Selection per wire type: largest val on-cut omit_m2 gap
among gains passing the state gate (val p1_cos < 0.90) and
stable (r1_max < 10); tie-break smallest gain. If no gain
passes the gate, the largest gap among stable gains wins
and the gate failure is flagged. The random wire's test gain
is the ALIGNED winner's gain (matched control).

Test: 5 nets per (task, seed), 11 lines per (task, seed),
660 test lines total. Conditions: aligned on/cut/shuffle,
learned on/cut/shuffle, random on/cut/shuffle (at the
aligned gain), wire-off, frozen-aligned.

Bars (exact Student-t, t(19) = 2.0930, correcting the
round-17/18 z=1.96 convention; lag-2 uses the SAME
0.10/0.05 floors as lag-1, fixing the round-17 analyzer
bug). B' (aligned wire is causal): B1 on-cut omit_m2 >=
+0.05 with lower t-CI > 0; B2 on-shuffle >= +0.05; B3
p1_cos mean < 0.90; B4 on-off CI excludes 0; B5 explaining
away (err_mag_last on < off, upper t-CI < 0); B6
cos_p1_omit_m2 on < cut with on mean < 0.20; B7 guard
(normal_acc >= 0.80x wire-off). B8 (secondary): aligned
gap exceeds random gap by >= +0.05. B9 (secondary):
learned on-cut gap >= +0.05. B10 (diagnostic): learned gap
minus aligned gap. B' passes iff B1-B6 pass and B7 passes.
D (rollout): D1-D3 sine rollout_acc on-cut / on-shuffle /
on-off >= +0.10; D4 sine late rollout_acc mean >= 0.30 with
lower t-CI > 0.20 (chance 0.031); D5-D6 planted rollout_comp
on-cut / on-shuffle >= +0.20; D7 (secondary) learned sine
rollout >= +0.10; D8 guard frozen sine rollout < 0.10. D
passes iff D1-D6 pass and D8 passes. Thesis-level rule
(AGY): predictive-coding causality is established iff B'
AND D both pass.

Pre-registered "causal but small" outcome: if B1 (or B2,
or B9) has mean > 0 with lower t-CI > 0 but the mean is
below the floor, the round reports "causal but small" as a
distinct outcome (neither pass nor fail).

Continuous re-validation is SPLIT into round 21
(Convention-A Lorenz/var-sine re-validation plus NARMA-10
and Mackey-Glass tau 17, column-standardized ridge probes,
local parameter-free continuous readout). It is sketched
in the protocol, not implemented in round 20.

### Smoke results (VM, seed 302, rpc_round20_smoke.py)

- S1 occlusion: deterministic=True, 1235 blanks in the
  200-sequence train split, 0 at motif positions, 0 at the
  probe position. PASS.
- S2 repro gain-0: driver normal_acc 0.195652, omit_m2
  0.875000, identical to the round-17 code path to 6
  decimals. PASS.
- S3 aligned gate: p1_cos 0.8033 (< 0.90), on-cut gap
  +0.0167, normal_acc 0.2383 (5 epochs, gain 1.0, occluded).
  PASS. Note: the occluded single-seed gap (+0.017) is
  SMALLER than the clean gap (+0.034 on seed 241), so the
  B1 floor (0.05) is above both measured single-seed gaps
  and the round may report "causal but small" rather than
  a pass.
- S4 shuffle RMS: on 0.2781, shuffle 0.2756, relative
  difference 0.0090 (within 5%). PASS.
- S5 clamp_frac: 0.0000. PASS.
- S6 learned stable: normal_acc 0.2296, fb_align 0.912,
  fb_norm 17.20, r1_max 0.97, err_last 1.6213 (5 epochs,
  gain 1.0, occluded). PASS.
- S7 rollout: sine rollout_acc 0.160, rollout_acc_late 0.030
  (n=20), planted rollout_comp 0.680 (n=50), frozen sine
  rollout_acc 0.043 (near chance 0.031). PASS. The sine
  rollout decays to chance by the late steps; the frozen
  rollout is near chance, so the D8 guard should hold.
- S8 CLI resumable: val lines 1 (expect 1), test lines 11
  (expect 11) after two runs each. No duplicate lines. PASS.

### Files

- PROTOCOL-R20-ALIGNED.md (frozen Nov 27, 2026; 11-point
  self-critique).
- rpc_round20_aligned.py (driver; val and test phases;
  resumable JSONL; one BLAS thread per process).
- r20/r20_analyze.py (analyzer; exact-t CIs; bar table and
  verdict; writes r20/r20_analysis.md).
- r20/monitor.py (val -> winner pick -> test -> analysis;
  9 val workers, 6 test workers, 660 test lines; detached
  workers, marker at launch time; Beast venv).
- rpc_round20_smoke.py (8 checks, seed 302).
- rpc_v2.py (A29: fb_wire, blank token, corrected drive
  ratio, composite learned rule).

### Two sharpest risks in this design

1. The B1 floor (0.05) is above every measured single-seed
   gap (+0.034 clean, +0.017 occluded). If the 20-seed mean
   follows the single-seed pattern, B1 fails and the round's
   headline is "causal but small" (if the CI excludes 0) or
   a null. The design handles this honestly (the
   pre-registered "causal but small" outcome), but the
   headline pass is genuinely uncertain, and D (the rollout)
   is the more sensitive test.
2. The learned wire is our fix, not AGY's formula. The
   composite rule is stable and aligned in smoke, but it has
   never been shown to beat the fixed aligned wire on any
   bar, and its fixed point (a map about 1.2x the aligned
   norm after 5 epochs) may still be growing when training
   stops. If the learned wire's bars (B9, D7) fail while the
   aligned wire's pass, the round cannot say whether a
   longer-trained learned wire would have won; that is a
   round-21 question, not a round-20 conclusion.

### Next step

Launch on Beast via r20/monitor.py (val -> winner -> test
-> analysis). Estimated about 6 CPU-hours, about 1 hour
wall with 6 workers. After completion: verify the analysis
by independent recomputation from the raw JSONL, write the
verdict as section 59, update README Status and MEMORY.md,
log a tracking entry on goal_a9a871e81255, remove the
r20-monitor cron, and deliver the verdict to Reza.
