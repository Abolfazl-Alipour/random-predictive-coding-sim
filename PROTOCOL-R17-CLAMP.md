# Round 17 protocol: the top-down clamp (experiment B)

Frozen Nov 26, 2026, before any round-17 test result is seen.

Question: does the top-down prediction causally shape the lower
module? Experiment B of the deep-research report
(~/workspace/research_notes/predictive-coding-local-learning-rules-20261127-0039/report.md,
sections 1.4 and 1.5). Round 16 showed the trained delta network
holds a stimulus-specific prediction that survives a missing
input. Round 17 asks whether that prediction does any work:
cut it, shuffle it, and see whether the lower module notices.

## 0. The architectural fact that sets this round

In the current model there is NO top-down pathway. d1, the
distal drive (module 2's prediction of module 1's code), is
computed in `_step_core` (rpc_v2.py), used for the prediction
readout and for the delta learning rule, and never fed back
into either module. Module 1's reservoir state depends only
on the bottom-up input and its own recurrence
(`ReservoirModule.forward`: `tanh(W_in @ e_sparse + W_res @ r
- inhib)`). A literal "clamp the top-down projection" test
cannot run on this architecture: there is nothing to clamp,
and cutting d1 would change only the readout, which proves
nothing.

So round 17 has two parts. Part 1 (assumption A26) ADDS the
top-down feedback wire. Part 2 runs the clamp test on the
wired network. A pass means the wire is causal; a fail means
it is decorative. Either way the round settles whether this
architecture can do predictive coding in the Rao-Ballard
sense, or whether the prediction is a readout-only epiphenomenon.

## 1. Assumption A26: the top-down feedback wire

A fixed random feedback matrix W_fb (n_res x n_out), entries
randn / sqrt(n_out), the same scale logic as W_in, created at
network init from the seeded RNG state, never trained. At
each step, after d1 is computed and before module 1 runs:

    fb = fb_gain * (W_fb @ d1)

and module 1's reservoir update becomes:

    r_new = (1 - alpha) * r + alpha * tanh(
        W_in @ e_sparse + W_res @ r + fb - inhib)

`fb_gain` is a constructor parameter, default 0.0. At 0.0 the
wire is off and every previous result reproduces exactly.
d1 at step t is computed from y2_prev (module 2's code of the
previous token), so the feedback arriving while module 1
processes token t is the prediction of token t: prediction
and input coincide at the lower level, which is the
Rao-Ballard form. The error (p1 - d1) still drives only the
W_dist1 delta update; module 1 has no plastic weights in this
config (learn_prox off), so "learning of the lower level
changes" is measured through W_dist1 and the omission
measures, not through module-1 weights (see self-critique 1).

Scale (measured, not assumed): trained delta d1 norm is
1.6-1.9 (round-16 records, r16_*_test_*.jsonl, d1_norm_*);
the bottom-up drive W_in @ e_sparse has norm about 2.3
(k_e=32 sparse expansion, same W_in scale as W_fb). So
fb_gain 1.0 makes the feedback term the same order as the
bottom-up drive, 0.3 makes it about a fifth, 0.1 about a
twentieth. The driver records the RMS ratio
RMS(fb) / RMS(bottom-up drive) per run, so the winning gain
is interpretable as a drive fraction, not a bare number.

Test-time clamp modes, set per step call (`fb_mode`):
- "on": the wire as trained (default).
- "cut": fb = 0. The prediction is removed, the bottom-up
  input and the readout are untouched.
- "shuffle": fb = fb_gain * (W_fb @ perm(d1)), where perm is
  a fixed random permutation of d1's coordinates, drawn once
  at network init from the seeded RNG state. The feedback
  keeps its magnitude and statistics but carries no
  information about the prediction. Cut vs shuffle separates
  drive magnitude from content (see self-critique 2).

The feedback loop is now closed: d1 shapes r1, r1 shapes
p1, p1 shapes the error, the error shapes W_dist1, W_dist1
shapes d1. Stability is monitored on every run (max |r1|,
NaN check, clamp_frac). A runaway is a finding, reported
plainly, not a crash to hide.

## 2. Tasks and config

Tasks: planted lag-1 and planted2 (lag-2), K = 16, length 24,
train 200, test 50, same generator as rounds 15-16
(baselines.make_data / make_data_lag). Seeds 241..260 (20
fresh seeds; round 16 used 221..240, round 15 used
201..220). Config: round-15 winner for both planted tasks
(alpha 1.0, smooth_sigma 0.0, eta_d 0.05), n_out 256,
n_res 200, full_code True, use_confirmation False,
s_inh 0.0, unmodified p1-prototype cosine readout, 5 epochs.
Validation seeds 241..245, test seeds 241..260.

## 3. Validation: pick fb_gain

Grid fb_gain in {0.1, 0.3, 1.0}, planted lag-1 only, val
seeds 241..245, wire-on training, wire-on testing. Winner =
highest mean val normal accuracy. Guard: the winner's val
accuracy must be at least 0.80 x the fb_gain = 0.0 val
accuracy on the same seeds. If no gain passes the guard, the
round STOPS with "the wire destabilizes learning" as the
finding; no test phase runs. The guard is frozen here, not
chosen after seeing test results.

## 4. Test conditions

All trained from scratch per condition, 5 epochs, same
seeds. Conditions 2 and 3 reuse condition 1's trained net
(train once, test three ways), so the test phase is 3 nets
per (task, seed), not 5:

1. wire-on (fb_gain = winner), tested wire-on. Reference.
2. wire-on, tested wire-cut (fb_mode=cut at test, weights
   frozen). The causal test.
3. wire-on, tested wire-shuffled (fb_mode=shuffle at test).
   The content control.
4. wire-off (fb_gain = 0.0), tested wire-off. The round-16
   config retrained on the new seeds: the control that the
   wire changed anything at all.
5. frozen wire-on (no plasticity, wire on). The round-16
   frozen control, with the wire.

## 5. Measures

Per run, the round-16 measures plus the new clamp measures
(all from the same test passes, one JSON line per
(task, condition, seed)):

- normal_acc: standard next-token accuracy, wire as tested.
- omit_m2, omit_m1, omit_filler, cont_m1, margin_m2,
  shuf_m2: the round-16 omission measures (same code path,
  same motif detection by pattern, same frozen weights).
- p1_cos: mean cosine between p1 with the wire on and p1
  with the wire cut, on identical test sequences (paired
  state pass: run each test sequence twice on the same
  trained net, once fb_mode=on, once fb_mode=cut, reset
  between; compare p1 step by step for steps after the
  first). This is the direct "the state actually changes"
  measure.
- err_mag: mean ||p1 - d1|| over training (per epoch
  recorded, final epoch reported). Rao-Ballard predicts the
  top-down prediction explains away input, so the error
  should fall with the wire on relative to wire-off.
- fb_rms_ratio: RMS(fb) / RMS(bottom-up drive W_in @
  e_sparse) over the test passes at the tested mode.
- r1_max: max |r1| over training and test (stability).
- clamp_frac and mean_burst: as in rounds 15-16.

## 6. Bars (lag-1 is the primary task; lag-2 reported)

- C1: wire-on to wire-cut degrades omit_m2 by at least
  0.10 (paired mean over 20 seeds, 95% CI above 0).
- C2: wire-on to wire-shuffled degrades omit_m2 by at
  least 0.10 (paired mean, 95% CI above 0).
- C3: p1_cos mean below 0.90 (the lower module's state
  actually changes when the wire is cut).
- C4: wire-off omit_m2 differs from wire-on omit_m2
  (paired 95% CI excludes 0). The wire shapes learning,
  not just test-time state.
- C5 (guard, not a pass bar): wire-on normal accuracy at
  least 0.80 x wire-off normal accuracy on the same seeds.
  If C5 fails, the wire destabilizes learning; that is the
  finding and the round reports it instead of a pass.

Experiment B passes if C1, C2, C3 and C4 all pass. C1
without C2 means the effect is drive magnitude, not
content. C3 without C1 means the state changes but the
prediction does not matter for the task. C4 without C1
means the wire changes training but the trained network
does not use the prediction at test time. All four are
needed for "the top-down prediction causally shapes the
lower module".

## 7. Smoke test (VM only, before any Beast run)

1 seed (241), 1 epoch, planted lag-1, all five conditions
(conditions 2-3 reuse condition 1's net). Checks:
1. fb_gain=0.0 reproduces the round-16 config exactly
   (same seed, same data: normal accuracy and omit_m2
   match the round-16 driver's output on seed 241 within
   floating-point noise). This proves A26 changes nothing
   when off.
2. Wire-on changes p1: p1_cos (on vs cut) is below 1.0
   and above 0.0 on the smoke seed (the wire does
   something, and does not replace the state).
3. All metrics finite; no NaN; r1_max bounded (below 10);
   clamp_frac reported.
4. The shuffle mode runs without error and its fb RMS
   matches the on-mode fb RMS within 5% (same magnitude,
   different content).

## 8. Execution

Driver: rpc_round17_clamp.py. CLI:
python rpc_round17_clamp.py <task> <val|test> --seeds A-B
--out FILE [--winners FILE] [--epochs N] [--fb-gain G]
Tasks: planted1, planted2. One JSON line per
(task, condition, seed) in the output file; a worker skips
lines already present, so a killed worker resumes cleanly.
Validation: 4 gains (0.0, 0.1, 0.3, 1.0) x 5 seeds x 1 task
= 20 training runs, one worker per gain, one file per gain.
Test: 3 nets x 2 tasks x 20 seeds = 120 training runs
(conditions 2-3 are test-only passes on condition 1's net),
5 output lines per (task, seed), 200 test lines total,
about 4 minutes per training run, about 8 CPU-hours. One BLAS thread
per process (OMP/MKL/OpenBLAS/NumExpr = 1,
torch.set_num_threads(1)), the same fix as rounds 15-16.
The full run waits for Reza's launch decision. The analysis
script is written at r17/r17_analyze.py (following
r16/r16_analyze.py) and smoke-tested on 1-seed driver
output: it reads r17/r17_{task}_val_*.jsonl for the gain
pick (with the guard) and r17/r17_{task}_test_{a,b}.jsonl
for the test phase, skips structurally empty (None)
measures, computes the paired contrasts, checks bars
C1-C5, and writes r17/r17_analysis.md.

## 9. Self-critique (frozen before any test result)

1. The wire feeds the prediction d1 into module 1's
   reservoir, not the error. The error (p1 - d1) still
   only drives W_dist1 learning. Module 1 has no plastic
   weights in this config, so "learning of the lower level
   changes" is measured through W_dist1 and the omission
   measures, not through module-1 weights. If the wire
   matters for module-1 learning, that is a round-18
   question with learn_prox on, not this round's claim.
2. Cut vs shuffle: the shuffle preserves drive magnitude
   but also changes d1's alignment with the prototypes.
   C1 without C2 means magnitude, not content. The
   protocol reports both and the conjunction decides.
3. fb_gain is picked on val normal accuracy, which can
   pick a gain too small to matter. C1 and C2 guard this:
   if the cut does nothing, the wire is decorative at
   that gain and the round reports a fail, not a pass.
4. The omission decode at t still equals the normal
   prediction by construction (round-16 limit carries
   over). The clamp measures use the same omission code
   path, so the same caveat applies to them.
5. Seeds 241..260 are fresh. The validation seeds
   (241..245) are a subset of the test seeds, as in round
   15: the gain is picked on validation accuracy and the
   bars are judged on the full 20-seed test, which
   includes those 5 seeds. This is the same mild overlap
   round 15 accepted; the paired clamp contrasts (C1, C2,
   C4) compare conditions on identical seeds, so the
   overlap cannot manufacture a clamp gap.
6. If the wire-on network learns WORSE than wire-off on
   normal accuracy but the clamp gap is large, the round
   reports both facts: the wire is causal AND costly. A
   causal wire that hurts learning is still a causal wire;
   it is not a failed test, it is a trade-off to report.
