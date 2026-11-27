# Round 16 protocol: the omission test (frozen weights)

Frozen Nov 26, 2026, before any round-16 test result is seen.
Question: does the trained delta network hold a stimulus-specific,
predictability-scaled prediction that survives a missing input?
This is experiment A of the deep-research report
(~/workspace/research_notes/predictive-coding-local-learning-rules-20261127-0039/report.md,
section 6.2): the cleanest single test of whether the delta rule is
predictive coding, rather than signed-error learning on a fixed
random reservoir. Round 15 (section 46) showed delta learns
discrete token tasks; round 16 asks whether it predicts in the
predictive-coding sense.

## 1. Step alignment (verified from rpc_v2.py, not assumed)

Read _step_core (lines 1036-1197), run (lines 1199-1243),
ReservoirModule.forward (lines 487-500), _encode (lines 864-874),
_finalize_prototypes (lines 813-833), predict_from_distal (lines
982-1020). The facts below are code facts, and the design rests
on them.

1. At step t with token x_t, the distal drive d1 is computed FIRST,
   from y2_prev, the module-2 code for token t-1:
   d1(t) = W_dist1 @ y2_prev. The current token is not yet fed in.
   So d1(t) predicts token t from the code of token t-1 only.
2. p1(t) = mod1.forward(x_t) is the module-1 drive for the CURRENT
   token t. The delta update trains d1(t) to match p1(t):
   err = p1 - d1. After the step, y2_prev becomes y2, the code
   for token t, so d1(t+1) = W_dist1 @ y2(t).
3. pred(t) = predict_from_distal(d1(t)) is the argmax cosine of d1
   against the unit prototypes (_proto_norm). run() scores
   pred == tok for positions i > 0, so the readout at step t
   predicts token t.
4. Prototypes are mean p1 per token from TRAIN data only
   (build_prototypes, proto_n = 200). _proto_norm is the unit
   version. center_proto is off, so _proto_mean is zero.
5. Blank input: x = zeros(K) (K = 16). mod1.forward(zeros) gives
   e = 0 and e_sparse = 0, so the reservoir state updates as
   r_new = (1-alpha) * r + alpha * tanh(W_res @ r). With the
   winner alpha = 1.0, r_new = tanh(W_res @ r): the state
   contracts through the recurrent matrix (spectral radius
   rho = 0.9) on each blank step. The state decays, it does not
   reset. mod2 reads only x and r1 (mod2_direct is off), so the
   blank reaches module 2 only through the decayed r1.
6. At the omission step t, d1(t) and pred(t) are computed from
   y2_prev (token t-1's code) BEFORE the blank is fed in, so they
   are UNAFFECTED by the blank. The omission decode at step t is
   therefore identical to the normal prediction at that position.
   The smoke test verifies this directly (section 9).
7. d1 does NOT feed back into the reservoirs. _mod2_input uses x
   and r1 only. There is no error unit and no top-down feedback
   into the state at test time. The omission response in this
   architecture IS the prediction d1 itself, not an error-driven
   response. Consequence: the test measures whether a
   stimulus-specific, predictability-scaled prediction exists as
   an internal state and persists after a missing input. It does
   NOT test error-unit segregation or top-down causality. Those
   are experiments B and C of the research report, separate rounds.
8. At test, plasticity is off, so the blank cannot teach or unteach
   anything. use_confirmation has no effect at test (it lives
   inside the train block). s_inh = 0.0, so inhibition is zero.
   Calibration and prototypes are fit on train data only, exactly
   as in round 15.

## 2. Tasks and splits

1. Planted lag-1: baselines.make_data, K = 16, length 24.
2. Planted lag-2: baselines.make_data_lag (lag = 2), K = 16,
   length 24. Motif tokens sit at positions pos, pos+2, pos+4
   (one random filler between each pair).
3. Train 200 sequences, test 50 sequences. Draw order per seed:
   np.random.seed(s); train 200, val 50, test 50, in that order
   from one stream, identical to round 15 splits(). The val 50
   are drawn and unused: the config is frozen (section 4), so no
   selection happens and no validation split is needed. Drawing
   them keeps the data pipeline identical to round 15.
4. No sine, Lorenz, or var-sine task: the omission test needs a
   discrete motif structure with known predictable and
   unpredictable positions, which only the planted tasks have.

## 3. Motif detection (corrects round 15's lag-2 definitions)

Motifs are (1,2,3), (5,6,7), (9,10,11). Positions are detected by
pattern in the ORIGINAL sequence, before any omission.

1. Lag-1: a motif occurrence is an index i with
   (seq[i], seq[i+1], seq[i+2]) in the motif set. m1 position =
   i+1 (token b, the middle motif element). m2 position = i+2
   (token c, the motif completion). Cue position = i (token a).
2. Lag-2: a motif occurrence is an index i with
   (seq[i], seq[i+2], seq[i+4]) in the motif set. m1 position =
   i+2. m2 position = i+4. Cue position = i.
3. A position that is both an m1 and an m2 position for different
   detected motifs is ambiguous and is dropped from both lists.
   A motif broken by a later overwrite does not match the pattern
   and is excluded, which is correct. A motif pattern that occurs
   by chance in the filler is included, which is correct: it is a
   real pattern in the sequence.
4. Round 15 applied the lag-1 definitions (prev, tok) and
   (prev2, prev, tok) to lag-2 data, where they mostly count
   chance filler patterns. Its lag-2 m1 (0.264) and m2 (0.050)
   are not motif accuracies. The true lag-2 motif accuracies were
   never measured. This driver also reports the normal prediction
   accuracy at the detected m1/m2 positions, which gives the true
   lag-2 motif accuracies as a byproduct.
5. Filler position: the first index i >= 2 that is not a cue, m1,
   or m2 position. The token there is a uniform random draw,
   unpredictable from context. One filler event per sequence.

## 4. Config (frozen from round 15, no selection)

Winner config for both planted tasks (r15/winners.json, verified
Nov 26 20:44 PT): alpha 1.0, smooth_sigma 0.0, eta_d 0.05.
All conditions: n_out = 256, n_res = 200, full_code = True,
use_confirmation = False, s_inh = 0.0, graded_conf = False,
norm_rows = False, w_max = 2.0, 5 epochs, unmodified p1-prototype
cosine readout. The driver loads r15/winners.json when a
--winners path is given and asserts the planted configs equal
the values above; a mismatch fails loudly instead of running
with the wrong config.

Conditions:
1. delta (A24): learn_rule = "delta", eta_d = 0.05.
2. frozen: plasticity = False, same config. No learning; the
   omission decode should sit at chance.

## 5. Omission procedure

For each test sequence, run one pass per omission event. A pass
resets the network, presents the sequence with exactly one
position replaced by a blank (x = zeros(K), fed to
net._step_core(x, train=False, use_confirmation=False)), and
presents every other token normally (net.step(tok, train=False,
use_confirmation=False)). Events per sequence: every detected m1
position, every detected m2 position, and the one filler
position. Also run one normal pass per sequence (no omission)
for the within-experiment reference.

Record per event:
1. At the omission step t: d1(t), pred(t), p1(t), ||d1(t)||,
   ||p1(t)||. The omitted token is seq[t].
2. Continuation (m1 events only): decode the motif completion
   c at the step where it occurs, t+1 for lag-1 and t+2 for
   lag-2 (lag-2 has a filler between b and c). The drive at
   that step is formed from the post-blank state before c is
   fed in, so this is a genuine prediction from the context
   (a, blank, and for lag-2 a filler), not leakage.
3. At t+1 and t+2: d1 and pred, counted only when the target
   position is NOT a motif position, so these are clean
   filler-target controls. A t+1/t+2 target that is another
   motif's element would be predictable and would contaminate
   the control. (For m1 events the continuation position is a
   motif position by construction and is counted in item 2,
   not here.)
4. In the normal pass: pred and d1/p1 norms at every position,
   for the reference accuracies at the same m1/m2/filler
   positions and the norm reference.

## 6. Decode measures (per task, per condition, per seed)

1. omit_acc: fraction of omission events where pred(t) equals the
   omitted token. Reported separately for m1, m2, and filler.
2. margin: mean over events of cos(d1(t), proto[omitted]) minus
   the mean cos(d1(t), proto[k]) over all k != omitted. Cosines
   are against the unit prototypes (_proto_norm). A positive
   margin means d1 points at the omitted token specifically.
3. omit_acc_t1, omit_acc_t2: fraction where pred(t+1) equals
   seq[t+1] and pred(t+2) equals seq[t+2], counted only when
   the target position is not a motif position (section 5
   item 3). These are filler-target controls and should sit
   at chance. Reported separately for m1, m2, and filler
   events.
4. omit_acc_cont: for m1 events only, the fraction where the
   prediction at the continuation step (t+1 for lag-1, t+2
   for lag-2) equals the motif completion c. This is the
   generative-continuation measure.
5. omit_acc_shuf: decode with a fixed per-seed label permutation
   pi (drawn from default_rng(seed + 777), re-rolled until no
   motif token in {2,3,6,7,10,11} is a fixed point). The shuffled
   prediction is pi^{-1}(pred). Reported for m1, m2, and filler
   at step t. It must sit at chance if the signal is in the
   label alignment rather than a generic property of d1.
6. Diagnostics: ||d1|| at t, t+1, t+2 (omission vs normal);
   ||p1|| at t (omission vs normal); cos(p1(t), proto[omitted]);
   err_proj, the signed projection of (p1(t) - d1(t)) onto the
   unit omitted prototype. The error projection tests the
   report's claim that the omission error is negative and
   proportional to the violated prediction. Note it is nearly
   redundant with the d1 cosine, because p1(t) is small at the
   omission step, so err(t) is approximately -d1(t).
7. Normal reference: normal_acc at the same m1/m2/filler
   positions from the normal pass. The omission decode at t
   must equal this reference (section 1 item 6); a mismatch
   means the driver is broken, not that the network learned
   something.
8. Also recorded per run: clamp_frac and mean_burst, as in
   round 15.

## 7. Seeds

Seeds 221..240 (20 fresh seeds). Justification: the winner
config was selected on seeds 201..205 and round 15's test used
201..220. Reusing those seeds for a new test after seeing round
15's results on them would be a mild form of peeking: the
config and the data would both be already observed. Fresh seeds
keep the omission test fully independent of round 15. The data
seed equals the network-init seed (the round-15 convention; the
confound is noted, same as round 15 section 8.1). The driver
computes the normal-prediction reference on the same fresh
seeds, so the omission-vs-normal comparison is
within-experiment.

## 8. Bars (20-seed test; CI = mean +/- 1.96 * std / sqrt(20);
chance = 1/16 = 0.0625)

Expected values from round 15 (lag-1 delta): the omission decode
at t should equal the normal prediction, so omit_acc_m2_t is
expected near 0.956 and omit_acc_m1_t near 0.680. The frozen
decode is expected near 0.06. The filler decode is expected near
chance. The t+1 and t+2 decodes have no prior measurement; no
numeric expectation is set for them, only the direction of the
contrast.

Primary bars, lag-1 (the task where delta is validated):
1. A1, stimulus-specific decode (m2): delta omit_acc_m2_t mean
   >= 0.80, lower 95% CI > 0.50.
2. A2, predictability scaling: delta (omit_acc_m2_t minus
   omit_acc_filler_t), paired per-seed mean > 0.30, 95% CI
   excluding 0.
3. A3, no response before learning: frozen omit_acc_m2_t mean
   < 0.15, frozen margin_m2_t mean < 0.05.
4. A4, label specificity: delta omit_acc_m2_t_shuf mean < 0.15.
5. A5, m1 decode: delta omit_acc_m1_t mean >= 0.50, lower 95%
   CI > 0.25.

Secondary bars, lag-1 persistence (reported, do not gate A):
6. A6, generative continuation: delta omit_acc_cont_m1 (the
   decode of the motif completion c at t+1 after omitting b)
   lower 95% CI > 0.0625.
7. A7, persistence specificity: delta (omit_acc_cont_m1 minus
   omit_acc_t1_filler), paired 95% CI excluding 0. A generic
   echo predicts the same level after every omission type;
   a generative internal model predicts the motif completion
   after an m1 omission but falls to chance after a filler
   omission.

Lag-2 bars (secondary replication, same structure, modest bars):
8. B1: delta omit_acc_m2_t lower 95% CI > 0.0625.
9. B2: delta (omit_acc_m2_t minus omit_acc_filler_t), paired
   95% CI excluding 0.
10. B3: frozen omit_acc_m2_t mean < 0.15.
11. B4, lag-2 continuation: delta omit_acc_cont_m1 (the decode
    of c at t+2 after omitting b, across the intervening filler)
    lower 95% CI > 0.0625. This is a harder test than A6: the
    network must hold the motif position across the blank AND
    a normally presented filler.

Verdict rule: experiment A passes if A1-A5 pass on lag-1. A6,
A7, and the lag-2 bars are reported separately. If A passes,
the network holds a stimulus-specific, predictability-scaled
prediction that survives a missing input. That is NECESSARY for
predictive coding but NOT sufficient for the full Rao-Ballard
claim: this architecture has no error units, no top-down
feedback into the state, and no hierarchy, so experiment B (the
top-down clamp) is still needed before the word "predictive
coding" is claimed. If A fails, the conclusion is: local
signed-error learning on a fixed random reservoir. The network
predicts in the engineering sense but does not implement
predictive coding in the Rao-Ballard sense. That is a
publishable negative result.

## 9. Smoke test (VM only, before any Beast run)

1 seed (221), 1 epoch, planted lag-1, both conditions. Checks:
1. pred(t) in the omission pass equals pred(t) in the normal
   pass at the same position (section 1 item 6).
2. Motif detection finds about 2-3 m1 and 2-3 m2 positions per
   sequence on average.
3. All metrics are finite; no NaN, no exception.
4. The frozen omission decode is reported (noisy on 1 seed, no
   bar applied).
5. The lag-2 detection path runs without error on 1 seed,
   1 epoch (it costs one extra training run in the smoke test).

## 10. Execution

Driver: rpc_round16_omission.py. CLI:
python rpc_round16_omission.py <task> --seeds A-B --out FILE
[--winners FILE] [--epochs N]
Tasks: planted1, planted2. One JSON line per (task, condition,
seed). A worker skips lines already present, so a killed worker
resumes cleanly. Runs: 2 tasks x 2 conditions x 20 seeds = 80
training runs, about 4 minutes each, about 5.5 CPU-hours.
One BLAS thread per process (OMP/MKL/OpenBLAS/NumExpr = 1,
torch.set_num_threads(1)), the same fix as round 15. The full
run waits for Reza's launch decision. The analysis script is
written at r16/r16_analyze.py (following r15/r15_analyze.py)
and smoke-tested on 1-seed driver output: it reads
r16/r16_{task}_test_{a,b}.jsonl, skips structurally empty
(None) measures, computes the paired contrasts, checks bars
A1-A7 and B1-B4, and writes r16/r16_analysis.md.

## 11. Self-critique (design decisions and known limits, frozen)

1. At the omission step t, the decode is identical to the normal
   prediction (section 1 item 6). A1 and A5 at step t therefore
   do not measure an "omission response" generated by the blank;
   they measure the prediction that was already formed. The
   novel part of the test is the motif-vs-filler contrast (A2),
   the frozen control (A3), the shuffled control (A4), and the
   time course (A6, A7). This is stated plainly so the result
   is not over-claimed.
2. Because d1 does not feed back into the reservoirs, the network
   cannot generate an error-driven omission response at test
   time. A pass of A means the prediction exists as an internal
   state; it does not demonstrate error units or top-down
   causality.
3. The t+1 decode has a reservoir-echo confound: y2(t) is
   computed from the decayed reservoir state, so a fading-memory
   trace of the pre-omission context reaches d1(t+1). The
   contrast in A7 separates the two accounts: a generative
   internal model predicts the motif completion after an m1
   omission but falls to chance after an m2 or filler omission,
   while a generic echo predicts the same level after every
   omission type. The p1 diagnostics (section 6 item 6) give the
   same separation from the input side.
4. The t+1 decode after an m1 omission is a genuine prediction,
   not leakage: d1(t+1) is computed from the blank-step code
   y2(t) before token t+1 is fed in. The token c is presented at
   t+1, but it cannot affect d1(t+1).
5. The seed confound (data seed = network-init seed) is carried
   over from round 15.
6. The omission test uses only the planted tasks. It says nothing
   about the sine, Lorenz, or var-sine tasks, where delta's
   status is already settled (sine wins, continuous tasks fail).
7. The err_proj diagnostic is nearly redundant with the d1
   cosine (section 6 item 6). It is kept because the research
   report explicitly calls for the error-signal check, and a
   mismatch between the two would itself be informative.
8. The t+1/t+2 filler-target restriction and the separate
   continuation measure were added after the first smoke test,
   before any full-run result. The first smoke run showed the
   t+1 target after an m2 omission is sometimes another
   motif's element (motifs cover about 9 of 24 positions), so
   an unrestricted t+1 rate mixes predictable and
   unpredictable targets. It also showed the lag-2
   continuation sits at t+2, not t+1, because a filler sits
   between b and c at lag 2. Both fixes are design
   corrections, made before the full run, and are frozen here.
9. Smoke observation (1 seed, 1 epoch, not a result): the
   frozen omit_acc_m2 was 0.112 on lag-1 and 0.160 on lag-2,
   above chance (0.0625) on both. The frozen margin was
   small (0.019-0.022), so the bias is weak, but bars A3 and
   B3 (frozen mean < 0.15) may be tight on lag-2. The bars
   stand as written; the margin bar is the sharper check.
   The frozen condition does no training, so its 1-epoch
   smoke value is representative of the full run on that
   seed up to seed variation.
10. Smoke observation (1 seed, 1 epoch, not a result): the
    frozen continuation was 0.172 on lag-1 and 0.084 on lag-2,
    above chance (0.0625) on both, while the frozen filler
    t+1 was 0.053 and 0.000. Part of the continuation signal
    may therefore be a generic bias (the random readout favors
    motif tokens), not a learned prediction. The analyzer
    reports the delta-minus-frozen continuation contrast
    alongside A7, so the two can be separated: a learned
    continuation must beat the frozen continuation, not just
    the filler control.
