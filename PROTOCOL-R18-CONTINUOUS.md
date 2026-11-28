# Round 18 protocol: the continuous-task wall (scoring correction + readout/input separation)

Frozen Nov 26, 2026, before any round-18 test result is seen.
Design by Muse (design agent) under Reza's loop order (Nov 26
16:38 PST) and standing order (Nov 26 23:03 PST): keep iterating,
do not stop for approval, until the question is completely clear
and killed. Round 17 (top-down clamp) runs on Beast in parallel;
round 18 is built so the loop never stalls.

## 1. Question

Does the delta rule fail on continuous prediction, or did round 15
score it wrong? Round 15 reported a continuous-task wall: Lorenz
delta 0.164 and variable sine delta 0.150, neither beating
persistence or Markov-1 on accuracy or MSE (section 46.4-46.5).
This round tests the hypothesis that the wall is, in whole or in
part, a scoring artifact, and separates the remaining causes
(readout, input, learning target) with a factorial design.

## 2. Hypothesis (one sentence)

Round 15's continuous-task wall is a scoring artifact: run_cont
scores the model's current-token prediction against the next token
(a two-step-ahead task) while all baselines are scored one-step-ahead,
so correctly scored delta beats persistence and Markov-1 on accuracy
on both tasks; whatever MSE wall survives on Lorenz is caused by the
discrete prototype readout and/or the bin-only input, which the ridge
probe and the continuous-bump input separate.

## 3. The scoring finding (verified from code, then empirically)

### 3.1 Code facts

1. In rpc_v2.py `_step_core`, d1 is computed from y2_prev (the
   module-2 code of token t-1) BEFORE the current token is fed in,
   and the delta rule trains d1(t) to match p1(t), the code of the
   current token. So out["pred"] at step t predicts token t, given
   tokens 0..t-1: a one-step-ahead prediction. Section 47 of the
   record states the same alignment ("d1(t) predicts token t from
   the code of token t-1").
2. rpc_v2.py `run()` scores it that way: `if i > 0: correct +=
   int(out["pred"] == tok)`, pred(i) vs token(i). The sine task
   (round 15, 0.868) and both planted tasks use this convention.
3. rpc_round15.py `run_cont` collects p[t] = pred at step t and
   scores it with baselines.py `accuracy`, which compares p[t] to
   seq[t+1]. That pairs a prediction of token t with token t+1:
   from history up to t-1, predict token t+1. A TWO-step-ahead
   task. The same shift applies to its MSE (`mse_from_bin_preds`
   compares decode(pred(t)) to cont[t+1]) and its within-1 metric.
4. All baselines in `compute_baselines` are true one-step-ahead:
   Markov pred[t] is argmax P(next | seq[t]), scored against
   seq[t+1]; ESN state after token t predicts token t+1; persistence
   predicts token t+1 = token t. So round 15 compared RPC's
   two-step-ahead accuracy and MSE against the baselines'
   one-step-ahead accuracy and MSE, on Lorenz and variable sine
   only. The planted and sine tasks were scored correctly.
5. Consequence: the round-15 continuous numbers (Lorenz delta acc
   0.164, MSE 66.1; var-sine delta acc 0.150, MSE 0.172) are
   two-step-ahead numbers. They do not show that delta fails at
   one-step continuous prediction. The wall as stated is
   unproven until re-measured with the correct convention.

### 3.2 Empirical check (diagnostic, 1 seed 261, NOT results)

Script /tmp/r18_scoring_check.py (VM, delta, round-15 winner
config, 5 epochs) scores the SAME test predictions both ways:

1. Lorenz seed 261: convention A (pred(t) vs token t) acc 0.285,
   MSE 35.1, within1 0.580; convention B (run_cont) acc 0.149,
   MSE 74.6, within1 0.368. Baselines on the same test: Markov-1
   0.211 acc / 13.5 MSE, Markov-2 0.539, persistence 0.154 acc /
   13.6 MSE, ESN 0.362 acc / 5.3 MSE. Under A, delta beats
   Markov-1 and persistence on accuracy, but its MSE (35.1) is
   still far worse than persistence (13.6).
2. Variable sine seed 261: convention A acc 0.319, MSE 0.042,
   within1 0.636; convention B acc 0.146, MSE 0.142, within1
   0.309. Baselines: Markov-1 0.279 acc / 0.133 MSE, Markov-2
   0.519, persistence 0.125 acc / 0.057 MSE, ESN 0.364 acc /
   0.048 MSE. Under A, delta beats Markov-1 and persistence on
   accuracy AND beats persistence on MSE (0.042 vs 0.057).
3. Sine control seed 261: convention A 0.864 (matches round 15's
   0.868 via run()), convention B 0.124. The same predictions
   score 0.74 apart under the two conventions. This is the
   decisive control: the conventions differ, and B is the wrong
   one for this model.

Expectation (not a bar): the var-sine wall dissolves under
correct scoring; the Lorenz accuracy wall dissolves but the
Lorenz MSE wall survives. The 20-seed run decides; the 1-seed
pilot is not a result.

## 4. The three candidate causes, weighed

After the scoring correction, three causes can still explain a
surviving Lorenz MSE wall. Each is a distinct, testable claim
about where continuous information is lost:

1. (a) Readout: `predict_from_distal` returns argmax cosine
   against 32 mean-p1 prototypes, one per bin. A wrong argmax can
   jump to a distant bin, so a single error costs many bins of
   squared error, while persistence is wrong by only the
   step-to-step movement. The readout cannot interpolate between
   bins. Test: a continuous ridge readout from the same trained
   d1 (condition C2). If the ridge MSE beats the prototype MSE,
   the information is in the drive and the discrete readout is
   the bottleneck.
2. (b) Input: `_encode` maps a token to a one-hot (or a Gaussian
   bump centred on the integer bin). The input carries no
   within-bin value. Test: assumption A27, a Gaussian bump
   centred on the TRUE continuous value, fed through `step_vec`
   (condition C3). If the bump-input MSE beats the one-hot MSE,
   within-bin input information is the bottleneck.
3. (c) Learning target: the delta rule trains d1 to match p1,
   and with a one-hot input p1 is a function of the bin code
   only, so the drive cannot carry within-bin value except
   through history. Test: by elimination. If C2 (ridge) and C3
   (bump) and C4 (both) all fail to fix the Lorenz MSE while C1
   accuracy is good, the discrete target is implicated: no
   readout or input change can put within-bin value into a drive
   trained to match a bin code. The next round would then need a
   continuous target (a new assumption, not this round).

Causes (a) and (b) are separable by the factorial design below
(C2 vs C3 vs C4). Cause (c) is tested only by elimination, and
the protocol says so plainly (see self-critique 3).

## 5. Tasks and config

Tasks: Lorenz and variable sine only (the two wall tasks).
Lorenz: 200/50/50, K=32, length 24, baselines.make_lorenz
(dt 0.01, stride 10, effective dt 0.10, range [-25, 25]).
Variable sine: 100/20/20, K=32, length 100,
rpc_round15.make_var_sine_cont (frequency uniform 0.02..0.08
per sequence; continuous values kept for MSE).
Seeds 262..281 (20 fresh seeds). Seed 261 is excluded: it was
used in the section 3.2 diagnostic. Val seeds 262..266.
Round-15 winners frozen (r15/winners.json): Lorenz alpha 0.7 /
smooth_sigma 0.0 / eta_d 0.2; var-sine alpha 1.0 /
smooth_sigma 2.0 / eta_d 0.02. n_out=256, n_res=200,
full_code=True, use_confirmation=False, s_inh=0.0,
norm_rows=False, 5 epochs, unmodified p1-prototype cosine
readout for the prototype conditions. The one-hot winner config
is NOT re-validated under the correct scoring: re-validating
would change two things at once (scoring + config) and H0
(the scoring claim) would be confounded. A later round can
re-validate; this round tests the scoring claim with the
config held fixed.

## 6. Validation: pick the bump sigma (A27)

Grid bump_sigma in {1.0, 2.0, 3.0}, per task, val seeds
262..266, delta rule, bump input, prototype readout,
convention-A MSE as the selection metric. Winner = lowest mean
val MSE per task. The bump sigma is a NEW dimension (the
continuous-bump input is new), so it needs a val pick; the
one-hot config is already frozen (section 5). Tie-break: first
in grid order (1.0, 2.0, 3.0). The winner is frozen per task
and used for all bump test conditions on that task.

## 7. Test conditions

Four nets trained from scratch per (task, seed), 5 epochs each.
Every model line reports acc, within1, and mse under convention
A (pred(t) vs token t; MSE decode(pred(t)) vs cont[t]), plus
mean_burst and clamp_frac. The ridge lines report the ridge
readout's acc/within1/mse under the same convention A.

1. N1 delta, one-hot input, winner config. Yields C1 (prototype
   readout) and C2 (ridge readout from N1's d1).
2. N2 delta, bump input (winner bump sigma), winner alpha/eta_d.
   Yields C3 (prototype readout) and C4 (ridge readout from
   N2's d1).
3. N3 frozen, one-hot input, winner config. Yields C5 (prototype
   readout) and C2f (ridge readout from N3's d1).
4. N4 frozen, bump input, winner bump sigma. Yields C6
   (prototype readout) and C4f (ridge readout from N4's d1).
5. Baseline line per (task, seed): Markov-1/2, persistence, ESN
   acc + MSE + within1 on the identical test sequences
   (rpc_round15.compute_baselines).

Conditions and their codes in the output files:
- C1_onehot_delta_proto, C2_onehot_delta_ridge,
  C3_bump_delta_proto, C4_bump_delta_ridge,
  C5_onehot_frozen_proto, C6_bump_frozen_proto,
  C2f_onehot_frozen_ridge, C4f_bump_frozen_ridge, baseline.

The ridge probe (C2, C4, C2f, C4f): after training, run the
TRAIN sequences through the frozen net, collect (d1(t), v(t))
pairs for t >= 1, fit a centred ridge regression (lambda 1.0,
as in the ESN baseline) mapping d1 to the continuous next
value v(t). At test, predict v_hat(t) from d1(t) and score MSE
against v(t) and bin accuracy of bin(v_hat(t)) against bin(v(t)).
The ridge is fit on train data only, never on test. It is a
DIAGNOSTIC PROBE, not a model claim: a C2 pass shows the
information is in the drive, not that the fixed prototype
readout can do continuous prediction. A theory-faithful
continuous readout is a later round, not this one.

## 8. Metrics (per task, per condition, per seed)

- All model lines: exact-bin accuracy (convention A),
  within-1-bin accuracy (convention A), continuous MSE
  (convention A: decoded value vs true continuous value at
  the same step), mean_burst, clamp_frac.
- Baseline line: markov1_acc, markov2_acc, persist_acc,
  esn_acc, markov1_mse, markov2_mse, persist_mse, esn_mse,
  esn_within1 (all one-step-ahead, as in round 15).
- Val lines: val_mse and val_acc (convention A, prototype
  readout, bump input) per (task, bump sigma, seed).

## 9. Bars (20-seed held-out test, paired per-seed differences)

Bars are judged on paired per-seed differences (mean paired
difference with 95% CI = mean +/- 1.96 * std / sqrt(20)).
"Beats X" means the paired mean difference is on the right
side of zero with the 95% CI excluding zero. Accuracy bars
use exact-bin accuracy (convention A); MSE bars use
continuous MSE (convention A).

1. D1 (H0, Lorenz accuracy): C1 acc beats persistence AND
   Markov-1 (both paired comparisons, CIs exclude 0).
2. D2 (H0, var-sine accuracy): C1 acc beats persistence AND
   Markov-1 (both paired comparisons, CIs exclude 0).
3. D3 (H0, var-sine MSE): C1 mse beats persistence mse
   (paired, CI excludes 0).
4. D4 (residual Lorenz MSE): C1 mse beats persistence mse
   (paired, CI excludes 0). Expected to FAIL on the pilot
   evidence (seed 261: 35.1 vs 13.6); this bar tests whether
   the Lorenz MSE wall survives correct scoring.
5. D5 (readout, Lorenz): C2 mse < C1 mse (paired, CI
   excludes 0).
6. D6 (input, Lorenz): C3 mse < C1 mse (paired, CI
   excludes 0).
7. D7 (interaction, Lorenz): C4 mse < C2 mse AND C4 mse <
   C3 mse (both paired, CIs exclude 0). Judged only if D5
   or D6 passes.
8. D8 (learning adds continuous information, Lorenz): C2 mse
   < C2f mse (paired, CI excludes 0).

Verdict rule. H0 passes if D1, D2, D3 pass: the wall was
scoring, except possibly Lorenz MSE. If D4 also passes, the
wall was entirely scoring. If D4 fails, the Lorenz MSE wall
is real, and the residual cause is: "readout" if D5 passes
and D6 fails; "input" if D6 passes and D5 fails;
"both/interacting" if D7 passes; "target (c) by elimination"
if D1 passes but D5, D6, D7 all fail (accuracy good, MSE bad,
and neither readout nor input fixes it). D8 distinguishes
whether delta adds any continuous information at all: if C2
matches C2f, the reservoir alone carries it and delta adds
nothing continuous. The D5-D8 analogues on var-sine are
reported as secondary (no bars): the var-sine wall is
expected to dissolve under H0, so the separation bars are
primary on Lorenz only.

## 10. Smoke test (VM only, before any Beast run)

1 seed (262), 1 epoch, both tasks, all conditions. Checks:
1. The bump at a bin centre equals _encode with the same
   sigma for that bin (exact, to floating-point tolerance):
   the continuous bump formula reduces to the smooth input.
2. cont_bin reproduces the generator's bins exactly on the
   splits data for both tasks (compare against the bins from
   rpc_round15.splits).
3. Convention A accuracy exceeds convention B accuracy on the
   smoke seed for both tasks, and convention A MSE is below
   convention B MSE: the scoring gap is real and in the
   expected direction (same check as section 3.2, on a fresh
   seed).
4. The ridge fit runs and produces finite MSE on train and
   test for every net; the in-sample train ridge MSE is below
   the prototype MSE on train (the ridge machinery works).
5. All metrics finite; no NaN; clamp_frac reported; the bump
   net's p1 differs from the one-hot net's p1 on identical
   inputs (the input actually changes the code).
6. The driver CLI path works with --seeds 262-262 --epochs 1
   and the output is valid JSONL with resumable keys (run
   twice, the second run writes nothing new).
7. The analyzer runs on the 1-seed output and writes the
   analysis file (bars will mostly FAIL on 1 seed; the smoke
   checks that it runs, not the bars).

## 11. Execution

Driver: rpc_round18_continuous.py. CLI:
python rpc_round18_continuous.py <task> <val|test> --seeds A-B
--out FILE [--winners FILE] [--epochs N]
Tasks: lorenz, varsine. One JSON line per (task, condition,
seed) in the output file; a worker skips lines already present,
so a killed worker resumes cleanly. Validation: 2 tasks x 3
bump sigmas x 5 seeds = 30 training runs, one worker per
(task, sigma), one file per (task, sigma). Test: 4 nets x 2
tasks x 20 seeds = 160 training runs, 9 lines per (task, seed)
(8 model + 1 baseline), 360 test lines total, about 1-2 min
per training run on one CPU, about 3-5 CPU-hours. One BLAS
thread per process (OMP/MKL/OpenBLAS/NumExpr = 1,
torch.set_num_threads(1)), the same fix as rounds 15-17.
The full run waits for the launch decision. The analysis
script is written at r18/r18_analyze.py (following
r16/r16_analyze.py) and smoke-tested on 1-seed driver output:
it reads r18/r18_{task}_test_{a,b}.jsonl, computes means,
medians, IQRs, 95% CIs and paired differences, checks bars
D1-D8, and writes r18/r18_analysis.md. The monitor is written
at r18/monitor.py (following r17/monitor.py): val -> pick bump
winner per task (lowest mean val MSE) -> write winner.json ->
launch test -> analysis at 360 lines. Idempotent via marker
files; exits 0 always.

## 12. Self-critique (frozen before any test result)

1. The scoring correction is the primary finding of this round's
   design, not a result yet. The 1-seed pilot (261) is NOT a
   result; the 20-seed run decides. If the 20-seed run under
   convention A reproduces round 15's failure, the pilot was
   noise and the wall stands as scored.
2. The ridge readout is a diagnostic probe, not a model claim.
   It is fit on train data only, but it IS a trained readout
   (ridge regression), so a C2 pass does not show the fixed
   prototype readout can do continuous prediction; it shows the
   information is in the drive. A theory-faithful continuous
   readout is a later round.
3. The bump input changes what p1 contains, not just the input:
   the delta target p1 is now a function of the continuous value,
   so C3 tests input and target-content jointly, not input alone.
   A pure input-vs-target separation would need a fixed discrete
   target with a continuous input, which this architecture cannot
   express (the target IS p1). The protocol reports this honestly:
   C3 tests (b) as realized through the target, and (c) is
   implicated only by elimination if C2, C3, C4 all fail.
4. Bump sigma is validated on val MSE over 5 seeds, a small
   validation set; the winner may be noisy. The test bars are
   paired comparisons, so a suboptimal bump sigma weakens C3/C4
   but cannot manufacture a pass.
5. Seeds 262..281 are fresh. Seed 261 is excluded (used in the
   section 3.2 diagnostic). The diagnostic used the round-15
   winner config on seed 261 only.
6. The one-hot winner config is frozen from round 15 (picked
   under convention-B scoring). This keeps H0 clean (same config,
   only scoring changes), but the config may be suboptimal under
   convention A. A later round can re-validate under convention
   A; this round tests the scoring claim, not the config.
7. Lorenz MSE is dominated by large errors when the argmax jumps
   to a distant bin; the ridge and bump conditions address this,
   but a chaotic system also has an intrinsic predictability
   limit (the 1-step error is not zero even for a perfect model
   of the quantised dynamics). The bars compare against
   persistence and Markov-1, not against zero, so this limit
   cannot manufacture a pass or a fail by itself.
8. The frozen ridge controls (C2f, C4f) use the same ridge
   procedure on frozen nets. If the frozen ridge matches the
   trained ridge, delta adds no continuous information and the
   reservoir alone carries it (an ESN-like result, not a delta
   result). D8 is the bar for this.
9. The val phase picks bump sigma on val MSE, but the test bars
   D5-D7 are paired MSE comparisons on test. A bump sigma that
   wins on val MSE by a small margin may not be the best for the
   paired test comparisons; that is the standard validation
   compromise, and the protocol does not re-pick on test.
