# PROTOCOL-R21-CONTINUOUS.md — Round 21: continuous re-validation

Frozen Nov 27, 2026. Protocol for the continuous re-validation of
the random predictive coding (RPC) model: Convention-A
re-validation of Lorenz and var-sine, plus NARMA-10 and
Mackey-Glass as discriminating continuous tasks, with
column-standardized ridge probes, a local parameter-free
continuous readout, and the aligned wire tested as a factor.

Related records: `simulation-results.md` sections 52 (round 18
verdict), 57 (round-17/18 critiques), 58-59 (round-20 design and
verdict). Round-18 protocol: `PROTOCOL-R18-CONTINUOUS.md`.
Round-20 protocol: `PROTOCOL-R20-ALIGNED.md` (bar style and CI
convention followed here). Driver:
`rpc_round21_continuous.py`. Analyzer: `r21/r21_analyze.py`.
Monitor: `r21/monitor.py`. Smoke test:
`rpc_round21_smoke.py`.

## 1. Question and hypothesis

Round 18 concluded that a "continuous wall" survives correct
scoring: the delta-rule model could not beat persistence on
Lorenz or var-sine with the frozen round-15 configuration
(Convention B). Section 57 showed that conclusion rested on a
hyperparameter carryover: with Convention-A hyperparameters
from round 15's own grid, the prototype readout beats
persistence on both tasks (Lorenz MSE 5.07 vs persistence 13.05,
reproduced exactly; var-sine MSE 0.0370 vs persistence 0.0548,
reproduced exactly with the round-21 driver in the smoke test).
Round 21 asks whether that finding survives valid validation on
fresh seeds, and whether it extends to continuous tasks that
need multi-step memory and cannot be solved by a 2-bin lookup.

Hypothesis H21: the signed-error (delta) rule learns
continuous trajectories. It predicts (a) the Convention-A
prototype readout beats persistence on Lorenz and var-sine on
fresh seeds, (b) the win is due to learning (delta beats the
frozen twin on the same net), (c) the same holds on NARMA-10
and Mackey-Glass, including beating Markov-2, and (d) a local
parameter-free continuous readout (the prototype centroid)
beats the hard prototype argmax. The null for each bar is that
the paired per-seed difference is zero or favours the control.

What round 21 does NOT test: predictive coding itself. Round
20 failed the revised clamp (B') and the generative rollout
(D), so even a complete wall-fall here establishes only that
the signed-error rule learns continuous trajectories, not
that the model implements predictive coding. The verdict
section states this explicitly.

## 2. Background: what the design rests on

1. Round 15 validated delta as a learning rule for discrete
   token tasks under the fixed p1-prototype cosine readout
   (sine 0.868, planted lag-2 0.222). The trained-readout ESN
   still beats delta on every task.
2. Round 18 scored Lorenz/var-sine correctly (Convention A:
   pred(t) predicts the value at step t) but carried over the
   round-15 Convention-B configuration (Lorenz alpha 0.7,
   smooth 0.0, eta_d 0.2; var-sine alpha 1.0, smooth 2.0,
   eta_d 0.02) and used an unstandardized fixed-lambda ridge.
   Its D3/D4 fails are a hyperparameter artifact, not a wall:
   the Convention-A configs beat persistence (section 57).
3. The D8 frozen-ridge win is a regularization artifact: on
   Lorenz seed 262 the trained drive norm is 1.025 vs 5.651
   frozen (5.52x); the raw lam=1.0 ridge gives 14.904 trained
   vs 14.073 frozen, while the column-standardized ridge ties
   at 13.976. Never an unstandardized fixed-lambda ridge
   again: every ridge in round 21 is column-standardized,
   with the raw ridge reported only for backward
   comparability.
4. Markov-2 trivially solves Lorenz and var-sine: 99.50% /
   99.82% of test 2-grams appear in training, and two bins
   explain R squared 0.989 / 0.995 of next-step variance.
   A discriminating task needs multi-step memory or noise:
   NARMA-10 and Mackey-Glass (tau 17) are the reviewers'
   recommendation.
5. Round 20: the aligned subtractive wire (A29) is causal but
   small on lag-1 omission (B1 +0.014, B2 +0.033, both CIs
   above 0 but below the +0.05 floor), has no explaining-away
   signature (B5 wrong sign on both lags), and the generative
   rollout fails with the wrong sign (on 0.147 < cut 0.184 <
   off 0.223). The wire costs 23 points of one-step sine
   accuracy (0.666 vs 0.892 wire-off). The learned wire fails
   everywhere. Thesis-level predictive-coding causality is NOT
   established.

## 3. Tasks and data

Four continuous tasks, all K = 32 bins, all scored by
Convention-A one-step-ahead prediction (pred(t) vs the value
at step t, t >= 1; L-1 transitions per sequence).

1. Lorenz x (`lorenz`): `baselines.make_lorenz`, RK4, dt 0.01,
   sigma 10, rho 28, beta 8/3, length 24, bins over [-25, 25].
   Splits 200 train / 50 val / 50 test sequences per seed
   (exactly `rpc_round15.splits`).
2. Variable-frequency sine (`varsine`):
   `rpc_round15.make_var_sine_cont`, length 100, bins over
   [-1, 1]. Splits 100 / 20 / 20 (exactly
   `rpc_round15.splits`).
3. NARMA-10 (`narma10`): NEW, `baselines.make_narma10`.
   u(t) iid Uniform(0, 0.5); y(t+1) = 0.3*y(t) +
   0.05*y(t)*sum_{i=0}^{9} y(t-i) + 1.5*u(t-9)*u(t) + 0.1
   (Atiya and Parlos 2000). Length 100, bins over [0, 1].
   Splits 100 / 20 / 20, same draw order as lorenz
   (global RNG, train then val then test).
4. Mackey-Glass (`mackeyglass`): NEW,
   `baselines.make_mackeyglass`. Discrete Euler map with
   delay D = 17 steps: x(t+1) = x(t) +
   0.2*x(t-17)/(1+x(t-17)^10) - 0.1*x(t). This is the standard
   reservoir-computing benchmark form, NOT the continuous
   delay-differential equation with tau = 17 time units
   sampled at dt = 0.1 (that form has a 170-step delay).
   Length 100, bins over [0, 1.5]. Splits 100 / 20 / 20.

NARMA-10 stability (found in the smoke test, Nov 27): the
recurrence has an unstable fixed point near y = 1.02
(mean-field roots 0.38 and 1.02). A run of large inputs can
push y past it, after which the quadratic term explodes:
measured about 1 in 1000 sequences diverges (seed 329,
sequence 46 went to inf during burn-in). A divergent draw is
not a NARMA-10 trajectory, so the generator resamples it;
the retained sequences are the canonical system conditioned
on staying in its stable basin, which is where every
reported literature run lives. Draws leaving the bin range
are also resampled, so returned values never clip at the bin
edges (measured 10-seed range [0.171, 0.974] over 1400
sequences, 27/32 bins hit). Mackey-Glass is globally bounded
(the 0.2*x/(1+x^10) term vanishes for large x); measured
10-seed range [0.368, 1.345], 22/32 bins hit.

## 4. Model, readouts, and the wire factor

Network: `RPCNetwork`, K = 32, n_res 200, n_out 256 (full
code), theta_p = theta_d = 1.0, eta_b = eta_c = 0.05,
learn_rule delta, eta_d from the config, full_code True,
use_confirmation False, s_inh 0.0, graded_conf False,
norm_rows False, proto_n 200 (lorenz) / 100 (all other
tasks). Training: 5 epochs over the train split, step()
for token input, step_vec() for bump input. One BLAS thread
per process in every driver (OMP/MKL/OpenBLAS/NumExpr = 1,
torch.set_num_threads(1)).

Readouts, all three computed from the SAME run (same net,
same seed, same training), so the comparisons are exactly
matched:

1. Prototype: hard argmax over the p1 prototypes by cosine,
   the round-15 readout. Primary readout; the W/N bars use
   it.
2. Centroid: softmax-weighted prototype centroid, v_hat =
   sum_k softmax(beta * cos(d1, mu_k)) * centre_k, with
   centre_k the bin centre. Beta = 10 is the primary value,
   fixed in advance; beta = 5 and 20 are reported from the
   same collected similarities as a sensitivity check (no
   extra runs). The centroid adds no fitted weights beyond
   the prototypes (which the prototype readout already uses)
   and no new learned parameters; beta is a fixed constant.
   It tests whether the "wall" was partly a hard-argmax
   artifact: a continuous interpolation over the same
   prototypes.
3. Ridge: ridge regression d1 -> continuous value, lam = 1.0,
   fit on TRAIN d1 only (collected on a frozen pass after
   training, so the ridge sees the final W_dist1).
   COLUMN-STANDARDIZED: each d1 column is centred and scaled
   by its train mean/std (ddof=0; columns with std < 1e-12
   get sd = 1e-12). The unstandardized (raw) ridge is also
   computed and reported, but no bar uses it.

Frozen control: a TWIN net per arm, same seed, same init,
same calibration and prototypes, never trained. Building it
costs two frozen collection passes instead of a second
5-epoch run, and the comparison is exactly matched by
construction. The smoke test verifies the twin equals a
standalone frozen invocation exactly (proto MSE to 12
decimals).

Baselines (per task, per seed, on the identical test
sequences): Markov-1, Markov-2 (counts from train only,
add-alpha 0.1, order-2 backs off to order-1), persistence
(repeat the last value), train-mean (constant), ESN
class-argmax (`baselines.ESN`, 300 units, RandomState
2000+seed), and ESN scalar ridge (the same ESN reservoir
with a continuous ridge readout: fit on train states after
the washout with target cont[t+1], applied to every test
transition; both raw and column-standardized versions).
The ESN scalar ridge is the strongest comparator: AGY
measured Lorenz 3.287 (one-hot) / 2.097 (bump) and
var-sine 0.0191 / 0.0072 with the raw ridge.

## 5. The wire decision (the round-20 open question, resolved)

DECISION: test the aligned wire on/off as a factor. Wire OFF
is the primary configuration; wire ON (aligned, gain 1.0) is
a secondary arm (`token_w`) on the token input only.

Reasons, in order of weight:

1. Round 20 measured a 23-point one-step sine cost for the
   aligned wire at gain 1.0 (0.666 vs 0.892 wire-off). One-step
   MSE is exactly what the continuous bars measure; carrying
   the wire as primary would confound "does delta learn
   continuous trajectories" with "does the wire tax one-step
   prediction".
2. Round 20's D verdict: the wire makes generative rollout
   WORSE, with the wrong sign (on 0.147 < cut 0.184 < off
   0.223). Nothing in round 20 suggests the wire helps any
   forward-prediction quantity.
3. The wire's only established causal effect is a small
   omission effect on lag-1 token tasks (B1/B2 below the
   +0.05 floor, no explaining-away). The burden of proof is
   on the wire to show it helps continuous prediction.
4. Testing on/off as a factor answers the question cleanly
   with one extra arm per task: if wire-on MSE beats
   wire-off MSE on any task (paired, exact-t CI above 0),
   that is new evidence FOR the wire in the continuous
   domain; if it loses, the wire's cost generalizes beyond
   tokens.

The wire-on arm uses the token input only, not the bump
input: a wire-by-bump interaction is out of scope for this
round (stated limit, section 10). The wire is the ONLY
difference between the token and token_w arms (same wire
type "aligned", gain 0.0 vs 1.0; gain 0 sets fb=None in
_step_core). The wire-live gate (bar F2) requires mean
fb_drive_ratio > 0.05 on the wire-on arm; the smoke test
measures 0.2982 on lorenz seed 322 (fb_rms_ratio 0.4273),
so the gate sits 6x below the measured value. Single-seed
calibration (same seed/config): wire-on proto MSE 3.7809
vs wire-off 6.6547, the wire HELPED one-step prediction
here, against round 20's token-task cost. One seed proves
nothing; that is why F1 is a three-way measurement over
20 seeds rather than a pass/fail bar.

## 6. Conditions, validation, and test arms

Preregistered primaries (the Convention-A configs):

1. Lorenz: alpha 1.0, smooth_sigma 2.0, eta_d 0.2.
   Reproduced exactly on seeds 262-266 with the correct
   scorer (acc 0.4285, MSE 5.073, per-seed
   [5.68, 4.62, 3.75, 5.83, 5.48]).
2. Var-sine: alpha 0.7, smooth_sigma 2.0, eta_d 0.2.
   AGY reviewer-reported (MSE 0.0370, acc 0.3230, per-seed
   [0.0423, 0.0368, 0.0371, 0.0207, 0.0479]) and now
   REPRODUCED EXACTLY with the round-21 driver on the same
   seeds 262-266 (smoke test: mean MSE 0.0370, acc 0.3230,
   per-seed [0.0423, 0.0368, 0.0371, 0.0207, 0.0479],
   diff -0.0000 vs AGY). The var-sine primary is no longer
   reviewer-reported-only.
3. NARMA-10 / Mackey-Glass: no prior; the primary is the
   val winner (procedure-preregistered, not a config).

Val token grid (5 configs x 5 val seeds 322-326, per task;
selection metric: mean val prototype MSE):

1. G1: alpha 1.0, smooth 2.0, eta_d 0.2 (lorenz primary).
2. G2: alpha 0.7, smooth 2.0, eta_d 0.2 (varsine primary).
3. G3: alpha 0.7, smooth 3.0, eta_d 0.2 (AGY's best var-sine
   variant, MSE 0.0317).
4. G4: alpha 1.0, smooth 3.0, eta_d 0.2 (symmetric neighbour).
5. G5: alpha 1.0, smooth 2.0, eta_d 0.02 (Convention-B
   carryover reference: what round 18 actually ran).

Winner: lowest mean val proto MSE; tie-break smallest
eta_d, then smallest smooth_sigma, then smallest alpha
(deterministic, stated in advance). Val uses the prototype
readout only; the centroid/ridge are test-only diagnostics.

Val bump grid (2 sigmas x 5 val seeds, per task; alpha/eta_d
from the token winner): bump_sigma 2.0 / 3.0, lowest mean
val proto MSE wins, tie-break smaller sigma. The bump arm
is secondary; its sigma is picked on 2 values only (a
stated limit: if the bump arm beats the token arm anywhere,
a follow-up round re-validates its sigma on a wider grid).

Test arms (20 test seeds 327-346, per task; each arm emits
a delta row and a frozen-twin row):

1. `token`: winner config, wire off. Primary arm.
2. `token_w`: winner config, wire on (aligned, gain 1.0).
3. `bump`: winner alpha/eta_d + picked bump_sigma, wire off.
4. `token_primary` (CONDITIONAL): the preregistered primary,
   wire off, only when the val winner differs from it
   (lorenz / varsine only; the new tasks have no separate
   primary). The primary verdict stands even if the winner
   differs; the winner is reported as a secondary.
5. `baseline`: Markov-1/2, persistence, train-mean, ESN
   argmax, ESN scalar ridge (raw + standardized), once per
   (task, seed).

Rows per (task, seed): 6 arm rows + 1 baseline = 7, plus 2
conditional primary rows when they apply. Total test lines:
560 base, up to 600 with both conditionals. Val lines: 100
token + 40 bump = 140. Round total: 700-740 lines.

## 7. Bars

All test bars use paired per-seed differences over the 20
test seeds with EXACT Student-t 95% CIs (t(19) = 2.0930,
correcting the round-17/18 z = 1.96 convention). A bar
passes when the paired difference favours the first-named
side with its lower 95% CI above 0, unless the bar says
otherwise. "MSE" is the prototype MSE unless a readout is
named. n = 20 for every bar; a bar with fewer paired seeds
fails loudly in the analyzer (it reports the count).

W bars: Convention-A re-validation (lorenz, varsine).

1. W1: lorenz token_delta proto MSE < persistence MSE.
2. W2: varsine token_delta proto MSE < persistence MSE.
3. W3: lorenz token_delta proto MSE < ESN argmax MSE.
4. W4: varsine token_delta proto MSE < ESN argmax MSE.
5. W5: lorenz token_delta proto acc > Markov-1 acc.
6. W6: varsine token_delta proto acc > Markov-1 acc.
7. W7: lorenz token_delta proto MSE < token_frozen proto MSE.
8. W8: varsine token_delta proto MSE < token_frozen proto MSE.
9. W9: lorenz token_primary_delta proto MSE < persistence
   MSE (n/a when the winner == primary: no separate arm).
10. W10: varsine token_primary_delta proto MSE <
    persistence MSE (n/a when the winner == primary).

N bars: new discriminating tasks (narma10, mackeyglass).

1. N1: narma10 token_delta proto MSE < persistence MSE.
2. N2: mackeyglass token_delta proto MSE < persistence MSE.
3. N3: narma10 token_delta proto MSE < ESN argmax MSE.
4. N4: mackeyglass token_delta proto MSE < ESN argmax MSE.
5. N5: narma10 token_delta proto MSE < token_frozen proto MSE.
6. N6: mackeyglass token_delta proto MSE < token_frozen
   proto MSE.
7. N7: narma10 token_delta proto MSE < Markov-2 MSE.
8. N8: mackeyglass token_delta proto MSE < Markov-2 MSE.

R bars: standardized ridge, all four tasks (T in
{lorenz, varsine, narma10, mackeyglass}).

1. R1 (per task): token_delta ridge_std MSE <
   token_frozen ridge_std MSE. (Delta adds continuous
   information beyond the frozen reservoir, under matched
   standardization. This is the clean D8 answer.)
2. R3 (per task): token_delta ridge_std MSE < persistence
   MSE.
3. R4 (per task): token_delta ridge_std MSE < ESN
   scalar-std ridge MSE. (The fair reservoir-vs-reservoir
   comparison AGY demanded.)

C bars: centroid, all four tasks.

1. C1 (per task): token_delta centroid (beta=10) MSE <
   token_delta proto MSE. (The local parameter-free readout
   improves on hard argmax.)
2. C2 (per task): token_delta centroid (beta=10) MSE <
   persistence MSE.

Beta sensitivity (reported, not a bar): centroid MSE at
beta 5 / 10 / 20. If the best beta differs from beta=10 by
more than 20% relative MSE, the verdict flags
"beta-sensitive" and C1 is downgraded to a readout
observation, not a win for the interpolation idea.

F bars: wire factor, all four tasks.

1. F1 (per task): wire effect on proto MSE, measured as the
   paired difference (off - on) with its exact-t CI.
   Three-way, not pass/fail: HELPS if the CI is above 0,
   HURTS if below 0, NULL if it covers 0.
2. F2 (per task): wire-live GATE: mean fb_drive_ratio on
   the token_w_delta rows > 0.05. If the gate fails, the
   F1 measurement for that task is void (the arm did not
   carry a live wire) and the analyzer says so.

B bars: bump input, all four tasks.

1. B1 (per task): bump_delta proto MSE < token_delta proto
   MSE. (Continuous bump input beats discrete token input.)
2. B2 (per task): bump_delta proto MSE < persistence MSE.

## 8. Verdict rules

1. The continuous wall (round 18's claim, narrowed by
   section 57) FALLS for the re-validation tasks if W1 AND
   W2 AND W7 AND W8 pass: the Convention-A prototype
   readout beats persistence on fresh seeds AND the win is
   due to learning (delta beats frozen).
2. The wall is PARTLY DOWN if W1/W2 pass but W7/W8 fail:
   the readout beats persistence, but only because the
   frozen reservoir already does; the credit goes to the
   reservoir, not to learning.
3. The wall SURVIVES if W1 or W2 fails: the Convention-A
   prototype readout does not beat persistence on fresh
   seeds.
4. The new tasks DISCRIMINATE if N7 AND N8 pass (RPC beats
   Markov-2 on both): NARMA-10 and Mackey-Glass are genuine
   multi-step-memory tasks and RPC competes on them. If only
   one of N7/N8 passes, the verdict names which task (beats
   Markov-2 on one task only). If N1/N2 pass but N7/N8 fail,
   the tasks need memory but a 2-gram lookup still wins
   (partial discrimination). If only one of N1/N2 passes, the
   verdict names which task (mixed).
5. Thesis-level: even if the wall falls, round 21 does NOT
   establish predictive coding. Round 20 failed B' and D;
   a wall-fall here establishes only that the signed-error
   rule learns continuous trajectories.

What would change the verdict:

1. If W1/W2 fail but the val winner passes its own
   persistence bar (W9/W10 logic applied to the winner):
   the Convention-A config does not generalize, but a new
   config search is warranted; the wall stands for the
   validated config only.
2. If C1 passes with a >20% relative MSE cut on any task:
   the "wall" was partly a readout artifact, and the
   local-readout story strengthens (the centroid, not the
   prototype, is the right continuous readout).
3. If F1 reads HELPS on any task: the wire's role must be
   revisited; round 20's token-task cost does not
   generalize, and a wire-plus-continuous follow-up is
   warranted.
4. If N7/N8 pass: NARMA/Mackey-Glass are genuine
   discriminating tasks and RPC competes with Markov-2,
   which strengthens the "delta learns" claim beyond the
   trivial tasks.
5. If R1 fails everywhere (standardized ridge: frozen beats
   trained): delta actively harms the continuous
   representation; the D8 story flips, and any prototype
   wins are a readout artifact, not learning.

## 9. Seeds, statistics, and launch

1. Seeds: val 322-326 (5), test 327-346 (20). Fresh: no
   overlap with rounds 15-18 (262-281) or round 20
   (302-321). The val seeds pick configs; the test seeds
   decide bars. The Convention-A primaries were selected
   on seeds 262-266 and are tested here on fresh seeds,
   so a val-lucky config on 262-266 cannot pass a bar by
   itself.
2. Statistics: exact Student-t CIs throughout, t(19) =
   2.0930 for the 20-seed test bars, t(4) = 2.7760 for
   5-seed val summaries. scipy is absent from the venv, so
   the constants are hardcoded from the standard t table.
   Paired per-seed differences, keyed by seed (never by
   list position). Means, medians, and IQRs are reported
   alongside every bar.
3. Launch: copy the driver, analyzer, monitor, and protocol
   to Beast; Beast smoke (one val + one test invocation);
   monitor val launch; create the r21-monitor cron (every
   30 min, same pattern as r20-monitor). Monitor phases:
   token val (100 lines) -> pick token winners ->
   winners_token.json -> bump val (40 lines) -> pick bump
   sigmas -> winner.json -> test (560-600 lines) ->
   concat + analysis. Detached workers, marker written at
   launch time, idempotent via marker files. The monitor
   stops (no new launches) if any JSONL contains a
   traceback.
4. Cost estimate (measured in the smoke test, VM seconds per
   invocation; Beast is faster per core but shared): one
   lorenz token arm (train + twin + collections) takes ~23s;
   one length-100 token arm takes ~48s; val runs are cheaper
   (no twin, ~15s lorenz / ~35s length-100). Total round is
   700-740 lines; each test invocation is one arm x one seed
   (train 5 epochs + train/test collections + frozen-twin
   collections). Scaling: about 4 CPU-hours total; about
   30-60 minutes wall with 8 parallel workers on an idle
   machine (Beast shares cores with connectome work, so the
   monitor should be given a full day, not an hour).

## 10. Self-critique (frozen before any test result)

1. The wire decision cuts both ways. Wire-off primary risks
   missing a wire benefit that only appears on continuous
   tasks (the wire was never tested on continuous input
   before round 21). The wire-on arm on every task mitigates
   this, but only on the token input: if the wire helps only
   with bump input, round 21 misses it. That interaction is
   out of scope by design, not by oversight, and the verdict
   must not claim "the wire does nothing on continuous
   tasks" if only the token arm was run.
2. Lorenz Convention-A carryover risk. The lorenz primary
   (alpha 1.0, smooth 2.0, eta_d 0.2) was selected on seeds
   262-266 and reproduced exactly on those same seeds; the
   fresh-seed test (327-346) is the honest check, but the
   config itself was still chosen using 262-266 performance.
   The var-sine primary is now in the same position: AGY's
   number was reviewer-reported, but the smoke test
   reproduced it exactly with the round-21 driver on the
   same seeds 262-266 (diff -0.0000), so the remaining risk
   for both tasks is only that the config was selected on
   262-266, not that the number was scored differently. If
   W1 or W2 fails on fresh seeds, the config does not
   generalize; the verdict must not re-blame the scorer.
3. Val-overfitting. The val grid picks the winner on 5
   seeds; with 5 configs the winner's val MSE is
   optimistically biased, and the test verdict for the new
   tasks rests entirely on that pick (they have no
   preregistered config to fall back on). The test uses 20
   fresh seeds and the bars require the CI to clear 0, but
   a val-lucky config can still pass a weak bar. The
   lorenz/varsine primaries exist precisely to avoid this;
   the new tasks have no such anchor, so a new-task pass
   is weaker evidence than a re-validation pass.
4. The centroid beta=10 is fixed without any validation. The
   prototypes are near-parallel (off-diagonal cosine 0.890
   on the cycle task), so the cosine spread differs across
   tasks; beta=10 may be too peaked (collapses to argmax)
   or too flat (collapses to the mean) on some task. The
   5/10/20 sensitivity check catches it but does not fix
   it; a wrong beta makes C1 a weak test of the
   interpolation idea. If the sensitivity check flags a
   task, the honest conclusion is "beta was wrong for this
   task", not "the centroid fails".
5. The bump arm's sigma is val-picked on 2 values only
   (2.0, 3.0) and the bump arm is wire-off only. A
   bump-plus-wire interaction is untested, and a bump loss
   (B1 fails) could mean the sigma was wrong, not that the
   bump input is worse. The protocol states this as a limit;
   the verdict must not overclaim a bump failure.
6. The frozen twin shares the trained net's seed and init,
   which is exact by construction, but the twin's
   prototypes are built from the same train data with the
   same proto_n, and its calibration scales are measured
   pre-training, identically to the delta net's. If any of
   those steps consumed RNG differently between the two
   builds, the twin would silently differ from a standalone
   frozen net. The smoke test verifies exact equality
   (proto MSE to 12 decimals) before launch; if it ever
   fails, the round falls back to standalone frozen
   invocations (separate driver runs, same seed).
7. NARMA-10 and Mackey-Glass are new generators written for
   this round. A bug in either (wrong recurrence, wrong
   burn-in, range outside [lo, hi]) would silently change
   the task. Mitigations: the smoke test checks range,
   clipping, bin coverage, and Markov-2-vs-persistence on
   both, and the driver asserts the bin mapping agrees with
   the generator on every bump pair. The NARMA-10
   divergence resampling (section 3) is a selection on the
   input draws, not on the model, and its rate (~1 in 1000
   sequences) is measured and reported.
8. Compute risk. The round is 700-740 lines; each test
   invocation is one arm x one seed (5-epoch train plus
   collections). If Beast is loaded (connectome work runs
   there), the round may take many hours; the monitor is
   idempotent and the cron polls every 30 minutes, but a
   slow round delays the precision (C) design. The measured
   per-invocation time is in section 9.
9. The ESN scalar-ridge baseline is the strongest
   comparator (AGY: Lorenz 2.097, var-sine 0.0072) and RPC
   is unlikely to beat it. If R4 fails everywhere while
   W1/W2 pass, the correct reading is "delta learns
   continuous trajectories but the reservoir-plus-ridge is
   a better continuous predictor", not a wall-fall for the
   model. The verdict must not overclaim.
10. Horizon. Round 21 tests ONE-STEP prediction only
    (h=1). The round-20 sketch mentioned horizon h=5 for
    Mackey-Glass; the reviewers' NARMA/Mackey-Glass
    recommendation was about multi-step MEMORY (the input
    needs history), not multi-step-ahead prediction. Round
    21 does not test h>1 rollout: round 20's D showed
    rollout fails with the wrong sign on tokens, and a
    continuous rollout would inherit that. Multi-step
    rollout is a follow-up, not this round.

## 11. Appendix: numbers the design rests on

1. Lorenz Convention-A (alpha 1.0, smooth 2.0, eta_d 0.2):
   prototype MSE 5.073 vs persistence 13.05, acc 0.4285,
   5 seeds 262-266, reproduced exactly by the section-57
   spot check (`r1718_spotcheck.py`, correct Convention-A
   scorer). Per-seed MSE [5.68, 4.62, 3.75, 5.83, 5.48].
2. Var-sine Convention-A (alpha 0.7, smooth 2.0, eta_d 0.2):
   prototype MSE 0.0370, acc 0.3230, 5 seeds 262-266,
   first reported by AGY (`r1718-agy-critique.md` Check 4)
   and REPRODUCED EXACTLY with the round-21 driver in the
   smoke test (mean MSE 0.0370, acc 0.3230, per-seed
   [0.0423, 0.0368, 0.0371, 0.0207, 0.0479], diff -0.0000).
   Persistence 0.0548, Markov-1 0.2826 acc, ESN 0.0483 MSE
   on the same 5 seeds.
3. Var-sine Convention-A variant (alpha 0.7, smooth 3.0,
   eta_d 0.2): MSE 0.0317, beating persistence and ESN on
   5/5 seeds (AGY, reviewer-reported).
4. Round-18 correct-scored C1 (Convention B): Lorenz MSE
   22.390 vs persistence 12.605; var-sine MSE 0.064 vs
   persistence 0.056. The gap between these and (1)/(2) is
   the carryover the round re-validates.
5. D8 (Lorenz seed 262, C1 config): mean ||d1|| trained
   1.025 vs frozen 5.651 (5.52x); raw lam=1.0 ridge MSE
   14.904 trained vs 14.073 frozen; column-standardized
   ridge ties at 13.976. All four numbers re-measured
   exactly by the smoke test with the round-21 driver
   (std 13.976 / 13.976, raw 14.904 / 14.073; proto
   trained 24.877 vs frozen 166.261 on the same config).
6. ESN scalar ridge (AGY, reviewer-reported, raw ridge):
   Lorenz 3.287 (one-hot input) / 2.097 (bump input);
   var-sine 0.0191 / 0.0072. Round 21 re-measures both raw
   and standardized versions on fresh seeds.
7. Markov-2 triviality (AGY, verified): 99.50% (Lorenz)
   and 99.82% (var-sine) of test 2-grams appear in
   training; two bins explain R squared 0.989 / 0.995.
8. Round-20 wire numbers: B1 +0.014 CI [0.005, 0.022], B2
   +0.033 CI [0.022, 0.044]; B5 wrong sign (err_mag_last
   on-off +0.0138 lag-1, +0.0181 lag-2); D sine rollout on
   0.147 < cut 0.184 < off 0.223; one-step sine cost 23
   points (0.666 vs 0.892).
9. New-task calibration (smoke test, single seed 322, G1
   config alpha 1.0 / smooth 2.0 / eta_d 0.2; calibration
   only, not a verdict). NARMA-10: persistence MSE 0.0097,
   Markov-2 0.0125 (persistence wins: the driving input u
   is unobserved, so no bin-only method predicts the
   u-driven part); delta proto 0.0124, centroid beta=10
   0.0081, ridge_std 0.0077. Mackey-Glass: persistence
   0.0012, Markov-2 0.0009; delta proto 0.0020, centroid
   beta=10 0.0021, ridge_std 0.0027. On this seed/config
   the prototype readout loses to persistence on BOTH new
   tasks while the continuous readouts win on NARMA-10
   only: one-step prediction on these tasks is
   persistence-dominated, so the N1/N2 bars are genuinely
   hard and a fail there means persistence wins, not that
   the machinery is broken.
10. Wire and centroid calibration (smoke test, lorenz seed
    322, G1 config). Wire-on (aligned, gain 1.0): mean
    fb_drive_ratio 0.2982, fb_rms_ratio 0.4273, proto MSE
    3.7809 vs wire-off 6.6547 (the wire helped on this
    seed). Centroid beta 5/10/20 MSE: 10.5741 / 5.0133 /
    4.4831 vs proto 6.6547 (beta=5 is worse than proto,
    beta=10 and 20 are better; the temperature curve is
    sane and the centroid genuinely improves on argmax).
    Frozen twin equals a standalone frozen invocation
    exactly (proto MSE diff 0.00e+00 on lorenz seed 322).
