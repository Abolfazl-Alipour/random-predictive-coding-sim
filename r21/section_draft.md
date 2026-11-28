## 60. Round 21: continuous re-validation (Convention-A + NARMA-10 / Mackey-Glass) -- DESIGN, NOT LAUNCHED

(Nov 27, 2026. Design subagent report; protocol frozen in
PROTOCOL-R21-CONTINUOUS.md. VM smoke only; NO Beast launch
in this turn. Beast launch, monitor, and cron are the next
step for the parent agent; see the launch checklist at the
end of this section.)

### Why this round exists

Round 18 concluded a "continuous wall" survives correct
scoring: the delta model could not beat persistence on
Lorenz or var-sine with the frozen round-15 configuration
(Convention B). Section 57 showed that conclusion rested on
a hyperparameter carryover, not a wall: with Convention-A
hyperparameters from round 15's own grid, the prototype
readout beats persistence on both tasks (Lorenz MSE 5.07 vs
persistence 13.05; var-sine MSE 0.0370 vs persistence
0.0548). Round 21 asks whether that finding survives valid
validation on fresh seeds, and whether it extends to
continuous tasks that need multi-step memory and cannot be
solved by a 2-bin lookup (Markov-2 trivially solves Lorenz
and var-sine: 99.50% / 99.82% of test 2-grams appear in
training). Round 20 settled the wire question for tokens
(B' FAILS, D FAILS, section 59); round 21 does NOT relitigate
it, it tests the wire as a factor on continuous tasks.

### Design (protocol PROTOCOL-R21-CONTINUOUS.md)

1. Tasks (all K = 32, Convention-A one-step scoring):
   lorenz (200/50/50, len 24), varsine (100/20/20, len 100),
   narma10 NEW (100/20/20, len 100), mackeyglass NEW
   (discrete map, delay 17 steps; 100/20/20, len 100).
2. Preregistered primaries: lorenz alpha 1.0 / smooth 2.0 /
   eta_d 0.2; varsine alpha 0.7 / smooth 2.0 / eta_d 0.2;
   new tasks take the val winner (procedure-preregistered).
   Val token grid G1-G5 (the two primaries, the sigma
   neighbours, and the Convention-B carryover reference);
   val bump grid sigma 2.0 / 3.0. Val seeds 322-326,
   test seeds 327-346 (20 fresh; no overlap with 262-281
   or 302-321).
3. Wire decision (the section-59 open question, RESOLVED):
   test aligned wire on/off as a factor; wire OFF is
   primary, wire ON (aligned, gain 1.0) is a secondary arm
   on the token input only. Reasons: round 20 measured a
   23-point one-step sine cost for the wire and a negative
   rollout sign, so carrying it as primary would confound
   "does delta learn continuous trajectories" with "does the
   wire tax one-step prediction"; the burden of proof is on
   the wire. If wire-on beats wire-off on any task (paired,
   exact-t CI above 0), that is new evidence FOR the wire.
4. Readouts, all three from the SAME run (exactly matched):
   prototype (hard argmax, primary), centroid
   (softmax-weighted prototype centroid, beta = 10 primary,
   5 and 20 as sensitivity, no new fitted weights), and
   ridge (lam = 1.0, COLUMN-STANDARDIZED -- the D8 lesson;
   the raw ridge is also computed and reported, no bar uses
   it). Frozen control is a TWIN net (same seed/init, never
   trained), verified bit-identical to a standalone frozen
   run.
5. Baselines per task/seed: Markov-1/2, persistence,
   train-mean, ESN class-argmax, and ESN scalar ridge (raw
   AND standardized; AGY's strongest comparator: Lorenz
   3.287 / 2.097, var-sine 0.0191 / 0.0072).
6. Bars (54 rows; paired per-seed differences, exact
   Student-t 95% CIs, t(19) = 2.0930): W1-W10 (Convention-A
   re-validation), N1-N8 (new tasks, incl. beats-Markov-2),
   R1/R3/R4 x4 (standardized ridge), C1/C2 x4 (centroid),
   F1 x4 (wire effect, three-way HELPS/HURTS/NULL) + F2 x4
   (wire-live gate, mean fb_drive_ratio > 0.05), B1/B2 x4
   (bump input). Verdict: wall FALLS iff W1, W2, W7, W8 all
   pass; DISCRIMINATION iff N7 and N8 pass. Even a
   wall-fall does NOT establish predictive coding (round 20
   failed B' and D); it establishes only that the
   signed-error rule learns continuous trajectories.
7. Self-critique (protocol section 10, 10 points): the wire
   decision cuts both ways (bump-by-wire interaction
   untested); both primaries were selected on seeds 262-266
   (var-sine's number is now reproduced exactly, so the
   remaining risk is selection, not scoring); val-overfitting
   on 5 seeds (new tasks have no preregistered anchor);
   centroid beta=10 unvalidated; bump sigma picked on 2
   values; twin exactness verified; new-generator risk;
   compute (~4 CPU-hours); ESN scalar ridge likely unbeaten;
   one-step horizon only (h>1 rollout is a follow-up).

### Smoke results (VM, `rpc_round21_smoke.py`, 777s, zero failures)

1. Lorenz Convention-A reproduces EXACTLY: acc 0.4285,
   MSE 5.073, per-seed [5.68, 4.62, 3.75, 5.83, 5.48].
2. Var-sine Convention-A reproduces AGY EXACTLY with the
   round-21 driver: acc 0.3230, MSE 0.0370, per-seed
   [0.0423, 0.0368, 0.0371, 0.0207, 0.0479], diff -0.0000.
   The var-sine primary is no longer reviewer-reported-only.
3. D8 re-measured exactly: standardized ridge ties
   13.976 / 13.976 (trained/frozen); raw 14.904 trained /
   14.073 frozen (the frozen-ridge win reproduces, and it
   is the artifact); proto trained 24.877 vs frozen 166.261.
4. New tasks sane: narma10 range [0.182, 0.939], 25/32 bins,
   persist 0.0097, Markov-2 0.0125 (persistence WINS: the
   driving input u is unobserved, so Markov-2 is
   non-trivial here, not unlearnable); mackeyglass range
   [0.370, 1.344], 22/32 bins, persist 0.0012, Markov-2
   0.0009. Single-seed G1 delta: narma10 proto 0.0124,
   centroid10 0.0081, ridge_std 0.0077; mackeyglass proto
   0.0020, centroid10 0.0021, ridge_std 0.0027. On this
   seed/config the prototype readout loses to persistence
   on BOTH new tasks: one-step prediction there is
   persistence-dominated, so N1/N2 are genuinely hard.
5. Driver CLI resumable (10 lines, re-run adds 0).
   Frozen twin bit-identical to standalone frozen
   (proto MSE diff 0.00e+00).
6. Centroid sane: argmax(sims) == pred on all 1150 steps;
   beta 5/10/20 MSE 10.5741 / 5.0133 / 4.4831 vs proto
   6.6547 (lorenz seed 322): beta=5 hurts, beta=10/20
   help; the centroid genuinely improves on argmax.
7. Wire gate: mean fb_drive_ratio 0.2982 (fb_rms 0.4273)
   on lorenz seed 322, 6x above the 0.05 gate. Single-seed
   calibration: wire-on proto MSE 3.7809 vs wire-off
   6.6547 (the wire HELPED here). One seed proves nothing;
   F1 decides over 20 seeds.
8. Bump path verified AFTER the smoke run (the smoke
   script only exercised token arms): one bump test arm
   (lorenz seed 322, sigma 2.0) gives bump_delta proto
   6.914 / centroid10 5.529 / ridge_std 6.740 and
   bump_frozen proto 134.384; the val-bump CLI writes both
   sigma rows (val MSE 5.8291 at sigma 2.0, 5.0521 at
   sigma 3.0). The bump machinery works; the smoke script
   does not cover it (noted for the next smoke revision).

### Two fixes found during the smoke test

1. NARMA-10 divergence (real bug, fixed before launch).
   The recurrence has an unstable fixed point near y = 1.02
   (mean-field roots 0.38 and 1.02); about 1 in 1000
   sequences diverges (seed 329, sequence 46 went to inf
   during burn-in). The generator now resamples divergent
   draws and draws leaving the bin range (canonical system
   conditioned on its stable basin; measured 10-seed range
   [0.171, 0.974] over 1400 sequences, no warnings). The
   u-term lag was also corrected to the canonical 9
   (the first version used lag 10). Mackey-Glass is
   globally bounded and needed no fix.
2. Three smoke-check premises were wrong and were
   corrected, not the code under test: (a) check 4c assumed
   Markov-2 must beat persistence for a task to be
   learnable -- false on NARMA-10, where the unobserved
   driving input makes Markov-2 lose while a learned
   readout still wins (now a report, not a gate); (b) check
   7b assumed centroid beta=20 should equal the prototype
   readout -- false, the softmax still mixes neighbours and
   beats argmax by 33% on lorenz seed 322 (now a factor-of-2
   sanity gate); (c) check 8 crashed with KeyError because
   the wire-on arm yields token_w_* rows, not token_* rows
   (fixed: run both arms for the comparison). The smoke
   test was re-run in full after the fixes and passed with
   zero failures; the deterministic re-run reproduced every
   number identically.

### Analyzer / monitor verification (synthetic data, then deleted)

The analyzer and the monitor's pick functions were verified
on fabricated data with known expected outcomes before
launch (files deleted afterwards; r21/ holds only the
deliverables). Analyzer: 54 bars, 36 pass / 13 fail / 5 n/a,
every outcome matched the design, including the F1
three-way (HURTS on three tasks, HELPS on one) and the
W10 n/a case. One logic gap found and fixed: the verdict
had no mixed case (N1 passes on one new task only), and its
text wrongly read "does not beat persistence on the new
tasks"; the analyzer now names the winning task
("BEATS MARKOV-2 ON ONE TASK ONLY" / "MIXED"). Monitor
picks verified: token winners (lorenz G1, varsine G2,
narma10 G1, mackeyglass G3), bump sigmas, final winner.json
shape, expected test lines 560 (no conditional primary arm
when winner == primary).

### Launch checklist (for the parent agent; NOT done here)

1. Copy to Beast: `rpc_round21_continuous.py`,
   `r21/r21_analyze.py`, `r21/monitor.py`,
   `PROTOCOL-R21-CONTINUOUS.md`, and the updated
   `baselines.py` (new generators) to the Beast rpc-sim
   copy. Do NOT copy `rpc_round21_smoke.py` results or the
   /tmp smoke files.
2. Beast smoke: one val invocation (lorenz, seeds 322-323)
   and one test-arm invocation, confirming the driver runs
   under the Beast venv.
3. Launch the monitor (it drives val -> winners -> test ->
   analysis) and create the r21-monitor cron (every 30 min,
   r20-monitor pattern).
4. Expected line counts: token val 100, bump val 40, test
   560 base (up to 600 with conditional primary arms).
   Cost about 4 CPU-hours; about 30-60 minutes wall with
   8 parallel workers on an idle machine.
5. On completion: write the verdict as the next free section
   (61), update the README Status and MEMORY.md, log a
   tracking entry on goal_a9a871e81255, and report to Reza.
