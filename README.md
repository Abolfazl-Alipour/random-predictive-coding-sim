# Random predictive coding: simulation tests

This repo holds the simulation tests for random predictive coding
(RPC), the theory in Chapter 3 of Abolfazl Alipour's PhD thesis
(Indiana University, August 2023). RPC says a cortical module
predicts the next input through fixed random distal wires, and a
burst marks a confirmed prediction, which then drives local
Hebbian learning.

## Verdict

1. The label "random predictive coding" is new in neuroscience.
   The idea is not. The closest prior art is Hierarchical Temporal
   Memory (HTM), and the most damaging is Chen et al. 2022/2024
   (CA3 predicts, CA1 signals error), which predates the thesis
   and is not cited in it.
2. The model learns noiseless deterministic token sequences:
   repeating cycle 0.843 and pure motif 0.934 at reservoir leak
   rate 1.0 (5 seeds, held-out test).
3. It fails in noise and on continuous signals under the
   unsigned Hebbian rule: planted motifs 0.113 at best
   (learned proximal projection), fixed sine 0.218 at best
   (nearest-neighbour readout), variable sine 0.108,
   Lorenz 0.093, 1D track 0.098. Every fix tried so far
   moves the number by 0.01 to 0.03 except the smooth input,
   and none passes the bar. Under the signed delta rule
   (round 14, section 43) the same model learns in noise:
   fixed sine 0.484 and planted motifs 0.221 with motif-only
   0.966, both passing AGY's bars. The unsigned rule was the
   binding failure, not the code or the readout.
4. The bottleneck under the unsigned rule is the learning
   rule itself. The module-1 reservoir holds the current
   token perfectly (linear probe 1.000), and the frozen
   distal drive already decodes the next token at 0.917
   (ridge probe, section 43.3). The unsigned Hebbian update
   is all positive, so the distal map drifts into a positive
   corner and the drive becomes nearly identical on every
   step; the fixed prototype readout then cannot use what
   the drive holds. The signed delta rule rotates the drive
   into the prototype space (trained ridge 0.913, unchanged)
   and the same readout scores 0.484. A trained ridge readout
   on the same distal drive scores 0.556 to 0.673 on the
   fixed sine under the old rule, so the information was in
   the drive all along and the prototype readout could not
   extract it until the drive was aligned.

## Key numbers

All numbers are held-out test accuracy, 5 seeds, chance 0.0625
unless noted. Pass bar: accuracy 0.2625 and gain 0.10 over the
no-plasticity control.

1. Pure repeating cycle, leak 1.0: 0.843 (control 0.147).
2. Pure motif, leak 1.0: 0.934 (control 0.322).
3. Planted motifs in noise, leak 1.0: 0.073 (control 0.064).
4. Planted motifs, learned ridge readout: 0.157 (frozen control
   0.163, so the readout does the work, not RPC learning).
5. Planted motifs, best fix so far (learned proximal
   projection, A12): 0.113 (control 0.067).
6. Baselines on planted motifs: Markov-1 0.278, echo state
   network 0.274, LSTM 0.283.
7. Fixed sine, smooth input sigma 3.0: 0.199 at 5 seeds
   (control 0.022), best 0.210 at 3 seeds with leak rate 0.7.
   One-hot input scores 0.091.
8. Variable-frequency sine, smooth input sigma 3.0: 0.108
   (control 0.036). One-hot input scores 0.035.
9. Lorenz x, smooth input sigma 3.0: 0.069 (control 0.041).
   One-hot input scores 0.044. Best Lorenz score is 0.093
   (graded confirmation plus eta_b 0.15, alpha 0.7).
10. Fixed sine, graded confirmation (A16) plus eta_b 0.15,
    alpha 0.7: 0.202 at 5 seeds (control 0.031). Graded
    confirmation at the same learning rate hurts everywhere;
    with eta compensation it ties binary at 5 seeds.
11. Fixed sine, dot-product readout (A17): graded plus eta_b
    0.15, alpha 0.7 scores 0.217 at 5 seeds (control 0.036),
    best 5-seed score yet; binary dot scores 0.209. Prototype
    norms range 0.55 to 0.75. The readout story is complete:
    cosine 0.202, dot 0.217, trained ridge 0.556 to 0.673.
12. Trained ridge readout on the distal drive (round 9): fixed
    sine 0.556, variable sine 0.244, Lorenz 0.251, planted
    motifs 0.157. The frozen no-plasticity control matches or
    beats the trained network everywhere (fixed sine 0.606,
    variable sine 0.234, Lorenz 0.295, planted 0.163), so the
    learning adds nothing to the readout's score on any task.
13. Round 10: the top-6 code and the prototype readout are my
    assumptions, not the theory. Full graded code (A18, all
    cells project back): fixed sine 0.105, unstable, 2 seeds
    work and 3 collapse to chance; planted motifs 0.065.
    Nearest-neighbour readout (A19, no averaging): fixed sine
    0.218, best 5-seed score yet, gain +0.190, still 0.013
    short of the 0.231 bar; planted motifs 0.072, ties cosine.
    The prototypes were mean p1 all along, not mean d1, so the
    readout already matched distal against proximal. Round 11
    tests the fix for the code-versus-learning tension:
    sharpen the learning pre-term (A20) so the update
    concentrates while the drive stays rich, and the clean-code
    combination of learned W_prox plus kNN readout.
14. Round 11: sharpening the learning pre-term (A20) changes
    nothing. Fixed sine 0.099 (p=2) and 0.110 (p=3), both
    still bimodal; planted motifs 0.070. The dilution
    explanation is wrong as stated: the drive stays degraded
    (d1 decode 0.478 vs top-6 0.509) even with a concentrated
    update, so the degradation sits in the drive read, not the
    learning write. The fixes do not stack on sine: learned
    W_prox plus kNN readout scores 0.133 (worse than kNN
    alone, 0.218), both under coactivity. CORRECTION: the
    planted stacking comparison (0.074 vs A12 alone 0.113) is
    confounded, because A12 ran under coactivity and A12+knn
    ran under the default burst rule; the clean test is round
    13. Round 12 tests the sharpened drive read (A21) and a
    per-seed probe of the next-token decode from the drive on
    working versus collapsed seeds.
15. Round 12: sharpening the drive read (A21) changes
    nothing. Fixed sine 0.119 (p=2) and 0.129 (p=3), both
    still bimodal; planted motifs 0.076. The per-seed probe
    decodes the next token from the drive: top-6 baseline
    0.507 on every seed; full code 0.408, per-seed [0.462,
    0.414, 0.495, 0.295, 0.373]. Seed 2 decodes at 0.495 but
    scores 0.035, so the drive holds the information and the
    fixed-space readout extracts none of it. The learned map
    does not reliably land in the prototype space. Round 13
    tests the burst rule on the full code and the
    prototype-span projection readout (A22).

## How to run

1. Python 3 with PyTorch (CPU build is enough). See
   `requirements.txt`.
2. `python rpc_v2.py` runs the base model. Each `rpc_*.py`
   driver runs one experiment and prints one line per condition.
3. `simulation-results.md` is the full experiment record, one
   numbered section per test, with the exact reproduce command.

## Files

1. `rpc_v2.py`: the model. All assumptions are labelled A1 to
   A20 in the header.
2. `rpc_toy.py`: the first version (v1). It never fires, because
   the distal drive starts at zero. Kept for the record, do not
   overwrite.
3. `baselines.py`: task data and the Markov, echo state network,
   and LSTM baselines.
4. `rpc_simple.py`, `rpc_alpha.py`, `rpc_alpha2.py`,
   `rpc_planted_alpha.py`, `rpc_dense.py`, `rpc_readout.py`,
   `rpc_norm.py`, `rpc_delay.py`, `rpc_probe.py`, `rpc_probe2.py`,
   `rpc_direct.py`, `rpc_fix.py`, `rpc_wide.py`, `rpc_prox.py`,
   `rpc_alpha_sine.py`, `rpc_sine_diag.py`, `rpc_sine_fix.py`,
   `rpc_sine_fix2.py`, `rpc_sine_fix3.py`, `rpc_sine_fix4.py`,
   `rpc_sine_fix5.py`, `rpc_sine_fix5b.py`, `rpc_sine_fix6.py`,
   `rpc_sine_fix6.py`, `rpc_sine_fix6b.py`, `rpc_sine_fix6c.py`,
   `rpc_sine_fix7.py`, `rpc_sine_fix8.py`, `rpc_readout_gen.py`,
   `rpc_sine_fix10.py`, `rpc_planted_fix10.py`,
   `rpc_sine_fix11.py`, `rpc_sine_fix11b.py`,
   `rpc_planted_fix11.py`, `rpc_sine_fix12.py`,
   `rpc_planted_fix12.py`, `rpc_sine_fix13.py`,
   `rpc_planted_fix13.py`:
   one driver per experiment. Each has its own docstring with
   the question, the design, and the result.
5. `simulation-results.md`: the full record.
6. `reviewer-pack.md`: a copy-paste adversarial prompt for other
   AI reviewers, with instructions to paste the full Chapter 3
   text in one message after the prompt.
7. `thesis.txt`, `ch3.txt`: extracted text of the thesis and of
   Chapter 3. Not in the repo by default; add them only if you
   want the thesis text public.

## Status

Boundary firm as of Nov 26, 2026: fixed sine about 0.22,
variable sine about 0.11, Lorenz about 0.09, all below the
bars. The nearest-neighbour readout (A19) is the best 5-seed
score at 0.218, still below the bar. Round 9 shows the trained
ridge generalises in score (variable sine 0.244, Lorenz 0.251)
but the frozen control matches or beats it everywhere, so the
learning adds nothing. Round 10 shows the full graded code
hurts (0.105, unstable) while the kNN readout helps a little
(0.218). Rounds 11-12 show sharpening the learning update
(A20) and the drive read (A21) change nothing. Round 13
finished: burst on the full code scores 0.022, and A22 is a
proven no-op for cosine and dot.

CORRECTION (Nov 26, external critique by Kimi and AGY, full
record in simulation-results.md section 41): the root-cause
story in the sections above is wrong in specific places.
1. The orthogonal-information explanation is dead. The 31
   percent was a norm ratio, not energy (true orthogonal
   energy 52.5 percent, chance level). The real collapse
   cause is the unsigned Hebbian rule: every update is
   positive, so the distal map drifts into a positive corner
   and the drive becomes nearly identical on every step.
   On collapsed seed 2, 65 percent of the drive's energy is
   inside the prototype span, more than the working seed.
2. The 200 to 64 bottleneck is real. At n_out 256 a fixed
   random projection loses nothing (p2 0.976, full code
   0.867). The round-22 "dimensionality is dead" verdict
   was wrong; the wide code was never probed.
3. The planted ceiling is 0.287, not 0.36. Each motif is
   planted once per sequence. The 0.2625 bar is 92 percent
   of the optimum.
4. The "no-confirmation" control never turned off distal
   learning, only the module-2 confirmation update. The
   burst is per-cell coincidence, not token correctness.
   The planted stacking test is still confounded.
5. A23 on sine top-6, 5 seeds, DONE Nov 26 08:55 PST: trained
   drive-proto 0.281 vs frozen drive-proto 0.444. AGY's warning
   confirmed at full scale: A23 hands the frozen control a
   supervised nearest-centroid classifier and learning is a net
   loss (-0.163). Reza ordered "do it all" Nov 26 08:36: A23
   diagnostic AND the delta rule, both running as round 14.
6. Third external critique (Nov 26, external-critique-3.md):
   the model tests no Chapter 3 predictive-coding signature
   (no error population, hierarchy, precision, omission);
   alpha 1.0 and smooth sigma were selected on the test set,
   so round 15 needs a validation split; "loses a third" is
   decode accuracy, not information loss; A22 is not a no-op
   for kNN or score values. The critic repeats two errors:
   the gate is per-cell coincidence, not token correctness,
   and the planted ceiling is 0.287, not 0.36.
7. The "same drive" ridge range 0.556 to 0.673 mixes
   networks. The matched gap is 0.478 on the one-hot
   network (ridge 0.556 vs prototype 0.078). The ridge on
   the best smooth network was never measured.

Round 14 DONE Nov 26, 10:31 PST (record section 43). A23
diagnostic: all three versions fail trained-vs-control (sine
top-6 0.281 vs frozen 0.444; sine A18 0.185 vs frozen 0.447;
planted top-6 0.132 vs frozen 0.129, flat). Thesis-claim
version (a), all wires fixed random with a fixed readout, is
dead. A24 delta rule, n_out 256, unmodified p1-prototype
cosine readout, all other wires fixed random: sine-fixed
0.484 vs frozen 0.031, gain +0.453, BOTH AGY bars pass
(0.46, +0.40), first theory-faithful config to beat Markov-1
(0.459); planted 0.221 vs frozen 0.062, gain +0.159,
motif-only 0.966, all three AGY bars pass (0.22, +0.15,
0.75), marginally on accuracy and gain; strict bar 0.2625
NOT met. Mechanism (ridge probe): trained drive decodes
next token at 0.913 vs frozen 0.917, so delta adds no
information; the win is readout alignment. Controls:
coactivity and burst collapse to chance at n_out 256
(sine 0.024/0.028, planted 0.059/0.059, identical per-seed
on planted, a shared constant-prediction collapse);
centring alone recovers partway (sine 0.151, planted
0.154). Only the error term reaches the bars. Open:
planted accuracy marginal; variable sine, Lorenz, and the
1D track untested under delta; alpha 1.0 and smooth 3.0
were selected on the test set, so round 15 needs a
validation split and 20 seeds before the win is claimed.

Round 15 COMPLETE Nov 26, 21:36 PST (record section 46,
analysis r15/r15_analysis.md). Preregistered 20-seed
replication, seeds 201..220, validation grid 18 configs,
winners frozen per task, 500 test lines, 0 errors. Winners:
planted1 0.229, planted2 0.218, sine 0.863, Lorenz 0.164,
varsine 0.152 (val mean acc); validation picked one-hot
(sigma 0.0) on 4 of 5 tasks. Test verdict, 6/18 bars pass:
sine delta 0.868 (CI 0.851-0.886) beats Markov-1 0.456 and
Markov-2 0.798, clean win; planted lag-1 delta 0.223 (CI
0.216-0.230) misses the lower-CI 0.22 bar by 0.004, m1 0.680
misses 0.80, m2 0.956 passes 0.90; planted lag-2 delta 0.222
passes both bars; Lorenz delta 0.164 and varsine delta 0.150
fail to beat persistence or Markov-1 on accuracy or MSE.
Delta beats frozen on all five tasks (paired CIs all above
zero); tempcov and burst collapse everywhere (0.04-0.08), so
the signed error term is the active ingredient. The trained
readout (ESN) still beats delta on every task, so the readout
gap survives. What is settled: delta is a validated learning
rule for discrete token tasks under a fixed prototype readout.
What is not: continuous prediction, the first motif step, and
whether the mechanism is predictive coding (omission test
still unrun). Round 16 (the omission test, PROTOCOL-R16-
OMISSION.md) COMPLETE Nov 26, 2026 about 22:45 PT: 11/11 bars
pass. Experiment A passes all five lag-1 bars plus A6, A7 and
all four lag-2 bars. Planted lag-1 delta omit_m2 0.966 (CI
0.955-0.977), omit_m1 0.667, continuation 0.610; lag-2 delta
omit_m2 0.943, omit_m1 0.678, continuation 0.467; frozen
0.048/0.058. The trained network holds a stimulus-specific
prediction that survives a missing input and generates the
completion. Limit stands: no error unit and no top-down
feedback at test time, so this is necessary but not sufficient
for the Rao-Ballard claim; next discriminator is the top-down
clamp (experiment B). The r16-monitor cron is removed; the
analysis is delivered. Round 17 (the top-down clamp,
PROTOCOL-R17-CLAMP.md) LAUNCHED Nov 26, 2026 about 23:05 PT
on Reza's order ("do all together, do not stop for my
approval, keep going until the question is completely clear
and killed"): A26 adds the wire (fb = fb_gain * (W_fb @ d1)
into module 1's reservoir) and the round clamps it (cut /
shuffle / wire-off / frozen). Beast smoke reproduced the VM
numbers exactly. Four validation workers running (planted1,
seeds 241-245, gains 0.0/0.1/0.3/1.0); the r17-monitor cron
picks the winner with the 0.80x guard and launches the test
phase. VAL WINNER Nov 26, 2026, 23:44 PT (verified from the
Beast val files): fb_gain 0.1, val_acc 0.226 vs gain-0.0
0.227, guard PASS; strong feedback (1.0) drops to 0.182.
Test phase launched (4 workers, planted1/planted2, seeds
241-260, fb_gain 0.1); analysis at 200 lines. Round 18 (continuous-task wall) COMPLETE Nov 27, 2026, 00:15 PT
(record section 52; analysis r18/r18_analysis.md,
verified by independent recomputation from the raw
JSONL: all 16 condition means match to 3 decimals, 0
duplicate keys, 360 lines). Correct scoring: Lorenz C1
accuracy 0.333 (was 0.164 mis-scored), var-sine C1 0.295
(was 0.150). Bars 5/8 pass: D1, D2, D5, D6, D7 pass; D3,
D4, D8 fail. H0 fails (D3: var-sine C1 MSE 0.064 against
persistence 0.056) and D4 fails (Lorenz C1 MSE 22.390
against persistence 12.605), so the wall survives
correct scoring. Readout and input both help (ridge
halves Lorenz MSE, bump cuts it by two thirds,
combined C4 3.795 beats the ESN's 4.308), but D8 fails:
the frozen ridge (C2f 10.818) beats the trained ridge
(C2 11.403), so delta adds no continuous information
beyond what the frozen reservoir already holds. The
r18-monitor cron is removed; the analysis is delivered.
Per the Nov 27 00:00 PT redirection, the hierarchy
build and the data mapping come next; the generative
test (experiment D) follows the round-17/18 verdicts.
Round 17 (top-down clamp, experiment B) COMPLETE Nov 27,
2026, 00:17 PT (record section 53; analysis
r17/r17_analysis.md, verified by independent
recomputation from the raw JSONL: all condition means
match to 3 decimals, paired diffs to 4 decimals, 0
duplicate keys, 200 lines). Experiment B FAILS on lag-1:
C1 FAIL (cut degrades omit_m2 by +0.005, needs >= 0.10),
C2 FAIL (shuffle -0.001), C3 FAIL (p1_cos 0.998, needs
< 0.90), C4 FAIL (wire-off +0.004, CI includes 0), C5
guard PASS (0.228 vs 0.229). Lag-2: only C1 passes
(+0.004, a 0.4-point effect). 2/9 bars pass. The wire
is present, small, and inert: it moves module 1's code
by 0.2 percent and the final error magnitude by 0.002.
The prediction lives in module 2's state and is never
fed back into module 1. Strong fixed feedback is worse
(gain 1.0 drops validation to 0.182), so a learned
feedback wire is the untested option, now a hierarchy
question. Two of the three research discriminators are
decided: omission passes, clamp fails. The r17-monitor
cron is removed; the analysis is delivered. Round 19
(hierarchy, experiment E) DESIGNED Nov 27, 2026, 00:19 PT
(record section 54; protocol PROTOCOL-R19-HIERARCHY.md
frozen; model rpc_hier.py, assumption A28; rpc_v2.py
untouched). Three levels, one delta-trained distal wire
and one fixed random feedback wire per level, leak rates
1.0 / 0.3 / 0.1. New two-timescale task: 4 blocks of 12
steps, context token per block, two motifs per context
with shared starters. Seeds 282..301, val picks levels
{2,3} and feedback gains on the hier task only. Bars
H1-H9: hierarchy beats flat on late m1 by 0.20 or more
and reaches 0.45, plus the old tasks (planted m1 0.80)
and mechanism checks (cut, frozen). Smoke passes on the
VM (nested init exact, flat reproduction within 0.05,
ladder holds: lag-11 context decode L1 0.70, L2 0.97,
L3 0.93). Honest dent in the premise, recorded in
section 54.10: level 1 already holds context at 0.70,
so the flat model may beat the 0.25 late-m1 guess and
compress the gap H2 demands. The main agent re-runs the
VM smoke independently, then launches on Beast (6 val
workers, about 5 h wall); r19-monitor drives val ->
winner -> test -> analysis at 480 lines (420 if the
winner is 2 levels). Monitor patch (main agent, Nov 27,
before launch): launch_val and launch_test now launch
detached (start_new_session, no wait) and write their
marker at launch time, because the test phase is about
5 h wall and a blocking launch would outlive the cron run
and stall the round. LAUNCHED Nov 27, 2026, 00:25 PT:
Beast smoke passed with VM-identical numbers; 6 val
workers running (one per config, seeds 282-286, hier
task); val_launched marker written; r19-monitor cron
created (every 30 min). VAL WINNER Nov 27, 2026, 00:44 PT
(record section 54.12): L2g010, 2 levels, fb_gain1 0.1,
mean val acc 0.268. The 3-level config L3g010_g010 ties
at 0.268 and loses on the tie-break, so the third level
adds nothing on validation and H4 is not judged. Test
phase running: 6 detached workers, target 420 lines,
about 5 h wall. Round 17/18 critiques LAUNCHED
Nov 27, 2026, 00:34 PT (record section 55; brief
CRITIQUE-BRIEF-R1718.md, 10 checks, numbers verified
against sections 52/53): Kimi PID 534794 and AGY Gemini
PID 534795 on Beast, pack synced to the review dir,
Opus not attempted (quota-blocked until about Dec 1;
round-14 retry scheduled Dec 2). Collection job
r1718-critique-check watches every 30 min.

Round 19 VERDICT Nov 27, 2026, 01:15 PT (record section
56; analysis r19/r19_analysis.md, 420/420 test lines, 20
seeds, no Traceback, headline means independently
recomputed to 3 decimals). 1 of 9 bars pass (H9 only).
The hierarchy does not win: hier acc cand_on 0.248 vs
flat 0.255; planted1 m1 0.561 vs flat 0.682; sine acc
0.679 vs flat 0.874. Feedback is inert at test time
(H8 +0.002, shuffle gap +0.001). The flat model beats
the hierarchy on every task and beats Markov-1 on the
hier task only by losing less (0.255 vs 0.301). The
r19-monitor cron is removed; its job is done.

Rounds 17/18 critiques LANDED Nov 27, 2026, about 01:20 PT (record section
57; r1718-kimi-critique.md and r1718-agy-critique.md, md5-verified from
Beast). Both reviewers agree experiment B fails for the tested wire, but the
general "no causal role" claim is too strong: the fixed random additive wire
at gain 0.1 was the wrong geometry, and AGY's aligned subtractive control
(-W_prox.T, gain 1.0, seed 241) reaches omit_m2 1.000 and clears C3. The
wire is live in the code (verified); the lag-2 C1 pass is an analyzer
artifact (the 0.10 floor was dropped for lag-2), so the honest tally is 1
of 9. The continuous wall splits: AGY's Convention-A revalidation of
round 15's own grid dissolves D3/D4, reproduced exactly here (Lorenz
discrete-bin A14 config acc 0.4285, MSE 5.07 vs persistence 13.05, 5
seeds), and D8's frozen ridge win is a regularization artifact (seed 262:
drive norms 1.025 trained vs 5.651 frozen; standardized ridge ties at
13.976). D2 passes only under the z 1.96 rule (CI lower bound +0.0003)
and fails under exact t. Next experiment: reviewers disagree (Kimi:
precision-weighted causal feedback; AGY: fix the wire, then generative
rollout), but both agree the wire must be fixed first with a state-effect
gate. The r1718-critique-check cron is removed; its job is done.

Round 20 LAUNCHED Nov 27, 2026, 02:12 PT (record section 58;
protocol PROTOCOL-R20-ALIGNED.md): aligned subtractive feedback
clamp (revised experiment B) plus multi-step omission rollout
(experiment D), seeds 302-321. A29 in rpc_v2.py: fb_wire
aligned/learned/random, blank token, corrected drive ratio.
The aligned wire reproduces AGY's seed-241 control exactly
(normal_acc 0.2330, omit_m2 1.000, p1_cos 0.8565); the learned
wire uses a local composite delta rule, NOT AGY's literal
formula, which was measured to collapse (normal_acc 0.061 at
gain 1.0). Smoke 8/8 PASS on the VM and on Beast with identical
numbers (gate p1_cos 0.8033, on-cut gap +0.0167). 9 validation
workers running (3 wire types x 3 gains, planted1-occ, seeds
302-306); r20-monitor cron created (every 30 min). Honest
risks: the B1 floor (0.05) sits above both measured single-seed
gaps, so the round may report "causal but small"; the sine
rollout decays to chance late (0.030 vs the 0.30 bar), so D4
may fail. Beast is loaded by the connectome Allen binwidth
runs, so wall time may stretch past the about 1 hour estimate.
Continuous re-validation is split into round 21. VAL WINNERS
Nov 27, 2026, 02:44 PT (record section 58.2): aligned gain 1.0
(gap +0.017, p1_cos 0.805), learned gain 1.0 (gap -0.002, p1_cos
0.751), random gain 1.0 (gap -0.088, p1_cos 0.754); all gates
PASS. The gate does real work: gains 0.1 and 0.3 fail it for
every wire (p1_cos 0.93 to 0.99), so only gain 1.0 counts as a
live wire. Random at gain 1.0 actively hurts (val_acc 0.190 vs
aligned 0.236). Round 20 VERDICT Nov 27, 2026, 04:14 PT
(record section 59): B' FAILS and D FAILS, 9 of 28 bars pass;
"causal but small" applies on lag-1 only (B1 +0.014, B2 +0.033);
B5 fails both lags with the wrong sign (no explaining away);
D is a clean negative (sine rollout on 0.147 < cut 0.184 <
wire-off 0.223); learned wire fails everywhere; the wire costs
23 points of one-step sine accuracy. Thesis-level predictive-
coding causality NOT established.

Round 21 LAUNCHED Nov 27, 2026, 05:26 PT (record section 60;
protocol PROTOCOL-R21-CONTINUOUS.md): continuous
re-validation, Convention-A Lorenz/var-sine plus NARMA-10 and
Mackey-Glass, standardized ridge probes, aligned wire tested
on/off as a factor (wire OFF primary, wire ON a secondary arm
on the token input only). VM smoke re-run independently by the
main agent: 785 s, exit 0, zero failures, every number
reproduces the design exactly (Lorenz 0.4285 / 5.073; var-sine
0.3230 / 0.0370, diff -0.0000 vs AGY; D8 standardized ridge
ties 13.976). Beast smoke PASSED (lorenz val 10 lines, test
seed 327 7 rows). 4 token-val workers running (seeds 322-326);
r21-monitor cron created (every 30 min). Beast load 24.20 at
launch (connectome Allen runs); give the round a full day.
Verdict in section 61. TOKEN VAL WINNERS Nov 27, 2026, 06:44 PT
(record section 60.2): lorenz G4 (alpha 1.0, smooth 3.0,
eta_d 0.2, val MSE 4.6268); varsine G4 (same config, 0.0240);
narma10 G5 (alpha 1.0, smooth 2.0, eta_d 0.02, 0.0109);
mackeyglass G1 (alpha 1.0, smooth 2.0, eta_d 0.2, 0.0021).
Lorenz and varsine winners differ from the preregistered
primaries, so the test runs conditional primary arms and
the target is 640 lines (7 rows per task-seed plus 2
conditional rows per seed on lorenz and varsine). BUMP VAL
WINNERS Nov 27, 2026, 07:15 PT: lorenz sigma 2.0 (4.3965
vs 3.0 at 4.6281); varsine sigma 3.0 (0.0236 vs 2.0 at
0.0390); narma10 sigma 3.0 (0.0103 vs 2.0 at 0.0108);
mackeyglass sigma 2.0 (0.0022 vs 3.0 at 0.0025). Test phase
running: 8 workers, seeds 327-346, 0 lines at launch.
Beast load about 15.7. Verdict in section 61.

Round 20 COMPLETE Nov 27, 2026, 04:14 PST (record section
59, analysis r20/r20_analysis.md). Aligned-feedback clamp
(revised experiment B) plus multi-step omission rollout
(experiment D), 20 fresh seeds 302-321, 660 test lines, 0
errors, analysis verified by independent recomputation
(r20/verify_r20.py, all headline means match to 3
decimals). Val winners (planted1-occ, seeds 302-306): all
three wire types at fb_gain 1.0, all state gates PASS
(aligned gap +0.017, p1_cos 0.805; learned -0.002, 0.751;
random -0.088, 0.754). Bar table: 9/28 pass. B' FAILS
(B1/B2/B5/B9 lag-1, B1/B2/B4/B5/B8/B9 lag-2). D FAILS
(D1-D7 all fail, all signs negative: sine rollout_acc on
0.147 < cut 0.184 < wire-off 0.223; late rollout 0.051 vs
bar 0.30). Pre-registered "causal but small" outcome
APPLIES on lag-1 only: B1 +0.014 CI [0.005,0.022], B2
+0.033 CI [0.022,0.044], both real (lower CI > 0) but below
the +0.05 floor. B8 lag-1 passes (+0.071): aligned beats
random by more than the floor, so direction matters. B6
passes both lags (cos_p1_omit_m2 on-cut -0.391/-0.381), so
the wire rotates module-1 state as predicted, but B5 fails
both lags with the wrong sign (err_mag_last up +0.0138/
+0.0181 with the wire on): no explaining away. Learned
wire fails everywhere (B9 +0.004 lag-1 null, -0.027 lag-2;
D7 -0.035). The wire costs 23 points of one-step sine
accuracy (0.666 vs wire-off 0.892) and nothing on planted.
Thesis-level rule (B' AND D) not met: predictive-coding
causality is NOT established. Next: round 21 (continuous
re-validation), sketched in PROTOCOL-R20-ALIGNED.md; open
question is whether to carry the aligned wire at gain 1.0
given the sine cost and the negative rollout sign.

Round 21 COMPLETE Nov 27, 2026, 09:14 PST (record section
61, analysis r21/r21_analysis.md). Continuous
re-validation, Convention-A Lorenz/var-sine plus NARMA-10
and Mackey-Glass, standardized ridge probes, aligned wire
tested on/off as a factor, 20 fresh seeds 327-346, 640 test
lines, 0 duplicates, 0 errors, analysis verified by
independent recomputation (r21/verify_r21.py, all headline
means match to 3 decimals). Val winners (seeds 322-326):
lorenz G4 (alpha 1.0, smooth 3.0, eta_d 0.2, 4.6268);
varsine G4 (same config, 0.0240); narma10 G5 (alpha 1.0,
smooth 2.0, eta_d 0.02, 0.0109); mackeyglass G1 (alpha 1.0,
smooth 2.0, eta_d 0.2, 0.0021). Bump sigmas: lorenz 2.0,
varsine 3.0, narma10 3.0, mackeyglass 2.0. Bar table: 26/54
pass. WALL FALLS for the re-validation tasks: W1 +8.3039
CI [7.5853,9.0225], W2 +0.0290 CI [0.0219,0.0360], W7
+232.5637 CI [188.6496,276.4778], W8 +0.9033 CI
[0.8092,0.9974], all pass; the preregistered primaries
pass too (W9 +7.6045, W10 +0.0187), so val selection is
not doing the work. This does NOT establish predictive
coding (round 20 failed B' and D); it establishes only
that the signed-error rule learns continuous trajectories.
The wall survives on the new tasks: prototype does not
beat persistence on NARMA-10 (N1 -0.0026) or Mackey-Glass
(N2 -0.0008), and beats Markov-2 on NARMA-10 only (N7
+0.0011; N8 -0.0011). R1 fails on all 4 tasks BY TIE, not
by a frozen win, so protocol section 8 item 5 does not
trigger; the ridge probe shows no learning signal while
the prototype shows a large one, matching the round-14
alignment story. R4 fails everywhere (ESN scalar-std ridge
still wins). F1: wire NULL on lorenz/NARMA-10, HURTS on
var-sine (-0.0178) and Mackey-Glass (-0.0015), while F2
passes everywhere (fb_drive_ratio 0.31 to 0.34): the wire
is live and a net tax on continuous one-step prediction.
The r21-monitor cron is removed; its job is done. Next
decision is Reza's: a multi-step continuous rollout
(h > 1) as the next round, or close the continuous
chapter on the one-step wall-fall plus the round-20
negatives.

Round 22 LAUNCHED Nov 27, 2026, 10:14 PT (record section
62; protocol PROTOCOL-R22-PRECISION.md): the precision
test (experiment C), the last unrun discriminator.
Precision is test-time input reliability (noise on the
probe token); training is always clean; the NLMS
denominator is explicitly NOT precision. Design probe
(seed 347): the interaction exists wire-on but persists
wire-cut at 30-45%, so the wire amplifies it 2-3x but
does not create it; Kimi's strict bar (cut removes it)
is predicted to FAIL. VM smoke re-run independently by
the main agent: 585 s, exit 0, all checks pass, numbers
identical (sine interaction 0.3622 rel_e, frozen
-0.0088). Beast smoke PASSED (noise val 5 lines,
interaction val 8 lines, test 8 lines). 3 noise-val
workers running (seeds 347-351); r22-monitor cron
created (every 30 min). Beast load 10.13 at launch.
NOISE VAL Nov 27, 2026, 10:44 PT: c_low planted1 0.25,
planted2 0.35, sine 0.25 (75/75 lines, no duplicates).
Interaction val running. INTERACTION VAL Nov 27, 2026,
11:14 PT: floors sine P1 0.079, planted1 P1 0.027,
planted2 P1 0.020 (minimum). Val interaction sine 0.158,
planted1 0.054, planted2 0.006 (CI covers 0, P1 expected
to fail). Wire-cut interaction persists 60% on planted1
and 50% on sine. Test running: 6 workers, seeds 352-371.
Verdict in section 63. The h>1 rollout is queued after
round 22. Scope note (Nov 27, alignment synthesis): the
no-stop order covers parallel work on already-designed,
preregistered rounds only, never a new design; round 22
continues to completion, no new design until Reza lands
and confirms scope. Round 22 COMPLETE Nov 27, 2026, 12:15
PT (record section 63; analysis r22/r22_analysis.md,
verified by independent recomputation from the raw JSONL:
all headline means match to 3 decimals, 0 mismatches, all
seven floors per task re-derived from the val files, 0
mismatches, 480 unique test records from 240 nets, no
Traceback). Verdict: C STRONG PASS, 24 of 34 bars. Sine P1 0.164, P2 0.242, P3
0.249, P4 pass; planted1 P1 0.042 pass; sine W1 0.081 and W2
0.096 pass; W5 gate passes on sine. But the strict Kimi bar
fails: the interaction persists wire-cut at 51% of wire-on
on sine (0.084 vs 0.164) and 42% on planted1, so the wire
amplifies the interaction about 2x and does not create it,
and the top-down precision reading is NOT strengthened.
Worse on sine: wire-off interaction (0.225) EXCEEDS wire-on
(0.164, W4 -0.061 fail), so W1 holds only against the cut
control, not against no wire; and the wire costs 28% of
normal accuracy on sine (W6 ratio 0.720 fail; planted
ratios 1.07/1.03 pass). Lag-2 replication fails: magnitude
interaction -0.005 (P1 fail, P3 fail) while specificity
stays 0.311 (P2 pass), so the interaction survives only
1-step expectation. S1/S2 pass on both planted tasks
wire-off (no p1 suppression by expectation); the mapping
report's falsifiable divergence is NOT falsified. W2 passes
but the shuffle magnitude confound stands; the cut is the
primary control and it passes. Thesis-level: no outcome
establishes predictive coding; C passes only as
precision-like behaviour from input noise plus error
selection, wire as amplifier. Discriminators now: omission
passes, clamp B fails, aligned clamp B' fails, generative D
fails, precision C passes with the wire as amplifier. Next
queued: the multi-step (h > 1) continuous rollout; no new
design until Reza lands and confirms scope. The r22-monitor
cron is removed; the analysis is delivered.
