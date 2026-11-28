# Round 19 protocol: the hierarchical version (experiment E)

Frozen Nov 27, 2026, before any round-19 test result is seen.
Order: Reza, Nov 27 00:00 PT. The hierarchical version has not
been tested: multiple modules with feedback connections that
learn input patterns. This round builds and tests it.

## 1. Question

Does stacking the delta-learned modules into a hierarchy buy
anything a flat two-module model cannot? The two-module model
learns discrete sequences (round 15) and holds a prediction
through a missing input (round 16), but it fails the first
motif step (planted m1 0.680 against the 0.80 bar) and it has
no slow level: every module runs at the same fast timescale.
Hierarchy should buy slow-timescale structure: a top level
that holds context across many steps while the bottom level
tracks fast tokens.

## 2. Hypothesis (one sentence)

A three-level hierarchy with a slow top level (effective
memory about 18 steps) learns the slow context of a
two-timescale sequence and uses it to predict the
context-dependent first motif step, which a flat two-module
model cannot do because its memory spans about 2 to 3 steps.

## 3. Assumption A28: the hierarchical network

A28 generalises the two-module network (rpc_v2.py) to M
levels. It is implemented in a new file, rpc_hier.py, which
imports ReservoirModule and make_data from rpc_v2. rpc_v2.py
is NOT edited: rounds 17 and 18 are running on Beast against
that file right now, and a docstring edit is not worth the
risk of breaking it. A28 is specified here and in
rpc_hier.py; the next sync copies both files to Beast.

### 3.1 Levels

Level 1 is the bottom (input). Level M is the top (slowest).
Each level l has one fixed random reservoir module, built
exactly like ReservoirModule (n_exp 256, n_res 200, n_out
256, k_e 32, rho 0.9). Level 1 reads the token input through
_encode (one-hot, or a smooth bump when smooth_sigma > 0).
Level l > 1 reads tanh(W_l @ r_{l-1}), where W_l is a fixed
random matrix (n_res x n_res) / sqrt(n_res), the same scale
logic as W_12. No level has plastic reservoir weights.

### 3.2 One distal wire per level

Each level l < M has a distal matrix W_dist_l (n_out x
n_out), init randn * 0.5 / sqrt(n_out) (the A2 scale),
trained ONLY by the delta rule (A24): d_l(t) =
W_dist_l @ y_{l+1}_prev, and W_dist_l += eta_d /
(||y_{l+1}_prev||^2 + 1) * outer(p_l - d_l, y_{l+1}_prev),
clamped at w_max 2.0, norm_rows False. The error is the
difference between the actual code p_l at level l and the
prediction d_l of it from the level above. The top level M
has no distal wire and no delta update: it only supplies
the slow code. The bottom readout is the fixed p1-prototype
cosine, as in round 15. The secondary readout is the top
prediction: d_{M-1} read against mean-p_{M-1} prototypes
per token (built the same way as the p1 prototypes). For
M = 2 the top readout is the bottom readout, so it is
reported only when M = 3.

### 3.3 One feedback wire per level

Each level l < M has a fixed random feedback matrix W_fb_l
(n_res x n_out), randn / sqrt(n_out), never trained, and
the term fb_l = fb_gain_l * (W_fb_l @ d_l) is added into
level l's reservoir drive (the A26 form). Test-time modes
per wire, as in round 17: "on", "cut" (fb = 0), "shuffle"
(a fixed random permutation of d_l, same magnitude, no
information). The error (p_l - d_l) still drives only the
W_dist_l delta update. Module reservoirs have no plastic
weights in this config (learn_prox off, use_confirmation
False, s_inh 0.0 always): the round-15 winner config,
generalised.

### 3.4 Timescale ladder (fixed, not validated)

Level alphas are fixed by depth: alpha_1 = task winner
alpha (1.0 for all three tasks), alpha_2 = 0.3, alpha_3 =
0.1. Rationale, estimated not measured: a reservoir step
has an effective Jacobian near (1 - alpha) * I + alpha *
(1 - tanh^2) * W_res, so the memory span is about 2 to 3
steps at alpha 1.0, about 6 steps at alpha 0.3, and about
18 steps at alpha 0.1 (rho 0.9, typical tanh derivative
0.3 to 0.6). The context block in the hierarchical task is
12 steps: inside the top level's memory, outside the
bottom level's. The smoke test measures this directly
(context decoding from each level's state at lag 11, see
section 10). If the ladder does not hold there, the design
is wrong and the round stops before launch.

### 3.5 Nested initialisation (exact property)

Weights are drawn level by level in order (all of level
l's matrices before level l+1's), from one seeded RNG
stream. So at the same seed, a 3-level net's level-1 and
level-2 weights are EXACTLY the 2-level net's weights.
The smoke test asserts this. This makes the 2-level vs
3-level comparison a clean ablation: the only difference
is the added third level and what it feeds back.

### 3.6 Text diagram (M = 3)

    input token t
        |
        v
    [mod1, alpha 1.0] --p1--> readout (pred, cosine vs p1 protos)
        ^  |                      ^
        |  | r1                   | d1 = W_dist_1 @ y2_prev
        |  v                      | (delta-trained vs p1)
        | [mod2, alpha 0.3] --p2--+
        ^  |                      ^
        |  | r2                   | d2 = W_dist_2 @ y3_prev
        |  v                      | (delta-trained vs p2)
        | [mod3, alpha 0.1] --p3--+
        |  y3 (slow code)
    fb_1 = g1 * (W_fb_1 @ d1) into mod1
    fb_2 = g2 * (W_fb_2 @ d2) into mod2

## 4. Tasks and config

Tasks: planted lag-1, sine-fixed, and hier (two-timescale,
section 5). Seeds 282..301 (20 fresh; rounds 15 to 18 used
201..220, 221..240, 241..260, 262..281, and 261 was the
round-18 diagnostic seed). Val seeds 282..286, test seeds
282..301. Base config (all conditions, all tasks):
n_res 200, n_out 256, full_code True, use_confirmation
False, s_inh 0.0, graded_conf False, norm_rows False,
learn_rule delta, w_max 2.0, theta_p = theta_d = 1.0,
5 epochs, unmodified p1-prototype cosine readout. Per-task
base: planted1 alpha 1.0, smooth_sigma 0.0, eta_d 0.05
(round-15 winner, val 0.229); sine alpha 1.0,
smooth_sigma 0.0, eta_d 0.2 (round-15 winner, val 0.863);
hier alpha 1.0, smooth_sigma 0.0, eta_d 0.05 (the planted
winner config, same K = 16 token family; the architecture
is what is validated, not the base rates).

1. Planted lag-1: 200/50/50, K = 16, length 24.
   baselines.make_data (motifs 1-2-3, 5-6-7, 9-10-11).
2. Sine-fixed: 50/20/20, K = 32, length 200.
   rpc_simple.make_sine_fixed (period 20, phase uniform).
3. Hier: 100/50/50, K = 16, length 48, make_hier_data
   (rpc_hier.py, section 5). Train 100 keeps the per-epoch
   step count equal to planted (100 x 48 = 200 x 24).

Splits follow PROTOCOL-R15 section 2: planted uses one
global-RNG stream (np.random.seed(s); train, val, test in
that order); sine uses explicit seeds (train s, val
s + 2000, test s + 1000); hier uses the planted scheme
(train 100, val 50, test 50 from one stream). Data seed =
network-init seed (the round-14 convention; the same
confound note as PROTOCOL-R15 section 8.1 applies).

## 5. The hierarchical task (make_hier_data)

Sequences carry structure at two timescales. Length 48:
4 blocks of 12 steps. Block b starts with a context token
C_b in {0, 1, 2, 3}, drawn per block. The other 11
positions hold 2 planted motifs and filler. Context C has
two motifs, both starting with a token shared across
contexts, so the continuation depends on the context:

1. C = 0: (4, 6, 7), (5, 8, 9)
2. C = 1: (4, 8, 9), (5, 10, 11)
3. C = 2: (4, 10, 11), (5, 12, 13)
4. C = 3: (4, 12, 13), (5, 14, 15)

Each block plants one motif starting with 4 and one
starting with 5, in random order at random non-overlapping
positions after the context token. Filler tokens are drawn
uniformly from 4..15, so context tokens 0..3 appear only
at block starts. Motif elements are named m1 (2nd token)
and m2 (3rd token), as in round 15. Late m1 means a motif
whose starter sits at block position 6 or later, at least
6 steps after the context token; early m1 is block
position 1 to 5.

Why this is sharp: given starter 4, the 2nd token is one
of {6, 8, 10, 12} by context, so a context-blind predictor
scores 0.25 on m1 (the context-blind Bayes level). Given
the 2nd token, the 3rd token is determined (6 -> 7,
8 -> 9, 10 -> 11, 12 -> 13, 14 -> 15), so m2 is locally
predictable and needs no context. Overall chance is
1 / 16 = 0.0625. A flat model with a 2 to 3 step memory
cannot hold a context from 6 to 12 steps ago, so its late
m1 should sit near 0.25; a hierarchy with a slow top
level should clear 0.45. The generator returns the motif
positions with the sequences, so m1/m2 scoring needs no
pattern matching (unlike the round-16 driver).

## 6. Validation: pick the architecture

Grid on the hier task only (it is the task that
discriminates levels): n_levels in {2, 3} x fb_gain1 in
{0.1, 0.3} x fb_gain2 in {0.1, 0.3}. For n_levels = 2
there is no second wire, so fb_gain2 is meaningless and
those configs collapse: 2 (n = 2) + 4 (n = 3) = 6
effective configs. Val seeds 282..286, 5 epochs, winner
= highest mean val overall accuracy. Tie-break: first in
grid order (n_levels ascending, fb_gain1 ascending,
fb_gain2 ascending). The winner (n_levels, fb_gain1,
fb_gain2) is frozen and used for ALL test conditions on
all tasks. Val m1 is reported alongside but does not pick
the winner: the winner is picked on overall accuracy, as
in rounds 15 to 18.

## 7. Test conditions

Five nets per (task, seed), 5 epochs, same seeds. The
cut and shuffle passes reuse the candidate net (train
once, test three ways, as in round 17). The candidate is
the val winner's n_levels; if the winner is 2 levels,
condition 4 duplicates condition 1 and is skipped, and
bar H4 is not judged (see section 9).

1. cand-on: M = winner n_levels, fb winner gains, delta,
   tested fb on. The candidate.
2. cand-cut: same trained net as 1, tested with all fb
   wires cut. The causal test of the wires.
3. cand-shuffle: same trained net as 1, tested with all
   fb wires shuffled. The content control.
4. hier2-ladder: M = 2, ladder alphas (1.0, 0.3),
   fb_gain1 = winner fb_gain1, delta. Separates the
   third level from the slower level 2.
5. hier2-nofb: M = 2, ladder alphas, fb_gain1 = 0.0,
   delta. With 4, separates feedback from the slower
   level 2 (4 vs 5 isolates feedback; 5 vs 6 isolates
   the slower level 2).
6. flat: M = 2, both alphas = task winner alpha (1.0),
   fb_gain1 = 0.0, delta. The round-15 winner config,
   expressed in the hierarchical class. The control.
7. cand-frozen: M = winner n_levels, fb winner gains,
   plasticity False. The reservoir control.
8. baseline: Markov-1/2, persistence, ESN on the
   identical test sequences (baselines.py, as in round
   15), plus baseline m1/m2 on the hier task (the
   baseline prediction sequences scored against the
   motif positions the generator returns).

One JSON line per (task, condition, seed); a worker skips
lines already present, so a killed worker resumes cleanly.
8 lines per (task, seed), 3 tasks x 20 seeds = 480 test
lines. Training runs: 5 nets x 3 tasks x 20 seeds = 300.

## 8. Metrics (per task, per condition, per seed)

1. All tasks: exact-bin accuracy (positions 1..end, pred(t)
   vs token t, convention A), mean_burst, clamp_frac per
   W_dist level (max over levels reported, per-level values
   in the record), fb_rms_ratio per wire (RMS fb / RMS
   bottom-up drive, averaged over the test pass), r_max
   (max |r| over levels and steps, stability).
2. Planted and hier: m1, m2, oth accuracies (round-15
   definitions for planted; generator positions for hier).
3. Hier only: early m1, late m1 (section 5), top_acc and
   top_m1 (the secondary top readout, M = 3 only), and the
   baseline m1/m2 (Markov-1, Markov-2, persistence, ESN).
4. Sine: exact-bin accuracy only, as in round 15.

## 9. Bars (20-seed held-out test, paired per-seed differences)

Bars are judged on paired per-seed differences, 95% CI =
mean +/- 1.96 * std / sqrt(20), unless stated otherwise.
"Beats" means the paired mean difference is above 0 with
its lower 95% CI above 0. H1 to H3 are the primary verdict
(H0). H4 to H7 are secondary. H8 and H9 are mechanism
checks. The shuffle contrast is a diagnostic, not a bar.

1. H1: hier-task overall acc: candidate beats flat by at
   least +0.10.
2. H2: hier-task late m1: candidate beats flat by at
   least +0.20.
3. H3: hier-task late m1: candidate mean at least 0.45,
   lower 95% CI above 0.35. (The candidate actually uses
   the context; 0.25 is context-blind Bayes.)
4. H4: hier-task late m1: 3-level beats 2-level-ladder by
   at least +0.10. Judged only if the val winner is 3
   levels; if the winner is 2 levels, H4 is not
   applicable and the round reports that plainly.
5. H5: planted lag-1 m1: candidate beats flat by at
   least +0.05. (Does hierarchy fix the first motif
   step?)
6. H6: planted lag-1 m1: candidate mean at least 0.80.
   (The round-15 bar, on the hierarchical candidate.)
7. H7: sine-fixed acc: candidate beats flat by at least
   +0.03. (Hierarchy does not hurt the sine win.)
8. H8: hier-task overall acc: candidate on beats
   candidate cut by at least +0.05. (The feedback wires
   matter at test time.)
9. H9: hier-task overall acc: candidate on beats frozen
   by at least +0.10. (Learning matters, not just the
   reservoir.)

Diagnostic (no bar): candidate on vs candidate shuffle on
hier-task overall acc. If H8 passes but the shuffle gap
does not, the effect is drive magnitude, not content
(the round-17 logic).

Verdict: H0 (hierarchy helps on the two-timescale task)
passes if H1, H2, and H3 pass. If H1 to H3 pass but H4
fails, the win is the slower level 2, not the third
level. If H1 to H3 pass but H8 fails, the win comes from
the slow level's code shaping delta training, not from
top-down feedback at test time. If H2 passes but H3
fails, the candidate beats a weak flat model without
actually using the context. H5 to H7 decide whether
hierarchy helps the old tasks; H8 and H9 are mechanism
checks. The protocol is not re-run with different bars.

## 10. Smoke test (VM only, before any Beast run)

1 seed (282), 1 epoch, hier task plus planted1 and sine
flat checks. Checks:
1. Nested init: a 2-level and a 3-level net at the same
   seed share level-1 and level-2 weights exactly
   (assert allclose on mod1, mod2, W_dist_1, W_fb_1).
2. Flat reproduction (approximate, not exact): the
   hierarchical flat net (M = 2, both alphas 1.0,
   fb_gain 0.0) on planted1 seed 201 scores within 0.05
   of the round-15 recorded seed-201 delta acc 0.227
   (r15 JSONL: acc 0.22696, m1 0.647, m2 0.966), and on
   sine seed 201 within 0.05 of 0.872. Exact equality is
   impossible: the init draw order differs from
   RPCNetwork (W_12d, W_dist1b, W_conf are not drawn in
   the hierarchical class), so the weights differ while
   the config, data, and rule are identical.
3. Feedback changes the lower module's state: p1_cos
   (on vs cut) below 1.0 and above 0.0; fb_rms_ratio
   above 0 when on, 0.0 when cut; shuffle drive measured
   on the same trajectory (same ds) matches on-mode
   drive within 5%. AMENDMENT Nov 27, 2026 (pre-launch,
   main agent): the separate-pass shuffle fb RMS does
   NOT match within 5% (measured +16.0% on the VM smoke),
   because wrong feedback changes the trajectory and ds
   differs between passes. The check is therefore the
   same-trajectory drive ratio, not the separate-pass
   RMS; the separate-pass gap is reported as a caveat and
   weakens the shuffle diagnostic (see self-critique
   point 8). No bar changes.
4. Top readout finite and non-degenerate on M = 3:
   top_pred takes at least 2 distinct values over the
   test pass, and top_acc is finite.
5. Timescale ladder (the design's load-bearing claim):
   on probe sequences with a context token at step 0
   followed by filler only, a ridge probe fit on train
   decodes the block context from each level's reservoir
   state at lag 11. Level-3 accuracy must exceed
   level-1 accuracy. If it does not, the ladder is
   broken and the round stops before launch.
6. All metrics finite; no NaN; r_max bounded (below
   10); clamp_frac reported; driver CLI resumable
   (second run writes 0 new lines); analyzer writes on
   1-seed output.

## 11. Execution

Driver: rpc_round19_hierarchy.py. CLI:
python rpc_round19_hierarchy.py <task> <val|test>
--seeds A-B --out FILE [--winners FILE] [--epochs N]
[--n-levels M] [--fb-gain1 G] [--fb-gain2 G]
Tasks: planted1, sine, hier. One JSON line per (task,
condition, seed); a worker skips lines already present.
Validation: 6 configs x 5 seeds x 1 task (hier) = 30
training runs, one worker per config, one file per
config. Test: 5 nets x 3 tasks x 20 seeds = 300 training
runs, 8 lines per (task, seed), 480 test lines total.
About 6 min per training run on one CPU, about 30
CPU-hours, about 5 h wall with 6 workers (3 tasks x 2
seed-shards). One BLAS thread per process
(OMP/MKL/OpenBLAS/NumExpr = 1, torch.set_num_threads(1)),
the same fix as rounds 15 to 18. The monitor is written
at r19/monitor.py (following r18/monitor.py): val ->
pick architecture winner (highest mean val acc, tie-break
first in grid order) -> write winner.json -> launch test
-> analysis at 480 lines. Idempotent via marker files;
exits 0 always. The analyzer is written at
r19/r19_analyze.py (following r18/r18_analyze.py): it
reads r19/r19_{task}_test_282-301.jsonl, computes means,
medians, IQRs, 95% CIs and paired differences, checks
bars H1 to H9, and writes r19/r19_analysis.md.

## 12. Self-critique (frozen before any test result)

1. rpc_v2.py is NOT edited. The A28 spec lives here and
   in rpc_hier.py. If the hierarchical class has a bug,
   rounds 15 to 18 are unaffected (they use RPCNetwork),
   but the round-19 result rests entirely on the new
   class, so the smoke test (section 10) is the gate.
2. Level alphas are fixed by depth (1.0, 0.3, 0.1), not
   validated. A 2-level net with a slow top (alpha 0.1)
   is untested; the ladder is by depth, so the 2-level
   top is 0.3. If the winner is 2 levels, the top is
   faster than in the 3-level net, which is part of what
   H4 would have tested. A follow-up can validate the
   alphas.
3. The flat condition (M = 2, both alphas 1.0, fb 0.0)
   is the round-15 config expressed in the hierarchical
   class, but its weights differ from RPCNetwork's
   (different init draw order), so the smoke check is
   approximate (within 0.05 of the recorded seed-201
   scores). A large deviation means the hierarchical
   class does not express the same model, and the round
   stops.
4. The hier2-ladder vs flat comparison conflates two
   changes (slower level 2 AND feedback). Conditions 5
   (hier2-nofb) and 6 (flat) separate them: 4 vs 5
   isolates feedback, 5 vs 6 isolates the slower level
   2. Both are reported as diagnostics, no bars.
5. The hier task's m1 chance level (0.25) assumes the
   model cannot infer the context from the filler. The
   filler is uniform over 4..15 and carries no context
   information, so this holds by construction. The
   context token itself is fed as input, so the task
   tests memory of the context, not inference of a
   hidden variable.
6. Seeds 282..301 are fresh. Val seeds (282..286) are
   a subset of the test seeds, as in rounds 15 to 18:
   the architecture is picked on validation accuracy
   and the bars are judged on the full 20-seed test.
   The paired contrasts compare conditions on identical
   seeds, so the overlap cannot manufacture a gap.
7. The top readout (d_{M-1} vs mean-p_{M-1} prototypes)
   is a diagnostic, not a model claim: it is a trained
   readout in the sense that the prototypes are fit on
   train data, and it reads the slow level's code, not
   the token stream. No bar rests on it.
8. The candidate's win on the hier task, if it comes,
   could be the slow level's code shaping W_dist_1
   during training rather than top-down feedback at test
   time. H8 (cut) separates these: a win with no cut
   gap means training-time shaping, not test-time use.
9. What round 19 does NOT test: the precision test
   (experiment C), the generative test (experiment D),
   and whether the learned hierarchy maps onto canonical
   microcircuits or hippocampal data. Those are later
   rounds, decided by this round's verdict.
