# Draft section 54 for simulation-results.md (design only, NOT launched)

## 54. Round 19: hierarchy (experiment E) -- DESIGN ONLY, NOT LAUNCHED

Status: protocol frozen Nov 27, 2026 (PROTOCOL-R19-HIERARCHY.md).
Driver rpc_round19_hierarchy.py, model rpc_hier.py, analyzer
r19/r19_analyze.py, monitor r19/monitor.py, smoke
rpc_round19_smoke.py. Nothing launched on Beast. The main agent
launches.

### 54.1 Question

Does stacking fixed random reservoirs into a hierarchy, with
one delta-trained distal wire per level and one fixed random
feedback wire per level, learn structure that a flat two-module
model cannot learn? The sharpest version: can the hierarchy
learn the first motif step (m1), which sits at 0.680 against
the 0.80 bar in round 15, when the first step depends on a
slow context the flat model cannot hold?

### 54.2 Hypothesis (one sentence)

A three-level hierarchy with a slow top level (effective memory
about 18 steps) learns the slow context of a two-timescale
sequence and uses it to predict the context-dependent first
motif step, which a flat two-module model cannot do because its
memory spans about 2 to 3 steps.

### 54.3 Architecture (A28, new file rpc_hier.py; rpc_v2.py untouched)

1. M modules stacked (M = 2 or 3): input feeds module 1, module
   2 predicts module 1, module 3 predicts module 2.
2. Each level l < M has a distal wire W_dist_l trained ONLY by
   the delta rule (signed error between the level prediction and
   the actual code below it, NLMS denominator, w_max 2.0).
3. Each level l < M has a fixed random feedback wire W_fb_l into
   the module below it (A26 form, never trained), with per-level
   gain fb_gain_l and test modes on / cut / shuffle.
4. Level leak rates are fixed by depth: alpha 1.0, 0.3, 0.1
   (estimated effective memories about 2-3, 6, 18 steps at
   rho 0.9). No confirmation, no inhibition.
5. Readout stays the fixed p1-prototype cosine at the bottom
   level; the top-level prediction against mean-p_{M-1}
   prototypes is a secondary readout (M = 3 only, no bar).

Nested initialisation: weights are drawn level by level from
one seeded RNG stream, so a 3-level net's levels 1-2 weights
are EXACTLY a 2-level net's at the same seed (smoke-tested).

### 54.4 Hierarchical task (make_hier_data)

Length 48, 4 blocks of 12 steps. Each block starts with a
context token C in {0,1,2,3}. Each context has two motifs with
shared starters 4 and 5: C=0: (4,6,7),(5,8,9); C=1: (4,8,9),
(5,10,11); C=2: (4,10,11),(5,12,13); C=3: (4,12,13),(5,14,15).
Each block plants one motif starting with 4 and one with 5 at
random non-overlapping starts; filler is uniform from 4..15.
m1 context-blind Bayes is 0.25; m2 is determined by the 2nd
token; overall chance 1/16 = 0.0625. Late m1 = starter at block
position 6 or later. Flat model expected score: late m1 about
0.25 (context-blind), overall acc about 0.35-0.40. Bar: the
candidate beats flat on late m1 by at least +0.20 and reaches
late m1 mean at least 0.45 (lower CI above 0.35).

### 54.5 Tasks, seeds, conditions

Tasks: planted lag-1 (200/50/50, K=16), sine-fixed (50/20/20,
K=32), hier (100/50/50, K=16). Seeds 282..301 (20 fresh; val
seeds 282..286). Base configs are the round-15 winners
(planted1 alpha 1.0 / sigma 0.0 / eta_d 0.05; sine alpha 1.0 /
sigma 0.0 / eta_d 0.2; hier alpha 1.0 / sigma 0.0 / eta_d 0.05).
5 epochs, n_res 200, n_out 256.

Conditions (5 nets per task per seed, 8 lines per task per
seed, 480 test lines): cand_on (val-winner architecture,
delta, feedback on), cand_cut (same net, feedback cut),
cand_shuffle (same net, feedback shuffled), hier2_ladder
(M=2, ladder alphas, winner fb_gain1, delta), hier2_nofb
(M=2, ladder alphas, no feedback, delta), flat (M=2, both
alphas = task alpha, no feedback, delta: the round-15 winner
config), cand_frozen (winner architecture, no plasticity),
baseline (Markov-1/2, persistence, ESN on identical test
sequences). If the val winner is 2 levels, hier2_ladder
duplicates cand_on and is skipped (420 lines), and H4 is not
judged.

Validation (hier task only): n_levels {2,3} x fb_gain1
{0.1,0.3} x fb_gain2 {0.1,0.3} = 6 effective configs, 5 val
seeds, 30 lines. Winner = highest mean val overall accuracy,
tie-break first in grid order, frozen for all test conditions.

### 54.6 Bars (20-seed test, paired per-seed differences, 95% CI)

1. H1: hier overall acc, cand_on - flat >= +0.10.
2. H2: hier late m1, cand_on - flat >= +0.20.
3. H3: hier late m1, cand_on mean >= 0.45, lower CI above 0.35.
4. H4: hier late m1, cand_on - hier2_ladder >= +0.10 (3-level
   winner only).
5. H5: planted lag-1 m1, cand_on - flat >= +0.05.
6. H6: planted lag-1 m1, cand_on mean >= 0.80.
7. H7: sine-fixed acc, cand_on - flat >= +0.03.
8. H8: hier overall acc, cand_on - cand_cut >= +0.05.
9. H9: hier overall acc, cand_on - cand_frozen >= +0.10.
Shuffle contrast is a diagnostic, not a bar. Verdict: H0
passes if H1, H2, H3 pass; H4 decides third level vs slower
level 2; H5-H7 decide whether hierarchy helps the old tasks;
H8-H9 are mechanism checks.

### 54.7 Smoke test (VM, seed 282, 1 epoch unless noted)

All checks pass (rpc_round19_smoke.py, 2m31s on the VM,
Nov 27, 2026). 1. Nested init exact: a 3-level net's levels
1-2 weights are identical to a 2-level net's at the same seed.
2. Flat reproduction (5 epochs, seed 201): hierarchical flat
net scores planted1 0.207 vs round-15 recorded 0.227 (diff
-0.020) and sine 0.903 vs 0.872 (diff +0.031), both within the
0.05 bar. 3. Feedback changes the lower module's state: mean
r1 cosine on vs cut 0.9725 (below 1.0, above 0.0); fb_rms_ratio
on 0.2513, cut 0.0000; shuffle drive measured on the same
trajectory (same ds) is 1.0306x the on drive (bar 0.90-1.10);
the separate-pass shuffle fb RMS is 0.2914 vs on 0.2513 (gap
+16.0 percent), which is trajectory divergence, not a magnitude
bug, and it weakens the shuffle contrast as a diagnostic.
4. Top readout non-degenerate: top_pred takes 16 distinct
values over 480 steps. 5. Timescale ladder: lag-11 context
decode from the untrained net is L1 0.70, L2 0.97, L3 0.93
(chance 0.25; L3 exceeds L1, so the ladder holds); lag-1
sanity is 1.00 at all levels. Note: L1 retains context at
0.70, so the flat model may beat the 0.25 late-m1 guess. 6.
Guardrails: all metrics finite, no NaN, r_max 0.869 (below
10), clamp_frac 0.0000 per level [0.0, 0.0]; driver CLI
resumable (1 line after two runs); analyzer writes
r19_analysis.md on 1-seed output (exits 1 on coverage, as
designed).

### 54.8 Execution

About 30 CPU-hours (300 training runs at about 6 min each),
about 5 h wall with 6 workers (3 tasks x 2 seed-shards).
Monitor r19/monitor.py drives val -> winner pick -> test ->
analysis at 480 lines (420 if the winner is 2 levels).

### 54.9 Self-critique (from the frozen protocol)

1. rpc_v2.py is untouched; rounds 17 and 18 keep running
   against it.
2. Level alphas are fixed by depth, not validated.
3. Flat reproduction in the hierarchical class is approximate
   (within 0.05), because the init draw order differs.
4. hier2-ladder vs flat conflates a slower level 2 with
   feedback; conditions 5 and 6 separate them as diagnostics.
5. Val seeds are a subset of test seeds (same mild overlap as
   rounds 15-18).
6. A win with no cut gap means training-time shaping, not
   test-time feedback use.
7. Round 19 does NOT test precision (C), generative (D), or
   the mapping onto canonical microcircuits or hippocampal data.
