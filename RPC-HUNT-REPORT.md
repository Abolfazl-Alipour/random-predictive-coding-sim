# Random predictive coding: what I started with and what I did

Report for external critique. Author: Muse (Muse Spark), working
directly for Abolfazl Alipour. Date: Nov 26, 2026. Full record:
`simulation-results.md` (sections 1-40) and `README.md` in
`~/workspace/thesis-rpc/`. Code: `rpc_v2.py` plus the drivers
`rpc_*.py`. This report is the summary the critics should judge;
the record is the evidence.

## 1. What I started with

### 1.1. The thesis idea

Random predictive coding (RPC) is Chapter 3 of Abolfazl
Alipour's PhD thesis (Indiana University, August 2023). The
idea, in plain English: a cortical module predicts the next
input through fixed random wires from another module. When
the prediction is confirmed, a burst of activity marks the
confirmation, and that burst drives local Hebbian learning
(cells that fire together wire together). The module then
predicts better next time. The "random" part is the claim
that the wires do not need to be learned or structured: fixed
random connections are enough.

### 1.2. The formal model I built (RPC v2, `rpc_v2.py`)

1. The input is a token, a number from 0 to K-1. It is coded
   as a one-hot vector (a vector with a 1 in one position and
   0 elsewhere), except where the smooth input is used.
2. Module 1 is a reservoir, a recurrent network of 200 units
   with a leak rate alpha. It receives the current token and
   holds a trace of recent tokens.
3. Module 2 is a second reservoir of 200 units. It receives
   the current token through a fixed random proximal
   projection (a matrix that maps the token into 64 numbers).
4. The distal drive is the prediction. It is computed by
   multiplying the module-2 code (the activity pattern from
   the previous step) by the distal weight matrix W_dist1.
   W_dist1 is the only matrix that learns in the base model.
5. The readout compares the distal drive against prototypes.
   A prototype is the average proximal drive for one token,
   computed from the training data. The predicted token is
   the one whose prototype is closest to the distal drive by
   cosine similarity (the angle between two vectors).
6. The burst is a binary gate. It fires when the prediction
   matches the actual input: both the distal drive and the
   proximal drive exceed a threshold on the same cells.
   When the burst fires, the confirmation rule allows the
   Hebbian update of W_dist1. The update strengthens the
   connection between the previous module-2 code and the
   cells that are active for the current input.
7. A coactivity rule is the ungated version: it strengthens
   W_dist1 on every co-occurrence, with no burst gate. This
   is not the thesis rule; it is a comparison rule.
8. Row-norm preservation: after each Hebbian update, each row
   of W_dist1 is rescaled to its previous length, so a row's
   total weight is fixed and the update only redistributes it.

### 1.3. The test setup (fixed before any result)

1. Tasks. Cycle: tokens 0-7 repeating, K=8, chance 0.125.
   Pure motif: the sequence 1,2,3 repeating, K=4. Planted
   motifs: three motifs (1,2,3), (5,6,7), (9,10,11) hidden
   in random filler tokens, K=16, chance 0.0625. Sine:
   a sine wave binned to 32 bins, K=32, chance 0.031,
   fixed frequency 0.05 with random phase per sequence, and
   a harder variable-frequency version. Lorenz: the Lorenz
   chaotic system, x binned to 32 bins. 1D track: a position
   moving on a line, binned.
2. Pass bars, set in advance. Token tasks: accuracy 0.2625
   and gain 0.10 over the no-plasticity control. Sine tasks:
   accuracy 0.231 and gain 0.10. The accuracy bar is the
   baselines' level; the gain bar proves the learning adds
   something beyond a frozen network.
3. Controls. No-plasticity: the same network with learning
   off. No-confirmation: learning on but the confirmation
   gate removed. Both are run on every condition.
4. Held-out test. The network trains on 50 sequences and is
   tested on 20 unseen sequences. 5 seeds (different random
   initialisations), 5 epochs unless noted.
5. Baselines. Markov-1 (predict from the previous token
   only), echo state network (a fixed reservoir with a
   trained readout), and LSTM (a standard recurrent network).
   These set the bar for what a working model scores.

### 1.4. The literature position (from my review, Nov 25)

The label "random predictive coding" is new in
neuroscience. The idea is not. The closest prior art is
Hierarchical Temporal Memory (HTM), and the most damaging
is Chen et al. 2022/2024 (CA3 predicts, CA1 signals error),
which predates the thesis and is not cited in it.

## 2. What I did, round by round

### 2.1. Rounds 1-4: the first sweep, the v1 scale failure, and the simple tests (Nov 25)

1. First broad sweep (rpc_toy.py, v1): motifs, sine, Lorenz,
   and the track at reservoir leak rate 0.3. Nothing passed
   the bars. Motif gains +0.013 to +0.028. Sine gain -0.007.
   Lorenz gain -0.001. Baselines beat RPC 3x to 9x.
2. V1 had a scale bug: the distal drive was identically zero
   at the start, so no learning could happen. V2 fixed the
   scales.
3. Simple tests first, at Reza's order: pure cycle 0.126
   (chance 0.125), pure motif 0.334, sine 0.071, all at
   leak 0.3. Diagnosis: the cycle prototypes were
   near-parallel (off-diagonal cosine 0.890), so the readout
   could not tell them apart. Centering the prototypes did
   not fix it.

### 2.2. Round 5: the alpha proof (Nov 25)

Raising the leak rate alpha from 0.3 to 1.0 separates the
cycle prototypes (off-diagonal cosine 0.890 to 0.147) and
RPC learns the deterministic cycle: accuracy 0.843 (range
0.728 to 0.989, 5 seeds) against control 0.147. Alpha 0.7
also passes both bars (0.608 vs 0.165). Scope, stated at
the time: alpha 1.0 makes the reservoir track the current
token, so the cycle is close to a fixed token-to-token
map, and Markov-1 scores 1.000. This proves the
burst-gated rule can learn a deterministic sequence when
the code separates. It does not prove the slow-reservoir
thesis model learns temporal structure.

### 2.3. Round 6: motif and sine at alpha 1.0, and the planted boundary (Nov 25)

1. Pure motif at alpha 1.0: 0.934 (control 0.322), gain
   +0.612. Both bars met.
2. Sine at alpha 1.0: 0.091 (control 0.022). No leak rate
   fixes the sine (0.3/0.5/0.7/1.0 give 0.071/0.055/0.079/
   0.091).
3. Planted motifs at alpha 1.0 and 0.7: 0.073 and 0.077,
   against controls 0.064 and 0.060. The alpha fix changes
   nothing in noise. Boundary: RPC learns only noiseless
   deterministic token sequences and fails on planted motifs,
   sine, Lorenz, and the track.

### 2.4. Round 7: dense plastic wiring (Nov 25)

Reza proposed that module 2 is wired all-to-all to module 1
at first, so its outputs must hit module-1 activity, and
the hit solidifies the connection. The dense random wiring
was already in v2; the new part was a coactivity learning
rule on the distal wires (assumption A8). Planted motifs:
0.072, identical to the burst rule's 0.073. Pure motif
sanity check: 0.999 (better than the burst rule's 0.934).
The rule works and learns clean sequences better, but
changes nothing in noise.

### 2.5. Round 8: learned readout, normalization, delay, probes (Nov 25)

1. Trained ridge readout on the distal drive: planted
   motifs 0.157, double the prototype readout's 0.072, but
   the frozen control scores 0.163, higher than both
   trained conditions. The readout does the work, not the
   learning. Gain -0.006.
2. Normalization screen (A9): surround subtraction 0.068,
   divisive sharpening 0.076, top-k post term 0.066. All
   fail.
3. Delay line (A10): a second distal matrix reading the
   code from two steps ago. 0.066. Fail.
4. Code probe: the module-2 code decodes the previous token
   at 0.361 frozen and 0.330 after training. The reservoir
   holds the token perfectly (1.000). The module-2 pathway
   destroys it, and training makes it slightly worse.
5. Module-2 direct input (A11): feeding module 2 the clean
   token through a fixed random projection. 0.088. A clean
   input does not fix it, so the degradation sits inside
   module 2.
6. Stage probe (frozen): input 1.000, expansion 0.999,
   reservoir 0.984, proximal drive 0.711, output code
   0.332. The loss sits at the fixed random proximal
   projection (0.984 to 0.711) and the top-6 output
   selection (0.711 to 0.332).
7. Fixes: feedforward module 2 (no recurrence) 0.090; more
   output winners (16 of 64) 0.076; wide proximal layer
   (256 units) 0.067, gain 0.000. All fail.
8. Ceiling correction: the planted task has three motifs,
   so a one-step model scores about 0.36 (Markov-1 proves
   0.278), not 0.2625 from one motif. Memory span is not
   the bottleneck.

### 2.6. Round 9: learned proximal projection (Nov 25)

Learned W_prox (A12): fit module 2's proximal projection
by ridge regression from the reservoir state onto a fixed
random code per token, train data only. The proximal drive
then decodes the token at 0.984 (against 0.711 random).
Planted motifs: 0.113, the best planted score, but the
confirmation rule destroys it (0.076 with confirmation
on), and the clean code alone gives nothing (control
0.067). Bars not met. Gain +0.046.

### 2.7. Rounds 10-12: why the sine fails, and the fix rounds (Nov 25-26)

1. Sine diagnosis: Markov ceilings on sine-fixed are
   Markov-1 0.459, Markov-2 0.802, Markov-3 0.833. Every
   test step's previous bin has more than one successor
   (ambiguous-prev-bin fraction 1.000). Frozen stage probes
   (ridge decode of the next bin / direction): reservoir
   0.953 / 0.984, module-2 code 0.607 / 0.748, distal drive
   0.602 / 0.740. The reservoir holds the answer, the top-6
   code destroys a third of it, the learned drive transmits
   what the code contains. Learning is not the bottleneck;
   the code is. The trained network scores 0.091 while its
   own drive decodes at 0.602, so the learned map plus the
   prototype readout lose about 0.5. Mechanism: one row of
   the distal map can only point at one successor, but the
   same previous code must map to two successors (rising
   and falling), so the row learns their average and matches
   neither. 74 to 83 percent of errors are 6 or more bins away.
2. Smooth input (A14): encode each token as a Gaussian bump
   over neighbouring bins instead of a one-hot vector. This
   is the only fix that moved the sine by more than 0.03.
   It raises the code's direction decode from 0.748 to
   about 0.915. Fixed sine at sigma 3.0: 0.197 to 0.199 at
   5 seeds (control 0.022), best 0.210 at 3 seeds with leak
   0.7. Sigma 2.0 with 10 epochs: 0.220 at 3 seeds.
3. Learned module-1 projection (A13): fit module 1's
   proximal projection the same way as A12. Sine 0.028,
   below its own control (0.046) and below chance; the
   burst rate collapses to 0.038. Dead.
4. Split prototypes (A15): three prototypes per bin
   (rising, falling, flat). 0.183 with smooth input, no
   better than plain. Dead.
5. Graded confirmation (A16): scale the Hebbian update by
   g, the cosine match between the prediction and the actual
   input, clipped to [0,1]. At the same learning rate it
   hurts everywhere (g is small where the model is weak:
   planted 0.048, one-hot 0.061, so grading starves exactly
   the cases that need learning). With the learning rate
   raised 5x to compensate, it ties binary at 5 seeds
   (0.196 to 0.202). Dead. The confirmation rule itself
   helps: turning it off drops graded 0.230 to 0.178 and
   binary 0.210 to 0.186 at 3 seeds.
6. Dot-product readout (A17): compare the distal drive
   against the unnormalised prototypes, so each prototype
   is weighted by its stored strength. Fixed sine 0.217
   at 5 seeds (control 0.036, gain +0.181), best 5-seed
   score at the time. The readout story: cosine 0.202, dot
   0.217, trained ridge 0.556 to 0.673.
7. Trained ridge readout, generalisation (round 9): fixed
   sine 0.556, variable sine 0.244, Lorenz 0.251, planted
   0.157, with the frozen control equal or better
   everywhere. Accepting the trained readout would prove
   the fixed random reservoir (the echo state network
   result), not RPC learning.
8. Generalisation of the smooth fix: variable-frequency
   sine 0.108, Lorenz 0.093 (best). The ceiling falls as
   the task needs more history.

### 2.8. Rounds 13-15: the code and the readout are my assumptions, not the theory (Nov 26)

Reza ordered: the top-6 code and the prototype readout are
assumptions I made, not the theory. Fix them and keep
identifying the root cause until the simulation is mature.

1. Full graded code (A18): every cell above the step mean
   projects back, graded, no rank cutoff, with per-step
   standardisation inside the code (the first naive version
   blew up about 15x in scale during training and scored
   0.013; the top-6 code had hidden the drift because rank
   is scale-invariant). Fixed sine 0.105, bimodal: seeds 0
   and 1 work (0.213, 0.218), seeds 2, 3, 4 collapse to
   chance (0.028, 0.037, 0.028). Planted motifs 0.065.
   Code probe: the full code carries more token
   information (decode 0.709 against top-6 0.516) but the
   learned drive carries less (0.457 against 0.509).
2. Nearest-neighbour readout (A19): store every training
   proximal-drive vector in a bank (up to 4000) and predict
   the token of the closest bank vector by cosine; no
   averaging. Fixed sine 0.218 (range 0.129 to 0.304,
   control 0.028, gain +0.190), the best 5-seed score,
   tied with the dot readout, 0.013 short of the bar.
   Planted motifs 0.072, ties cosine.
3. Sharpened learning pre-term (A20): raise the code to
   power 2 or 3 before the Hebbian update, so the update
   concentrates while the drive stays rich. Fixed sine
   0.099 (p=2) and 0.110 (p=3), both still bimodal. Dead.
   This killed the dilution explanation as stated: the
   drive stays degraded (0.478 against top-6 0.509) even
   with a concentrated update.
4. Sharpened drive read (A21): raise the code to power 2
   or 3 when computing the distal drive, while learning
   keeps the raw code. Fixed sine 0.119 (p=2) and 0.129
   (p=3), both still bimodal. Dead. Per-seed probe: the
   drive decodes the next token at 0.408 mean against
   top-6 0.507, and on collapsed seed 2 it decodes at
   0.495 while the network scores 0.035. The drive holds
   the information and the fixed-space readout extracts
   none of it. The learned map does not reliably land in
   the prototype space.
5. Burst rule on the full code (round 13): fixed sine 0.022,
   below chance 0.031, gain -0.001. The thesis gate cannot
   bootstrap the full code: it only learns on steps where
   the prediction already matches, and with the full code
   the early predictions are too noisy for it to ever start.
   Per-seed probe on the burst drive: mean 0.462, per-seed
   [0.546, 0.488, 0.543, 0.348, 0.387]. The burst drive is
   MORE informative to a trained decoder than the coactivity
   drive (0.408), yet its network score is 0.022, an
   extraction ratio of 0.05. The gate prevents
   error-correction, so the map develops an idiosyncratic
   encoding a trained decoder can read but the fixed readout
   cannot.
6. Prototype-span projection readout (A22): project the
   distal drive onto the span of the prototypes before
   comparing. Proven a MATHEMATICAL no-op for the cosine
   readout (diagnostic `rpc_proj_check.py`): the projection
   removes 31% of the drive's energy, yet the per-seed
   scores are exactly identical to the plain full code,
   because (P d1).proto = d1.proto for every prototype in
   the span and cosine normalises the scale change. The
   same argument holds for the dot readout. Dead for cosine
   and dot; only non-trivial for the kNN readout, whose bank
   vectors are not in the prototype span. The orthogonal
   part of the drive (31% of its energy) is exactly the
   information the ridge probe sees but cosine cannot.
7. Clean planted tests (round 13): full code under
   coactivity 0.075 (control 0.060); learned projection
   plus kNN under coactivity 0.075 (control 0.070),
   against the learned projection alone under coactivity
   (0.113). The fixes genuinely do not stack, even under
   the same rule. Full code plus projection on planted:
   0.065, identical to the plain full code, as the
   linear algebra predicts.

### 2.9. What is implemented but not yet run

Assumption A23, drive prototypes: rebuild the prototypes
after training from the trained network's own distal drive
(mean d1 per next-token), so the comparison vectors live in
d1-space and the cosine readout can see the orthogonal
part. It is a nearest-centroid readout in drive space, using
train data only, with no trained weights. The code is in
`rpc_v2.py` (flag `drive_proto`, method
`build_drive_prototypes`, hook in `run()`). No run has been
launched yet, because this report and the critique were
ordered first.

## 3. The boundary as it stands

1. RPC learns only noiseless deterministic token sequences:
   cycle 0.843, pure motif 0.934 (burst rule) and 0.999
   (coactivity rule), all at leak rate 1.0, 5 seeds,
   held-out test.
2. It fails on planted motifs (best 0.113, learned proximal
   projection), fixed sine (best 0.218, kNN readout),
   variable sine (best 0.108), Lorenz (best 0.093), and the
   1D track (best 0.098). All below the bars.
3. Baselines pass the planted bar: Markov-1 0.278, echo
   state network 0.274, LSTM 0.283.
4. The learning gain passes the bar everywhere on the sine
   (up to +0.203), so learning does something; the accuracy
   bar is what fails. On planted motifs the gain is small
   (+0.015 to +0.049).
5. The three theory-faithful readouts cluster at 0.20 to
   0.22 on the fixed sine (cosine 0.202, dot 0.217, kNN
   0.218), while the trained ridge scores 0.556 to 0.673
   on the same drive. The readout is no longer the binding
   constraint in the sense that three different readouts
   agree; the binding constraint is that no fixed
   comparison against p1 patterns can see the information
   in the drive's orthogonal part.

## 4. The root-cause chain, as measured

1. The module-1 reservoir holds the current token
   perfectly (linear probe 1.000).
2. The fixed random proximal projection loses a third of
   it (0.984 to 0.711).
3. The top-6 output selection loses another half (0.711
   to 0.332). The full graded code recovers some of it
   (0.709) but the learned drive then carries less
   (0.457 against 0.509).
4. The learned distal drive holds the next token in a way
   a trained decoder can read (ridge 0.408 to 0.507 on
   the sine, 0.462 under the burst rule), but the
   information sits largely in components orthogonal to
   the prototype span (31% of the drive's energy).
5. No fixed comparison against p1 patterns can see the
   orthogonal part, because p1 patterns (nearly) span only
   the prototype space. Cosine, dot, and kNN all fail the
   same way on the collapsed seeds.
6. The burst gate makes the misalignment worse, because it
   only learns from steps where the prediction already
   matches and never corrects a misaligned map.
7. The smooth input helps because it puts direction
   information (rising vs falling) into the code, raising
   the code's direction decode from 0.748 to about 0.915.

## 5. My mistakes and corrections (the honest part)

1. V1 scale bug: the distal drive was identically zero at
   the start. Fixed in v2.
2. Prototype misstatement: I told Reza the prototypes were
   "the average distal drive for that token". They were
   always the average proximal drive. Corrected in the
   record and in the report to him.
3. Dilution explanation: I claimed the full code hurts
   because each Hebbian update spreads across about 32
   columns instead of 6, diluting each row's fixed mass.
   A20 (concentrate the update) changed nothing, so that
   explanation is wrong as stated. The degradation sits in
   the drive read, not the write. Then A21 (concentrate
   the read) also changed nothing, so the degradation is
   not a concentration problem at all. The per-seed probe
   replaced both with the space-misalignment finding.
4. Planted stacking claim: I told Reza the fixes do not
   stack on planted motifs (0.074 vs 0.113), but the two
   runs used different learning rules (burst vs
   coactivity), so the comparison was confounded. I
   withdrew it. Round 13 reran the clean test under the
   same rule and the stacking failure is real (0.075 vs
   0.113), so the claim is reinstated on clean evidence.
5. A22 design error: I proposed projecting the drive onto
   the prototype span to strip the misaligned part, but the
   projection is a mathematical no-op for the cosine
   readout. I should have seen that from the linear
   algebra before running it. The diagnostic proved it
   and the run confirmed it.
6. 1-epoch smoke tests are not predictive for full-code
   conditions: a 1-epoch score of 0.274 became 0.113 at
   5 epochs. I stopped trusting smoke tests for verdicts
   after that.
7. Ceiling error: I first analysed the planted task as if
   it had one motif. It has three, so the one-step ceiling
   is about 0.36, not 0.2625 from a single motif. Corrected.
8. The smooth-input 1.5 result was overstated at 3 seeds
   (0.145, gain +0.114) and regressed at 5 seeds (0.129,
   gain +0.098). I corrected it in the record.
9. Graded confirmation at the same learning rate hurts; I
   only found the compensated version ties binary after
   raising the learning rate 5x. The 3-seed 0.220 and 0.230
   bumps both regressed at 5 seeds.

## 6. What is still open

1. A23 (drive prototypes) is implemented but not run. It is
   the next test, unless the critique says otherwise.
2. The GitHub repo
   (https://github.com/Abolfazl-Alipour/random-predictive-coding-sim,
   private) is at commit 9b168028 (sections 1-35). Sections
   36-40 and assumptions A18-A23 are not yet pushed; the
   push needs Reza's approval because the last attempt
   timed out.
3. The story walkthrough artifact
   (`rpc-story-walkthrough`, v3) covers rounds 1-8 and
   needs chapters for rounds 9-13.
4. The thesis PDF and text stay out of the repo unless
   Reza says to add them.

## 7. The brief for the critics (Kimi and agy)

Read this report, then read the full record
(`simulation-results.md`, sections 1-40) and the code
(`rpc_v2.py`, the drivers `rpc_*.py`, and the `*_results.txt`
files). Then answer, in plain English, with numbers where
you can check them by running the code:

1. Critique the work. Where is the method wrong? Which
   assumptions did I make that the theory does not make,
   and which of those actually matter? Where did I
   interpret a number wrong? Where is the test setup
   unfair to the theory, or too kind to it? Name every
   error you find, including ones I already listed in
   section 5, and say whether my correction is right.
2. Check the root-cause chain in section 4. Is each step
   actually supported by the measurements I cite? Is there
   a step I skipped, or a simpler explanation for the
   numbers? In particular: is the "orthogonal part of the
   drive" claim right, and does it actually explain why
   cosine, dot, and kNN all fail on the collapsed seeds?
3. Tell me how to do it right. What is the correct way to
   test whether random predictive coding can learn? What
   experiment would you run next, exactly (task, code
   change, readout, bars)? What is the single most
   promising change to the model, and why? If you think the
   theory as formalised cannot work, say so plainly and
   give the reason and the smallest change that would make
   it work.
4. Is A23 (drive prototypes) the right next test, or is it
   another no-op like A22? Work through the linear
   algebra before answering. If it is a no-op, say so and
   do not let me waste a run on it.
5. The thesis claims fixed random wires are enough. My
   results say the fixed random proximal projection is
   where a third of the information is lost, and a learned
   proximal projection fixes the code but the confirmation
   rule then destroys the gain. Is there a version of this
   theory that keeps the random wires and still works, or
   does the theory need learned wires somewhere? If it
   needs them, where exactly, and what does that do to the
   thesis claim?

Rules for the critique: judge the work as it is, not as I
describe it. Run the code if you need to check a claim.
Quote the exact file and line or the exact results file
when you say something is wrong. Do not invent numbers.
If you cannot check something, say so. Write the critique
as a report file, not as chat.
