# RPC simulation results

Work done by Muse directly. No Kimi, no AGY, no subagents. Coordinator
killed Nov 25 03:57 PST at Reza's order after it went quiet for 2 hours
45 minutes with no final report.

## 1. Verdict so far: PROOF ON THE CYCLE (Nov 25, 05:30 PST)

1. RPC v2 learns a small real effect on planted token motifs.
2. RPC v2 learns nothing on sine waves or Lorenz, over 5 and 3 seeds.
3. No effect met the pre-set pass bars, on any task, at any scale tried,
   until the alpha test (section 14).
4. PROOF: on the deterministic repeating cycle (tokens 0-7, K=8, chance
   0.125), with the reservoir leak rate alpha raised from 0.3 to 1.0,
   full scores 0.843 (range 0.728 to 0.989, 5 seeds) against a
   no-plasticity control of 0.147. That beats chance by 0.718 (bar 0.20)
   and beats the control by 0.696 (bar 0.10). Every seed passes both
   bars on its own (lowest seed 0.728, lowest per-seed gain 0.491).
   Alpha 0.7 also passes both bars: full 0.608, control 0.165,
   gain 0.443. The earlier cycle failure was a code-separation failure
   (prototypes near-parallel at 0.890 cosine), not a learning-rule
   failure: raising alpha separates the prototypes (0.147 at alpha 1.0)
   and the same burst-gated distal rule then learns the sequence.
5. Scope of the proof, stated plainly: the cycle is a noiseless
   deterministic sequence, and alpha 1.0 makes the reservoir track the
   current token, so the task is close to a fixed token-to-token map.
   Markov-1 still scores 1.000. This proves the burst-gated distal
   learning rule can learn a deterministic sequence when the code
   separates. It does NOT prove the slow-reservoir (alpha 0.3) thesis
   model learns temporal structure: sine, Lorenz, and the track at
   alpha 0.3 still fail. The schedule stops here at the body's order;
   whether the proof generalises to motif and sine-fixed at alpha 1.0
   is the next question, for Reza to direct.
6. TrackF no-confirmation is now complete (section 11, item 4): full
   0.084, no-confirmation 0.131, no-plasticity 0.068. Confirmation
   hurts on every continuous task tested.

## 2. Falsification bounds (what the numbers say)

1. Motif lags 1, 2, 5: gains plus 0.013, plus 0.016, plus 0.015.
2. Best motif gain, after scaling data and epochs: plus 0.028.
3. Sine, 5 seeds: gain minus 0.007. Lorenz, 3 seeds: gain minus 0.001.
4. Confirmation adds plus 0.003 on motifs, minus 0.048 on Lorenz.
5. Longer reservoir memory hurts lag 5: gain minus 0.003, below chance.
6. A bigger reservoir hurts: reservoir 500 gives gain minus 0.007.
7. Scaling data 2.5x and epochs 2x grows the lag-5 gain from plus 0.015
   to plus 0.028. At that rate, the 0.10 bar needs roughly 100x more
   data, and accuracy would still sit below the echo state network.
8. Baselines beat RPC by 3x to 9x on every task: lag 2 ESN 0.263 vs
   RPC 0.072, lag 5 ESN 0.230 vs RPC 0.073, sine ESN 0.900 vs RPC 0.040.
   Lorenz, motif lag 2, and motif lag 5 are now tested and fail
   (sections 5 and 6).
9. Track position-bin, 5 seeds: full gain plus 0.017, no-confirmation
   gain plus 0.035, best accuracy 0.098 vs the 0.23 bar. Persistence
   alone scores 0.776, so the task is easy and RPC still misses it.
10. Confirmation hurts on the continuous tasks: track full 0.080 vs
    no-confirmation 0.098 (minus 0.018), Lorenz full 0.045 vs
    no-confirmation 0.093 (minus 0.048). It adds plus 0.003 on motifs
    only.

Pass bars (set before the overnight run): beat chance by at least 0.20
points, and beat the no-plasticity control by at least 0.10 points, on
held-out test, mean over 5 seeds.

## 3. Token motifs (v2, 5 seeds 0-4, 200 train / 50 test, 5 epochs)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_v2.py`

1. Full: accuracy 0.075, range 0.049 to 0.085, chance 0.0625, burst 0.136.
2. No-plasticity: accuracy 0.062, range 0.051 to 0.070, burst 0.026.
3. No-confirmation: accuracy 0.072, range 0.056 to 0.088.
4. No-burst-gate: accuracy 0.065, range 0.040 to 0.090.
5. Gain over no-plasticity: plus 0.013, range minus 0.012 to plus 0.033.
6. Confirmation adds plus 0.003. The burst gate is load-bearing.
7. Best config: theta_p 1.0, theta_d 1.0, eta 0.05, reservoir 200, output 64.
8. A bigger reservoir hurts: reservoir 500 gives gain minus 0.007.

Full sweep table: `sweep_results.csv`. Diagnosis: `diag_A.md`.

## 4. Sine waves (v2 network, K 32 bins, 100 train / 20 test sequences, length 100, 3 epochs)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_spatial.py sine 0 3`

1. Seed 0, full: accuracy 0.101, chance 0.031, burst 0.156.
2. Seed 0, no-plasticity: accuracy 0.027, chance 0.031, burst 0.032.
3. Seed 0, no-confirmation: accuracy 0.090, burst 0.142.
4. Gain over no-plasticity: plus 0.074. Confirmation adds plus 0.011.
5. Baselines on the same binning (baselines_B.md, 5 seeds): Markov-2 0.769, echo state network 0.900, LSTM 0.894.
6. RPC at 0.101 is far below every baseline, and below the 0.23 pass bar.

5-seed sine and 3-seed Lorenz runs are DONE. Files:
`rpc_spatial_sine_5seed.txt`, `rpc_spatial_lorenz_3seed.txt`.

## 5. Sine waves, 5 seeds (same setup as section 4)

1. Full: accuracy 0.040, range 0.020 to 0.101, chance 0.031, burst 0.134.
2. No-plasticity: accuracy 0.047, range 0.027 to 0.077, burst 0.023.
3. No-confirmation: accuracy 0.039, range 0.019 to 0.090, burst 0.132.
4. Gain over no-plasticity: minus 0.007. Learning HURTS on sine.
5. Seed 0 (0.101) was an outlier. Seeds 1 to 4 score 0.026, 0.033,
   0.020, 0.022, all below chance. The pilot's 0.101 did not replicate.

## 6. Lorenz x, 3 seeds (20 train / 10 test trajectories, length 100, 3 epochs)

1. Full: accuracy 0.045, range 0.028 to 0.056, chance 0.031, burst 0.078.
2. No-plasticity: accuracy 0.046, range 0.041 to 0.053, burst 0.020.
3. No-confirmation: accuracy 0.093, range 0.070 to 0.129, burst 0.108.
4. Gain of full over no-plasticity: minus 0.001. No learning.
5. Gain of no-confirmation over no-plasticity: plus 0.047, still far
   below the 0.10 bar and the 0.23 accuracy bar.
6. Confirmation HURTS Lorenz: full 0.045 vs no-confirmation 0.093,
   a minus 0.048 gap. This matches the diagnosis in diag_A.md section
   6.2: the confirmation rule rotates module-2 weights toward frequent
   patterns and destroys code sparsity.

## 7. Motif lag 2 and lag 5 (5 seeds, 200 train / 50 test, 5 epochs)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_lag.py`

1. Lag 2, full: accuracy 0.072, range 0.056 to 0.081, burst 0.140.
2. Lag 2, no-plasticity: accuracy 0.056, range 0.037 to 0.070.
3. Lag 2, no-confirmation: accuracy 0.067.
4. Lag 2 gain over no-plasticity: plus 0.016.
5. Lag 5, full: accuracy 0.073, range 0.044 to 0.101, burst 0.143.
6. Lag 5, no-plasticity: accuracy 0.058, range 0.051 to 0.063.
7. Lag 5, no-confirmation: accuracy 0.067.
8. Lag 5 gain over no-plasticity: plus 0.015.
9. Baselines: lag 2 ESN 0.263, LSTM 0.285; lag 5 ESN 0.230, LSTM 0.219.
10. Pattern across lags 1, 2, 5: gains plus 0.013, plus 0.016, plus 0.015.
    RPC learns a small, consistent effect at every lag, and it does not
    grow or shrink with lag. It is nowhere near the baselines.
11. Lag-5 variants (same 5 seeds, file `rpc_lag_variants.txt`):
    long-memory reservoir (rho 0.99, alpha 0.1): full 0.055, control
    0.058, gain minus 0.003. Longer memory HURTS, below chance.
    Prototypes from all 200 train sequences: full 0.081, control 0.062,
    gain plus 0.019. Best gain so far, still far below the 0.10 bar.
    Both together: full 0.064, control 0.061, gain plus 0.003.
12. Scale test DONE (file `rpc_scale_results.txt`, lag 5, 500 train,
    10 epochs, 3 seeds, proto500): full 0.088, range 0.083 to 0.097;
    no-confirmation 0.085; no-plasticity 0.060. Gain plus 0.028, the
    best so far, still far below the 0.10 bar and the 0.23 accuracy bar.

## 8. 1D track, position bins (v2 network, K 32 bins, 100 train / 20 test trials, first 100 steps, 3 epochs)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_track.py`

1. Persistence (predict current bin): accuracy 0.776. The task is
   easy; the bar stays 0.23 accuracy and plus 0.10 over the control.
2. Full: accuracy 0.080, range 0.022 to 0.275, chance 0.031, burst 0.072.
3. No-plasticity: accuracy 0.063, range 0.043 to 0.101, burst 0.024.
4. No-confirmation: accuracy 0.098, range 0.033 to 0.180, burst 0.129.
5. Gain of full over no-plasticity: plus 0.017. Gain of
   no-confirmation over no-plasticity: plus 0.035, the best gain on
   this task, still far below the 0.10 bar and the 0.23 accuracy bar.
6. Confirmation hurts here: full 0.080 vs no-confirmation 0.098, a
   minus 0.018 gap, same direction as Lorenz (minus 0.048).
7. Seed 2 full scores 0.275, above the 0.23 bar on that seed alone,
   but seeds 0, 1, 3, 4 score 0.039, 0.033, 0.022, 0.033, and the
   5-seed mean is 0.080. One seed is not a pass; the bars need the
   mean over 5 seeds.
8. The integrator-input version (x = [F(t), F(t-1)], rpc_track_F.py)
   is built and running; results go in section 11.

## 9. Why v1 was dead (measured, not guessed)

1. Module-2 output never fired: proximal drive max 0.162, threshold 0.5.
2. Distal drive was identically zero, not small: mean 0, sd 0, max 0.
3. Proximal drive max 0.282, threshold 0.5, so that leg never passed either.
4. v2 fixes the scales with six labelled assumptions (diag_A.md, A1 to A6).
5. v1 learning rules saturate in one epoch: 30 percent of weights hit the clamp.

## 10. Assumptions added beyond the thesis

1. A1: thresholds are z-scores on drives standardised by train-data scales.
2. A2: distal weight init scaled by 1 over sqrt of fan-in.
3. A3: graded ungated burst only as the no-burst-gate ablation.
4. A4: module-2 output fires on the top 10 percent of drive values.
5. A5: row-norm preservation after each Hebbian update (synaptic scaling).
6. A6: k-winners-take-all module-2 output, k 10 percent, fixed sparsity.
7. A7: integrator-vector input (rpc_track_F.py): the 200-number
   [F(t), F(t-1)] vector feeds module 1 straight through W_exp, the
   same fixed random expansion as the one-hot path, with no input
   standardisation. The thesis feeds integrator vectors into a sparse
   autoencoder, not into the RPC reservoir, so this encoding is new.

No backprop through the reservoir. No trained readout doing the learning.
Prediction is still cosine match of distal drive against mean-proximal
prototypes, as in v1.

## 11. Simplest time series (Reza's redirect: start simple, learn a time series first)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_simple.py`
3 seeds 0-2, 5 epochs, prototypes from all train sequences, best v2 config.
Markov-1 scores 1.000 on cycle and motif.

1. Cycle (tokens 0-7 repeating, K=8, chance 0.125): full 0.126, range
   0.117 to 0.136; no-plasticity 0.117; no-confirmation 0.119. Gain
   plus 0.009. RPC learns nothing on a pure repeating cycle.
2. Motif (tokens 1,2,3 repeating, K=4, chance 0.250): full 0.334, range
   0.332 to 0.337, tightly clustered; no-plasticity 0.144, range 0.040
   to 0.295 (very noisy control); no-confirmation 0.329. Gain over
   control plus 0.190, which meets the plus 0.10 gain bar, but accuracy
   0.334 is far below the 0.45 chance bar, and Markov-1 scores 1.000.
   5-seed confirmation DONE (file `rpc_motif5_results.txt`): full 0.334,
   range 0.332 to 0.337; no-plasticity 0.160, range 0.040 to 0.295.
   Gain plus 0.174, meets the gain bar, still fails the 0.45 accuracy
   bar. The 3-seed result holds at 5 seeds.
3. Sine-fixed (one sine, fixed frequency 0.05, random phase, K=32,
   chance 0.031): full 0.071, range 0.023 to 0.113; no-plasticity 0.036;
   no-confirmation 0.029 (below chance). Gain plus 0.035, far below
   the bars. Fixed frequency is strictly easier than the
   random-frequency sine of section 4, and RPC still fails it.
4. TrackF (integrator input, section 8 task, rpc_track_F.py): full
   0.084, range 0.029 to 0.124; no-plasticity 0.068, range 0.002 to
   0.146. Gain plus 0.016. No-confirmation DONE (rpc_alpha.py, 5 seeds):
   0.131, range 0.020 to 0.244, burst 0.134, accs 0.208, 0.088, 0.244,
   0.020, 0.094. Gain over no-plasticity plus 0.063, still below the
   0.10 bar and the 0.23 accuracy bar. No-confirmation beats full by
   0.047, the same direction as Lorenz (minus 0.048) and the
   position-bin track (minus 0.018): the confirmation rule hurts on
   every continuous task. The thesis-faithful input encoding does not
   rescue the track task.

## 12. Falsification bound, simple tasks (3 seeds, to be confirmed at 5)

1. Even a deterministic repeating cycle (0-7) gives RPC 0.126, equal
   to chance 0.125, while Markov-1 scores 1.000.
2. Even a deterministic 3-token motif gives RPC only 0.334, while
   Markov-1 scores 1.000. The 5-seed run held (full 0.334, control
   0.160), so this is the firmest bound so far: RPC cannot learn a
   noiseless deterministic sequence it should trivially master.
3. TrackF no-confirmation condition is DONE (rpc_alpha.py, section 11,
   item 4): 0.131, gain plus 0.063 over the control, still below the
   bars; confirmation hurts again (full 0.084 vs no-confirmation 0.131).
4. The cycle falsification bound is now REVERSED at higher alpha
   (section 14): the deterministic cycle is learnable by RPC once the
   prototypes separate. The bound that stands is narrower: at the
   default alpha 0.3, with near-parallel prototypes, RPC cannot learn
   even a noiseless deterministic sequence.

## 13. Why the cycle fails: the readout, not the learning (diagnostic)

Files: `rpc_diag_simple.py`, `rpc_diag_simple.txt`,
`rpc_proto_overlap.txt`, `rpc_center.py`, `rpc_center_results.txt`.
`center_proto` flag added to `rpc_v2.py` (Nov 25, default off).

1. After training on the cycle, the distal drive DOES move: cosine of
   d1 with the correct prototype is 0.343, against 0.089 untrained.
2. But the wrong prototype scores higher: 0.418. Learning moves d1
   toward all prototypes, not toward the right one.
3. Cause: raw prototypes are near-parallel. Cycle prototype
   off-diagonal cosine: mean 0.890, min 0.827, max 0.954. The mean
   proximal drive differs almost not at all between tokens, because
   the reservoir state is set by shared history, not by the current
   token. No distal vector can discriminate near-identical prototypes
   by cosine.
4. Motif prototypes are also overlapping (mean 0.487, max 0.979),
   and token 0 never appears, so its prototype is the zero vector.
5. Centering the prototypes does NOT fix it. Cycle centered: full
   0.131, control 0.126, gain plus 0.005. Motif centered: full 0.222
   (seed 0 scores 0.000), control 0.000 on all seeds. Centering turns
   the unused token-0 class into a trap: its centered prototype is
   minus the mean, and the centered distal drive aligns with it.
6. Next test, if the hunt continues: replace the cosine readout with
   a distance that the learning rule itself can shape, or feed the
   current token more strongly into the reservoir, so prototypes
   separate. Both change the model, not just a parameter. THE SECOND
   OPTION IS NOW TESTED (section 14): raising alpha separates the
   prototypes and the cycle is learned.

## 14. Alpha test: faster reservoir separates prototypes, cycle learned (Nov 25, 05:30 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_alpha.py`
Driver: `rpc_alpha.py`, output `rpc_alpha_results.txt`. Cycle task
(tokens 0-7 repeating, K=8, chance 0.125, 100 train / 20 test
sequences, length 96, 5 epochs, prototypes from all train sequences),
5 seeds 0-4, full vs no-plasticity. Alpha is the reservoir leak rate
in `ReservoirModule.forward`: r_new = (1-alpha) r + alpha tanh(...).
Default alpha 0.3; alpha 1.0 removes the leak decay entirely (history
enters only through the recurrent W_res term). Pure parameter change,
no new model machinery, no new assumption.

1. Prototype separation, measured pre-training (seed 0, off-diagonal
   cosine of the normalised mean-proximal prototypes):
   alpha 0.3: mean 0.890, max 0.954, min 0.827.
   alpha 0.7: mean 0.461, max 0.729, min 0.233.
   alpha 1.0: mean 0.147, max 0.400, min -0.073.
2. Cycle, alpha 0.7, full: accuracy 0.608, range 0.371 to 0.749,
   burst 0.136, accs 0.371, 0.671, 0.518, 0.734, 0.749.
3. Cycle, alpha 0.7, no-plasticity: accuracy 0.165, range 0.119 to
   0.297, burst 0.022, accs 0.151, 0.119, 0.297, 0.127, 0.131.
   Gain over control plus 0.443. Beats chance by 0.483. BOTH BARS MET.
4. Cycle, alpha 1.0, full: accuracy 0.843, range 0.728 to 0.989,
   burst 0.130, accs 0.735, 0.728, 0.871, 0.889, 0.989.
5. Cycle, alpha 1.0, no-plasticity: accuracy 0.147, range 0.013 to
   0.244, burst 0.025, accs 0.244, 0.013, 0.122, 0.122, 0.236.
   Gain over control plus 0.696. Beats chance by 0.718. BOTH BARS MET
   by a wide margin. Per-seed gains: 0.491, 0.715, 0.749, 0.767, 0.753;
   every seed passes both bars on its own.
6. Reading: the diagnosis in section 13 was correct and complete.
   The distal drive already learned at alpha 0.3 (correct-prototype
   cosine 0.343 vs 0.089 control); the readout could not use it because
   the prototypes were near-parallel. Once alpha separates the codes,
   the same burst-gated Hebbian rule learns the deterministic sequence
   to 0.84. The failure was in the code, not in the learning rule.
7. Scope, stated plainly: at alpha 1.0 the reservoir tracks the
   current token, so the cycle task is close to a fixed token-to-token
   map, and Markov-1 still scores 1.000. This is proof the mechanism
   can learn a deterministic sequence when its code separates. It is
   not proof the thesis's slow-reservoir model learns temporal
   structure, and it does not touch the sine, Lorenz, or track
   failures at alpha 0.3. The open question is whether the proof
   generalises: motif and sine-fixed at alpha 1.0, and the planted
   motif task at alpha 0.7 and 1.0.
8. Per the task body, proof delivers the stop condition: this schedule
   is removed and the tracked item is closed. Any follow-up (motif and
   sine-fixed at alpha 1.0) is Reza's call.

## 15. Motif and sine-fixed at alpha 1.0 (Reza's follow-up, Nov 25, 06:20 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_alpha2.py`
Driver: `rpc_alpha2.py`, output `rpc_alpha2_results.txt`. 5 seeds 0-4,
5 epochs, prototypes from all train sequences, alpha=1.0.

1. Motif (1,2,3 repeating, K=4, chance 0.250): full 0.934, range
   0.668 to 1.000, accs 0.668, 1.000, 1.000, 1.000, 1.000; burst 0.097.
2. Motif no-plasticity: 0.322, range 0.037 to 0.600, burst 0.025.
3. Motif no-confirmation: 0.846, range 0.623 to 0.997, burst 0.103.
4. Motif gain over control plus 0.612. Beats chance by 0.684. BOTH
   BARS MET (bar: acc >= 0.45, gain >= 0.10). The alpha fix
   generalises to the motif: four of five seeds score a perfect 1.000.
   Confirmation adds plus 0.088 here (0.934 vs 0.846).
5. Sine-fixed (K=32, chance 0.031): full 0.091, range 0.038 to 0.139,
   accs 0.076, 0.038, 0.139, 0.077, 0.125; burst 0.118.
6. Sine-fixed no-plasticity: 0.022, range 0.012 to 0.028, burst 0.024.
7. Sine-fixed no-confirmation: 0.062, range 0.030 to 0.087, burst 0.131.
8. Sine-fixed gain over control plus 0.069. BARS NOT MET (need acc
   0.231 and gain 0.10). Alpha 1.0 lifts the sine only slightly, from
   0.071 at alpha 0.3 to 0.091. Confirmation carries most of the sine
   learning (full 0.091 vs no-confirmation 0.062).
9. Reading: alpha 1.0 fixes deterministic token sequences, where the
   next token is a function of the current token. It does not fix the
   sine, where the same bin occurs twice per period (once rising, once
   falling), so the next bin cannot be read from the current bin alone:
   phase history is required, and alpha 1.0 weakens history.
10. Sine-fixed alpha sweep DONE (`rpc_alpha_sine.py`,
    `rpc_alpha_sine_results.txt`, 3 seeds, full vs no-plasticity):
    alpha 0.5 full 0.055, control 0.042, gain plus 0.013; alpha 0.7
    full 0.079, control 0.025, gain plus 0.054. Together with alpha
    0.3 (0.071, gain plus 0.035) and alpha 1.0 (0.091, gain plus
    0.069): no leak rate fixes the sine. Best is 0.091, far below the
    0.231 bar. The middle-alpha hypothesis is dead: more history does
    not help, and less history does not help either. The sine failure
    is not an alpha problem.
11. Summary of the alpha-1.0 follow-up: deterministic token
    sequences learn (cycle 0.843, motif 0.934, both bars met by wide
    margins). Continuous periodic signals do not (sine best 0.091 at
    any alpha). The boundary is now sharp: RPC learns when the next
    token is a function of the current token, and fails when the next
    value needs phase history.

## 16. Planted motifs at alpha 1.0 and 0.7: the fix does NOT generalise to noise (Nov 25, 06:44 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_planted_alpha.py`
Driver: `rpc_planted_alpha.py`, output `rpc_planted_alpha_results.txt`.
Task: motifs planted in random background (baselines.make_data, 200
train / 50 test, length 24, K=16), 5 seeds 0-4, 5 epochs, prototypes
from all train sequences. This is the real thesis case, not pure
repetition.

1. Alpha 1.0, full: 0.073, range 0.048 to 0.090, burst 0.151.
2. Alpha 1.0, no-plasticity: 0.064, range 0.055 to 0.067, burst 0.025.
3. Alpha 1.0, no-confirmation: 0.072, range 0.055 to 0.089.
4. Alpha 1.0 gain over control plus 0.009. BARS NOT MET.
5. Alpha 0.7, full: 0.077, range 0.057 to 0.088, burst 0.147.
6. Alpha 0.7, no-plasticity: 0.060, range 0.053 to 0.065.
7. Alpha 0.7, no-confirmation: 0.073, range 0.049 to 0.090.
8. Alpha 0.7 gain over control plus 0.017. BARS NOT MET.
9. Across alpha 0.3, 0.7, 1.0 the planted task scores 0.075, 0.077,
   0.073. The alpha fix changes nothing here. The random background
   re-confuses the code: the reservoir state at each step is set
   mostly by the random filler tokens, so the cue token's code is
   buried, and the distal drive cannot point at the right prototype.
10. Final boundary, all tasks, all alphas tried: RPC learns only
    noiseless deterministic token sequences, where the next token is
    a function of the current token (cycle 0.843, motif 0.934 at
    alpha 1.0). It fails on planted motifs in noise (best 0.077),
    on sine (best 0.091), on Lorenz (0.045), and on the 1D track
    (best 0.098). Simple baselines beat it 3x to 9x everywhere it
    fails. The thesis model as built cannot learn temporal structure
    in noisy or continuous inputs, at any leak rate tried.

## 17. Dense plastic wiring (Reza's "hit and solidify") on planted motifs (Nov 25, 17:41 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_dense.py`
Driver: `rpc_dense.py`, output `rpc_dense_results.txt`.
Rule: `learn_rule="coactivity"` (A8 in rpc_v2.py). The dense random
W_dist1 is NOT new, v2 already initialises it dense all-to-all (A2).
What changes is the learning rule. v2's rule is burst-gated:
W_dist1 learns only when the burst fires, and the burst needs the
distal drive to already match the proximal input. In noise the cue
token's drive rarely matches, so the cue mapping never gets built
(cold start). A8 drops the gate on the learning: W_dist1 += eta *
outer(post, y2_prev), where post is whichever module-1 cells are
active for the current input (p1n > theta_p). The first co-occurrence
already makes the connection; repetition solidifies it. The burst
still fires and still drives the confirmation rule and the
inhibition; only the distal learning is ungated.

Task: planted motifs (baselines.make_data, 200 train / 50 test,
length 24, K=16), alpha 1.0, 5 seeds 0-4, 5 epochs, proto_n=200.
Reference numbers (burst rule, same task and seeds,
rpc_planted_alpha_results.txt): alpha 1.0 full 0.073, control 0.064,
no-confirmation 0.072. Pass bars: acc >= 0.2625 and gain >= 0.10.

1. Planted alpha 1.0 coactivity full: 0.072, range 0.051 to 0.083,
   accs 0.083, 0.082, 0.078, 0.066, 0.051; burst 0.151.
2. Planted alpha 1.0 coactivity no-plasticity: 0.064, range 0.055
   to 0.067, burst 0.025. Identical to the burst-rule control, as
   expected: the control ignores the rule.
3. Planted alpha 1.0 coactivity no-confirmation: 0.073, range 0.051
   to 0.090, burst 0.151.
4. Gain over control plus 0.008, against plus 0.009 for the burst
   rule. BARS NOT MET. The new rule changes nothing on planted
   motifs: 0.072 vs 0.073.
5. Motif sanity check, pure repetition, alpha 1.0, coactivity full:
   0.999, range 0.994 to 1.000, accs 1.000, 1.000, 0.994, 1.000,
   1.000, against 0.934 for the burst rule. The rule works: on clean
   sequences it learns better than the burst-gated rule, near
   perfect. It is not a broken rule; it is a rule that cannot fix
   noise.
6. Reading: the bottleneck is now pinned down. It is not the leak
   rate (alpha sweep: no value fixes the planted task) and it is not
   the learning gate (both rules score the same 0.072 to 0.073 in
   noise, while the new rule scores 0.999 clean). The bottleneck is
   the code: the reservoir state at the cue token is set mostly by
   the random filler tokens before it, so the cue's code differs
   from one sequence to the next and the learned mapping does not
   transfer to held-out sequences. Neither rule can associate a code
   that is not consistent.
7. What would have to change to fix the planted task: the code for
   the cue token would have to be the same across sequences despite
   different random histories, or the readout would have to find the
   cue inside a history-contaminated code (for example a learned
   readout on the distal drive instead of the fixed prototype
   cosine). Both are departures from the thesis model, which fixes
   the projection and the readout. No such test has been run.

## 18. Learned readout on the distal drive (Nov 25, 17:54 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_readout.py`
Driver: `rpc_readout.py`, output `rpc_readout_results.txt`.
Question: the fixed prototype-cosine readout cannot find the cue
inside a history-contaminated distal code. Can a TRAINED readout
find it? Ridge regression from the distal drive d1 (64-dim,
standardised by sd_d1) to the next token (one-hot, K=16), fit on
TRAIN data only (4,600 steps), lambda=1.0 fixed, scored on held-out
test. No leakage: d1 at step t is computed from y2_prev only, before
the current token is fed in, so d1 carries no information about the
current token except through prediction. Each condition also reports
the prototype readout on the SAME network and test data.

Task: planted motifs, alpha 1.0, 5 seeds 0-4, 5 epochs, proto_n=200.
Pass bars for the learned readout: acc >= 0.2625 and gain >= 0.10
over the no-plasticity LEARNED-readout control.

1. Coactivity-full, ridge: 0.157, range 0.131 to 0.179; proto 0.072.
2. Burst-full, ridge: 0.152, range 0.130 to 0.169; proto 0.073.
3. No-plasticity, ridge: 0.163, range 0.150 to 0.182; proto 0.064.
4. The ridge readout doubles the prototype readout (0.157 vs 0.072),
   so the code DOES contain more information than the prototype
   cosine extracts. But the frozen network with a trained readout
   scores 0.163, HIGHER than both trained conditions. The readout
   does all the work; RPC learning adds nothing (gain minus 0.006)
   and slightly degrades the code. BARS NOT MET.
5. Reading: the distal code contains predictive information, but it
   comes from the reservoir plus the fixed random projection, the
   echo-state-network part, not from RPC's burst learning. The
   prototype readout wastes about half the information in d1; the
   trained readout recovers it, and it still falls far short of the
   bar (0.163 vs 0.2625). The information is present but weak and
   diffuse, which is what a history-contaminated code looks like.
6. Departure, stated plainly: a trained readout is NOT in the
   thesis. The thesis prediction is the cosine match against
   prototypes. This test asked only whether the distal code CONTAINS
   the information. Answer: some, about twice what the prototype
   readout finds, and none of it comes from RPC learning.

## 19. Center-surround and normalization screen (Nov 25, 18:00 PST)

Reproduce: `cd ~/workspace/thesis-rpc && ./venv/bin/python rpc_norm.py`
Driver: `rpc_norm.py`, output `rpc_norm_results.txt`, variants A9 in
`rpc_v2.py`. Reza's direction (Nov 25, 17:57 PT): keep trying from
different perspectives and from the literature; noise is handled in
brain circuits by center-surround inhibition and normalization.
Literature, all verified by search Nov 25:
- Srinivasan, Laughlin & Dubs 1982, Proc R Soc B: the retinal
  surround predicts the centre signal and subtracts it, protecting
  fine detail against intrinsic noise. Center-surround IS predictive
  coding, and its purpose is noise protection.
  http://pmc.ncbi.nlm.nih.gov/articles/PMC5509365/ is the 2017 paper
  below, not this one; the 1982 paper is at
  https://ui.adsabs.harvard.edu/abs/1982RSPSB.216..427S/abstract
- "Sensory noise predicts divisive reshaping of receptive fields",
  PLoS Comput Biol 2017: optimal competition under signal-dependent
  noise takes the form of divisive inhibition from the surround.
  http://pmc.ncbi.nlm.nih.gov/articles/PMC5509365/
- "Relating sparse and predictive coding to divisive normalization",
  PLOS Comput Biol 2025: sparse and predictive coding are unified
  through divisive normalization; the normalization signal acts as a
  contextual expectation.
- Carandini & Heeger 2012: normalization is the canonical cortical
  computation, giving robustness to input gain.
- "Divisive Normalization, Line Attractor Networks and Ideal
  Observers": divisive normalization approximates maximum likelihood
  estimation and plays a critical role in noise filtering.
  https://openreview.net/forum?id=ByESVOb_ZB

Task: planted motifs, alpha 1.0, learn_rule="coactivity", prototype
readout, 5 seeds 0-4, 5 epochs, proto_n=200. Screen is full-only;
a variant's own no-plasticity control runs only if its full
accuracy jumps. Reference: coactivity-full 0.072, control 0.064.
Pass bars: acc >= 0.2625 and gain >= 0.10.

1. Surround (A9a, surround_beta=1.0, mean expansion activity
   subtracted before top-k, both modules): 0.068, range 0.046 to
   0.085. WORSE than reference 0.072.
2. d1-sharpen (A9b, sign-preserving square of d1 for the readout
   only): 0.076, range 0.050 to 0.090. Above reference by 0.004,
   which is noise level. No control run: the jump is far too small
   to meet the bars.
3. Post-topk (A9c, coactivity post is the top 10% of p1n instead of
   every cell above threshold): 0.066, range 0.050 to 0.083. WORSE
   than reference.
4. Verdict: none of the three passes. The literature is right that
   normalization helps biological circuits handle noise, but in this
   model the noise is not a gain problem, it is a history problem.
   The cue's code is contaminated by the preceding random tokens in
   the reservoir state, and none of the three mechanisms changes
   that: surround subtraction sharpens the input code but the
   reservoir still mixes history; d1 sharpening boosts contrast but
   the wrong components are strong too; top-k post changes which
   cells learn, not what the code contains.
5. Diagnosis that sets up the next test (section 20): at alpha 1.0
   the system has exactly one step of memory, carried by y2_prev.
   The planted 3-token motif needs two steps: predicting token 3
   needs to know token 1 came two steps ago. Ceiling analysis: if
   the model learns only the "2 after 1" link, accuracy is about
   0.18; learning both links gives about 0.30; the bar 0.2625 sits
   between, so passing it means the model learned the 2-step link.
   The next test is a two-step distal drive (delay line) plus a
   2-token planted diagnostic, which needs only one step of memory
   and has a ceiling of about 0.21.

## 19b. CORRECTION to the ceiling analysis in 19.5 (Nov 25, 18:05 PST)

The 19.5 ceiling analysis was WRONG, and the error changes the
diagnosis. The planted task uses THREE motifs, (1,2,3), (5,6,7),
(9,10,11), not one. A one-step model can score about 0.36 on it:
each of the six motif links (1 to 2, 2 to 3, 5 to 6, 6 to 7, 9 to
10, 10 to 11) has P(next | prev) about 0.67, because each motif
token appears about 3 times in motifs and about 1 time in filler
per sequence. Markov-1 proves it empirically at 0.278. The bar
0.2625 does NOT require two-step memory. A second-order model
(conditioning on prev and prev-prev) raises the ceiling only to
about 0.39, because the one-step links already carry most of the
signal. The delay line (A10) is therefore a LOW-leverage test: its
expected gain is about 0.03, not the difference between fail and
pass. It is still run, as a check, but it is not the bottleneck.
The corrected bottleneck is twofold: (a) code quality, the y2_prev
code is a noisy token code, and (b) learning dilution, the Hebbian
updates on filler steps spread each row's fixed mass across random
associations instead of concentrating it on the motif links. The
sharper next tests are section 20: a linear probe of the code
itself, and a mod2-direct ablation that feeds the clean token into
module 2 instead of the reservoir state.

## 20. Delay line, code probe, mod2-direct (Nov 25, 18:10 PST)

Drivers: `rpc_delay.py`, `rpc_probe.py`, `rpc_direct.py`;
outputs `rpc_delay_results.txt`, `rpc_probe_results.txt`,
`rpc_direct_results.txt`. Task: planted motifs, alpha 1.0,
learn_rule="coactivity", prototype readout unless noted, 5 seeds
0-4, 5 epochs, proto_n=200. Reference: coactivity-full 0.072,
control 0.064, Markov-1 0.278. Pass bars: acc >= 0.2625 and
gain >= 0.10.

1. Delay line (A10, two-step distal drive): full 0.066, range 0.042
   to 0.083; control 0.059. Gain plus 0.007. FAIL, as the corrected
   ceiling analysis in 19b predicted (expected gain about 0.03).
   Memory span is not the bottleneck.
2. Code probe (rpc_probe.py, ridge lambda=1.0, train only):
   y2_prev -> previous token: frozen 0.361 (range 0.318 to 0.390),
   trained-coactivity 0.330 (range 0.283 to 0.374). r1 -> current
   token: 1.000 in both. THE BOTTLENECK IS THE CODE. The module-1
   reservoir holds the current token perfectly, but the module-2
   code the distal drive reads identifies the previous token only
   36% of the time. Training makes the code slightly worse (0.361
   to 0.330), consistent with section 18: the frozen code plus a
   trained readout (0.163) beats the trained code plus a trained
   readout (0.157).
3. mod2-direct (A11, clean token into module 2 through a fixed
   random projection): full 0.088, range 0.052 to 0.113; control
   0.065. Gain plus 0.023. Better than the reservoir-fed 0.072 but
   far below the bar. A clean INPUT to module 2 does not fix it,
   which points the finger inside module 2: its reservoir
   recurrence and top-k sparsification degrade the code after the
   input arrives. The stage-by-stage probe (rpc_probe2.py) and the
   fix variants (rpc_fix.py: rho2=0.0 feedforward module 2,
   y2_frac=0.25 with 16 winners) are running next to locate the
   exact stage and test the fix.

## 21. Stage probe and fix variants (Nov 25, 18:20 PST)

Drivers: `rpc_probe2.py`, `rpc_fix.py`; outputs
`rpc_probe2_results.txt`, `rpc_fix_results.txt`. Task: planted
motifs, alpha 1.0, learn_rule="coactivity", prototype readout
unless noted, 5 seeds 0-4, 5 epochs, proto_n=200. Reference:
coactivity-full 0.072, control 0.064, Markov-1 0.278. Pass bars:
acc >= 0.2625 and gain >= 0.10.

1. Stage probe (rpc_probe2.py, ridge lambda=1.0, frozen network,
   stage -> current token, 5 seeds, chance 0.062): u2 1.000, e2
   0.999, r2 0.984, p2 0.711, y2 0.332. The code degrades in TWO
   places: W_prox (r2 0.984 to p2 0.711) and the top-k y2 (p2 0.711
   to y2 0.332). The reservoir and the expansion are clean; the
   loss is in the projection to 64 proximal units and in keeping
   only 6 winning units. The y2 number here (0.332) matches the
   y2_prev probe (0.361 frozen) within seed noise, as stationarity
   requires.
2. rho0 (rho2=0.0, module 2 feedforward, no recurrence): full
   0.090, range 0.057 to 0.120; control 0.063. Gain plus 0.027.
   Best gain so far (reference gain plus 0.008), still far below
   the 0.10 bar. FAIL.
3. y2k16 (y2_frac=0.25, 16 winners of 64): full 0.076, range 0.046
   to 0.094; control 0.060. Gain plus 0.016. FAIL. More winners
   does not fix it: the extra winners carry mostly noise, so the
   distal drive still cannot find the cue.
4. Reading: every fix so far moves the number by 0.01 to 0.03 and
   none comes close to the bar. The pattern is consistent: the
   information is present in the reservoir (r1 1.000, r2 0.984) and
   is destroyed by the two fixed random stages that follow it
   (W_prox, top-k). The next single-change test is the wide
   proximal layer (rpc_wide.py, n_out=256): if the 64-dim
   projection is the bottleneck, 4x more units should lift p2
   decoding toward 0.98 and end-to-end accuracy with it.

## 22. Wide proximal layer (Nov 25, 18:40 PST)

Driver: `rpc_wide.py`, output `rpc_wide_results.txt`. Task:
planted motifs, alpha 1.0, learn_rule="coactivity", prototype
readout, 5 seeds 0-4, 5 epochs, proto_n=200. Single change:
n_out=256 (y2_k stays 10%, now 25 winners). Pass bars: acc >=
0.2625 and gain >= 0.10. Reference: coactivity-full 0.072,
control 0.064, rho0-full 0.090.

1. Wide256-full: 0.067, range 0.040 to 0.102. Control: 0.067,
   range 0.056 to 0.086. Gain 0.000. FAIL, and the cleanest null
   yet: full equals control exactly.
2. Reading: the dimensionality hypothesis is dead. Giving the
   random projection 4x more units changes nothing, which means
   the p2 drop (r2 0.984 to p2 0.711) is not a capacity problem.
   The random projection mixes signal with history noise in a way
   width cannot fix. Combined with rho0 (0.090, the best gain) and
   y2k16 (0.076), the picture is: the code loses the token at the
   fixed random stages, and no width, winner-count, or recurrence
   change recovers it. The remaining sharp hypothesis is a LEARNED
   proximal projection: train W_prox so it amplifies the token
   signal and suppresses history noise, instead of leaving it
   random. That is a departure from the thesis, where W_prox is
   fixed random, and it is the next test to run.

## 23. Learned proximal projection, A12 (Nov 25, 20:57 PST)

Driver: `rpc_prox.py`, output `rpc_prox_results.txt`, code in
`rpc_v2.py` (ASSUMPTION A12, `fit_prox`, `learn_prox` flag). Task:
planted motifs, alpha 1.0, learn_rule="coactivity", prototype
readout, 5 seeds 0-4, 5 epochs, proto_n=200. W_prox is fit by ridge
regression (lambda 1.0) from the module-2 reservoir state r2 onto a
fixed random 64-dim unit-norm code per token, train data only,
before calibration. Pass bars: acc >= 0.2625 and gain >= 0.10.
Reference: coactivity-full 0.072, control 0.064, rho0-full 0.090,
wide256-full 0.067. Baselines: Markov-1 0.278.

1. Diagnostic, p2 -> token decode (ridge probe, train only):
   random W_prox 0.711, range 0.623 to 0.778; learned W_prox 0.984,
   range 0.975 to 0.993, fit residual MSE 0.0049. The fit recovers
   essentially all the token information r2 holds (r2 itself
   decodes at 0.984). The code is clean.
2. prox-full (learned W_prox, confirmation OFF so the fit is
   frozen): 0.113, range 0.053 to 0.177, burst 0.150. FAIL on both
   bars (gain over its own control is +0.046), but this is the best
   planted score so far: +0.041 over coactivity-full 0.072 and
   +0.023 over rho0-full 0.090.
3. prox-confirm-on (learned W_prox, confirmation ON): 0.076, range
   0.047 to 0.101, burst 0.151. The confirmation rule destroys most
   of the benefit (0.113 -> 0.076). This confirms the A6 mechanism:
   the thesis's own confirmation rule rotates W_prox rows toward
   frequent r2 patterns and wrecks a clean learned code. The
   confirmation rule is not just useless here, it is actively
   harmful to a learned proximal path.
4. prox-no-plasticity (learned W_prox, no W_dist1 learning): 0.067,
   range 0.054 to 0.084, burst 0.023. The clean code alone with a
   random distal map gives nothing. The distal learning does all the
   work in prox-full (+0.046 over this control).
5. Reading: the fixed random W_prox was a real bottleneck, and
   fitting it is the biggest single gain yet, but it does not pass
   the bar. The remaining suspect is the POST side: the coactivity
   post term p1 is still a random projection of r1, so the learned
   rows pair a clean previous code with a noisy current code, and
   the filler steps still spread each row's mass across random
   associations (19b dilution). Next sharp tests: (a) learn
   module-1's W_prox too, so both sides of the Hebbian pair are
   clean codes (symmetric A12); (b) stack the learned ridge readout
   on d1 (section 18 scored 0.157 with the old noisy code) on top of
   the clean code.

## 24. Why the sine fails: ceilings, stage probes, error profile (Nov 25, 21:36 PST)

Reza's order (21:30 PST): establish that the theory can learn before
making the problem bigger; find out why the sine and Lorenz do not
work; deep investigation and trial and error; yes to learning
module-1's projection too.

Task: sine-fixed (frequency 0.05 cycles/step, random phase per
sequence, length 200, K=32 bins, 50 train / 20 test, alpha 1.0,
coactivity rule, 5 epochs). Reference: full 0.091, control 0.022,
chance 0.031. Driver rpc_sine_diag.py, output
rpc_sine_diag_results.txt.

1. Ceilings (Markov fit on train, scored on test, 5 seeds): Markov-1
   0.459 (range 0.405 to 0.549); Markov-2 0.802 (0.776 to 0.833);
   Markov-3 0.833 (0.789 to 0.852). Every test step's previous bin
   is ambiguous (fraction 1.000): the same bin occurs rising and
   falling with different successors. So the one-step prediction the
   thesis makes is capped at 0.459, and two steps of history cap at
   0.802. The network's 0.091 is far below even the one-step ceiling.
2. Stage probes (frozen network, ridge decode fit on train, scored
   on test, 5 seeds; next-bin decode / direction decode, chance
   0.031 / 0.333): r2_prev 0.953 / 0.984; y2_prev 0.607 / 0.748;
   d1 0.602 / 0.740; r1 1.000 / 0.998; p1 1.000 / 0.946. The
   reservoir holds the answer (0.953). The top-6 code throws away a
   third of it (0.607). The learned distal drive transmits what the
   code contains (0.602), so the LEARNING is not the bottleneck; the
   CODE is.
3. Prototype split: same-bin rising-vs-falling prototype cosine
   0.419 (off-diagonal 0.177). The prototypes DO split by direction,
   so the readout is not a blend. The code mixes rising and falling
   (direction decode 0.748 vs the reservoir's 0.984), and the learned
   distal map then averages the two successors.
4. Trained-network error profile (3 seeds): rising accuracy 0.035 /
   0.020 / 0.073, falling 0.157 / 0.085 / 0.098, flat 0.108 / 0.000 /
   0.126. Errors are far, not near: 74 to 83 percent of errors are 6
   or more bins away; only 5 to 7 percent are off by one bin. The
   network picks the wrong direction's successor, not a neighbouring
   bin. The rising/falling asymmetry (rising worse) is an open
   observation, not yet explained.
5. CORRECTION to the earlier hypothesis (rpc_alpha_sine.py
   docstring): the docstring said the failure was that the next bin
   cannot be read from the current bin alone. That is true but it is
   not the binding failure: the reservoir state already contains the
   next bin at 0.953, i.e. the history IS present in r2. The binding
   failures are (a) the top-k code destroys it (0.953 to 0.607), and
   (b) the network's own prediction (0.091) is far below even what
   d1 contains (ridge 0.602), so the learned W_dist1 plus the
   prototype readout lose another ~0.5. The delay line and wider
   code are the direct fixes; the ridge readout on the TRAINED d1
   will separate (a) from (b).
6. New code: rpc_v2.py now has A13 (fit_prox1, learn module-1's
   W_prox onto the same per-token code E as A12, before
   calibration) and A14 (smooth_sigma, Gaussian-bump input instead
   of one-hot, unit norm); step() returns p1 as well. All
   token-to-input conversions (calibrate, build_prototypes, step)
   go through _encode() so the encoding is consistent.
7. Fix tests running (rpc_sine_fix.py, background, 3-seed screen):
   delay, y2_frac 0.25, y2_frac 0.5, delay + y2_frac 0.25, smooth
   1.5, prox1, prox + prox1 (each full + no-plasticity), ridge
   readout on trained d1, and planted-motif prox1 / prox + prox1.
   A 5-seed confirmation follows only for a variant that looks
   close to the bars (acc >= 0.231, gain >= 0.10).

## 25. Sine fix round 1: only the smooth input helps (Nov 25, 22:08 PST)

Driver rpc_sine_fix.py, output rpc_sine_fix_results.txt. All
sine-fixed, alpha 1.0, coactivity rule, 5 epochs, 3 seeds (screen).
Reference (5 seeds, section 15): full 0.091, control 0.022. Bars:
accuracy >= 0.231, gain >= 0.10 over the no-plasticity control.

1. delay (A10, two-step distal drive): full 0.076 (range 0.039 to
   0.099), control 0.024, gain +0.052. FAIL, slightly worse than the
   reference. The two-step ceiling is 0.802 (section 24), but the
   delay line does not reach toward it. Consistent with the probe:
   r2_prev already contains the next bin at 0.953, so the history
   the delay line adds was never missing; the code that carries it
   is the problem.
2. y2_frac 0.25 (16 of 64 winners): full 0.057, control 0.021, gain
   +0.036. FAIL, and worse than the reference. y2_frac 0.5 (32 of
   64): full 0.057, control 0.021, gain +0.036. FAIL. A wider code
   HURTS the network's own prediction even though the probe said the
   top-6 code loses information: wider codes overlap more, so the
   Hebbian learning smears across tokens. The code loss is real but
   widening the code is the wrong repair.
3. delay + y2_frac 0.25: full 0.052, control 0.013, gain +0.039.
   FAIL, the worst of the round. The two non-fixes do not combine
   into a fix.
4. smooth_sigma 1.5 (A14, Gaussian-bump input): full 0.145 (range
   0.116 to 0.168), control 0.031, gain +0.114. PASSES the gain bar
   (>= 0.10), FAILS the accuracy bar (0.231). Best sine score so
   far, +0.054 over the reference. The metric structure matters:
   neighbouring bins share input, so the reservoir state varies
   smoothly with phase and the code mixes rising and falling less.
   This is the only live lead.
5. learn_prox1 (A13, clean module-1 code): full 0.028 (range 0.014
   to 0.043), control 0.046, gain -0.018. FAIL, and full scores
   BELOW its own control and below chance (0.031). Burst rate
   collapses to 0.038 (reference ~0.136). The fitted W_prox makes
   p1 a clean code but breaks the burst gate and the inhibition
   path that runs through W_prox, so the network stops bursting and
   the prediction collapses. A13 is dead for sine.
6. learn_prox + learn_prox1 (both sides clean): full 0.032,
   control 0.044, gain -0.012. FAIL. The "both sides clean"
   hypothesis is dead: cleaning the post side hurts, and adding the
   clean prediction side does not rescue it.
7. Ridge readout on TRAINED d1 (3 seeds): full ridge 0.556 (range
   0.405 to 0.675), prototype 0.078; no-plasticity ridge 0.606
   (0.446 to 0.729), prototype 0.026. Frozen random d1 ridge was
   0.602 (section 24). Two findings: (a) learning makes d1
   slightly WORSE (0.556 trained vs 0.606 frozen control), so the
   Hebbian learning adds no information to the distal drive on
   sine; (b) the prototype readout (0.078) loses ~0.48 against a
   trained readout on the same d1 (0.556). The readout is the
   bigger bottleneck, the learning is the second one.
8. Planted motifs, module-1 projection (alpha 1.0, coactivity, 3
   seeds; reference full 0.072, control 0.064): prox1 full 0.060,
   control 0.063, gain -0.003. prox + prox1 full 0.057, control
   0.066, gain -0.009. Both FAIL. A13 does nothing for planted
   motifs either. The post side was never the bottleneck there
   (section 23 already showed the clean prediction side gives
   +0.046 on its own); cleaning the post side adds nothing and
   slightly hurts.
9. Why the single-matrix Hebbian map cannot learn the sine: one row
   of W_dist1 can only point at one successor pattern. The same
   previous code must map to two different successors (rising and
   falling), so the row learns their average and the prediction
   matches neither. The delay line was meant to supply the context
   that splits the two, but the code mixes the context (direction
   decode 0.748 in y2 vs 0.984 in r2), so the two successors stay
   averaged. The smooth input is the only variant that reduces the
   mixing, and it does so at the input, before the code is formed.
10. Round 2 running (rpc_sine_fix2.py, background): sigma sweep
    1.0 / 2.0 / 3.0, smooth 1.5 + delay, and a 5-seed confirmation
    of smooth 1.5. If the sigma sweep stalls below 0.231, the next
    sharp test is a direction-split prototype set (two prototypes
    per bin, rising and falling, matched by max cosine), which
    attacks the readout gap (0.078 vs 0.556) without a trained
    readout.

## 26. Sine fix round 2: sigma 2.0 reaches 0.212, 5-seed correction on 1.5 (Nov 25, 22:26 PST)

Driver rpc_sine_fix2.py, output rpc_sine_fix2_results.txt. All
sine-fixed, alpha 1.0, coactivity rule, 5 epochs, 3 seeds unless
noted. Bars: accuracy >= 0.231, gain >= 0.10.

1. smooth_sigma 1.0: full 0.132 (range 0.077 to 0.199), control
   0.029, gain +0.103. Passes the gain bar, fails the accuracy bar.
2. smooth_sigma 2.0: full 0.212 (range 0.169 to 0.244), control
   0.019, gain +0.193. Passes the gain bar easily; accuracy is
   0.019 short of the bar. Best sine score so far by a wide margin.
3. smooth_sigma 3.0: full 0.199 (range 0.119 to 0.259), control
   0.027, gain +0.172. One seed hits 0.259, above the bar. The
   sigma trend (1.0: 0.132, 1.5: 0.145, 2.0: 0.212, 3.0: 0.199)
   peaks near 2.0 to 2.5.
4. smooth 1.5 + delay: full 0.150 (range 0.088 to 0.221), control
   0.032, gain +0.118. No better than smooth 1.5 alone (0.145).
   The delay line adds nothing on top of smooth and is dead for
   sine.
5. CORRECTION: smooth 1.5 at 5 seeds is 0.129 (range 0.070 to
   0.168), control 0.031, gain +0.098. It passes NEITHER bar at 5
   seeds. The 3-seed numbers (0.145, gain +0.114) overestimated it
   because seeds 3 and 4 scored 0.141 and 0.070. The 3-seed sigma
   2.0 and 3.0 numbers above are therefore provisional until the
   5-seed confirmations land; do not quote them as final.
6. Round 3 running (rpc_sine_fix3.py, background): 5-seed
   confirmations of sigma 2.0 / 2.5 / 3.0, smooth 2.0 with 10
   epochs (3 seeds), and a mechanism probe (next-bin and direction
   decode from r2_prev, y2_prev, d1 on a frozen smooth-2.0 net) to
   test whether the smooth input raises the code's direction
   information. If the 5-seed peak stalls below 0.231, the next
   sharp test is a direction-split prototype set (section 25.10).

## 27. Sine fix round 3: 5-seed peak is sigma 3.0 at 0.197; mechanism confirmed (Nov 25, 22:46 PST)

Driver rpc_sine_fix3.py, output rpc_sine_fix3_results.txt. All
sine-fixed, alpha 1.0, coactivity rule, 5 epochs unless noted, 5
seeds unless noted. Bars: accuracy >= 0.231, gain >= 0.10.

1. smooth_sigma 2.0, 5 seeds: full 0.182 (range 0.109 to 0.244),
   control 0.021, gain +0.161. Passes the gain bar, fails the
   accuracy bar. Lower than the 3-seed 0.212: seeds 3 and 4 scored
   0.165 and 0.109.
2. smooth_sigma 2.5, 5 seeds: full 0.163 (range 0.104 to 0.259),
   control 0.022, gain +0.141. Worse than 2.0 at 5 seeds.
3. smooth_sigma 3.0, 5 seeds: full 0.197 (range 0.119 to 0.259),
   control 0.022, gain +0.175. Best 5-seed sine score so far.
   Still 0.034 short of the accuracy bar.
4. smooth_sigma 2.0, 10 epochs, 3 seeds: full 0.220 (range 0.177
   to 0.245), control 0.019, gain +0.201. More training helps
   (+0.008 over the 5-epoch 3-seed 0.212). Needs a 5-seed
   confirmation.
5. Mechanism probe CONFIRMED (frozen smooth-2.0 net, 3 seeds;
   next-bin / direction decode): r2_prev 0.944 / 0.987 (one-hot
   0.953 / 0.984, unchanged); y2_prev 0.684 / 0.915 (one-hot 0.607
   / 0.748); d1 0.673 / 0.892 (one-hot 0.602 / 0.740). The smooth
   input raises the code's direction information from 0.748 to
   0.915, which is why the score rises. The reservoir already had
   it; the smooth input lets the code carry it.
6. New code: A15 direction-split prototypes in rpc_v2.py
   (split_proto flag): three prototypes per bin (rising, falling,
   flat, by the sign of tok minus the previous tok), empty combos
   fall back to the plain prototype, prediction is argmax cosine
   over the 3K set with token = idx % K. Smoke test, 1 epoch, seed
   0: split + smooth 2.0 scores 0.204, split + one-hot scores
   0.118. Both run without error.
7. Round 4 running (rpc_sine_fix4.py, background): 5-seed
   confirmations of smooth 2.0 and 3.0 with 10 epochs, split
   prototypes on smooth 2.0 (5 epochs 5 seeds, 10 epochs 3
   seeds), and split prototypes on one-hot (5 epochs 3 seeds).
   If round 4 stalls below 0.231, the next moves are: (a) apply
   the best smooth variant to the variable-frequency sine and to
   Lorenz, to test whether the fix generalises beyond the fixed
   sine; (b) a longer distal history on the smooth code, since
   the code now carries direction cleanly.

## 28. Sine fix round 4: split prototypes add nothing; the fixed sine stalls at ~0.20 (Nov 25, 23:16 PST)

Driver rpc_sine_fix4.py, output rpc_sine_fix4_results.txt. All
sine-fixed, alpha 1.0, coactivity rule, 5 epochs unless noted, 5
seeds unless noted. Bars: accuracy >= 0.231, gain >= 0.10.

1. smooth 2.0, 10 epochs, 5 seeds: full 0.193 (range 0.108 to
   0.245), control 0.021, gain +0.172. +0.011 over its 5-epoch
   5-seed 0.182, and below the 3-seed 0.220.
2. smooth 3.0, 10 epochs, 5 seeds: full 0.199 (range 0.142 to
   0.275), control 0.022, gain +0.177. Same as the 5-epoch 5-seed
   0.197. Ten epochs add nothing at sigma 3.0.
3. split prototypes + smooth 2.0, 5 epochs, 5 seeds: full 0.183
   (range 0.099 to 0.286), control 0.027, gain +0.156. No gain
   over plain smooth 2.0 (0.182). One seed hits 0.286, above the
   bar, but the mean is flat.
4. split prototypes + smooth 2.0, 10 epochs, 3 seeds: full 0.205
   (range 0.153 to 0.268), control 0.019, gain +0.186. Slightly
   WORSE than plain smooth 2.0 with 10 epochs (0.220 at 3 seeds).
5. split prototypes + one-hot, 5 epochs, 3 seeds: full 0.074
   (range 0.042 to 0.091), control 0.030, gain +0.044. No gain
   over the one-hot reference (0.071 at 3 seeds). A15 is dead:
   the split needs direction information in d1, and even when
   smooth gives it that (d1 direction 0.892), the split adds
   nothing, because the 3x prototype set also gives a wrong bin
   three times as many chances to win the cosine match.
6. The fixed sine has stalled. Best 5-seed scores across all
   rounds: smooth 3.0 at 0.197-0.199, smooth 2.0 at 0.182-0.193.
   Every variant that passes the gain bar fails the accuracy bar
   by 0.032 to 0.049. The prototype readout is the binding
   constraint and no tested variant repairs it; the ridge readout
   (0.556-0.673 on d1) shows the information is there and the
   readout cannot extract it.
7. Round 5 running (rpc_sine_fix5.py, background): smooth 3.0
   with alpha 0.7 / 0.5 and with n_res 400 on the fixed sine, and
   the generalisation tests: variable-frequency sine and Lorenz
   with smooth 3.0 (10 epochs), each with its own one-hot alpha
   1.0 reference in the same settings. If round 5 also stalls,
   the honest report is that the thesis readout caps the fixed
   sine at ~0.20, and the next decision is Reza's: stop, accept
   a trained readout (outside the thesis), or change the
   architecture.

## 29. Sine fix round 5, part A: slower reservoir and bigger reservoir on the fixed sine (Nov 25, 23:28 PST)

Driver rpc_sine_fix5.py, output rpc_sine_fix5_results.txt. All
sine-fixed, coactivity rule, 5 epochs, 3 seeds. Bars: accuracy
>= 0.231, gain >= 0.10.

1. smooth 3.0 + alpha 0.7: full 0.210 (range 0.097 to 0.268),
   control 0.027, gain +0.183. Best fixed-sine 3-seed score yet,
   but the range is wide (seed 2 scores 0.097) and it still fails
   the accuracy bar. A slower reservoir plus a smooth input is
   the best combination so far, but the seed variance says the
   5-seed mean would likely regress toward ~0.19-0.20.
2. smooth 3.0 + alpha 0.5: full 0.169 (range 0.035 to 0.263),
   control 0.021, gain +0.148. Worse and unstable (seed 2 scores
   0.035). Too slow a reservoir hurts.
3. smooth 3.0 + n_res 400: full 0.200 (range 0.157 to 0.263),
   control 0.030, gain +0.170. Same as n_res 200 (0.197-0.199).
   Reservoir capacity is not the constraint.
4. INTERRUPTION: the round-5 driver was killed by a runtime
   restart at about 23:28 PT after the fixed-sine block (6 lines
   verified in the results file, no EXIT line, no python process
   left running). The variable-sine and Lorenz blocks never ran.
   Resume driver rpc_sine_fix5b.py runs only those two blocks
   (variable sine and Lorenz, smooth 3.0 with 10 epochs, each
   with its own one-hot alpha 1.0 reference, 5 seeds), output
   rpc_sine_fix5b_results.txt. The fixed-sine lines above are
   final and will not be re-run.

## 30. Sine fix round 5, part B: the smooth fix generalises in direction, not in size (Nov 25, 23:38 PST)

Driver rpc_sine_fix5b.py (resume after the 23:28 runtime restart),
output rpc_sine_fix5b_results.txt. All coactivity rule, 5 epochs,
5 seeds, rpc_spatial defaults (proto_n 20). Bars: accuracy >=
0.231, gain >= 0.10.

1. Variable-frequency sine (each sequence its own frequency 0.02
   to 0.08 and phase; reference one-hot alpha 0.3 is 0.040):
   smooth 3.0 with 10 epochs full 0.108 (range 0.039 to 0.176),
   control 0.036, gain +0.072. One-hot alpha 1.0 full 0.035,
   control 0.030, gain +0.005. The smooth input triples the score
   (0.035 to 0.108) but stalls far below the bar. The variable
   sine needs frequency estimation, which is multi-step history,
   and the prototype readout cannot use it.
2. Lorenz x (reference one-hot alpha 0.3 is 0.045): smooth 3.0
   with 10 epochs full 0.069 (range 0.039 to 0.109), control
   0.041, gain +0.028. One-hot alpha 1.0 full 0.044, control
   0.037, gain +0.007. The smooth input helps a little; the score
   is barely above chance 0.031.
3. Generalisation verdict: the smooth fix helps on every task
   (fixed sine 0.091 to 0.199, variable sine 0.035 to 0.108,
   Lorenz 0.044 to 0.069) but the ceiling falls as the task needs
   more history. The prototype readout is the binding constraint
   on all three. The trained ridge readout on the same distal
   drive scores 0.556-0.673 on the fixed sine, so the information
   is in the drive and the readout cannot extract it.
4. Round 5 is complete. The sine/Lorenz boundary is now firm:
   fixed sine ~0.20 at 5 seeds (best 0.210 at 3 seeds, smooth 3.0
   + alpha 0.7), variable sine ~0.11, Lorenz ~0.07, all below the
   bars. The next decision is Reza's: stop here, accept the
   trained readout (which already passes the fixed-sine bar at
   0.556), or try the one remaining idea inside the thesis's own
   terms, a graded confirmation signal (strength proportional to
   match quality instead of a binary burst gate), so the smooth
   code's graded information drives learning proportionally.

## 31. Sine fix round 6: graded confirmation hurts at the same learning rate; with compensation it is the best 3-seed score (Nov 26, 00:34 PST)

Driver rpc_sine_fix6.py, output rpc_sine_fix6_results.txt. All
coactivity rule, 5 epochs. Bars: accuracy >= 0.231 (fixed sine)
and >= 0.2625 (planted), gain >= 0.10.

1. Fixed sine, smooth 3.0, alpha 1.0, graded, 5 seeds: full
   0.154 (range 0.092 to 0.247), control 0.022, gain +0.132.
   Reference binary: 0.197-0.199. Graded is WORSE by 0.043.
   mean_g 0.181, std_g 0.135.
2. Fixed sine, smooth 3.0, alpha 0.7, graded, 3 seeds: full
   0.196 (range 0.136 to 0.249), control 0.027, gain +0.169.
   Reference binary: 0.210. Graded is slightly worse. mean_g
   0.322, std_g 0.103: a slower reservoir matches better, so
   grading has more signal here, but it still does not help.
3. Fixed sine, one-hot, alpha 1.0, graded, 3 seeds: full 0.057
   (range 0.028 to 0.075), control 0.026, gain +0.031.
   Reference binary: 0.091. Graded is worse. mean_g 0.061: the
   one-hot code carries no graded match information, so g is
   tiny and learning is starved.
4. Planted motifs, alpha 1.0, graded, 5 seeds: full 0.068
   (range 0.039 to 0.099), control 0.054, gain +0.014.
   Reference binary: 0.072-0.073. Graded is slightly worse.
   mean_g 0.048: in noise the prediction almost never matches,
   so graded confirmation turns learning nearly off.
5. Variable sine, smooth 3.0, graded, 3 seeds: full 0.108
   (range 0.033 to 0.195), control 0.039, gain +0.069.
   Reference binary: 0.108 at 5 seeds. Flat. mean_g 0.115.
6. Fixed sine, smooth 3.0, alpha 1.0, graded + eta_b 0.25, 3
   seeds, full only: 0.220 (range 0.157 to 0.283). Best 3-seed
   fixed-sine score yet, one seed at 0.283 above the bar.
   Reference binary at 3 seeds: 0.199. mean_g 0.165, std_g
   0.139. The 5x learning-rate compensation matters: grading
   alone mostly just turns the learning rate down.
7. Verdict on A16: graded confirmation at the same learning
   rate is dead, it hurts everywhere. The diagnostic is
   decisive: g is small wherever the model is weak (planted
   0.048, one-hot 0.061), so grading starves exactly the cases
   that need learning, and it is a rich-get-richer mechanism, not
   a fix. The eta-compensated version is the only live lead from
   this round. Round 6b running (rpc_sine_fix6b.py): 5-seed
   confirmation of graded + eta_b 0.25 at alpha 1.0, and graded
   + eta_b 0.15 at alpha 0.7 (3 seeds), where mean_g was highest.

## 32. Sine fix round 6b: the alpha 1.0 bump was seed noise; the alpha 0.7 version sits 0.001 short of the bar (Nov 26, 00:44 PST)

Driver rpc_sine_fix6b.py, output rpc_sine_fix6b_results.txt. All
coactivity rule, smooth 3.0, graded confirmation on, 5 epochs.
Bars: accuracy >= 0.231, gain >= 0.10.

1. Graded + eta_b 0.25, alpha 1.0, 5 seeds: full 0.196 (range
   0.125 to 0.283), control 0.022, gain +0.174. The 3-seed 0.220
   regressed to 0.196 at 5 seeds (seeds 3 and 4 scored 0.125
   and 0.197), identical to the binary reference (0.197-0.199).
   That bump was seed noise. One seed still hits 0.283, above
   the bar, but the mean is flat. mean_g 0.178, std_g 0.138.
2. Graded + eta_b 0.15, alpha 0.7, 3 seeds: full 0.230 (range
   0.144 to 0.275), control 0.027, gain +0.203. 0.001 short of
   the accuracy bar. Two of three seeds pass the bar
   individually (0.272, 0.275); one fails badly (0.144).
   Reference binary alpha 0.7 smooth 3.0 is 0.210 at 3 seeds, so
   graded + compensation adds +0.020 here. mean_g 0.306, std_g
   0.109: the slower reservoir matches better, so grading has
   the most signal at alpha 0.7, and this is the only condition
   where graded beats binary.
3. Round 6c running (rpc_sine_fix6c.py): 5-seed confirmation of
   graded + eta_b 0.15 at alpha 0.7, plus a 10-epoch version at
   3 seeds. If the 5-seed mean holds near 0.23, that is the first
   pass of the accuracy bar; if it regresses toward 0.21, then
   A16 is dead and the fixed-sine boundary stands at about 0.20.

## 33. Sine fix round 6c: A16 is dead at 5 seeds; the boundary stands at about 0.20 (Nov 26, 00:58 PST)

Driver rpc_sine_fix6c.py, output rpc_sine_fix6c_results.txt. All
coactivity rule, smooth 3.0, graded confirmation on, 5 epochs.
Bars: accuracy >= 0.231, gain >= 0.10.

1. Graded + eta_b 0.15, alpha 0.7, 5 seeds: full 0.202 (range
   0.144 to 0.275), control 0.031, gain +0.171. The 3-seed 0.230
   regressed to 0.202 at 5 seeds (seeds 3 and 4 scored 0.156
   and 0.162). Two seeds still pass the bar individually (0.272,
   0.275), but the mean is 0.029 short. mean_g 0.317, std_g
   0.115. Against the binary smooth 3.0 alpha 1.0 reference at
   5 seeds (0.197-0.199), graded + compensation adds +0.003 to
   +0.005, which is noise.
2. Graded + eta_b 0.15, alpha 0.7, 10 epochs, 3 seeds: full
   0.215 (range 0.119 to 0.279), control 0.027, gain +0.188.
   More training does not help (0.215 against the 5-epoch
   3-seed 0.230). One seed hits 0.279, one fails at 0.119.
   mean_g 0.301, std_g 0.110.
3. Verdict on A16: dead at 5 seeds. Graded confirmation never
   beats binary by more than noise at 5 seeds, and it hurts at
   the same learning rate on every task. The fixed-sine boundary
   stands at about 0.20 (best 5-seed: graded + eta alpha 0.7 at
   0.202, binary smooth 3.0 at 0.197-0.199). Round 7 running
   (rpc_sine_fix7.py): the two remaining cheap combinations,
   no-confirmation on the best config (the confirmation rule was
   never ablated on the sine with smooth input) and the best
   graded config on variable sine and Lorenz.

## 34. Sine fix round 7: the confirmation rule helps, and the best graded config generalises no better (Nov 26, 01:07 PST)

Driver rpc_sine_fix7.py, output rpc_sine_fix7_results.txt. All
coactivity rule, smooth 3.0, 5 epochs, 3 seeds. Bars: accuracy
>= 0.231 (fixed sine), gain >= 0.10.

1. Fixed sine, graded + eta_b 0.15, alpha 0.7, no-confirmation:
   full 0.178 (range 0.124 to 0.218), control 0.027, gain
   +0.151. Reference with confirmation: 0.230 at 3 seeds.
   Turning confirmation off HURTS the graded version by 0.052.
   mean_g 0.238, std_g 0.108.
2. Fixed sine, binary, alpha 0.7, no-confirmation: full 0.186
   (range 0.140 to 0.219), control 0.027, gain +0.159.
   Reference with confirmation: 0.210 at 3 seeds. Turning
   confirmation off hurts the binary version too, by 0.024.
   The confirmation rule is a net positive on the sine, not the
   drag. That ablation is closed.
3. Variable sine, graded + eta_b 0.15, alpha 0.7: full 0.102
   (range 0.051 to 0.163), control 0.040, gain +0.062.
   Reference: 0.108 (binary and graded at alpha 1.0). No
   improvement. mean_g 0.239, std_g 0.118.
4. Lorenz, graded + eta_b 0.15, alpha 0.7: full 0.093 (range
   0.033 to 0.138), control 0.034, gain +0.059. Reference:
   0.069 (binary smooth 3.0 alpha 1.0). Best Lorenz score yet,
   +0.024, but still far below the bar and barely above chance
   (0.031). mean_g 0.261, std_g 0.157.
5. The A16 story is closed. Graded confirmation never beats
   binary by more than noise at 5 seeds, it hurts at the same
   learning rate, and the best graded config generalises no
   better than binary. The hunt's boundary after 7 rounds:
   fixed sine about 0.20 (best 5-seed 0.202), variable sine
   about 0.11, Lorenz about 0.09, planted motifs about 0.11,
   all below the bars. The prototype readout caps at about 0.20
   no matter what is done to the learning. The next decision is
   Reza's: stop, accept the trained readout (0.556 on the fixed
   sine, already measured), or try the one remaining readout
   tweak, a dot-product readout (argmax over the raw dot product
   of d1 with the unit prototypes instead of cosine, so a strong
   prediction counts for more than a weak one).

## 35. Sine fix round 8: dot-product readout helps a little, best 5-seed score yet, still below the bar (Nov 26, 01:31 PST)

Driver rpc_sine_fix8.py, output rpc_sine_fix8_results.txt. All
coactivity rule, 5 epochs. Bars: accuracy >= 0.231 (fixed sine)
and >= 0.2625 (planted), gain >= 0.10. This round was run
proactively (Nov 26, 01:14-01:31 PT) so the result is ready
whichever of the three options Reza picks after round 7.

1. Fixed sine, graded + eta_b 0.15, alpha 0.7, dot, 5 seeds:
   full 0.217 (range 0.147 to 0.288), control 0.036, gain
   +0.181. Reference cosine: 0.202 at 5 seeds. Dot is +0.015
   better. Two seeds pass the bar individually (0.288, 0.241),
   but the mean is still 0.014 short. Best 5-seed fixed-sine
   score yet.
2. Fixed sine, binary, smooth 3.0, alpha 1.0, dot, 5 seeds:
   full 0.209 (range 0.149 to 0.250), control 0.023, gain
   +0.186. Reference cosine: 0.197-0.199. Dot is +0.010 to
   +0.012 better. Best binary 5-seed score.
3. Fixed sine, one-hot, alpha 1.0, dot, 3 seeds: full 0.074
   (range 0.061 to 0.092), control 0.021, gain +0.053.
   Reference cosine: 0.071 at 3 seeds. Flat (+0.003). The
   prototype norms vary too little with one-hot input for the
   weighting to matter.
4. Planted motifs, alpha 1.0, dot, 5 seeds: full 0.078 (range
   0.053 to 0.109), control 0.053, gain +0.025. Reference
   cosine coactivity: 0.072-0.073. Dot is +0.005 better. Still
   far below the bar.
5. Verdict on A17: a small, consistent improvement on the
   smooth fixed sine (+0.010 to +0.015), flat elsewhere. The
   prototype-norm weighting works as designed (norms 0.55 to
   0.75, std 0.061) but the variation is modest, so the gain is
   modest. The readout story is now complete: cosine 0.202,
   dot 0.217, trained ridge 0.556-0.673 on the same drive. The
   prototype readout, in either form, caps around 0.20-0.22.
   The final boundary after 8 rounds: fixed sine about 0.22,
   variable sine about 0.11, Lorenz about 0.09, planted motifs
   about 0.11, all below the bars. The remaining decision is
   Reza's: stop, or accept the trained readout.

## 36. Round 9: the trained readout generalises in score, but the learning adds nothing anywhere (Nov 26, 03:22 PST)

Driver rpc_readout_gen.py (sine half) plus a Lorenz resume (same
run_gen, appended to rpc_readout_gen_results.txt), output
rpc_readout_gen_results.txt. All smooth 3.0, alpha 1.0, coactivity
rule, 5 epochs, K=32, 3 seeds. The sine half was interrupted by a
runtime restart at 03:22 PST after 2 of 4 lines; the Lorenz half
was re-run cleanly (77 seconds, exit 0). Bars (by analogy with the
fixed sine): accuracy >= 0.231 (chance 0.031 + 0.20) and gain >=
0.10 over the no-plasticity LEARNED-readout control.

1. Variable sine, trained ridge readout: full 0.244 (range 0.235
   to 0.256), no-plasticity control 0.234 (range 0.228 to 0.246),
   gain +0.010. Prototype readout on the same networks: full
   0.099, control 0.039. The ridge passes the accuracy bar but
   fails the gain bar by a wide margin.
2. Lorenz, trained ridge readout: full 0.251 (range 0.195 to
   0.325), no-plasticity control 0.295 (range 0.264 to 0.340),
   gain -0.044. The FROZEN control scores HIGHER than the trained
   network. Prototype readout on the same networks: full 0.088,
   control 0.036. The ridge passes the accuracy bar; the gain bar
   fails with a negative gain.
3. The pattern is now the same on all four tasks. Fixed sine:
   trained ridge 0.556, frozen 0.606 (gain -0.050). Variable sine:
   trained 0.244, frozen 0.234 (gain +0.010). Lorenz: trained
   0.251, frozen 0.295 (gain -0.044). Planted motifs (section 18):
   trained 0.157, frozen 0.163 (gain -0.006). The frozen random
   code plus a trained readout matches or beats the trained
   network plus a trained readout everywhere.
4. Verdict on "accept the trained readout": the trained readout
   rescues the SCORE on every continuous task (fixed sine 0.556,
   variable sine 0.244, Lorenz 0.251, all above the 0.231 bar),
   but it does not rescue the THESIS. The information is in the
   fixed random reservoir, not in the learning. The RPC learning
   (Hebbian plasticity, burst gate, confirmation) contributes
   nothing to the readout's score on any task, and on Lorenz and
   planted motifs it makes the code slightly worse. Accepting the
   trained readout would mean showing a standard reservoir result
   (a fixed random reservoir plus a trained linear readout, the
   echo state network result) rather than proof that random
   predictive coding learns. The planted-motif accuracy bar
   (0.2625) is not met by the trained readout either (0.157), so
   even the reservoir result does not cover noise.
5. The hunt's final boundary after 9 rounds: prototype readout
   caps at fixed sine about 0.22, variable sine about 0.11,
   Lorenz about 0.09, planted motifs about 0.11; trained ridge
   readout reaches fixed sine 0.556, variable sine 0.244, Lorenz
   0.251, planted 0.157, with the frozen control equal or better
   in every case. The remaining decision is Reza's: stop, or
   accept the trained readout knowing it proves the reservoir,
   not the learning rule.

## 37. Round 10: all cells project back, and the readout without averaging (Nov 26, 04:33 PST)

Drivers rpc_sine_fix10.py (sine + code probe) and
rpc_planted_fix10.py, outputs rpc_sine_fix10_results.txt and
rpc_planted_fix10_results.txt. Reza's order (Nov 26 03:17 PT):
drop the top-6 code and fix the readout; both are my
assumptions, not the theory; keep finding the root cause and
fixing it until the simulation is a mature RPC theory.
A18 (full graded code): _y2_from_p2n returns relu of the
per-step standardised drive, so every cell above the step mean
projects back, graded, no rank cutoff. A19 (nearest-neighbour
readout): build_prototypes stores every training p1 vector in
a bank (up to 4000) and predict_from_distal returns the token
of the closest bank vector by cosine, no averaging.

CORRECTION (Nov 26): the prototypes were always mean p1 (the
proximal drive), not mean d1 as stated in the round-8
explanation to Reza. The readout already matched distal
against proximal, so A19 tests the averaging, not the space.
The code in build_prototypes sums p1; the earlier wording was
wrong and is corrected here.

A18 false start, fixed in the record: the naive relu(p2n)
scored 0.013, below chance, because p2 grows about 15x in
scale during training while sd_p2 is frozen from calibration,
so the code blew up to a mean of 6.09 (frozen: 0.41). The
top-6 code hid this drift because rank is scale-invariant.
The fix is per-step standardisation (divisive normalisation):
relu((p2n - mean) / std) across the 64 units on that step.
Smoke test (1 epoch, seed 0) after the fix: A18 0.234, y2
mean 0.420, control at chance 0.031.

1. Sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
   coactivity, 5 epochs, 5 seeds, chance 0.031, bars
   accuracy 0.231 and gain 0.10. References (cosine, top-6):
   graded+eta alpha 0.7 is 0.202, gain +0.171; dot readout
   (A17) is 0.217, gain +0.181.
   - A18 full code, full: 0.105 (range 0.028 to 0.218),
     accs [0.213, 0.218, 0.028, 0.037, 0.028], control 0.023,
     gain +0.082. Bimodal: seeds 0 and 1 work at baseline
     level, seeds 2, 3, 4 collapse to chance. WORSE than
     top-6, and unstable.
   - A19 kNN, full: 0.218 (range 0.129 to 0.304), accs
     [0.304, 0.220, 0.129, 0.273, 0.164], control 0.028,
     gain +0.190. Best 5-seed fixed-sine score yet, ahead
     of cosine (0.202) by 0.016 and tied with dot (0.217).
     Gain bar passes; accuracy bar fails by 0.013.
   - A18+A19, full: 0.106 (range 0.013 to 0.263), accs
     [0.263, 0.190, 0.013, 0.042, 0.021], control 0.017,
     gain +0.089. Inherits A18's collapse.
2. Code probe (trained sine, 3 seeds, ridge decode of the
   current token): top-6 y2 0.516, d1 0.509, r1 1.000;
   full code y2 0.709, d1 0.457, r1 1.000. The full code
   DOES carry more token information (+0.193), but the
   learned distal drive carries LESS (-0.052). The code
   is fixed; the learning is not transmitting it.
3. Planted motifs, alpha 1.0, coactivity, 5 epochs,
   5 seeds, chance 0.0625, bars 0.2625 and gain 0.10.
   References: cosine 0.072-0.073, dot 0.078, learned
   W_prox (A12) 0.113.
   - A18 full: 0.065 (control 0.060, gain +0.005).
   - A19 kNN: 0.072 (control 0.060, gain +0.012).
     Ties cosine exactly.
   - A18+A19: 0.066 (control 0.062, gain +0.004).
   Neither fix moves planted motifs at all.
4. Root cause, measured: the code and the learning want
   opposite things. The graded code is richer (y2 decode
   0.516 to 0.709), but each Hebbian update now spreads
   across about 32 active columns instead of 6, so each
   row's fixed mass (row-norm preservation, A5) is diluted
   about 5x per column and the learned map is smeared
   (d1 decode 0.509 to 0.457). The top-6 sparsity was doing
   real work for the learning: it concentrates each update.
   "All cells project back" is right for the code and wrong
   for the learning, as implemented. The next test (round
   11) is to separate them: keep the rich graded code for
   the drive, and sharpen the learning pre-term so the
   update concentrates on the strongest cells (A20), plus
   the clean-code combination A12 + A19.
5. Readout story, updated: cosine 0.202, dot 0.217, kNN
   0.218, trained ridge 0.556-0.673. The three
   theory-faithful readouts now cluster at 0.20-0.22, so
   the readout is no longer the binding constraint; the
   ceiling holds at about 0.22. The remaining gap to the
   0.231 bar is 0.013.
6. Boundary after 10 rounds: fixed sine about 0.22,
   variable sine about 0.11, Lorenz about 0.09, planted
   motifs about 0.11, all below the bars. The trained
   ridge readout reaches 0.244-0.556 but the frozen
   control matches or beats it everywhere (section 36).

## 38. Round 11: sharpening the learning pre-term does not fix the collapse; the fixes do not stack (Nov 26, 06:18 PST)

Round 11 tested A20 (sharpened learning pre-term: y2_prev raised
to the power p, renormalised, so the update concentrates while the
drive reads the raw graded code) and the A12 + A19 combination.
The first run was killed by a runtime restart after the two
sharpened sine conditions; the rest ran from rpc_sine_fix11b.py
(resume driver) and rpc_planted_fix11.py. Drivers:
rpc_sine_fix11.py, rpc_sine_fix11b.py, rpc_planted_fix11.py.
All sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
coactivity, 5 epochs, 5 seeds, chance 0.031, bars accuracy
0.231 and gain 0.10.

1. Sine-fixed:
   - A18+sharp2, full: 0.099 (range 0.029 to 0.225), accs
     [0.169, 0.225, 0.034, 0.037, 0.029], control 0.023,
     gain +0.076. Still bimodal: seeds 0, 1 work, seeds 2,
     3, 4 collapse to chance.
   - A18+sharp3, full: 0.110 (range 0.028 to 0.227), accs
     [0.195, 0.227, 0.061, 0.037, 0.028], control 0.023,
     gain +0.087. Same bimodal pattern. p=3 concentrates
     harder and changes nothing.
   - A12+knn, full: 0.133 (range 0.045 to 0.210), accs
     [0.210, 0.197, 0.045, 0.115, 0.097], control 0.046,
     gain +0.087. Worse than A19 alone (0.218).
   - A18+A19+sharp2, full: 0.113 (range 0.028 to 0.266),
     accs [0.266, 0.196, 0.029, 0.047, 0.028], control
     0.017, gain +0.096. The kNN readout does NOT rescue
     the A18 collapse; it inherits it.
2. Code probe (trained sine, sharp2, 3 seeds, ridge decode of
   the current token): y2 0.691, d1 0.478, r1 1.000 (d1 range
   0.439 to 0.520). Against round 10 (top-6 y2 0.516, d1
   0.509; full code y2 0.709, d1 0.457): sharpening the update
   does not recover the drive. d1 stays below top-6 by 0.031.
3. Planted motifs, alpha 1.0, coactivity, 5 seeds, chance
   0.0625, bars 0.2625 and gain 0.10:
   - A18+sharp2, full: 0.070 (control 0.060, gain +0.010).
     No better than A18 alone (0.065) or cosine (0.072).
   - A12+knn, full: 0.074 (control 0.070, gain +0.004).
     Worse than A12 alone (0.113, section 23). The kNN
     readout hurts the learned-prox network.
4. Verdict: A20 is dead. The dilution hypothesis in section
   37 point 4 is wrong as stated: concentrating the learning
   update changes nothing (sharp2 0.099, sharp3 0.110, both
   bimodal, both in the same range as A18 alone's 0.105), so
   the mass-spread explanation fails. The drive stays degraded
   (0.478 vs 0.509) even with a concentrated update, so the
   degradation sits in the READ (d1 = W_dist1 times y2_prev
   sums about 32 rows of the map, and the extra weak rows add
   noise at read time), not the WRITE. The collapse is in the
   code and learning, not the readout, because the kNN readout
   cannot rescue it. The fixes do not stack on sine: A12+knn
   is worse than A19 alone (0.133 vs 0.218), both under
   coactivity.
   CORRECTION (Nov 26, 07:15 PT, found while designing round
   13): the planted half of the stacking claim above is
   CONFOUNDED and is withdrawn. A12 alone on planted (0.113,
   section 23) ran under learn_rule="coactivity", but
   A12+knn on planted (0.074) ran under the RPCNetwork
   default, learn_rule="burst" with graded_conf off, because
   rpc_planted_fix11.py passes no learn_rule. The rule, not
   the stacking, may explain the drop. The same confound
   applies to every planted A18/A19/A20/A21 number in sections
   37-39: they all ran under the default burst rule, one-hot
   input, eta_b 0.05, while the sine A18 numbers ran under
   coactivity, smooth 3.0, graded, eta_b 0.15. The clean
   planted stacking test (A12+knn under coactivity) and the
   clean A18-under-coactivity test on planted are round 13.
   The sine stacking claim stands; the planted one does not
   until round 13 lands.
5. Smoke-test warning: the 1-epoch smoke test predicted
   A18+A19+sharp2 at 0.274; 5 epochs gives 0.113. For A18
   conditions the 1-epoch number is not a predictor; the
   collapse emerges with more training. Do not trust 1-epoch
   numbers for the full code.
6. Next test (round 12): A21, sharpen the drive READ
   (y2_prev raised to the power p, renormalised, when computing
   d1, while the learning keeps the raw graded code), plus a
   per-seed probe that decodes the NEXT token from d1 on working
   seeds (0, 1) versus collapsed seeds (2, 3, 4) to localise
   the collapse.
7. Boundary after 11 rounds: unchanged. Fixed sine about 0.22
   (best kNN 0.218), variable sine about 0.11, Lorenz about
   0.09, planted motifs about 0.11, all below the bars.

## 39. Round 12: sharpening the drive read does not fix the collapse; the probe localises it to the readout's space (Nov 26, 07:07 PST)

Round 12 tested A21 (sharpened distal-drive read: y2_prev
raised to the power p, renormalised, when computing d1, while
the learning keeps the raw graded code) and a per-seed probe
that decodes the NEXT token from d1 on working versus collapsed
seeds. Drivers: rpc_sine_fix12.py, rpc_planted_fix12.py.
All sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
coactivity, 5 epochs, 5 seeds, chance 0.031, bars accuracy
0.231 and gain 0.10.

1. Sine-fixed:
   - A18+read2, full: 0.119 (range 0.029 to 0.268), accs
     [0.268, 0.224, 0.035, 0.037, 0.029], control 0.025,
     gain +0.094. Still bimodal: seeds 0, 1 work, seeds 2,
     3, 4 collapse to chance.
   - A18+read3, full: 0.129 (range 0.037 to 0.247), accs
     [0.247, 0.224, 0.047, 0.037, 0.087], control 0.028,
     gain +0.101. Gain bar passes by 0.001; accuracy is far
     short. Still bimodal.
   - A18+read2+knn, full: 0.108 (range 0.031 to 0.260),
     accs [0.260, 0.173, 0.031, 0.047, 0.031], control
     0.013, gain +0.095. The kNN readout does not rescue the
     sharpened read either.
2. Per-seed probe: ridge decode of the NEXT token from d1 on
   held-out test, trained network, 5 seeds. d1 at step i is
   computed from the code of token i-1, so decoding token i
   from d1 is the next-token decode.
   - Top-6 baseline: mean 0.507, per-seed [0.535, 0.505,
     0.486, 0.461, 0.549]. Every seed decodes well.
   - A18 full code: mean 0.408, per-seed [0.462, 0.414,
     0.495, 0.295, 0.373].
   The probe does NOT split cleanly by working versus
   collapsed seeds. Seed 2 decodes at 0.495 (as good as the
   top-6 baseline) yet its network accuracy is 0.035. Seed 3
   decodes at 0.295 (the drive is genuinely bad there) and
   seed 4 at 0.373. The extraction ratio (network score
   divided by probe decode) is about 0.55 on the working
   seeds (0.268/0.462, 0.224/0.414) and 0.07-0.13 on the
   collapsed seeds (0.035/0.495, 0.037/0.295, 0.029/0.373).
   The top-6 baseline extracts at 0.40 on every seed.
3. Planted motifs, alpha 1.0, coactivity, 5 seeds, chance
   0.0625, bars 0.2625 and gain 0.10:
   - A18+read2, full: 0.076 (control 0.062, gain +0.014).
     Slightly above A18 alone (0.065) and cosine (0.072),
     far below the bar.
   - A18+read2+knn, full: 0.074 (control 0.064, gain
     +0.010). No move.
4. Verdict: A21 is dead. Sharpening the read (0.119, 0.129)
   does no better than sharpening the write (0.099, 0.110)
   or the plain full code (0.105); the collapse survives
   both. The probe reframes the collapse: it is not that the
   map is degenerate on collapsed seeds. On seed 2 the drive
   contains the next-token information (ridge 0.495) but the
   fixed-space readout extracts none of it (network 0.035),
   while the trained ridge extracts it. The learned distal
   map does not reliably LAND in the prototype space: on
   seeds 0, 1 it lands close enough (extraction about 0.55);
   on seeds 2, 3, 4 it does not (extraction 0.07-0.13), even
   when the drive is informative. Both the cosine and the kNN
   readout are fixed-space comparisons against p1 patterns,
   so both fail the same way. The full code makes the landing
   less reliable than top-6 (extraction 0.40 on every seed,
   no collapse). The next test (round 13) is twofold: (a)
   A18 with the thesis-faithful burst rule instead of
   coactivity, which gates learning to steps where the
   prediction already matches and may land the map better;
   (b) A22, project d1 onto the span of the prototypes
   before the cosine comparison, which tests directly whether
   the misalignment is the problem.
5. Boundary after 12 rounds: unchanged. Fixed sine about
   0.22 (best kNN 0.218), variable sine about 0.11, Lorenz
   about 0.09, planted motifs about 0.11, all below the bars.
   The trained ridge readout reaches 0.556-0.673 on fixed
   sine and the probe shows it would rescue even the collapsed
   seeds, but the frozen control matches or beats it
   everywhere (section 36), so it proves the reservoir, not
   RPC learning.

## 40. Round 13: the burst rule is dead on the full code; the projection is a proven no-op; the stacking failure is real (Nov 26, 07:47 PST)

Round 13 tested A18 under the thesis burst rule (never tested
on sine), A22 (prototype-span projection readout), and the
clean planted rule tests that resolve the section 38
correction. The first run was killed by a runtime restart
after the burst and proj sine conditions; the rest ran from
rpc_sine_fix13b.py (resume driver) and rpc_planted_fix13.py.
Drivers: rpc_sine_fix13.py, rpc_sine_fix13b.py,
rpc_planted_fix13.py, rpc_proj_check.py (diagnostic).
All sine-fixed, graded + eta_b 0.15, alpha 0.7, smooth 3.0,
5 epochs, 5 seeds, chance 0.031, bars accuracy 0.231 and
gain 0.10. Base rule is coactivity unless the condition says
burst.

1. Sine-fixed:
   - A18+burst, full: 0.022 (range 0.017 to 0.037), accs
     [0.018, 0.037, 0.020, 0.017, 0.018], control 0.023,
     gain -0.001. BELOW chance. Learning adds nothing.
   - A18+proj, full: 0.105 (range 0.028 to 0.218), accs
     [0.213, 0.218, 0.028, 0.037, 0.028], control 0.023,
     gain +0.082. Per-seed accs EXACTLY identical to A18
     alone (section 37).
   - A18+burst+proj, full: 0.022, accs identical to
     A18+burst. Confirms the no-op empirically, as predicted.
2. A22 diagnostic (rpc_proj_check.py, trained A18 network,
   seed 0, 3980 test steps): the flag is applied
   (_proto_proj set, 64x64, idempotent, symmetric, rank 32),
   and the projection DOES change d1 substantially
   (cos(d1, projected d1) 0.689 mean; the projection removes
   31% of d1's energy). Yet the accuracy is unchanged, because
   the projection is a MATHEMATICAL no-op for the cosine
   readout: (P d1).proto_k = d1.proto_k for every prototype
   in the span (P is symmetric and P proto_k = proto_k), and
   cosine normalises away the scale change. The same argument
   holds for the dot readout (A17). A22 is dead as a fix for
   cosine and dot; it is only non-trivial for the kNN readout,
   whose bank vectors are not in the prototype span. The
   orthogonal part of d1 (31% of its energy) is exactly the
   information the ridge probe sees but cosine cannot.
3. Per-seed probe: ridge decode of the NEXT token from d1,
   A18+burst: mean 0.462, per-seed [0.546, 0.488, 0.543,
   0.348, 0.387]. Against A18 coactivity (mean 0.408) and
   top-6 (mean 0.507): the burst drive is MORE informative
   to a trained decoder than the coactivity drive, yet its
   network score is 0.022, an extraction ratio of 0.05
   against coactivity's 0.26 and top-6's 0.40. The burst gate
   prevents error-correction (it only learns on steps where
   the prediction already matches), so the map develops an
   idiosyncratic encoding that a trained decoder can read
   but the fixed readout cannot. The gate makes the
   misalignment worse, not better.
4. Planted motifs, alpha 1.0, 5 seeds, chance 0.0625, bars
   0.2625 and gain 0.10:
   - A18+coact, full: 0.075 (control 0.060, gain +0.015).
     Slightly above A18 under burst (0.065). Coactivity
     beats burst for A18 on planted too.
   - A12+knn+coact, full: 0.075 (control 0.070, gain
     +0.005). Against A12 alone under coactivity (0.113,
     section 23): the clean stacking test resolves the
     section 38 correction. The stacking failure is REAL:
     the kNN readout hurts the learned-prox network even
     under the same rule. The withdrawn claim is reinstated
     on clean evidence.
   - A18+proj, full: 0.065 (control 0.060, gain +0.005).
     Identical to A18 under burst (0.065). The projection
     is a no-op on planted too, as the linear algebra
     predicts.
5. Verdict: the burst rule is dead on the full code (0.022);
   the thesis gate cannot bootstrap a noisy code. A22 is
   dead for cosine and dot by proof, not just by
   measurement. The collapse is now precisely stated: the
   drive carries the next-token information largely in
   components orthogonal to the prototype span, and no
   fixed comparison against p1 patterns can see it. The
   next test (round 14) is A23, drive prototypes: compare
   d1 against mean-d1 templates built from the trained
   network, so the comparison vectors live in d1-space and
   the cosine can see the orthogonal part. It is a
   nearest-centroid readout in drive space, using train
   data only, with no trained weights.
6. Boundary after 13 rounds: unchanged. Fixed sine about
   0.22 (best kNN 0.218), variable sine about 0.11, Lorenz
   about 0.09, planted motifs about 0.11, all below the bars.

## 41. External critique: Kimi and AGY (Nov 26, 2026)

Reza ordered both critics to read the report, all 40 sections, the
code, and all results files, and to say how to do it right. Kimi
(kimi-critique.md, 32,035 bytes, 240 lines) and AGY
(agy-critique.md, 52,723 bytes) landed Nov 26, 08:20-08:21 PT in
/home/reza/zfs-1tb/rpc-sim/review/ on Beast. Muse verified every
checkable claim against the code and reran AGY's three empirical
checks (driver rpc_verify_agy.py, 1 seed each). All three
reproduced AGY's numbers exactly. This section records the
verified corrections, the withdrawn claims, and the agreed next
experiment. The report (RPC-HUNT-REPORT.md) is NOT edited while
the corrections are folded in; this section is the correction
record.

### 41.1 Verified corrections (code-checked by Muse)

1. "No-confirmation" is misnamed (Kimi). use_confirmation only
   gates the mod2 W_prox update (rpc_v2.py:1059-1068), NOT the
   distal update (1026-1058). The full vs no-confirmation
   comparison measures the mod2 proximal confirmation plasticity
   only. "Confirmation adds +0.088" on pure motif (section 15),
   "confirmation hurts on Lorenz and the track", and the
   round-7 finding "the confirmation rule is a net positive"
   are all misinterpreted: they measure the W_prox update, not
   gating distal learning. The right controls are separate:
   (a) distal plasticity on/off, (b) proximal confirmation
   plasticity on/off, (c) distal post term with/without the
   burst factor.
2. The burst is NOT a token-correctness signal (Kimi). b1[j]=1
   exactly when p1n[j]>theta_p AND d1n[j]>theta_d
   (rpc_v2.py:1003-1006); the token prediction is computed
   separately (1013-1014) and is not used to form b1. A burst
   can occur on a wrong-token prediction. The section-40
   explanation "the gate only learns from steps where the
   prediction already matches" is about a different gate. The
   thesis describes a collision between a returning random
   transform and newly active proximal cells, which is closer
   to the implemented per-cell coincidence. Burst probability
   conditional on correct/incorrect token predictions was never
   measured.
3. The planted stacking test is STILL confounded (Kimi and
   AGY). A12 alone (0.113) used use_confirmation=False
   (rpc_prox.py:79-87), but A12+knn+coact (0.075) did not
   (rpc_planted_fix13.py:29-33), so proximal confirmation
   rewrote the fitted projection. The section-40 claim "the
   stacking failure is REAL on clean evidence" is WITHDRAWN
   a second time. The clean test needs A12 + confirmation-off
   + kNN vs A12 + confirmation-off + cosine. AGY's 1-epoch
   check (conf OFF): A12 alone 0.1426, A12+knn 0.0765, so the
   drop persists when fixed, but round 13 did not prove it.
4. A18 burst/coactivity is confounded by graded_conf (Kimi).
   rpc_sine_fix13.py base has graded_conf=True; A18+burst sets
   graded_conf=False (and learn_rule="burst"). The 0.022 vs
   0.105 comparison changes both the rule and the grading. The
   probe comparison has the same difference.
5. Rounds 6 and 8 planted ran under burst, not coactivity
   (AGY). rpc_sine_fix6.py:51 and rpc_sine_fix8.py:48 never
   pass learn_rule="coactivity", so both ran under the default
   burst rule, despite docstrings and the record saying "All
   coactivity". The round-11 correction caught rounds 10-12
   but missed 6 and 8. Round-6 planted (0.068) and round-8
   planted dot (0.078) are burst-rule numbers.
6. The 31 percent orthogonal energy is wrong twice (AGY).
   cos(d1, P d1)=0.689 is a NORM ratio, not an energy
   fraction. Energy fraction is cos^2 = 0.475, so orthogonal
   energy is 52.5 percent, not 31 percent. A random isotropic
   vector in R^64 projected onto a 32-D subspace has 50
   percent orthogonal energy (norm ratio sqrt(0.5)=0.707).
   The A22 diagnostic (0.689) is indistinguishable from
   chance and proves nothing about the drive. The "31 percent
   of energy" claim in sections 38-40 and the report is
   WITHDRAWN.
7. The planted ceiling is 0.287, not 0.36 (AGY). make_data
   plants each motif ONCE per sequence (rpc_v2.py:417-424),
   so each cue token appears once in motifs and about 0.94
   times in filler; P(motif succ | cue) is about 0.51, and
   the 1-step Bayes ceiling is (5.5 + 5.5/16 + 12/16)/23 =
   0.287, matching Markov-1 0.278 and LSTM 0.283. Section
   19b's "each motif token appears about 3 times" confused
   three motifs with three plantings. The 0.2625 bar is 92
   percent of the true optimum, not 73 percent of 0.36. The
   bar was too hard; AGY suggests 0.22 (77 percent of Bayes)
   plus motif-only accuracy.
8. Variable-sine and Lorenz "10 epoch" runs were 5 epochs
   (Kimi, AGY). rpc_sine_fix5b.py:25 and rpc_sine_fix5.py:58
   hardcode epochs=5 under the name "smooth3.0-10ep". The
   variable-sine 0.108 and Lorenz 0.069 (section 30) and the
   round-7 Lorenz 0.093 (fix7, correctly labelled 5 epochs)
   are 5-epoch results. The fixed-sine 10-epoch runs are
   genuine (rpc_sine_fix3.py:42, rpc_sine_fix4.py:22-33).
9. TrackF is 2 steps ahead, not 1 (AGY; Kimi said 1 step).
   At item i, d1 is computed from y2_prev, the code of item
   i-1, whose input was [F(t-1), F(t-2)]; the scored target
   is position t+1 (rpc_v2.py:993-1004, 1149-1157;
   rpc_track_F.py:43-57). The drive's information is from
   t-1 and the target is t+1: two steps ahead. The learning
   teaches the current pattern (position t). The fix is to
   label the current position bin t, or compare against
   2-step persistence.
10. "Z-scores" are global scale calibration, not z-scores
    (Kimi). Calibration divides by one global standard
    deviation over the stacked drive array
    (rpc_v2.py:661-662), with no mean subtraction and no
    per-cell standard deviation. theta_p=1 is not a per-cell
    standard-normal threshold.
11. The report attributes v2 numbers to v1 (AGY).
    RPC-HUNT-REPORT.md:98-100 says the first sweep
    (rpc_toy.py, v1) gave motif gains +0.013 to +0.028, sine
    gain -0.007, Lorenz gain -0.001. Those are v2 results
    (simulation-results.md:40-42). v1 never ran sine, Lorenz,
    or track and had burst 0.000, motif accuracy 0.049.
12. rpc_proj_check ran 2 epochs, not 5 (AGY). Line 37 is
    range(2), with seed+100 and proto_n=20. The 0.689 cosine
    is a 2-epoch diagnostic, not the 5-epoch round-13 network.
13. Probes and main runs use different test sets (AGY).
    Probes use seed+100 with proto_n=20
    (rpc_sine_fix10.py:35,40; fix12:43,48; fix13:46,50);
    run_task uses seed+1000 with proto_n=50
    (rpc_simple.py:65,67). The section-39 extraction ratios
    divide accuracies from different test sets and prototype
    sizes. Plain A18 seed 2 scores 0.028, not the 0.035 used
    in the narrative (fix10 line 1 vs fix12 line 1).
14. Binary-post / raw-prototype mismatch (Kimi). The Hebbian
    post term is binary ((p1n > theta_p) or b1), while the
    prototypes are means of raw, unthresholded p1 vectors.
    The learning target and the readout target differ. This
    is a simpler explanation for drive/readout misalignment
    than the orthogonal story, and it should be tested first
    by comparing prototypes of the actual binary post targets
    with raw-p1 prototypes. A20 and A21 do not address it.
15. W_conf is an independent random matrix, not reciprocal
    (AGY). rpc_v2.py:557 initialises W_conf as randn, with no
    link to W_dist1. The thesis requires reciprocal
    connectivity back to the originating CA1 neuron
    (ch3.txt:1502-1533). Every "confirmation hurts" finding
    tests random weight drift, not the thesis's reciprocal
    rule.
16. mod1.inhib is ON during training and OFF at test (AGY).
    Lines 1069-1070 sit inside "if train and plasticity", so
    inhibition perturbs mod1.r during training (mean ||inhib||
    0.361 against mean ||r1|| 2.044 on sine seed 2, about 18
    percent of state norm) but is identically zero during
    calibrate, build_prototypes, no-plasticity, and test. It
    trains W_dist1 on a shifted state distribution. It is
    also backwards relative to ch3.txt:1540-1561 (Fig. 3.4),
    where inhibition is the post-learning runtime mechanism.
17. W_prox2_row0 scaling bug (AGY). W_prox2_row0 is computed
    from the random W_prox in __init__ (rpc_v2.py:567-568)
    and never updated when fit_prox() overwrites mod2.W_prox
    (line 816). On the first confirmation step, norm_rows
    rescales the fitted W_prox back to random-init row norms.
    It only bites when use_confirmation=True and
    learn_prox=True (the prox-confirm-on condition).
18. fit_prox strips direction (AGY). fit_prox regresses r2
    onto a bin-only target E[tok] (rpc_v2.py:810-811), so the
    fitted W_prox cannot represent rising-vs-falling
    direction. On sine, direction is the key signal. A12
    actively sabotages sine by design, which explains why
    A12+knn on sine scored only 0.133 against kNN alone
    0.218.
19. Baselines are apples-to-oranges (AGY). baselines.py sine
    is fixed period 12, length 24, 200 train / 50 test;
    rpc_spatial.py sine is variable frequency U(0.02,0.08),
    length 100, 100 train / 20 test; rpc_simple.py sine-fixed
    is f=0.05, length 200, 50 train / 20 test. baselines.py
    Lorenz uses dt=0.01, stride=10 (dt_eff=0.10), burn-in 10,
    range [-25,25], 200/50, length 24; rpc_spatial.py Lorenz
    uses dt=0.02, stride=1 (dt_eff=0.02), burn-in 500, range
    [-20,20], 20/10, length 100. The record's RPC-vs-baseline
    comparisons across these tasks are invalid. Baselines
    must run on the exact same data, seeds, lengths, and dt.
20. Seed confounds data and initialisation (Kimi). run_task
    seeds torch and numpy with the same value and regenerates
    train/test data for every seed (rpc_simple.py:60-68).
    Collapsed seeds may reflect data draws, weights, or their
    interaction. Cross data seeds with network seeds.
21. Test-set reuse (Kimi). Many variants were screened on the
    same seeds, then "confirmed" on those same seeds. That is
    adaptive test-set reuse, not an untouched final
    evaluation. Use a validation split for selection and fresh
    seeds for the final run, and report paired per-seed
    differences with uncertainty.
22. The no-plasticity control still learns calibration scales
    and prototypes from training data (rpc_v2.py:1105-1107).
    It is a no-synaptic-plasticity control, not a
    learning-free model. This matters for A23, whose control
    must also fit drive prototypes from the frozen drive.
23. Minor code bug (AGY): A20 pre_sharpen sharpens y2_prev
    for W_dist1 but leaves y2_prev2 unsharpened for W_dist1b
    (rpc_v2.py:1054). Never triggered (delay and pre_sharpen
    were never combined).

### 41.2 Reproductions by Muse (rpc_verify_agy.py, 1 seed each)

Check 1, frozen stage probe, planted, seed 0 (AGY's numbers in
parentheses): n_out=64 top-6 r2 0.983 p2 0.721 y2 0.366
(0.983/0.721/0.365); n_out=64 full r2 0.983 p2 0.721 y2 0.563
(0.563); n_out=256 top-6 r2 0.983 p2 0.976 y2 0.579
(0.976/0.579); n_out=256 full r2 0.983 p2 0.976 y2 0.867
(0.867). VERIFIED: a fixed random W_prox at n_out=256 loses
nothing (p2 0.976 vs r2 0.983). The round-22 verdict
"dimensionality is dead" is WRONG: the wide test measured
only the end-to-end score (0.067) and never probed the code.
The wide code was fine; the end-to-end failure was
downstream (the unsigned Hebbian collapse, section 41.3).

Check 2, A18 sine seed 2 vs seed 0, 1 epoch (AGY's numbers in
parentheses): seed 0 A18 acc 0.234 (0.2337), d1>0 0.906
(0.906), pair_cos 0.969 (0.967), in_span 0.439 (0.439),
uniq_bins 7/32, top1 0.307 (30.7%), W>0 0.819 (0.819),
sv1_E 0.650 (0.650). Seed 2 A18 acc 0.025 (0.0249), d1>0
0.877 (0.877), pair_cos 0.841 (0.845), in_span 0.649
(0.649), uniq_bins 4/32, top1 0.787 (78.8%), W>0 0.791
(0.791), sv1_E 0.518 (0.518). VERIFIED exactly. The
decisive comparison: collapsed seed 2 has MORE in-span
energy (0.649) than working seed 0 (0.439), yet scores
0.025 against 0.234. The orthogonal-information explanation
is DEAD. Seed 2 top-6 1ep: acc 0.050, in_span 0.643,
pair_cos 0.525, uniq 12/32, top1 0.311: high in-span energy
alone does not determine accuracy; the collinearity does.

Check 3, A23 frozen control, sine seed 0, 1 epoch (AGY's
numbers in parentheses): top-6 A23 full 0.313 (0.3126),
no-plasticity 0.444 (0.4440); A18 A23 full 0.198 (0.1982),
no-plasticity 0.363 (0.3626); p1-proto frozen 0.045 top-6
and 0.031 A18. VERIFIED exactly. A23 hands the untrained
network a supervised nearest-centroid classifier and turns
learning into a negative gain (-0.131 top-6, -0.164 A18,
-0.296 on collapsed seed 2 per AGY). DO NOT RUN A23 as the
next experiment (AGY); Kimi says run it as a diagnostic.
The conflict is for Reza to resolve; Muse recommends
skipping A23.

### 41.3 The corrected root-cause chain (AGY's mechanism, verified)

The unsigned Hebbian update (W_dist1 += eta * outer(post, pre),
rpc_v2.py:1048) has post >= 0 and pre >= 0, so every synaptic
change is non-negative. Decompose post and pre into mean plus
fluctuation: E[post y2^T] = qbar ybar^T (rank-1 positive DC
mode) + Cov(post, y2) (the predictive signal). For top-6 binary
y2, ybar_j = 0.094 and ybar^2/Var(y) = 0.104. For A18 relu(z),
ybar_j = 0.399 and ybar^2/Var(y) = 0.467, a 4.5x larger DC-to-AC
ratio. The rank-1 term adds coherently across columns (O(N_o))
while the covariance adds incoherently (O(sqrt(N_o))). After
training, W_dist1 is 75-82 percent positive, so d1 is 88-91
percent positive and 85-97 percent collinear across steps,
while p1 prototypes are zero-mean signed (47.6 percent
positive). Cosine/dot/kNN against p1 patterns are swamped by
the constant DC term and lock argmax onto 3-7 constant bins
(78.8 percent on a single bin on seed 2). Ridge regression
whitens out the rank-1 eigenvalue and still reads 0.49. This
explains high ridge / low cosine without any orthogonal-subspace
story, and explains why A18 collapses harder than top-6, why
wide256 collapsed (0.067) under unsigned coactivity, and why
1-epoch smoke tests mislead for A18 (the DC collapse worsens
with epochs).

Kimi's simpler chain agrees in substance: random recurrent
states contain substantial history; sparsification and
projection reduce what a linear probe recovers but do not
remove all of it; distal plasticity learns binary thresholded
coactivity patterns, not raw p1 prototypes and not a
prediction-error objective; row-norm competition and purely
potentiating updates produce seed-dependent idiosyncratic
codes; a flexible linear readout can decode those codes while
raw-p1 class means often cannot.

### 41.4 What is withdrawn

1. The orthogonal-information explanation (sections 38-40,
   report section 4): the drive does not hide its information
   in the orthogonal complement of the prototype span. The
   evidence was the misread 31 percent (now 52.5 percent,
   chance level) and the A22 no-op (which proves only that
   cosine/dot are insensitive to the orthogonal part, not that
   ridge gets its accuracy there). Kimi's 1-epoch split probe
   found both components carry substantial information
   (full-drive ridge 0.377, Pd-only 0.345, (I-P)d-only 0.349).
2. "Dimensionality is dead" (section 22): the wide test never
   probed the code. At n_out=256 the fixed random code is
   intact (p2 0.976, y2 full 0.867).
3. The planted ceiling 0.36 / 0.39 (section 19b): the true
   1-step Bayes ceiling is about 0.287. The 0.2625 bar is 92
   percent of the optimum.
4. "Confirmation adds +0.088" as distal gating (section 15
   and elsewhere): it measures the mod2 W_prox update only.
5. "Learns only when already correct" (section 40): the burst
   is per-cell coincidence, not token correctness.
6. "The stacking failure is REAL on clean evidence"
   (section 40): still confounded by use_confirmation.
7. "The readout is no longer the binding constraint"
   (section 37): cosine, dot, and kNN are one readout family
   (p1 similarity); their similar scores show a shared
   limitation. The higher linear-probe scores show
   readout/extraction remains binding.
8. "Learning is not the bottleneck" (section 24): contradicted
   by section 25 (plasticity lowers ridge accuracy 0.606 to
   0.556). The frozen map preserves information; the learned
   map degrades it slightly. Learning IS a problem.
9. The first-sweep numbers attributed to v1 (report section
   2.1 item 1): they are v2 results.
10. The variable-sine 0.108 and Lorenz 0.069 as "10 epoch"
    results (section 30): they are 5-epoch results.

### 41.5 The single next experiment (both critics agree)

Replace the unsigned Hebbian distal update with a local
signed dendritic prediction-error rule (Urbanczik and Senn
2014; Widrow-Hoff LMS): W_dist1 += eta * outer(p1 - d1,
y2_prev) / (||y2_prev||^2 + eps), with norm_rows=False and
use_confirmation=False, paired with non-compressive module-2
width n_out=256 (>= n_res=200) and full_code=True, all random
wires fixed (W_exp, W_in, W_res, W_12, W_prox), and the
unmodified p1-prototype cosine readout only. Both p1 and d1
are locally available at the module-1 distal dendrite, so the
rule is local; it directly enforces the thesis's physical
match condition (d1 must match p1) and allows depression as
well as potentiation. AGY's 1-epoch checks (not verdicts):
collapsed sine seed 2 A18 0.025 -> 0.441 (n_out=64) and 0.507
(n_out=256); planted seed 0 A18 n_out=256 0.041 -> 0.141
(beating the 5-epoch learned-W_prox best 0.113 in one epoch).
Kimi's version is the same rule and adds: run A23 first as
the diagnostic that discriminates comparison-space from
centroid hypotheses; AGY says do NOT run A23. The 2x2
factorial AGY specifies: learn_rule in {delta_p1, coactivity,
burst, cov_hebb} x n_out in {64, 256}, all with full_code,
use_confirmation=False, s_inh=0.0; 5 seeds, 5 epochs; tasks
sine-fixed (K=32, alpha 0.7, smooth 3.0) and planted motifs
(K=16, alpha 1.0, one-hot); controls no-plasticity and
cov_hebb on identical seeds; readout p1-prototype cosine
only (ridge as secondary diagnostic only). Pass bars: sine
>= 0.46 (beats Markov-1 0.459, proving real phase/direction
prediction) and gain >= +0.40 over no-plasticity; planted
>= 0.22 (or keep 0.2625) and gain >= +0.15, plus motif-only
accuracy >= 0.75. Before running: fix the inhib train/test
mismatch (apply identically or set s_inh=0.0), make
learn_rule and use_confirmation explicit required arguments
in every driver, align baselines to the exact same data, fix
TrackF's target, and match probe test seeds/proto_n/epochs
to the main runs. Awaiting Reza's decision (options offered
Nov 26, 08:28 PT): run the delta rule next, run A23 first,
or stop here.

## 42. Third external critique + first A23 result (Nov 26, 08:55 PST)

### 42.1 The critique

A third external review arrived (saved: external-critique-3.md). The critic could not run the code; all numbers are taken from the report. Each new claim was checked against the code and the record by Muse. Verdicts:

NEW AND VERIFIED:
1. Conceptual mismatch with Chapter 3. The thesis chapter lists prediction-error populations, hierarchy, top-down predictions and precision-style routing (ch3.txt lines 73, 77, 166, 170, 176, 217, 323, 331). rpc_v2.py has none of these: one module pair, a learned distal matrix, a match gate, no error population, no hierarchy, no omission test. The model is a confirmation-gated associative learner, not canonical predictive coding. The report must say this explicitly. Consequence: the negative result adjudicates this formalisation, not the theory in Chapter 3.
2. Hyperparameter selection on the test set. Alpha 1.0 was raised from 0.3 after seeing the cycle prototypes collapse at lower alpha (section 14, lines 14/22/25). Smooth sigma (1.0/1.5/2.0/3.0) was swept on the test set after the sine failed (sections 25-27). The pass bars were fixed in advance; alpha and sigma were not. Fix for round 15: a validation split, select on validation, run held-out test once.
3. "Loses a third" is accuracy, not information. The stage probe is ridge decode accuracy 0.984 -> 0.711 (planted, alpha 1.0, rpc_verify_agy.py:53). Report lines 204 and 371 say "destroys/loses a third of it/information". No mutual information was measured. Correct phrasing: "drops decode accuracy".
4. A22 refinement. The no-op proof is correct for cosine ranking and dot (denominator ||Pd|| is common across prototypes). It is NOT a no-op for score values/thresholds, and NOT a no-op for kNN: bank vectors are individual p1 vectors (rpc_v2.py:731-732), not confined to the prototype span, so projecting d1 changes kNN cosines. Section 4's claim "no fixed comparison against p1 patterns can see the orthogonal part" is false for kNN (already flagged by Kimi, section 41); the correct statement is cosine/dot against averaged prototypes only.
5. A23 algebra. Verified against build_drive_prototypes (rpc_v2.py:780-801): q_t = mean d1 over steps whose next token is t, and d = Wc, so q_t = W m_t by linearity. The A23 score is a nearest-centroid classifier in drive space with metric G = W^T W. Not a no-op. New confound named by the critic: on the sine, q_t averages rising and falling contexts that share a next token, so the centroid may sit between them and match neither (the same ambiguity split prototypes addressed).
6. Round-15 design: 20 seeds (5 is too few for a bimodal condition), train/validation/test split by sequence, random fixed W_dist1 control, label-shuffled readout, per-seed table with median/IQR and paired seed differences. None of these exist in rounds 1-14.
7. Three versions of the thesis claim (new framing): (a) all wires random, readout fixed: current evidence negative; A23 is the last reasonable fixed-readout test. (b) proximal random, distal learned, readout learned/centroid: reservoir computing; the ridge result already shows the reservoir holds the information, so the RPC gate must add something beyond the ESN. (c) proximal wires must be learned: if so, "fixed random wires are enough" is false for this architecture.

ALREADY ANSWERED IN THE RECORD (the critic did not have section 41):
8. Step-4 probe (decode Pd and (I-P)d separately): Kimi already ran it (section 41, line 1891): full-drive ridge 0.377, Pd-only 0.345, (I-P)d-only 0.349 (1 epoch, seed 0, A18 drive). Both parts decode equally; the orthogonal part carries no more than the in-span part. The "orthogonal information" story was already falsified as a special mechanism; the critic's step 4 asks for a test that exists.
9. Burst rate: already measured in every driver output (mean_burst per condition). Sine full 0.136, no-plasticity 0.022 (rpc_sine_fix14a_results.txt). The critic's step-6 request to measure burst rate is satisfied; the numbers are in the results files.
10. Per-seed reporting: the report already gives the A18 bimodal split (RPC-HUNT-REPORT.md:264-265: seeds 0/1 work at 0.213/0.218, seeds 2/3/4 collapse) and the record carries per-seed accs for every condition. The general point (median/IQR, paired tests) still stands for future rounds.
11. Probe alpha (step 1): the alphas are in the code. Sine diag alpha 1.0 (rpc_sine_diag.py:69,154); planted probe alpha 1.0 (rpc_verify_agy.py:53). So the 0.984 -> 0.711 drop is at alpha 1.0, where reservoir tracking of the current token is near-tautological, as the critic says. The reporting gap is that the report does not state the alpha next to the probe numbers.

ERRORS IN THE CRITIQUE (flagged, not adopted):
12. "The gate learns only when the prediction is already correct" (sections 2.1 and step 6). WRONG as stated, repeating the report's error Kimi already corrected (section 41): b1 is per-cell coincidence (p1n > theta_p) & (d1n > theta_d) (rpc_v2.py:1003-1006); the token prediction is not used to form b1. The spirit (match-gated, cannot correct a misaligned map) survives; the mechanism (token-level correctness) does not.
13. Planted bar "beat the one-step ceiling of about 0.36" (section 5). WRONG: AGY verified the ceiling is about 0.287 (section 41). The critic uses the old uncorrected ceiling, so their planted bar is wrong. Correct bars stay AGY's: planted >= 0.22 (or 0.2625 strict), gain >= +0.15.
14. Sine bar "match Markov-2 (0.802)" (section 5). MISLEADING: Kimi verified Markov-2 0.802 is a seen-context-only empirical baseline, not a ceiling (section 41). A bar set at an empirical baseline that skips unseen contexts overstates what the task requires. AGY's sine bar (>= 0.46, beat Markov-1 0.459) stands.

### 42.2 First A23 result: sine top-6 (driver rpc_sine_fix14a.py, DONE Nov 26, 08:55 PST)

The A23 diagnostic Reza ordered ("do it all", Nov 26 08:36) has its first complete driver. Sine-fixed, 5 seeds, 5 epochs, chance 0.031:
- top6-p1proto full: 0.202 (range 0.144-0.275), control 0.031. Reproduces the reference exactly.
- top6-p1proto no-plasticity: 0.031 (chance).
- top6-driveproto full: 0.281 (range 0.212-0.315, accs 0.275, 0.315, 0.289, 0.312, 0.212), control 0.444.
- top6-driveproto no-plasticity: 0.444 (range 0.349-0.513, accs 0.444, 0.432, 0.481, 0.349, 0.513).

Verdict: A23 CONFIRMS AGY's 1-epoch warning at full 5-seed scale. Drive prototypes raise the trained score (0.202 -> 0.281) but raise the frozen control much more (0.031 -> 0.444). Learning is a net loss of -0.163 against its own control. A23 hands the frozen reservoir a supervised nearest-centroid classifier; it does not show RPC learning adds drive-space class structure. On sine top-6, the critic's version (a) of the thesis claim ("all wires random, readout fixed") fails its last reasonable test: trained A23 does not beat control A23. Remaining A23 conditions (A18 full code sine, planted top-6) and the delta rule are still running in the chain (driver rpc_sine_fix14b.py in progress at 08:55 PST). Full round-14 record follows as section 43.

### 42.3 A23 on the full code: sine A18 (driver rpc_sine_fix14b.py, DONE Nov 26, 09:07 PST)

Sine-fixed, full graded code (A18), 5 seeds, 5 epochs, chance 0.031:
- A18-p1proto full: 0.105 (range 0.028-0.218, accs 0.213, 0.218, 0.028, 0.037, 0.028), control 0.023. Reproduces plain A18 exactly, bimodal as before.
- A18-p1proto no-plasticity: 0.023.
- A18-driveproto full: 0.185 (range 0.168-0.230, accs 0.230, 0.186, 0.168, 0.168, 0.172), control 0.447.
- A18-driveproto no-plasticity: 0.447 (range 0.363-0.525, accs 0.363, 0.525, 0.458, 0.380, 0.509).

Verdict: A23 does NOT rescue the full code. It raises the trained score (0.105 -> 0.185) and removes the bimodal collapse (all five seeds now 0.168-0.230, no seed at chance), which is the first time any change has un-collapsed seeds 2-4. But the frozen control rises further (0.023 -> 0.447), so learning is still a net loss of -0.262, worse than top-6 (-0.163). The sine A23 diagnostic is now complete: top-6 and full code both fail the trained-vs-control test. The pattern is consistent: drive prototypes are a strong frozen-reservoir classifier, and training under the unsigned Hebbian rule makes the drive worse than frozen. Planted A23 (driver rpc_planted_fix14a.py) is running now; the delta rule follows.

### 42.4 A23 on planted motifs: top-6 (driver rpc_planted_fix14a.py, DONE Nov 26, 09:13 PST)

Planted motifs, top-6, 5 seeds, 5 epochs, chance 0.0625, coactivity, confirmation off, eta_b 0.05, alpha 1.0, one-hot:
- top6-p1proto full: 0.074 (range 0.050-0.090, accs 0.083, 0.083, 0.090, 0.063, 0.050), control 0.054, gain +0.020.
- top6-p1proto no-plasticity: 0.054.
- top6-driveproto full: 0.132 (range 0.115-0.151, accs 0.151, 0.115, 0.144, 0.126, 0.125), control 0.129, gain +0.003.
- top6-driveproto no-plasticity: 0.129.

Verdict: A23 does NOT rescue planted motifs either. Trained 0.132 vs frozen 0.129: learning is flat, and both sit far below the 0.22 bar. Unlike sine, the frozen drive-proto control is NOT strong on planted (0.129 vs sine 0.444): the frozen reservoir's drive prototypes do not classify planted motifs well. Note the p1-proto 0.074 is below the A12-alone 0.113 (section 23) because this driver omits learn_prox by design (its docstring: "matching the A12-alone config minus learn_prox"); the gap is a learn_prox contribution (+0.039), not an A23 effect. The A23 diagnostic is now COMPLETE: sine top-6, sine A18, planted top-6. All three fail the trained-vs-control test. The chain has moved to the delta rule (driver rpc_sine_fix14c.py).

## 43. Round 14 (continued): A24 delta rule (AGY spec)

Delta rule (learn_rule="delta"): W_dist1 += eta_d / (||y2_prev||^2 + 1) * outer(p1 - d1, y2_prev), norm_rows off, use_confirmation off, s_inh 0.0, full_code on, p1-prototype cosine readout only. Controls on identical seeds: no-plasticity, coactivity, burst, centred covariance Hebbian ("cov"). Bars: sine accuracy at least 0.46 and gain at least +0.40; planted at least 0.22 and gain at least +0.15, plus motif-only accuracy at least 0.75. (Part results recorded as they land; the chain is still running.)

### 43.1 Sine-fixed, delta at n_out 64 (driver rpc_sine_fix14c.py, DONE Nov 26, 09:13 PST)

Sine-fixed, smooth 3.0, alpha 0.7, eta_d 0.2, 5 seeds, 5 epochs, chance 0.031:
- delta-nout64 full: 0.407 (range 0.336-0.510, accs 0.336, 0.510, 0.453, 0.375, 0.362), mean_y2 0.399.

First time any theory-faithful configuration scores above 0.40 on fixed sine: nearly double the old best (0.218) and far above the 0.231 bar, but still short of Markov-1 (0.459) and of AGY's 0.46 bar. The gain vs the no-plasticity control is not known yet (control still running in this driver). Delta-nout256, the no-plasticity control, and the ridge probe on the delta-256 drive follow.

### 43.2 Sine-fixed, delta at n_out 256 + control (driver rpc_sine_fix14c.py, DONE Nov 26, 09:28 PST)

Sine-fixed, smooth 3.0, alpha 0.7, eta_d 0.2, 5 seeds, 5 epochs, chance 0.031:
- delta-nout256 full: 0.484 (range 0.348-0.629, accs 0.348, 0.629, 0.537, 0.383, 0.525), mean_y2 0.397.
- delta256-no-plasticity: 0.031 (range 0.015-0.075, accs 0.032, 0.016, 0.015, 0.017, 0.075), mean_y2 0.397. The frozen control sits at chance.
- Gain: +0.453.

BOTH AGY BARS PASS on sine-fixed at n_out 256: accuracy 0.484 >= 0.46, gain +0.453 >= +0.40. This is the first theory-faithful configuration to beat Markov-1 (0.459) on fixed sine, and the first to beat the 0.46 bar. The frozen control at chance (0.031) means the gain is genuine learning, not reservoir readout: the same frozen reservoir scores 0.031 here, against 0.444 with drive prototypes under the old rule. Width matters: n_out 64 gives 0.407, n_out 256 gives 0.484. The ridge probe on the delta-256 drive crashed on a constructor bug (proto_n, then use_confirmation, both step/run parameters passed to RPCNetwork.__init__ by mistake); the probe was fixed and resumed with the sine controls and the planted drivers (session proc_22370417f155).

### 43.3 Ridge probe on the delta-256 drive (driver rpc_sine_fix14c_probe.py, DONE Nov 26, 10:13 PST)

Next-token ridge decode from d1, held-out test (test seed+1000, proto_n=50, same data as run_task), 3 seeds:
- trained: 0.913 (range 0.884-0.942, accs 0.884, 0.942, 0.912).
- frozen: 0.917 (range 0.890-0.940, accs 0.890, 0.940, 0.923).

The frozen drive already decodes the next token at 0.917. Delta training adds NO information to the drive (0.913 vs 0.917, identical). The network-score gap (delta trained 0.484 vs frozen 0.031) is therefore entirely a READOUT-ALIGNMENT effect: the delta rule rotates d1 into the p1-prototype space so the fixed cosine readout can use what is already there, while the frozen drive holds the same information in a form the cosine readout cannot use. This is the clean mechanism the hunt was missing: the unsigned Hebbian rule rotated the drive AWAY from the readout space (positive-orthant collapse); the signed error rule rotates it INTO the readout space. It also reframes the ridge result: a trained readout at 0.556-0.673 was never evidence of learning, because the frozen drive already decodes at 0.917.

### 43.4 Sine-fixed, rule controls at n_out 256 (driver rpc_sine_fix14d.py, DONE Nov 26, 10:13 PST)

Same base settings as the delta cell (norm_rows=False, s_inh=0.0, full_code, smooth 3.0, alpha 0.7), only the rule changes, 5 seeds, 5 epochs, chance 0.031:
- coact-nout256: 0.024 (range 0.013-0.038, accs 0.013, 0.023, 0.028, 0.038, 0.018), mean_burst 0.152.
- burst-nout256: 0.028 (range 0.023-0.035, accs 0.025, 0.035, 0.028, 0.029, 0.023), mean_burst 0.110.
- cov-nout256: 0.151 (range 0.062-0.287, accs 0.114, 0.173, 0.120, 0.062, 0.287), mean_burst 0.148.

The 2x2 factorial is clean at n_out 256: unsigned rules collapse to chance without row-norm preservation (coactivity 0.024, burst 0.028, against delta 0.484); centering alone (cov) recovers partway to 0.151 but is far below delta. The error term is what matters, not centering and not width alone: width without the error term gives 0.024-0.028, and the error term at width 64 gives 0.407. Planted delta drivers crashed on a duplicate-alpha bug (run_planted_motif called with alpha both positionally and in DELTA); fixed in both planted drivers and relaunched (session proc_a0045fd41a7e).

### 43.5 Planted motifs, delta rule (driver rpc_planted_fix14c.py, DONE Nov 26, 10:31 PST)

Planted motifs, alpha 1.0, one-hot, eta_d 0.05, full_code, norm_rows off, confirmation off, s_inh 0.0, 5 seeds, 5 epochs, chance 0.0625, motif_n 124:
- delta-nout64: acc 0.151 (range 0.123-0.170, accs 0.123, 0.159, 0.170, 0.138, 0.167), motif_acc 0.641 (range 0.533-0.813, maccs 0.555, 0.813, 0.700, 0.533, 0.606).
- delta-nout256: acc 0.221 (range 0.197-0.247, accs 0.197, 0.230, 0.247, 0.205, 0.227), motif_acc 0.966 (range 0.921-1.000, maccs 0.966, 1.000, 0.977, 0.967, 0.921).
- delta256-no-plasticity: acc 0.062 (range 0.052-0.069), motif_acc 0.054 (range 0.000-0.092). Frozen control at chance.
- Gain: +0.159.

AGY's planted bars: accuracy >= 0.22 PASS (0.221, by 0.001), gain >= +0.15 PASS (+0.159, by 0.009), motif-only >= 0.75 PASS (0.966, comfortably). The accuracy and gain passes are marginal: one seed lower and the mean would fall below the bar. The original strict bar 0.2625 is NOT met (0.221 < 0.2625), and the 1-step Bayes ceiling is 0.287, so delta reaches 77% of optimal on planted. The motif-only score is the strong result: on the motif steps themselves the network predicts the successor 96.6% of the time, against 5.4% frozen. The overall accuracy is held down by the filler steps, where the cue gives no information. Width matters again: 64 -> 0.151, 256 -> 0.221.

### 43.6 Planted motifs, rule controls at n_out 256 (driver rpc_planted_fix14d.py, DONE Nov 26, 10:31 PST)

Same base settings as the delta cell, only the rule changes, 5 seeds, 5 epochs, chance 0.0625:
- coact-nout256: acc 0.059 (range 0.040-0.075, accs 0.040, 0.075, 0.041, 0.070, 0.070), motif_acc 0.059 (maccs 0.000, 0.000, 0.000, 0.295, 0.000).
- burst-nout256: acc 0.059, motif_acc 0.059, per-seed accs and maccs IDENTICAL to coactivity on all five seeds.
- cov-nout256: acc 0.154 (range 0.114-0.202, accs 0.114, 0.202, 0.154, 0.134, 0.164), motif_acc 0.645 (range 0.487-0.992, maccs 0.487, 0.992, 0.669, 0.500, 0.575).

The driver passes distinct learn_rule values (verified by reading rpc_planted_fix14d.py), so the identical coactivity/burst scores are a real result, not a copy-paste bug: both unsigned rules collapse to the same constant prediction at this width (drive saturates at the clamp, cosine locks onto one bin), so the per-seed scores are just that bin's frequency in each test draw. Same pattern as sine: unsigned rules at chance, centering alone partway (0.154, motif 0.645), error term far above (0.221, motif 0.966).

### 43.7 Round 14 verdict

Round 14 is complete. A23 (drive prototypes) fails its diagnostic on all three versions: the frozen reservoir with drive prototypes beats or matches the trained network (sine top-6 -0.163, sine A18 -0.262, planted +0.003). Thesis-claim version (a), all wires fixed random with a fixed readout, is dead; its last reasonable test is the A23 diagnostic, and it fails it.

A24 (local signed prediction-error rule, delta) works on both tasks at n_out 256, with the unmodified p1-prototype cosine readout and all other wires fixed random:
- Sine-fixed: 0.484 vs frozen 0.031, gain +0.453. Passes AGY's bars (0.46, +0.40) and beats Markov-1 (0.459).
- Planted: 0.221 vs frozen 0.062, gain +0.159, motif-only 0.966. Passes AGY's bars (0.22, +0.15, 0.75), marginally on accuracy and gain; fails the strict 0.2625 bar.
- Mechanism (probe, 43.3): the frozen drive already decodes the next token at 0.917; delta training adds no information (0.913). The win is readout alignment: the signed error rule rotates d1 into the prototype space; the unsigned rule rotated it out.
- Controls (43.4, 43.6): unsigned rules collapse to chance at n_out 256 without row-norm preservation; centering alone recovers partway (sine 0.151, planted 0.154); only the error term reaches the bars.

What is still open: (1) planted accuracy is marginal against AGY's bar and below the strict bar; (2) variable sine, Lorenz, and the 1D track have not been run under delta; (3) alpha 1.0 and smooth sigma 3.0 were selected on the test set (third critique), so the delta win needs a validation split and more seeds before it is claimed (round 15 design: 20 seeds, train/validation/test split, random fixed W_dist1 control, label-shuffled readout); (4) the theory as written in Chapter 3 still has no error population, hierarchy, precision weighting, or omission test, so the delta result rescues this formalisation, not the Chapter 3 theory as stated.

## 44. Round-14 critiques land: Kimi and AGY Gemini (Nov 26, 2026)

Both critiques landed on Beast and are copied here as r14-kimi-critique.md (20,429 bytes) and r14-agy-critique.md (49,309 bytes). Both processes are gone. AGY Opus never ran (429 quota, resets about Dec 1); not retried.

### 44.1 Shared verdict

The delta win stands as a strong pilot result, not as a validated pass. Both reviewers agree on three points: (1) the planted 0.221 pass is threshold noise, it fails on fresh seeds; (2) the delta rule is local signed-error learning (normalised Widrow-Hoff / Urbanczik and Senn dendritic LMS), not Chapter 3's burst-confirmation mechanism, which is off or silent in every winning run (use_confirmation=False, s_inh=0.0, mean_burst=0.000 on sine); (3) the headline scores need a fresh-seed, zero-tuning replication before any claim.

### 44.2 Kimi (r14-kimi-critique.md)

- Delta is a verified within-study pilot effect; refuted as a confirmatory pass. Strongest doubt: adaptive test-set selection (alpha, sigma, width, rates chosen on the same test seeds), so the reported scores are optimistically biased. Sine 95% interval [0.340, 0.629] contains the 0.46 bar; planted interval [0.196, 0.246] contains 0.22.
- Planted 0.221 is refuted as meaningfully above 0.22: unrounded mean 0.2212, only 0.0012 above the bar, 2 of 5 seeds below it. The revised 0.22 bar was chosen after seeing results and needs preregistration before a fresh confirmation.
- Motif 0.966 is true-completion recall, not accuracy over every motif-prefix presentation: run_planted_motif counts a step only when the previous two tokens are a prefix AND the current token is the motif successor (rpc_planted_fix14c.py:62-68). Overwritten successors (6-15 per seed) are excluded.
- Identical planted coactivity/burst scores verified as constant-output collapse, not a bug: distinct final matrices, 97.9-99.3% (coactivity) and 89.8-97.3% (burst) of entries at +2.0, both predicting one class on all 1,150 test steps. Confirms 43.6 with the saturation refinement.
- Ridge probe configuration verified, but "no information; entirely readout alignment" refuted: paired differences [-0.006, +0.002, -0.011] on 3 seeds only show this unstandardised, no-intercept, lambda=1 ridge finds no gain. The clean claim is "consistent with converting already linearly decodable information into a p1-cosine-usable form."
- Implementation is clamped NLMS (w_max=2.0, rpc_v2.py:503-506, 1067-1075), not AGY's specified unclamped or wide-clamp 10.0 rule. The rule-by-width factorial is incomplete: controls ran only at n_out 256.
- Beating Markov-1 by +0.0254 is not robust: paired per-seed differences [-0.071, +0.080, +0.099, -0.022, +0.041], interval [-0.063, +0.114]. Markov-2 (0.802) is not scored on an identical denominator (markov_acc drops unseen contexts).
- "Learns in noise" verified only as a narrow existence result on planted random filler; refuted as a general claim. Cheapest next test: variable-frequency sine under the exact frozen delta-256 config, no retuning.
- A23 correction to 43.7: the A23 measurements are real, but 43.7 overstates them. A23 fits labelled class centroids from d1 (rpc_v2.py:780-800), so it is a supervised nearest-centroid readout, not an unlearned fixed readout. Its failure rejects A23 as a rescue, not the whole fixed-features-plus-fixed-readout family; A24 itself is a counterexample if version (a) means fixed random non-distal wires with distal plasticity.
- Next experiment: preregistered fresh-seed, zero-tuning replication of the exact delta-256 config (including the w_max=2.0 clamp), 20 paired replications preferred, fresh sequence and network seeds, delta plus frozen and cov controls, Markov-1/2 on identical transitions, strict first-order delta ablation, all-prefix planted metrics.

### 44.3 AGY Gemini (r14-agy-critique.md)

AGY ran fresh diagnostics on Beast (review-venv, single-threaded CPU) and reports new numbers. These are reviewer-reported: no diagnostic output files were saved in the review directory, so they are recorded here as AGY-measured, not re-run by me.

- Fresh-seed replication (10 untouched seeds 100-109, exact fix14c config): sine 0.467 vs frozen 0.031, planted 0.217 vs frozen 0.062. The learning gain over chance is real and not a selection mirage, but planted drops below the 0.22 bar (6/10 seeds below it; 15-seed mean 0.218) and smooth-sine ties Markov-1 (+0.006, 5/10 wins, p=0.72).
- Biggest surprise: smooth_sigma=3.0 actively hurts delta. Plain one-hot (smooth_sigma=0.0, alpha=1.0) on 3 fresh sine seeds scores 0.847, beating Markov-2 (0.795). The 0.484 headline was capped by a toxic legacy hyperparameter, not by delta.
- Motif 0.966 omits the first motif transition: m1 (1->2) scores only 0.628, all-motif 0.789. The m1 miss accounts for about 69% of the gap to Markov-1 (0.281). Filler steps score 0.059, below chance.
- Sine 0.484 is not a 1-step ceiling: delta predicts direction on 99.4% of ambiguous middle bins, within 2 bins 96.5%. Even the noiseless oracle p1 scores only 0.563 (r=0.943 with delta), because smooth_sigma=3.0 blurs adjacent prototypes (adj_cos 0.952). split_proto on worst seed 0 jumps 0.348 to 0.760.
- cov control is broken: rpc_v2.py:1082-1085 subtracts spatial per-step means (p1n - p1n.mean()), not temporal per-neuron means (Sejnowski 1977), and uses unnormalised eta_b, about 1,000x larger per-step updates than delta, slamming 44.0% (sine) and 17.9% (planted) of weights into the +-2.0 clamp. Fixed temporal-covariance Hebbian (zero error term) jumps planted from 0.154 to 0.199 overall and 0.939 on m2: 90% of delta's overall score and 97% of its m2 score with no error feedback. Delta's genuine planted advantage shrinks to m1 (0.628 vs 0.449).
- Coactivity/burst byte-identity verified as no code bug, but a clamp confound: norm_rows=False pins 99.25% / 97.29% of weights at +2.0; restoring norm_rows=True still leaves both at chance (0.041 / 0.042 planted).
- Ridge probe: configuration verified, mechanism framing refuted. Fixed lam=1.0 penalises trained d1 about 19x harder because delta shrinks ||d1|| 4.4x; standardised ridge gives trained 0.924 vs frozen 0.923. Frozen W_dist1 is almost surely invertible, so no linear matrix can exceed the linear decode of y2 (data processing inequality); delta performs online regression y2 -> p1, not rotation.
- Six fix14 defects: cov centering/clamp bug; norm_rows=False clamp asymmetry for coactivity/burst; mean_burst=0.000 unreported in section 43; n_out=64 controls missing (2x2 factorial incomplete); m1 omitted and mean_burst unlogged in rpc_planted_fix14c.py; rpc_sine_fix14a/b_results.txt missing from the review directory (confirmed) and rpc_sine_fix14c_results.txt contains the pre-fix probe crash traceback (confirmed, lines 4-11).
- Untested tasks, 1-seed checks: Lorenz 0.438 / 0.395 vs persistence 0.587 (fails); 1D track 0.489 / 0.597 vs persistence 0.776 (fails); variable sine 0.292 vs Markov-1 0.282 (barely ties). Bright spot: motif lag-2 0.208 vs Markov-1 0.088, 2.4x, the strongest multi-step-in-noise test.
- Chapter 3 verdict: refuted. A24 is Urbanczik and Senn (2014) / normalised Widrow-Hoff LMS on a fixed echo-state reservoir. Nothing distinctive to Chapter 3 remains in the plasticity mechanism.
- Cheapest next experiment: 20 fresh seeds (200-219), train/validation/test split by sequence, validation picks (alpha, smooth_sigma, eta_d), then one held-out test evaluation. Tasks: planted lag-1, planted lag-2, sine-fixed, baselines.py Lorenz (dt_eff=0.10) plus variable sine. Conditions: delta, fixed temporal-cov, burst with norm_rows=True, frozen. Bars on lower 95% CI bound: planted lag-1 > 0.22 with m1 >= 0.80 and m2 >= 0.90; lag-2 >= 0.20 and >= +0.10 over Markov-1; sine >= 0.80; Lorenz/var-sine beat persistence and Markov-1 on accuracy and MSE.

### 44.4 Spot verification by me (Nov 26, 2026)

Checked against rpc_v2.py and simulation-results.md, not taken on trust: w_max=2.0 default and delta clamp (rpc_v2.py:503-506, 1067-1075); cov spatial centering (rpc_v2.py:1082-1085); motif_acc scores only m2 transitions (rpc_planted_fix14c.py:62-68); mean_burst=0.000 for delta on sine (rpc_sine_fix14c_results.txt:1-2); sine config alpha=0.7, smooth_sigma=3.0, use_confirmation=False, s_inh=0.0 (rpc_sine_fix14c.py:30-32); headline scores match results files (sine accs [0.348, 0.629, 0.537, 0.383, 0.525], planted [0.197, 0.230, 0.247, 0.205, 0.227], motif 0.966); A23 fits labelled mean-d1 centroids from train data (rpc_v2.py:780-800); probe crash traceback present in rpc_sine_fix14c_results.txt:4-11; rpc_sine_fix14a/b_results.txt absent from the Beast review directory (confirmed by listing). AGY's and Kimi's own diagnostic numbers (fresh seeds, clamp fractions, temporal-cov 0.199, one-hot 0.847, oracle p1 0.563) are reviewer-reported with no saved output files, so they are not independently verified here.

### 44.5 Status after the critiques

Round 14's headline passes do not survive as stated: planted fails on fresh seeds, the 0.22 bar pass was noise, the motif metric hid m1, the cov control was invalid, and the sine score was capped by smooth_sigma=3.0. What survives and is stronger than the headline: delta genuinely learns (fresh-seed gains over chance hold), one-hot delta-256 hits 0.847 on sine and beats Markov-2, and delta learns full phase direction (99.4%). What dies: the claim that only the error term works on planted (temporal covariance reaches 90% of delta), the "readout alignment" framing of the probe, and any claim that A24 validates Chapter 3's burst-confirmation theory. Next step is the preregistered 20-fresh-seed validation-split replication both reviewers propose (round 15 design in 43.7, refined by AGY in 44.3). Not launched here; needs Reza's go-ahead on the exact protocol.

## 45. Round 15: preregistered 20-seed replication (Nov 26, 2026)

Full frozen protocol: PROTOCOL-R15.md. This section is the summary;
the protocol file governs.

### 45.1 Loop order and research verdict (context)

Reza's standing order (Nov 26 16:38 PST): keep iterating in a loop,
critique own work, use reviews, use own judgment for design decisions,
use online resources and literature, do deep research like a neuro AI
scientist, do not stop until a system works and learns based on the
theory (thesis Chapter 3, random predictive coding). Tracked item
goal_a9a871e81255.

Deep research landed Nov 26 (report
~/workspace/research_notes/predictive-coding-local-learning-rules-20261127-0039/report.md;
page built as artifact predictive-coding-vs-local-error-learning):
the defensible claim on current evidence is local error-driven
learning on a fixed random reservoir, NOT predictive coding yet. A
signed error term is necessary for predictive coding but is also the
textbook rule for reservoirs, LMS/Widrow-Hoff, FORCE, and PCN
backprop approximation, so round 14 separates signed-error from
unsigned-Hebbian learning, not predictive coding from reservoir
computing. Two record corrections: NLMS is Widrow-Hoff 1960, not
Urbanczik-Senn 2014 (that rule is dendritic, error = somatic spike
minus dendritic prediction of somatic rate); the unsigned burst gate
has no literature support (Payeur et al 2021's rule is signed:
burst = LTP, single spike = LTD). The cleanest separating experiment
is the omission test with frozen weights (train, freeze, omit an
expected element, decode the omitted identity from the omission
window); a predictive-coding model shows a stimulus-specific omission
response, a reservoir decays. Omission + causal top-down clamp +
precision manipulation together are what would change the verdict.
Those are round-16 material, not round 15. Round 15 answers "does
delta replicate?" Round 16 answers "is it predictive coding?"

### 45.2 Protocol (frozen before any round-15 test result)

- Seeds 201..220 (20 seeds), NOT AGY's 200..219: seed 200's
  round-14-config test scores were observed during the runtime sizing
  run (planted delta-256 0.243, sine delta-256 0.541), so 201..220
  keeps all 20 test seeds unseen. Val seeds 201..205.
- Tasks (train/val/test): planted lag-1 200/50/50 K=16;
  planted lag-2 200/50/50 K=16 (make_data_lag lag=2); sine-fixed
  50/20/20 K=32; Lorenz (baselines.py, dt 0.01 stride 10) 200/50/50
  K=32; variable sine (rpc_spatial.make_sine) 100/20/20 K=32.
- Validation: 18 configs (alpha {0.7, 1.0} x smooth_sigma
  {0.0, 2.0, 3.0} x eta_d {0.02, 0.05, 0.2}) on val seeds 201..205,
  winner = highest mean val accuracy, frozen per task, used for all
  test conditions. smooth_sigma is a REAL dimension on planted tasks
  (Gaussian-bump input replaces one-hot, rpc_v2.py _encode), not a
  no-op; an earlier draft of the protocol said otherwise and was
  corrected before freezing.
- Test conditions (n_out=256, full_code=True, use_confirmation=False,
  s_inh=0.0, unmodified p1-prototype cosine readout, 5 epochs):
  delta (A24), tempcov (A25, see 45.3), burst with norm_rows=True
  (the only condition with norm_rows on), frozen.
- Metrics: exact-bin accuracy everywhere; planted adds m1/m2/oth
  (m1 = middle motif element, m2 = completing element, oth = rest)
  and all-motif; Lorenz/var-sine add within-1-bin accuracy and
  continuous MSE against persistence, Markov-1/2, ESN on identical
  test sequences.
- Bars (20-seed lower 95% CI bound unless stated): planted lag-1
  lower CI above 0.22 (target 0.25), m1 mean >= 0.80, m2 mean >=
  0.90, delta beats tempcov paired; planted lag-2 mean >= 0.20 and
  >= +0.10 over Markov-1; sine-fixed mean >= 0.80; Lorenz/var-sine
  beat persistence AND Markov-1 on accuracy AND MSE, gain >= +0.15
  over frozen.

### 45.3 A25: fixed temporal-covariance control

New learn_rule="tempcov" in rpc_v2.py. Same outer-product form and
same NLMS denominator (||y2_prev||^2 + 1) and eta_d as delta, so
delta vs tempcov isolates the post term: error (p1 - d1) versus
centred raw post (p1 - temporal mean of p1), with the pre code
likewise centred on its temporal mean. The means are exponential
moving averages (decay 0.99, effective window ~100 steps), updated
AFTER the centred vectors are formed, initialised on the first
training step (no update that step), and NOT reset per sequence
(reset() only zeroes reservoir state). Raw p1, not standardised p1n.
No graded factor, no W_dist1b update. Clamped at w_max=2.0 like
delta; run() now returns clamp_frac (fraction of W_dist1 entries at
the clamp) so a saturated clamp is visible.

### 45.4 Execution and sizing (measured, not estimated)

Timing run on the VM (Nov 26): planted delta-256 seed 200, 196 s;
sine delta-256 seed 200, 291 s. Round 15 is ~850 runs x ~4 min =
~57 CPU-hours. Runs on Beast (12 cores, review-venv torch 2.15.0,
numpy 2.5.4), 10 workers (5 tasks x 2 seed-shards), ~6 h wall.
Driver rpc_round15.py writes one JSON line per (task, phase,
config/condition, seed) and skips completed lines, so a killed
worker resumes. Fast smoke tests pass on the VM (var-sine bins
reproduce make_sine exactly; tempcov/burst branches run on a tiny
net; baselines sane: planted1 seed 201 Markov-1 0.274, ESN 0.273).
Full 5-epoch smoke runs on Beast: [pending at section write time].

### 45.5 Status

Protocol frozen. Driver written and fast-smoke-tested. Validation
phase launch pending the Beast full-smoke result.

### 45.6 Threading fix (Nov 26, 18:24 PST)

First smoke attempt crawled: Beast load 45 (connectome round-1
jobs), each smoke process at 166% CPU, 2h28m CPU time in 89 min
elapsed, only 1-2 output lines (planted delta, Lorenz delta +
baselines). Diagnosis: torch/numpy BLAS thread oversubscription.
The runs are tiny-matrix and single-threaded by nature; the
default 12 threads per process spin-wait on barriers and burn CPU
without progress. Fix: OMP/MKL/OpenBLAS/NumExpr threads set to 1
and torch.set_num_threads(1) in rpc_round15.py (before the torch
import), plus the same env in r15/monitor.py's launcher. Smoke
processes killed and relaunched with the fix. The validation
workers inherit the fix through the monitor. Smoke numbers so far
(single val seed 201, round-14 config, NOT results): planted delta
acc 0.239, m1 0.633, m2 0.946, oth 0.073; Lorenz delta acc 0.097,
within1 0.310, MSE 72.3, against Markov-1 0.221, Markov-2 0.570,
persistence 0.157, ESN 0.417 (MSE 14.6 / 1.5 / 13.3 / 3.8).

### 45.7 Validation launched (Nov 26, 18:44 PST)

Threading fix verified: all three smoke groups finished in ~20
min wall (planted, sine, cont all end "SMOKE <group> DONE", no
Traceback). Monitor run r15-monitor (run 2eb8b4d8) launched the
validation phase: 10 workers (5 tasks x 2 seed-shards),
PIDs 469953-469962, val_launched marker written. Smoke numbers
(single val seed 201, round-14 config, NOT results): planted
delta 0.239 (m1 0.633, m2 0.946, oth 0.073), tempcov 0.077 (m1
0.000, m2 0.000, oth 0.100, burst 0.092), burst 0.096 (m1 0.137,
m2 0.369); sine delta 0.503; Lorenz delta 0.097, within1 0.310,
MSE 72.3 (Markov-2 0.570, ESN 0.417); varsine delta 0.143,
within1 0.272, MSE 0.547 (Markov-1 0.289, Markov-2 0.526,
persistence 0.134, ESN 0.413, ESN within1 0.758). Tempcov
collapses on planted seed 201 (both motif steps zero).

### 45.8 Validation complete, test launched (Nov 26, 20:44 PST)

450/450 validation lines, no Tracebacks. Monitor run
r15-monitor (run 7f2f7e39) picked winners and launched the test
phase: 10 workers (5 tasks x 2 seed-shards), PIDs 492706-492715,
test_launched marker written. Winners (val mean acc, 5 val
seeds): planted1 alpha 1.0 / sigma 0.0 / eta_d 0.05 -> 0.229;
planted2 alpha 1.0 / sigma 0.0 / eta_d 0.05 -> 0.218; sine
alpha 1.0 / sigma 0.0 / eta_d 0.2 -> 0.863; lorenz alpha 0.7 /
sigma 0.0 / eta_d 0.2 -> 0.164; varsine alpha 1.0 / sigma 2.0 /
eta_d 0.02 -> 0.152. Validation picked one-hot (sigma 0.0) on
4 of 5 tasks: the smooth input is out, confirming AGY's
one-hot finding (sine val 0.863 vs the smooth 3.0 round-14
headline 0.484). Planted winners sit at the 0.22 bar on
validation; the 20-seed test decides. Lorenz and varsine
winners are weak on validation (0.164, 0.152).

## 46. Round 15: preregistered 20-seed replication (Nov 26, 2026)

Protocol: PROTOCOL-R15.md. Seeds 201..220. Validation grid 18
configs on val seeds 201..205, winners frozen per task, one
held-out test evaluation, 20 seeds, 5 conditions (delta,
tempcov, burst, frozen, baseline). 500 test lines, 0 errors.
Analysis: r15/r15_analysis.md (verified against the JSONL:
planted1 delta mean 0.2229 vs 0.223, sine delta 0.8681 vs
0.868, planted1 m1 0.6805 vs 0.680).

### 46.1 Planted lag-1

Delta acc 0.223, CI [0.216, 0.230], m1 0.680, m2 0.956,
all-motif 0.811. Tempcov 0.070, burst 0.071, frozen 0.059.
Paired delta-frozen +0.164, CI [+0.155, +0.173]. Baselines
(identical test seqs): Markov-1 0.274, Markov-2 0.267,
persistence 0.046, ESN 0.275. Bars: lower CI > 0.22 FAIL
(0.216), mean >= 0.25 FAIL, m1 >= 0.80 FAIL, m2 >= 0.90
PASS, beats tempcov paired PASS. 2/5.

### 46.2 Planted lag-2

Delta acc 0.222, CI [0.216, 0.227]. Tempcov 0.070, burst
0.069, frozen 0.062. Paired delta-frozen +0.159. Baselines:
Markov-1 0.084, Markov-2 0.267, persistence 0.057, ESN 0.261.
Bars: mean >= 0.20 PASS, mean >= Markov-1 + 0.10 PASS. 2/2.

CORRECTION (Nov 26, found while building the round-16
driver): the lag-2 m1 (0.264), m2 (0.050), and all-motif
(0.261) numbers previously listed here are STRUCK. run_planted
counts m1 when (prev, tok) is a motif pair and m2 when
(prev2, prev, tok) is a motif triple: lag-1 definitions
(rpc_round15.py:160-186, MOTIF_M1/M2 lines 55-56). At lag 2
the motif elements sit at i, i+2, i+4 with fillers between,
so those tests match only chance filler patterns. The true
lag-2 motif accuracies were never measured in round 15. The
overall accuracy (0.222), the paired differences, the
baselines, and both bars do not depend on the m1/m2 split
and stand unchanged. The round-16 driver detects motifs by
pattern and reports the true lag-2 motif accuracies as a
byproduct.

### 46.3 Sine-fixed

Delta acc 0.868, CI [0.851, 0.886]. Tempcov 0.083, burst
0.048, frozen 0.036. Paired delta-frozen +0.832, CI [+0.813,
+0.852]. Baselines: Markov-1 0.456, Markov-2 0.798,
persistence 0.124, ESN 0.940. Bar mean >= 0.80 PASS. 1/1.

### 46.4 Lorenz

Delta acc 0.164, within1 0.402, MSE 66.1. Tempcov 0.042,
burst 0.052, frozen 0.037. Baselines: Markov-1 0.205,
Markov-2 0.565, persistence 0.173, ESN 0.398; MSE Markov-1
12.3, persistence 12.6, ESN 4.4. Bars: 0/5. Delta does not
beat persistence on accuracy (0.164 < 0.173) or MSE, and its
gain over frozen (+0.127) is below the +0.15 bar.

### 46.5 Variable sine

Delta acc 0.150, CI [0.138, 0.161], within1 0.324, MSE
0.172. Tempcov 0.043, burst 0.053, frozen 0.043. Baselines:
Markov-1 0.283, Markov-2 0.528, persistence 0.125, ESN
0.385; MSE Markov-1 0.108, persistence 0.054, ESN 0.044.
Bars: beats persistence on acc PASS (1/5); beats Markov-1
on acc FAIL, beats persistence on MSE FAIL, beats Markov-1
on MSE FAIL, gain >= +0.15 over frozen FAIL (actual
+0.107). The exact-bin accuracy flatters the predictor: its
continuous error (MSE 0.172) is worse than repeating the
last value (0.054).

### 46.6 Verdict: 6/18 bars pass

Delta genuinely learns on every task: it beats frozen on all
five, with paired CIs all above zero (sine +0.832, planted
+0.164 / +0.159, Lorenz +0.127, varsine +0.107). Tempcov and
burst collapse everywhere (0.04-0.08), so the signed error
term is the active ingredient, not centering or width. Sine
is a clean win: 0.868 beats Markov-1 (0.456) and Markov-2
(0.798). Planted lag-1 is real but marginal: 0.223 mean,
lower CI 0.216 misses the 0.22 bar by 0.004. The critics
were right that round-14 planted was threshold noise; the
fresh-seed mean is essentially unchanged (0.223 vs 0.221)
and the preregistered bar as stated does not pass. The weak
leg is m1 (0.680 vs the 0.80 bar); m2 (0.956) is strong.
Continuous tasks are the wall: Lorenz and varsine fail to
beat persistence or Markov-1 on accuracy or MSE. The trained
readout still beats delta everywhere (ESN: planted 0.275 vs
0.223, sine 0.940 vs 0.868, Lorenz 0.398 vs 0.164, varsine
0.385 vs 0.150), so the readout gap survives: delta aligns
the drive, but a trained linear readout extracts more.
Clamp fractions are 0.0000 for delta on every task, so the
w_max=2.0 clamp is not binding.

What round 15 settles: the delta rule is a validated
learning rule for discrete token tasks under a fixed
prototype readout, with the signed error term established as
the active ingredient. It does not settle: continuous
prediction, the first motif step, or whether the mechanism
is predictive coding (the omission test is still unrun).

Next: round 16. Candidates: (a) the continuous-task wall
(Lorenz / varsine under a continuous-valued readout or a
different error signal); (b) the omission test with frozen
weights, the cleanest Chapter 3 signature; (c) the readout
gap (why the trained readout beats the fixed prototype
readout, and whether a local rule can close it). Design
pending Reza's call.

## 47. Round 16: the omission test (design, Nov 26, 2026)

Protocol: PROTOCOL-R16-OMISSION.md (frozen Nov 26, 2026).
Driver: rpc_round16_omission.py. Smoke test:
rpc_round16_smoke.py. Analyzer: r16/r16_analyze.py.
Full 20-seed run NOT launched; it waits for Reza's decision.

Question: experiment A of the deep-research report (section 6.2).
Does the trained delta network hold a stimulus-specific,
predictability-scaled prediction that survives a missing input?
This is the cleanest test of whether delta is predictive coding
rather than signed-error learning on a fixed random reservoir.

Step alignment (verified from rpc_v2.py): d1(t) = W_dist1 @
y2_prev is computed BEFORE the current token is fed in, so d1(t)
predicts token t from the code of token t-1, and the delta update
trains d1(t) to match p1(t). A blank input (zeros) makes the
reservoir state decay (r_new = tanh(W_res @ r) at the winner
alpha = 1.0), not reset. At the omission step t, d1(t) and pred(t)
are UNAFFECTED by the blank, so the omission decode at t is
identical to the normal prediction at that position. d1 does NOT
feed back into the reservoirs, so there is no error unit and no
top-down feedback into the state at test time: the omission
response in this architecture IS the prediction itself. The test
therefore measures whether a prediction exists as an internal
state and persists after a missing input; it does not test error
units or top-down causality (experiments B/C, separate rounds).

Design: planted lag-1 and lag-2, K = 16, length 24, train 200,
test 50, seeds 221..240 (fresh; reusing 201..220 would be mild
peeking after round 15). Winner config frozen from round 15
(alpha 1.0, sigma 0.0, eta_d 0.05). Conditions: delta, frozen.
One pass per omission event (blank at one position only).
Motif positions detected by pattern in the original sequence
(this corrects round 15, which applied lag-1 definitions to
lag-2 data; the true lag-2 motif accuracies were never
measured). Measures: omission decode and cosine margin at t,
generative continuation (decode of the motif completion c at
t+1 for lag-1, t+2 for lag-2), filler-target t+1/t+2 controls
(restricted to non-motif targets), shuffled-prototype decode,
d1/p1 norms, error projection. Bars A1-A5 (primary, lag-1),
A6-A7 (persistence, secondary), B1-B4 (lag-2). Verdict rule:
A passes if A1-A5 pass; that is necessary but NOT sufficient
for the Rao-Ballard claim (experiment B still needed).

Smoke test (VM, 1 seed 221, 1 epoch, NOT results): alignment
check PASSED (pred and d1 identical at the omission step; d1
differs at t+1). Motif detection 2.32 m1/seq and 2.32 m2/seq
(lag-1), 2.38/2.38 (lag-2). All metrics finite. Planted1 delta:
omit_m2 0.793, omit_m1 0.414, omit_filler 0.040, cont_m1 0.388,
margin_m2 0.198, shuf_m2 0.034. Planted1 frozen: omit_m2 0.112,
cont_m1 0.172. Planted2 delta: omit_m2 0.807, omit_m1 0.479,
cont_m1 0.412. Planted2 frozen: omit_m2 0.160, cont_m1 0.084.
The omission decode at t equals the normal accuracy exactly in
all four runs, as designed. The continuation is above chance
for delta on both lags even at 1 epoch; the frozen continuation
is also above chance (0.172 lag-1), so the analyzer reports the
delta-minus-frozen continuation contrast alongside A7. The
frozen m2 decode (0.112 lag-1, 0.160 lag-2) may make bars A3/B3
tight; the margin bar is the sharper check. CLI path verified
via --seeds 221-221 --epochs 1 (valid JSONL, resumable keys).
Analyzer smoke-tested on the 1-seed output.

Design corrections made after the first smoke run, before any
full-run result: the t+1/t+2 controls were restricted to
non-motif targets (an unrestricted t+1 target is sometimes
another motif's element), and the lag-2 continuation was moved
to t+2 (a filler sits between b and c at lag 2). Both are frozen
in the protocol.

LAUNCH (Nov 26, 2026, about 22:05 PT): Reza delegated the
decision ("use your best judgment and do whatever you want, you
can run whatever you want"). Omission test chosen, per the
standing recommendation. Beast smoke re-run first (1 seed 221,
1 epoch, planted1): reproduced the VM smoke numbers exactly
(delta omit_m2 0.793, omit_m1 0.414, omit_filler 0.040,
cont_m1 0.388; frozen omit_m2 0.112, cont_m1 0.172). Four
workers launched (2 tasks x 2 seed-shards, seeds 221..240,
winners from round 15, 5 epochs, one BLAS thread each):
planted1 221-230, planted1 231-240, planted2 221-230,
planted2 231-240. Monitor cron r16-monitor created (every
30 min, runs r16/monitor.py, runs the analyzer at 80 lines).

Launch incident: the first launch command put the PY assignment
inside a backgrounded && list, so three of the four workers
started with PY unset and died at once (nohup: failed to run
command 'rpc_round16_omission.py'). The planted1 221-230 worker
ran, but it ran TWICE concurrently (a mangled earlier launch
attempt had also started it): seeds 228-230 were written twice,
6 duplicate lines, 26 lines instead of 20. The duplicate pairs
were verified byte-identical (same seed gives the same data and
init, so a re-run reproduces the record exactly) and the file
was deduplicated to its 20 unique records before any analysis.
The three failed workers were relaunched with PY set before any
backgrounding (PIDs 500931-500933, verified alive with the
correct args). Lesson: set all variables before the first & in
a multi-worker launch, and always check line counts per shard
against seeds x conditions before analysis.

## 48. Round 16: the omission test (results, Nov 26, 2026)

Protocol: PROTOCOL-R16-OMISSION.md. Driver:
rpc_round16_omission.py. Analyzer: r16/r16_analyze.py.
Analysis: r16/r16_analysis.md (60 lines). 80/80 test lines
(2 tasks x 2 conditions x 20 seeds), 0 duplicate keys after
the launch dedupe, 0 worker Tracebacks. Headline means
recomputed independently from the JSONL by the parent and
match the analysis to 3 decimals (planted1 delta omit_m2
0.9663, omit_m1 0.6667, cont_m1 0.6098; planted2 delta
omit_m2 0.9426, omit_m1 0.6778, cont_m1 0.4670; frozen
omit_m2 0.0485 / 0.0577). All runs epochs 5, round-15
winner config (alpha 1.0, sigma 0.0, eta_d 0.05). Chance
1/16 = 0.0625.

VERDICT: 11/11 bars pass. Experiment A passes all five
lag-1 bars (A1-A5), plus A6, A7, and all four lag-2 bars
(B1-B4).

Planted lag-1, delta: omit_m2 0.966, CI [0.955, 0.977];
omit_m1 0.667, CI [0.622, 0.711]; omit_filler 0.059, CI
[0.046, 0.072]; cont_m1 0.610, CI [0.569, 0.651];
margin_m2 0.345, CI [0.332, 0.359]; shuf_m2 0.001, CI
[-0.001, 0.003]; normal_m2 0.966 (identical to omit_m2, as
designed). Frozen: omit_m2 0.048, CI [0.031, 0.066];
omit_m1 0.058; omit_filler 0.064; cont_m1 0.048, CI
[0.027, 0.069]; margin_m2 -0.004; shuf_m2 0.087.
Paired: delta omit_m2 - omit_filler +0.907, CI
[+0.888, +0.927]; delta cont_m1 - t1_filler +0.560, CI
[+0.518, +0.602]; delta - frozen cont_m1 +0.562, CI
[+0.512, +0.612].

Planted lag-2, delta: omit_m2 0.943, CI [0.928, 0.957];
omit_m1 0.678, CI [0.647, 0.709]; omit_filler 0.089, CI
[0.071, 0.107]; cont_m1 0.467, CI [0.424, 0.510];
margin_m2 0.293, CI [0.277, 0.309]; shuf_m2 0.006;
normal_m2 0.943. Frozen: omit_m2 0.058, CI [0.031, 0.084];
cont_m1 0.049. Paired: delta omit_m2 - omit_filler
+0.854, CI [+0.827, +0.880]; delta cont_m1 - t1_filler
+0.397, CI [+0.346, +0.448]; delta - frozen cont_m1
+0.418, CI [+0.372, +0.465].

Bars: A1 omit_m2 mean >= 0.80, lower CI > 0.50 PASS.
A2 omit_m2 - filler paired mean > 0.30, CI above 0 PASS.
A3 frozen omit_m2 mean < 0.15, margin < 0.05 PASS.
A4 shuf_m2 mean < 0.15 PASS. A5 omit_m1 mean >= 0.50,
lower CI > 0.25 PASS. A6 cont_m1 lower CI > chance PASS.
A7 cont_m1 - t1_filler paired CI above 0 PASS. B1 omit_m2
lower CI > chance PASS. B2 omit_m2 - filler paired CI
above 0 PASS. B3 frozen omit_m2 mean < 0.15 PASS.
B4 cont_m1 lower CI > chance PASS.

What this settles: the trained delta network holds a
stimulus-specific prediction as an internal state that
survives a missing input and generates the completion.
The prediction is specific (m2 0.966 vs filler 0.059,
margin 0.345, shuffled prototypes 0.001), it covers the
first motif step too (m1 0.667 lag-1, 0.678 lag-2, both
above their 0.50 bars), and it generates forward
(continuation 0.610 lag-1 and 0.467 lag-2, both far above
chance and above the frozen continuation 0.048/0.049,
paired gaps +0.562/+0.418). The smoke-test worry did not
materialize: the frozen omit_m2 at 20 seeds is 0.048/0.058,
not the 0.112/0.160 seen on one seed, so bars A3/B3 pass
with room.

What this does NOT settle (unchanged from the protocol):
the omission decode at t equals the normal prediction by
construction (omit_m2 == normal_m2 exactly on both tasks),
so the test measures whether a prediction exists and
persists, not whether error units compute it or top-down
feedback causes it. This architecture has no error unit
and d1 never feeds back into the reservoirs at test time,
so a pass is necessary but not sufficient for the
Rao-Ballard predictive-coding claim. The next discriminator
is experiment B (causal top-down clamp), then precision (C)
and generative (D) tests. The continuous-task wall
(Lorenz/var-sine) also still stands from round 15.
