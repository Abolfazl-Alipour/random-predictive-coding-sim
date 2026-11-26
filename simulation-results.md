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
