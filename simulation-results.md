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

CORRECTION (Nov 26, found while designing round 18): the
Lorenz (46.4) and var-sine (46.5) numbers above are STRUCK
as scored. rpc_round15.py run_cont collects out["pred"] at
step t and scores it with baselines.py accuracy, which
compares p[t] to seq[t+1]. But _step_core computes d1(t)
from y2_prev before the current token is fed in
(rpc_v2.py:1122-1126), and run() scores out["pred"] against
the CURRENT token (rpc_v2.py:1330-1340). So pred(t)
predicts token t, and run_cont scored it against token t+1:
a two-step-ahead task. The baselines (Markov-1/2,
persistence, ESN) predict t+1 from tokens 0..t: true
one-step-ahead. The same shift sits in within1
(rpc_round15.py:219-222) and in mse_from_bin_preds
(baselines.py:322-330, whose docstring wrongly claims
"preds[t] predicts token t+1"). Round 15 therefore compared
RPC's two-step-ahead accuracy, within-1, and MSE against the
baselines' one-step-ahead accuracy, within-1, and MSE on
Lorenz and var-sine ONLY. The discrete-task numbers (46.1
planted, 46.3 sine) use run() and are unaffected. The
"continuous-task wall" verdict in 46.6 is withdrawn pending
round 18, which re-scores correctly at 20 seeds. The
1-seed diagnostic (seed 261, 5 epochs, round-15 winner
configs, NOT a result): Lorenz correct scoring acc 0.285 /
MSE 35.1 vs mis-scored 0.149 / 74.6; var-sine correct 0.319
/ 0.042 vs mis-scored 0.146 / 0.142; sine control correct
0.864 vs mis-scored 0.124. The wall may dissolve on
accuracy; the Lorenz MSE wall may survive. Round 18
decides.

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

## 49. Round 17: the top-down clamp (design, Nov 26, 2026)

Protocol: PROTOCOL-R17-CLAMP.md (frozen Nov 26, 2026).
Driver: rpc_round17_clamp.py. Analyzer: r17/r17_analyze.py.
Monitor: r17/monitor.py. Full 20-seed run NOT launched; it
waits for Reza's decision.

Question: experiment B of the deep-research report (sections
1.4-1.5). Round 16 showed the trained delta network holds a
stimulus-specific prediction that survives a missing input.
Round 17 asks whether that prediction does any work: cut it,
shuffle it, and see whether the lower module notices.

The architectural fact: in the current model there is NO
top-down pathway. d1 is computed, used for the readout and
for the delta rule, and never fed back into either module
(verified from rpc_v2.py _step_core and
ReservoirModule.forward). A literal clamp test cannot run on
it: there is nothing to clamp. So round 17 has two parts.
Part 1 (assumption A26) adds the wire; part 2 clamps it.

A26: fixed random matrix W_fb (n_res x n_out), entries
randn / sqrt(n_out), never trained. fb = fb_gain *
(W_fb @ d1), added into module 1's reservoir drive
alongside the bottom-up input. d1 at step t predicts token
t, so the feedback arriving with the input is the
prediction of that input: the Rao-Ballard form. Default
fb_gain = 0.0. W_fb and the shuffle permutation are drawn
after all other init draws, so fb_gain = 0.0 reproduces
every previous result exactly (verified: the round-16
smoke test reproduces its numbers exactly after the A26
edit, section 49.1). Test-time modes via step(fb_mode):
"on", "cut" (fb = 0), "shuffle" (fixed random permutation
of d1's coordinates, same magnitude, no information).

Scale (measured): trained delta d1 norm 1.6-1.9, bottom-up
drive norm about 2.3, so fb_gain 1.0 matches the
bottom-up drive, 0.3 is about a fifth, 0.1 about a
twentieth. The driver records RMS(fb) / RMS(bottom-up)
per run.

Design: planted lag-1 and lag-2, K = 16, round-15 winner
config (alpha 1.0, sigma 0.0, eta_d 0.05), n_out 256,
5 epochs, seeds 241..260 (fresh). Validation: fb_gain in
{0.1, 0.3, 1.0} on val seeds 241..245, planted lag-1,
winner = highest mean val normal accuracy, guard: winner
>= 0.80 x the fb_gain = 0.0 val accuracy, else the round
stops with "the wire destabilizes learning". Test
conditions (3 nets per (task, seed); conditions 2-3 are
test-only passes on condition 1's net): 1. wire-on tested
wire-on; 2. wire-on tested wire-cut; 3. wire-on tested
wire-shuffled; 4. wire-off (fb_gain = 0, round-16 config
on the new seeds); 5. frozen wire-on. Measures: the
round-16 omission measures plus p1_cos (mean cosine
between p1 wire-on and wire-cut on identical sequences),
err_mag (mean ||p1 - d1||, first and last epoch),
fb_rms_ratio, r1_max, clamp_frac. Bars: C1 cut degrades
omit_m2 by >= 0.10 (paired, CI above 0); C2 shuffle
degrades omit_m2 by >= 0.10; C3 p1_cos mean < 0.90;
C4 wire-off omit_m2 differs (paired CI excludes 0);
C5 guard wire-on normal accuracy >= 0.80 x wire-off.
Experiment B passes if C1-C4 pass on lag-1.

### 49.1 Smoke test (VM, 1 seed 241, 1 epoch, NOT results)

All four protocol checks pass. (1) fb_gain = 0.0
reproduces round 16 exactly: normal_acc_m2 0.855 and
omit_m2 0.855, identical to the round-16 driver on the
same seed. (2) Wire-on changes p1: p1_cos (on vs cut) =
0.962. (3) All metrics finite, no NaN, r1_max = 0.91,
clamp_frac = 0.0000. (4) fb_rms on = 0.277, shuffle =
0.281 (within 5%), cut = 0.000. The analyzer was
smoke-tested on the 1-seed driver output (val + test,
planted1 only): it picks the winner, applies the guard,
and writes the analysis file with real numbers.

Smoke numbers (1 seed, 1 epoch, NOT results): wire-on
omit_m2 0.735, omit_m1 0.402, cont_m1 0.453,
normal_acc 0.167; wire-cut omit_m2 0.803, omit_m1 0.496;
wire-shuffle omit_m2 0.761; wire-off omit_m2 0.855;
frozen omit_m2 0.043. Honest note: at 1 epoch the wire
HURTS the omission decode (cut beats on by 0.068) and the
wire-on val accuracy (0.181) sits below wire-off (0.194).
That is expected at 1 epoch: W_dist1 is barely trained,
so d1 is a poor prediction feeding noise into module 1.
Whether the trained prediction helps or hurts is what the
5-epoch, 20-seed run decides; C1/C5 are built to catch
either outcome. Do not read the smoke numbers as a result.

LAUNCH (Nov 26, 2026, about 23:05 PT): Reza ordered "do all
together, do not stop, keep going, do not stop for my approval,
continue until the question is completely clear and killed".
Beast smoke re-run first (rpc_round17_smoke.py): reproduced
the VM smoke numbers exactly (wire-on omit_m2 0.735, cut
0.803, shuffle 0.761; checks 3 and 4 passed, r1_max 0.91,
clamp_frac 0.0000, fb_rms on 0.277 / shuffle 0.281 / cut
0.000). Four validation workers launched (planted1, seeds
241-245, fb_gain 0.0/0.1/0.3/1.0, PIDs 517179-517182,
verified alive with the correct args; the round-16 lesson
held, variables set before the first &). Monitor cron
r17-monitor created (every 30 min; drives val -> winner
pick with the 0.80x guard -> test launch -> analysis at 200
lines). The monitor launches the test phase, not the main
agent. VAL WINNER (Nov 26, 2026, 23:44 PT, verified from the
Beast val files by the main agent): 20/20 val lines.
fb_gain 0.0 val_acc 0.227, 0.1 0.226, 0.3 0.221, 1.0 0.182.
Winner fb_gain 0.1, guard PASS (0.226 >= 0.80 x 0.227).
Strong feedback hurts (1.0 drops to 0.182); gentle
feedback (0.1) is a tie with no wire on validation
accuracy. Test phase LAUNCHED (4 workers, planted1 and
planted2, seeds 241-260, fb_gain 0.1, PIDs 524053-524056,
verified alive with the correct args; 0 test lines yet,
workers started about 1 minute before the check). The
monitor drives the rest (analysis at 200 lines). Round 18
val runs in parallel on Beast.

## 50. Round 18 design: the continuous-task wall (scoring correction + readout/input separation) (Nov 26, 2026)

Question: does the delta rule fail on continuous prediction,
or did round 15 score it wrong? Code fact, verified by the
main agent against the files: in _step_core, d1(t) is
computed from y2_prev before token t is fed in, and delta
trains d1(t) to match p1(t), so out["pred"] at step t
predicts token t, and run() scores it that way
(rpc_v2.py:1122-1126, 1330-1340). Bug: rpc_round15.py
run_cont collects p[t] and scores it with baselines.py
accuracy, which compares p[t] to seq[t+1]: a two-step-ahead
task, while all baselines are true one-step-ahead predictors.
The same shift sits in within1 and in mse_from_bin_preds
(baselines.py:322-330, whose docstring wrongly claims
"preds[t] predicts token t+1"). Section 46.4-46.5 numbers
are STRUCK as scored; the continuous-wall verdict is
withdrawn pending this round. Diagnostic (1 seed 261, 5
epochs, round-15 winner configs, NOT a result): Lorenz
correct scoring acc 0.285 / MSE 35.1 vs mis-scored 0.149 /
74.6; var-sine correct 0.319 / 0.042 vs mis-scored 0.146 /
0.142; sine control correct 0.864 vs mis-scored 0.124.
Hypothesis: the wall is a scoring artifact, and whatever
MSE wall survives on Lorenz is caused by the discrete
prototype readout and/or the bin-only input.

Design (protocol PROTOCOL-R18-CONTINUOUS.md, frozen before
any test result): 4 nets per (task, seed), seeds 262..281,
5 epochs, round-15 winner configs frozen. N1 delta one-hot
gives C1 (prototype readout, H0) and C2 (ridge readout from
d1, tests the readout). N2 delta with the A27
continuous-bump input gives C3 (prototype) and C4 (ridge,
tests the input). N3/N4 frozen give C5/C6 (prototype) and
C2f/C4f (frozen ridge, tests whether delta adds continuous
information). Val picks the bump sigma from {1.0, 2.0, 3.0}
on seeds 262..266. Bars D1-D8 (20-seed paired differences,
95% CI): D1 Lorenz C1 acc beats persistence and Markov-1;
D2 var-sine C1 acc beats persistence and Markov-1; D3
var-sine C1 MSE beats persistence; D4 Lorenz C1 MSE beats
persistence (expected FAIL on pilot evidence); D5 Lorenz C2
ridge MSE < C1; D6 Lorenz C3 bump MSE < C1; D7 Lorenz C4
MSE < C2 and < C3 (judged only if D5 or D6 passes); D8
Lorenz C2 ridge MSE < C2f frozen-ridge MSE. Verdict: H0
passes if D1-D3 pass; if D4 also passes the wall was
entirely scoring; if D4 fails the residual cause is readout
(D5 only), input (D6 only), both/interacting (D7), or
target by elimination (D1 passes, D5-D7 fail). Cause (c) is
tested only by elimination, because the bump input changes
p1 content jointly with the input. A27 (cont_bump /
cont_bin helpers in rpc_v2.py) is additive only; the
helpers are used only by the round-18 driver and its smoke
test.

Smoke (seed 262, 1 epoch, VM, 36/36 checks pass,
re-run by the main agent, 213 s): Lorenz convention A
0.272 / 35.18 vs convention B 0.160 / 70.67; var-sine
convention A 0.217 / 0.1155 vs convention B 0.136 / 0.2467;
bump at a bin centre equals _encode exactly; cont_bin
matches the generator bins; all metrics finite; driver CLI
resumable; analyzer writes. One analyzer bug found and
fixed in smoke testing (stats on empty data).

FIX found by the main agent before launch (val sharding):
the driver's val loop ran ALL THREE bump sigmas per worker,
so the monitor's 6 workers (one per task x sigma) would
have duplicated each other's work 3x (90 lines instead of
30) and the monitor's 30-line threshold would have fired
early, picking a winner from partial data (the monitor
crashes on the missing task and retries, so it would have
self-healed, but with wasted compute). Fixed with a
--bump-sigma flag on the driver (default all sigmas,
preserving smoke behaviour) and the monitor passing it per
worker. Verified on the VM: flagged run writes exactly 1
line (Lorenz sigma 1.0 seed 262, val_mse 8.1314, val_acc
0.397); default run writes 3 lines. The test phase was
checked and has no such bug (sharded by seed only, 4
workers x 10 seeds x 9 lines = 360).

Status: LAUNCHED Nov 26, 2026, about 23:37 PT, on Reza's
standing order (do all together, do not stop for approval).
Beast smoke re-run first (rpc_round18_smoke.py): passed
36/36 on Beast with the fixed driver. Six validation
workers launched (lorenz and varsine, seeds 262-266, bump
sigma 1.0/2.0/3.0, one worker per task x sigma with the
--bump-sigma flag, PIDs 521666-521671, verified alive with
the correct args; val_launched marker written). Monitor
cron r18-monitor created (every 30 min; drives val ->
winner pick per task -> test launch -> analysis at 360
VAL WINNERS (Nov 26, 2026, 23:45 PT, verified from the
Beast val files by the main agent): 30/30 val lines, 5 per
file, so the --bump-sigma fix held in production (no
duplication). Lorenz: sigma 1.0 mean val MSE 8.0389, 2.0
7.0820, 3.0 9.0691; winner 2.0. Var-sine: sigma 1.0 0.1971,
2.0 0.0675, 3.0 0.0356; winner 3.0. Test phase LAUNCHED (4
workers, lorenz and varsine, seeds 262-281, PIDs
524058-524061, verified alive with the correct args and
--winners r18/winner.json). The monitor drives the rest
(analysis at 360 lines). Round 17 test runs in parallel
on Beast.

## 51. Redirection: the model-data match is the contribution (Nov 27, 2026, 00:00 PT)

Reza's order, in his words: "No mofo. The architecture
resembles canonical microcircuits and hippocampus. And the
match between data and our findings would be the biggest
contribution and then you can go and fucking figure out
the mapping or a phenomena or several of them that our
model can explain and then we have a theory and a data
that we can explain. Then that means we are onto
something." The "not predictive coding yet" framing is
rejected as the goalpost: the contribution is a model that
explains real data, whether or not other models explain it
too. The model's unique way, his words: "it is kinda like
evolution, you randomly predict the next input and that is
how you learn patterns." No hidden rule, no teacher, no
backprop, no labels beyond the readout prototypes; random
prediction plus error selection is the learning mechanism.
Second order: "you did not test the hierarchical version
where there are multiple modules all with feed back
connections here and there that can learn input patterns."

Two tracks launched on this order, both in parallel with
rounds 17 and 18 (which continue on Beast):
1. Deep research on the model-data mapping (research
folder ~/workspace/research_notes/rpc-model-data-mapping-20261127-0800):
map each model element onto canonical cortical
microcircuits and hippocampal circuits; list the
electrophysiology phenomena the model explains (omission
responses, mismatch negativity, repetition and expectation
suppression, predictive remapping, hippocampal replay and
predictive place cells, sharp-wave ripples, theta
sequences); state what is unique about random prediction
plus error selection against predictive coding networks,
reservoir computing and FORCE, HTM, successor
representation, and Hebbian sequence learning; name the
data that would discriminate this model from the
alternatives.
2. Hierarchical build (design agent, VM only, no Beast
launch yet): a multi-module version with feedback wires
between levels, each level's distal wire trained by the
delta rule, a protocol, driver, analyzer, monitor, and
smoke test. The design question is what hierarchy buys
that two modules cannot (first motif step m1, slow
timescale structure, continuous prediction).

Status: research running; hierarchical design building.
The round-19 scope note in the record (precision next if
D8 passes, generative next if D8 fails) is superseded in
order, not cancelled: the hierarchy and the data mapping
are now the next builds, and the precision/generative
tests follow the round-17/18 verdicts.

## 52. Round 18 verdict: the continuous wall survives correct scoring (Nov 27, 2026, 00:15 PT)

Round 18 COMPLETE. 360/360 test lines (20 seeds 262..281,
9 lines per task-seed), 0 duplicate keys, 0 worker failures,
analysis r18/r18_analysis.md verified by independent
recomputation from the raw JSONL: all 16 condition means
match the analysis to 3 decimals, baselines match
(Lorenz Markov-1 0.208, persistence 0.171 / MSE 12.605,
ESN 0.397 / MSE 4.308; var-sine Markov-1 0.284,
persistence 0.126 / MSE 0.056, ESN 0.393 / MSE 0.046),
clamp fraction 0.0 on every line.

Correct scoring changes the numbers a lot. Lorenz C1
accuracy 0.333 (CI 0.320-0.347) against the mis-scored
round-15 value 0.164; var-sine C1 0.295 against 0.150.
The pilot's wall was partly a scoring artifact, as the
round-18 hypothesis said. But the wall survives on the
error metric, which is what continuous prediction is
judged by.

Bars D1-D8: 5 pass, 3 fail.
1. D1 PASS. Lorenz C1 accuracy beats persistence (paired
diff 0.162, CI 0.149-0.175) and Markov-1 (0.125, CI
0.112-0.137).
2. D2 PASS, barely. Var-sine C1 accuracy beats
persistence (0.169, CI 0.157-0.181) and Markov-1 by
0.011 (CI 0.000-0.021, lower bound exactly zero).
3. D3 FAIL. Var-sine C1 MSE 0.064 against persistence
0.056 (paired diff -0.008, CI -0.018 to 0.002, includes
zero).
4. D4 FAIL, badly. Lorenz C1 MSE 22.390 against
persistence 12.605 (paired diff -9.785, CI -12.029 to
-7.540).
5. D5 PASS. The ridge readout halves Lorenz MSE:
C2 11.403 against C1 22.390 (paired diff 10.987).
6. D6 PASS. The continuous bump input cuts Lorenz MSE
by two thirds: C3 7.292 against C1 22.390 (paired diff
15.097).
7. D7 PASS. Combined, C4 3.795 beats C2 11.403 and C3
7.292, and beats the ESN's 4.308.
8. D8 FAIL. The frozen ridge is as good as the trained
ridge: C2f 10.818 against C2 11.403 (paired diff -0.584,
CI -0.735 to -0.434); C4f 3.615 against C4 3.795. Delta
adds no continuous information beyond what the frozen
reservoir already holds.

H0 (D1, D2, D3) FAILS because D3 fails. The verdict rule
in the protocol: if D4 also fails, the residual cause is
readout (D5 only), input (D6 only), both or interacting
(D7), or target by elimination (D1 passes, D5-D7 fail).
Here D5, D6, and D7 all pass, so readout and input both
contribute, and their combination is the best model
configuration. But D8 fails, so the improvement is not
learning: a ridge fitted on a frozen reservoir reaches
the same MSE. The delta rule learns a discrete code
(the learning gap is real: C1 0.333 against frozen
0.038 on Lorenz, 0.295 against 0.041 on var-sine), and
that discrete code carries no continuous information the
frozen reservoir lacks.

Two extra baselines the analysis table carries but the
bars do not use: Markov-2 on Lorenz scores accuracy
0.556 and MSE 1.416; on var-sine its MSE is 0.005. A
two-step lookup beats every model configuration on
continuous MSE by a wide margin, because both tasks are
smooth and nearly deterministic given two past bins.
The model is nowhere near that bar, and no readout or
input change tested here closes it.

Settled by round 18: (a) the round-15 continuous numbers
are replaced by the correct-scored numbers above; (b)
the delta rule does not learn continuous information;
(c) the best continuous configuration is bump input plus
ridge readout, and it is a readout of the frozen
reservoir, not of learned weights. Not settled: whether
any learned rule can add continuous information (the
generative test, experiment D, is the next discriminator
per the round-19 scope rule, since D8 failed). Per the
Nov 27 00:00 PT redirection, the hierarchy build and
the data mapping come first in order; the generative
test follows the round-17/18 verdicts. Round 17 (clamp)
is still in its test phase.

## 53. Round 17 verdict: the top-down wire carries no causal role (Nov 27, 2026, 00:17 PT)

Round 17 COMPLETE. Experiment B FAILS. 200/200 test lines
(20 seeds 241..260, 5 conditions per task-seed), 0 duplicate
keys, 0 worker failures, analysis r17/r17_analysis.md
verified by independent recomputation from the raw JSONL:
all condition means match the analysis to 3 decimals and
all paired differences to 4 decimals, clamp fraction 0.0
on every line.

Validation (planted lag-1, 5 seeds): fb_gain 0.0 val_acc
0.227, 0.1 0.226, 0.3 0.221, 1.0 0.182. Winner 0.1, guard
PASS (0.226 >= 0.80 x 0.227). Strong fixed feedback hurts:
gain 1.0 drops validation accuracy to 0.182.

Lag-1 bars (C1-C4 decide, C5 is a guard):
1. C1 FAIL. Cutting the wire degrades omit_m2 by +0.005
(CI +0.000 to +0.010); the bar needs >= 0.10. It fails
by a factor of 20.
2. C2 FAIL. Shuffling the wire changes omit_m2 by -0.001
(CI -0.006 to +0.005). A shuffled wire is as good as the
real one.
3. C3 FAIL. p1_cos (wire-on vs wire-cut) is 0.998; the
bar needs < 0.90. The wire moves module 1's code by
0.2 percent.
4. C4 FAIL. Wire-off omit_m2 differs by +0.004, CI
-0.000 to +0.009, which includes zero. Never building
the wire is the same as building it.
5. C5 guard PASS. Wire-on normal_acc 0.228 vs wire-off
0.229.

Lag-2 bars: C1 PASS, C2/C3/C4 FAIL. The lag-2 C1 bar is
weaker (any degradation with CI above zero, no 0.10
floor): cut degrades omit_m2 by +0.004, CI +0.0004 to
+0.0080, so the pass is real but a 0.4-point effect.
Total: 2/9 bars pass (lag-1 C5, lag-2 C1).

Numbers behind the verdict (lag-1 omit_m2): wire-on
0.953, wire-cut 0.948, wire-shuffle 0.953, wire-off
0.948, frozen 0.083. Lag-2: wire-on 0.924, cut 0.920,
shuffle 0.922, off 0.922, frozen 0.092. The omission
result itself reproduces (wire-on 0.953 vs round 16's
0.966 on different seeds); the wire adds nothing to
it. The error magnitude also does not move: first epoch
3.032, last epoch 1.619 wire-on vs 1.617 wire-off.
Feedback RMS is 6.5 percent of the reservoir drive, yet
module 1's code is 99.8 percent identical with the wire
on or cut. The wire is present, small, and inert.

What this kills: the fixed random top-down wire at gain
0.1 carries no causal role in the prediction. It does
not kill all feedback: gain 1.0 is worse, not better,
so stronger fixed feedback is not the fix; a learned
feedback wire is untested. That is a hierarchy question
now (round 19): in the hierarchical version each level's
feedback would carry a learned prediction, not a random
one.

What this does to the thesis question: two of the three
research discriminators are now decided. Omission (A)
passes, top-down clamp (B) fails. The model as it
stands is local error-driven learning on a fixed random
reservoir whose prediction is read out from d1 and never
fed back into module 1. Whether that counts as
predictive coding now rests on precision (C) and the
generative test (D), and on the hierarchy. This verdict
matches the mapping research: a fixed random feedback
wire cannot deliver a stimulus-specific template, and
the Kok/Aitken deep-layer template result is exactly
what such a wire cannot produce.

## 54. Round 19: hierarchy (experiment E) -- DESIGN ONLY, NOT LAUNCHED

Status: protocol frozen Nov 27, 2026
(PROTOCOL-R19-HIERARCHY.md). Driver
rpc_round19_hierarchy.py, model rpc_hier.py, analyzer
r19/r19_analyze.py, monitor r19/monitor.py, smoke
rpc_round19_smoke.py. Nothing launched on Beast. The
design agent numbered its draft section 51; it is
recorded here as 54, because 51 is the redirection, 52
is the round-18 verdict, and 53 is the round-17 verdict.

### 54.1 Question

Does stacking fixed random reservoirs into a hierarchy,
with one delta-trained distal wire per level and one
fixed random feedback wire per level, learn structure
that a flat two-module model cannot learn? The sharpest
version: can the hierarchy learn the first motif step
(m1), which sits at 0.680 against the 0.80 bar in round
15, when the first step depends on a slow context the
flat model cannot hold?

### 54.2 Hypothesis (one sentence)

A three-level hierarchy with a slow top level (effective
memory about 18 steps) learns the slow context of a
two-timescale sequence and uses it to predict the
context-dependent first motif step, which a flat
two-module model cannot do because its memory spans
about 2 to 3 steps.

### 54.3 Architecture (A28, new file rpc_hier.py; rpc_v2.py untouched)

1. M modules stacked (M = 2 or 3): input feeds module 1,
   module 2 predicts module 1, module 3 predicts module 2.
2. Each level l < M has a distal wire W_dist_l trained
   ONLY by the delta rule (signed error between the level
   prediction and the actual code below it, NLMS
   denominator, w_max 2.0).
3. Each level l < M has a fixed random feedback wire
   W_fb_l into the module below it (A26 form, never
   trained), with per-level gain fb_gain_l and test modes
   on / cut / shuffle.
4. Level leak rates are fixed by depth: alpha 1.0, 0.3,
   0.1 (estimated effective memories about 2-3, 6, 18
   steps at rho 0.9). No confirmation, no inhibition.
5. Readout stays the fixed p1-prototype cosine at the
   bottom level; the top-level prediction against
   mean-p_{M-1} prototypes is a secondary readout (M = 3
   only, no bar).

Nested initialisation: weights are drawn level by level
from one seeded RNG stream, so a 3-level net's levels
1-2 weights are EXACTLY a 2-level net's at the same seed
(smoke-tested).

### 54.4 Hierarchical task (make_hier_data)

Length 48, 4 blocks of 12 steps. Each block starts with
a context token C in {0,1,2,3}. Each context has two
motifs with shared starters 4 and 5: C=0: (4,6,7),(5,8,9);
C=1: (4,8,9), (5,10,11); C=2: (4,10,11),(5,12,13); C=3:
(4,12,13),(5,14,15). Each block plants one motif
starting with 4 and one with 5 at random non-overlapping
starts; filler is uniform from 4..15. m1 context-blind
Bayes is 0.25; m2 is determined by the 2nd token;
overall chance 1/16 = 0.0625. Late m1 = starter at block
position 6 or later. Flat model expected score: late m1
about 0.25 (context-blind), overall acc about 0.35-0.40.
Bar: the candidate beats flat on late m1 by at least
+0.20 and reaches late m1 mean at least 0.45 (lower CI
above 0.35).

### 54.5 Tasks, seeds, conditions

Tasks: planted lag-1 (200/50/50, K=16), sine-fixed
(50/20/20, K=32), hier (100/50/50, K=16). Seeds 282..301
(20 fresh; val seeds 282..286). Base configs are the
round-15 winners (planted1 alpha 1.0 / sigma 0.0 / eta_d
0.05; sine alpha 1.0 / sigma 0.0 / eta_d 0.2; hier alpha
1.0 / sigma 0.0 / eta_d 0.05). 5 epochs, n_res 200,
n_out 256.

Conditions (5 nets per task per seed, 8 lines per task
per seed, 480 test lines): cand_on (val-winner
architecture, delta, feedback on), cand_cut (same net,
feedback cut), cand_shuffle (same net, feedback
shuffled), hier2_ladder (M=2, ladder alphas, winner
fb_gain1, delta), hier2_nofb (M=2, ladder alphas, no
feedback, delta), flat (M=2, both alphas = task alpha,
no feedback, delta: the round-15 winner config),
cand_frozen (winner architecture, no plasticity),
baseline (Markov-1/2, persistence, ESN on identical test
sequences). If the val winner is 2 levels, hier2_ladder
duplicates cand_on and is skipped (420 lines), and H4 is
not judged.

Validation (hier task only): n_levels {2,3} x fb_gain1
{0.1,0.3} x fb_gain2 {0.1,0.3} = 6 effective configs, 5
val seeds, 30 lines. Winner = highest mean val overall
accuracy, tie-break first in grid order, frozen for all
test conditions.

### 54.6 Bars (20-seed test, paired per-seed differences, 95% CI)

1. H1: hier overall acc, cand_on - flat >= +0.10.
2. H2: hier late m1, cand_on - flat >= +0.20.
3. H3: hier late m1, cand_on mean >= 0.45, lower CI above 0.35.
4. H4: hier late m1, cand_on - hier2_ladder >= +0.10 (3-level winner only).
5. H5: planted lag-1 m1, cand_on - flat >= +0.05.
6. H6: planted lag-1 m1, cand_on mean >= 0.80.
7. H7: sine-fixed acc, cand_on - flat >= +0.03.
8. H8: hier overall acc, cand_on - cand_cut >= +0.05.
9. H9: hier overall acc, cand_on - cand_frozen >= +0.10.
Shuffle contrast is a diagnostic, not a bar. Verdict: H0
passes if H1, H2, H3 pass; H4 decides third level vs
slower level 2; H5-H7 decide whether hierarchy helps the
old tasks; H8-H9 are mechanism checks.

### 54.7 Smoke test (VM, seed 282, 1 epoch unless noted)

All checks pass (rpc_round19_smoke.py, 2m31s on the VM,
Nov 27, 2026). 1. Nested init exact: a 3-level net's
levels 1-2 weights are identical to a 2-level net's at
the same seed. 2. Flat reproduction (5 epochs, seed
201): hierarchical flat net scores planted1 0.207 vs
round-15 recorded 0.227 (diff -0.020) and sine 0.903 vs
0.872 (diff +0.031), both within the 0.05 bar. 3.
Feedback changes the lower module's state: mean r1
cosine on vs cut 0.9725 (below 1.0, above 0.0);
fb_rms_ratio on 0.2513, cut 0.0000; shuffle drive
measured on the same trajectory (same ds) is 1.0306x
the on drive (bar 0.90-1.10); the separate-pass shuffle
fb RMS is 0.2914 vs on 0.2513 (gap +16.0 percent), which
is trajectory divergence, not a magnitude bug, and it
weakens the shuffle contrast as a diagnostic. 4. Top
readout non-degenerate: top_pred takes 16 distinct
values over 480 steps. 5. Timescale ladder: lag-11
context decode from the untrained net is L1 0.70, L2
0.97, L3 0.93 (chance 0.25; L3 exceeds L1, so the
ladder holds); lag-1 sanity is 1.00 at all levels.
Note: L1 retains context at 0.70, so the flat model may
beat the 0.25 late-m1 guess. 6. Guardrails: all metrics
finite, no NaN, r_max 0.869 (below 10), clamp_frac
0.0000 per level [0.0, 0.0]; driver CLI resumable (1
line after two runs); analyzer writes r19_analysis.md on
1-seed output (exits 1 on coverage, as designed).

### 54.8 Execution

About 30 CPU-hours (300 training runs at about 6 min
each), about 5 h wall with 6 workers (3 tasks x 2
seed-shards). Monitor r19/monitor.py drives val ->
winner pick -> test -> analysis at 480 lines (420 if the
winner is 2 levels).

### 54.9 Self-critique (from the frozen protocol)

1. rpc_v2.py is untouched; rounds 17 and 18 keep running against it.
2. Level alphas are fixed by depth, not validated.
3. Flat reproduction in the hierarchical class is approximate (within 0.05), because the init draw order differs.
4. hier2-ladder vs flat conflates a slower level 2 with feedback; conditions 5 and 6 separate them as diagnostics.
5. Val seeds are a subset of test seeds (same mild overlap as rounds 15-18).
6. A win with no cut gap means training-time shaping, not test-time feedback use.
7. Round 19 does NOT test precision (C), generative (D), or the mapping onto canonical microcircuits or hippocampal data.
8. (Main agent, Nov 27): the shuffle diagnostic is weakened by the +16.0 percent separate-pass gap (see 54.7 point 3 and the protocol section 10 amendment).

### 54.10 Main-agent review (Nov 27, 2026, before launch)

Two points from the design agent's report, verified
against the files on disk: (a) the protocol section 10
check 3 was amended pre-launch (shuffle drive measured
on the same trajectory, not separate-pass RMS; no bar
changes); (b) the hypothesis premise is already dented
by the smoke test: level 1 decodes the block context at
lag 11 with 0.70 accuracy, so the flat model's memory
spans well past the assumed 2 to 3 steps. The flat
model may therefore beat the 0.25 late-m1 guess, which
compresses the +0.20 gap H2 demands and makes H3
harder. The round is still worth running: the bars
decide, and a flat win on late m1 is itself the finding
that hierarchy is unnecessary for this task. The main
agent re-ran the VM smoke test independently before
launch (result recorded in the launch entry below).

### 54.11 Launch (Nov 27, 2026, 00:25 PT)

Main agent re-ran the VM smoke independently: all checks
pass, numbers identical to the design agent's run
(3c 1.0306, 3d +16.0 percent, ladder L1 0.70 / L2 0.97 /
L3 0.93, r_max 0.869, clamp_frac 0.0000). Code copied to
Beast (tarball rpc-r19.tar.gz, protocol + rpc_hier.py +
driver + smoke + r19/). Beast smoke re-run: SMOKE PASSED,
all checks, numbers identical to the VM.

Monitor fix (main agent, before launch): r19/monitor.py
launched workers with Popen + wait, so the cron run that
launches the test phase would block for its whole ~5 h
wall (300 training runs at about 6 min, 6 workers) and
could be killed mid-phase, leaving the test_launched
marker set and the monitor waiting forever. Both
launch_val and launch_test now use start_new_session=True
with no wait and write their marker at launch time. The
patched monitor was copied to Beast. No protocol or
driver change.

Six validation workers launched by one monitor run at
00:25 PT (verified alive with correct args, one per
config: L2g010, L2g030, L3g010_g010, L3g010_g030,
L3g030_g010, L3g030_g030; seeds 282-286, hier task);
val_launched marker written. Monitor cron r19-monitor
created (every 30 min, drives val -> winner -> test ->
analysis at 480 lines, 420 if the winner is 2 levels).

## 55. Round 17/18 critiques -- LAUNCHED Nov 27, 2026, 00:34 PT

Brief CRITIQUE-BRIEF-R1718.md (10 checks, built by private
proactive work, verified by the main agent against
sections 52/53: every quoted number matches the record).
Sharpest questions: (1) whether the round-17 wire was
actually live in the code (feedback is 6.5 percent of the
drive yet module 1's code moves 0.2 percent); (2) whether
the round-18 frozen ridge beating the trained ridge means
delta actively harms the readout rather than adding
nothing. Also: the lag-2 C1 bar artifact, borderline D2/D3,
the scoring-correction audit, Markov-2's dominance, the
mapping tension (A passes, B fails), round 19's risk of
repeating round 17's null, and which of the generative (D)
or precision (C) tests settles more.

Pack synced to /home/reza/zfs-1tb/rpc-sim/review/
(simulation-results.md sections 44-54, both protocols,
PROTOCOL-R19-HIERARCHY.md, rpc_v2.py, both drivers, r17/
and r18/ with winners, test JSONLs and analyses, the
brief). Winner files were fetched from Beast first
(r17 fb_gain 0.1 val 0.22609 vs gain0 0.22661; r18 lorenz
bump 2.0 val_mse 7.082, varsine bump 3.0 val_mse 0.0356),
so the pack is complete. Launch script /tmp/launch_r1718.sh
(same pattern as round 14, brief pasted in ONE message):
Kimi PID 534794 (log r1718-kimi-critique.log, writing),
AGY Gemini PID 534795 (log r1718-agy-critique-gemini.log,
empty at 45 s, same slow start as round 14, process alive
with the full brief in its args). AGY Opus NOT attempted:
quota-blocked until about Dec 1, 2026; its round-14 retry
is separately scheduled Dec 2. Reviewers write
r1718-kimi-critique.md and r1718-agy-critique.md in the
review directory. Collection job r1718-critique-check
(every 30 min) fetches, verifies and records both when
they land.

### 54.12 Validation winner (Nov 27, 2026, 00:44 PT)

30/30 val lines (5 per config), no duplicates, no
Traceback. Mean val overall accuracy on the hier task
(seeds 282-286): L2g010 0.268, L2g030 0.263,
L3g010_g010 0.268, L3g010_g030 0.261, L3g030_g010 0.266,
L3g030_g030 0.259. Winner: L2g010 (2 levels, fb_gain1
0.1, val_acc 0.26826). L3g010_g010 ties the winner at
0.268; the tie-break (first in grid order) picks the
2-level model. The third level adds nothing on
validation: the slow top level does not help even before
the test phase. Consequence: the test is 420 lines (the
hier2_ladder condition duplicates the candidate and is
skipped), and bar H4 (3-level beats 2-level ladder) is
NOT judged. Test phase launched at 00:44 PT: 6 detached
workers (3 tasks x 2 seed-shards, seeds 282-301, winner
config), test_launched marker written, 0 lines at launch.
The r19-monitor body was corrected in the same turn: its
verdict instruction said "section 55", which is the round
17/18 critiques launch record, so both monitor bodies now
say "the next free section, checked against the file".

## 56. Round 19 verdict: the hierarchy does not win; the flat model wins everywhere (Nov 27, 2026, 01:15 PT)

Test phase complete and analysis done (monitor run 01:14 PT).
420/420 test lines (target 420, because the val winner is
2 levels), 20 seeds 282-301, 7 conditions x 3 tasks, no
Traceback in any log. The a/b shard files are exact copies
of the 282-301 files (identical rows, no conflicts), so the
420 unique (task, condition, seed) keys are the full test
set. Independent recomputation from the raw JSONL confirms
every headline mean to 3 decimals and every baseline mean
to 3 decimals; CI endpoints differ by at most 0.003 (the
analysis uses a different CI method), and no bar verdict
changes.

Val winner (record section 54.12): 2 levels, fb_gain1 0.1,
val_acc 0.268. Note the near tie: the best 3-level config
L3g010_g010 scored 0.26809, only 0.00017 below the winner,
so the winner is a tie-break artefact as much as a win.

Headline test numbers (20 seeds, mean with 95 percent CI):

1. hier task: cand_on acc 0.248 (0.238, 0.258), flat 0.255
   (0.247, 0.262); cand_on late_m1 0.278 (0.241, 0.315),
   flat late_m1 0.263 (0.241, 0.286). Baselines: Markov-1
   0.301, Markov-2 0.345, ESN 0.406. Every trained model
   sits below Markov-1.
2. planted1 task: cand_on acc 0.210, flat 0.226; cand_on m1
   0.561 (0.512, 0.609), flat m1 0.682 (0.633, 0.730).
   Baselines: Markov-1 0.280, Markov-2 0.272, ESN 0.279.
3. sine task: cand_on acc 0.679 (0.648, 0.710), flat 0.874
   (0.860, 0.887). Baselines: Markov-1 0.466, Markov-2
   0.787, ESN 0.946. The flat model beats Markov-2 and
   sits just below the ESN.
4. Frozen sanity: hier frozen acc 0.061 (0.052, 0.070),
   below the 0.25 ceiling. PASS.

Bars (protocol PROTOCOL-R19-HIERARCHY.md, 20-seed paired
differences): 1 of 9 pass.

1. H1 FAIL: hier acc cand_on minus flat -0.007 (-0.014,
   0.000). Bar +0.10 with lower CI above 0.
2. H2 FAIL: hier late_m1 cand_on minus flat +0.015 (-0.015,
   0.044). Bar +0.20 with lower CI above 0.
3. H3 FAIL: hier late_m1 cand_on 0.278 (0.241, 0.315). Bar
   mean at least 0.45 with lower CI above 0.35.
4. H4 FAIL (not judged): val winner is 2 levels, so the
   3-level ladder duplicates cand_on (protocol section 9).
5. H5 FAIL: planted1 m1 cand_on minus flat -0.121 (-0.148,
   -0.094). Bar +0.05 with lower CI above 0.
6. H6 FAIL: planted1 m1 cand_on 0.561 (0.512, 0.609). Bar
   mean at least 0.80.
7. H7 FAIL: sine acc cand_on minus flat -0.195 (-0.226,
   -0.163). Bar +0.03 with lower CI above 0.
8. H8 FAIL: hier acc cand_on minus cand_cut +0.002 (-0.001,
   0.004). Bar +0.05 with lower CI above 0. Feedback is
   not used at test time.
9. H9 PASS: hier acc cand_on minus cand_frozen +0.187
   (0.177, 0.198). Bar +0.10 with lower CI above 0.
   Learning matters, but it does not buy hierarchy.

Shuffle diagnostic (not a bar): hier acc cand_on minus
cand_shuffle +0.001 (-0.001, 0.004). Even if H8 had
passed, the near-zero shuffle gap would say the effect is
drive magnitude, not feedback content.

Verdict:

1. H0 FAILS. The hierarchy does not win the hierarchical
   task. The flat model beats the 2-level hierarchy on
   every task: hier acc by 0.007, planted1 m1 by 0.121,
   sine acc by 0.195.
2. The dented premise from section 54.10 is confirmed:
   level 1 already decodes block context at lag 11 with
   0.70 accuracy, so the flat model had the context the
   hierarchy was supposed to add, and the late-m1 guess
   of 0.25 was beaten by the flat model itself (0.263).
3. Depth adds nothing: the 3-level config ties the
   2-level winner on validation, and the 2-level winner
   loses to flat on test.
4. Mechanism: feedback is inert at test time (H8 fail,
   shuffle gap 0.001, cand_cut and cand_shuffle match
   cand_on to 0.002). The only passing bar is H9, which
   says the trained weights beat frozen weights, not that
   the hierarchy does anything the flat model does not.
5. The hierarchy experiment is closed with a flat win.
   The open question from round 14 stands: the delta
   rule learns, but nothing tried so far makes the
   top-down wire or the extra level earn its keep.

Files: r19/r19_analysis.md, r19/r19_hier_val_*.jsonl,
r19/r19_*_test_*.jsonl, r19/winner.json, fetched to
~/workspace/thesis-rpc/r19/. The r19-monitor cron is
removed; its job is done.

Correction (main agent, Nov 27, 2026, 01:16 PT,
independent recomputation): the "1 of 9" tally counts
H4 as a fail, but the protocol says H4 is not applicable
when the val winner is 2 levels, so the honest tally is
1 of 8 judged bars pass (H4 N/A). The verdict is
unchanged: H0 fails, and the only passing bar is H9.
The analyzer's r19_analysis.md keeps its own "1 / 9
pass (0 N/A)" line as the artifact of record; this
section is the corrected account.

## 57. Rounds 17/18 critiques land: Kimi and AGY Gemini (Nov 27, 2026, about 01:20 PT)

Both critiques landed on Beast in /home/reza/zfs-1tb/rpc-sim/review/ and both
reviewer processes are gone. Kimi finished first (file 29,157 bytes, 08:53 UTC);
AGY Gemini finished 09:15 UTC (file 59,433 bytes). Both files fetched to
~/workspace/thesis-rpc/ (r1718-kimi-critique.md, r1718-agy-critique.md),
md5-verified against the Beast copies. Both reviewers read sections 44-53,
both protocols, rpc_v2.py, both drivers and analyzers, and all 560 test JSONL
records. Both ran fresh diagnostics on Beast. This section records their
verdicts and what I verified myself. Where a number is reviewer-reported and
I did not re-run it, I say so.

### 57.1 Experiment B (round 17): the failure stands for the tested wire only

1. Both reviewers agree the preregistered failure stands for the exact
   mechanism tested: a fixed random additive wire at fb_gain 0.1. Lag-1 C1-C4
   all fail, and the lag-2 C1 pass is not a protocol-licensed pass (see 57.3).
2. Strongest reason to doubt the general conclusion ("the wire carries no
   causal role"): round 17 tested the wrong wire geometry. The feedback is a
   fixed random additive map (+W_fb @ d1). Rao and Ballard (1999) and
   Chapter 3 specify aligned subtractive feedback, minus W_prox transpose
   times d1. AGY ran that control on seed 241: at gain 1.0 the subtractive
   wire is stable, raises normal accuracy above wire-off (0.2330 vs 0.2217),
   reaches omit_m2 1.000, clears C3 (p1_cos 0.8565), and drops the omission
   code cosine from 0.401 to 0.036. Reviewer-reported, one seed, not
   re-checked by me. It shows the tested wire, not feedback in general,
   failed.
3. Wire liveness, verified by me in the code: the wire is live. Feedback
   enters module 1 before tanh and before the leak (rpc_v2.py:1127-1141,
   564-573), and round 17 uses alpha 1.0, so the leak attenuates nothing.
   There is no normalization or threshold that washes it out. The 6.5
   percent figure is fb over bottom-up drive only (rpc_v2.py:1286-1288); it
   omits the recurrent drive W_res @ r, so the true drive fraction is about
   3.9 percent (AGY). At that size, p1_cos 0.998 follows from plain geometry,
   not from a broken wire. Kimi's gain diagnostic (reviewer-reported) shows
   p1_cos 0.843 at gain 1.0, so the implementation can respond.
4. The wire is not only inert, it is slightly harmful. Both reviewers found
   the same unreported effect: enabling the wire raises final training error
   on 19 of 20 seeds (lag-1 +0.00203, p below 1e-7) and degrades lag-1
   omit_m1 by 0.0157 (p 0.007). Section 53 did not report this. The honest
   sentence is: the fixed random wire at gain 0.1 has a tiny causal effect
   in the wrong direction, not no effect.
5. Winner-pick bias, verified by me in the code: gain 0.0 can never win.
   r17_analyze.py only considers gains above 0 as winners, so the procedure
   must pick a positive wire even when no wire scores best (winner.json:
   0.22609 at gain 0.1 vs 0.22661 at gain 0.0). AGY's sweep shows the bias
   did not change round 17: at gains 0.3, 1.0, and 2.0, cutting the wire
   improves omit_m2, so no gain on the grid passes B. The bias matters for
   future designs, not for the round-17 outcome.

### 57.2 The continuous wall (round 18): split verdict, and the Lorenz wall falls

1. Kimi: the Lorenz C1 prototype MSE wall stands, but the var-sine MSE wall
   is unresolved, and "delta does not learn continuous information" is
   refuted. Delta massively improves prototype MSE over frozen nets on all
   four task/input combinations (20 of 20 seeds each). D8 only tests one raw
   fixed-lambda ridge probe.
2. AGY: the headline wall verdict is refuted. Round 18 froze hyperparameters
   selected under round 15's buggy two-step-ahead scoring, and D8 repeats
   round 14's unstandardized ridge confound.
3. I reproduced AGY's decisive Lorenz numbers exactly, from the same code
   and data (script in the goal hidden_files, run on this VM):
   discrete bin input only, no bump, no ridge, 5 seeds 262-266. The
   Convention-A config (alpha 1.0, smooth_sigma 2.0, eta_d 0.2) scores
   acc 0.4285, MSE 5.073, per-seed MSE [5.68, 4.62, 3.75, 5.83, 5.48].
   The frozen round-18 C1 config scores acc 0.3169, MSE 26.611, per-seed
   [24.88, 20.06, 25.52, 31.16, 31.44]. These match AGY's table to every
   printed digit. Conclusion: D4 (Lorenz C1 MSE 22.39 vs persistence 12.61)
   was a hyperparameter carryover artifact, not a wall. Under valid
   validation the delta prototype model beats persistence 5 of 5 seeds
   (MSE 5.07 vs 13.05) and beats the round-18 bump config C3 (7.73).
   AGY's var-sine Convention-A numbers (MSE 0.0370) are reviewer-reported;
   I did not re-run them.
4. D8, verified by me in the code and by a seed-262 re-run: fit_ridge
   centers the features but does not standardize them, with lam fixed at
   1.0 (rpc_round18_continuous.py:231-246). On seed 262, mean drive norm is
   1.025 trained vs 5.651 frozen (5.52x), exactly Kimi's numbers. Raw
   lam 1.0 ridge MSE is 14.904 trained vs 14.073 frozen (frozen wins).
   Column-standardized lam 1.0 ridge MSE is 13.976 vs 13.976, a tie. The
   frozen win is a regularization artifact from delta shrinking the drive
   norm, not evidence that delta adds no continuous information. AGY's
   stronger claim (trained beats frozen once scale-matched) depends on the
   matching method; my standardization gives a tie, not a trained win.
5. D2/D3 borderline, settled by my recomputation from the JSONLs: D2b mean
   +0.010808, 13 of 20 seeds positive. Under the analyzer's z 1.96 rule the
   CI is [+0.000312, +0.021304], a pass by 0.0003 of margin. Under exact
   Student t with 19 degrees of freedom the CI is [-0.000400, +0.022016],
   a fail. Kimi and AGY are each right under their own interval rule. D3
   fails under both (mean -0.008275, 7 of 20 seeds). H0 fails either way,
   but only because the frozen C1 config is bad, not because a var-sine
   MSE wall is established.
6. Scoring correction, verified: both reviewers confirm no remaining
   one-step shift in round 18, and I verified collect() scores pred(t)
   against token t for t at least 1, matching run(). One record correction:
   the bug was in rpc_round15.py run_cont, not in baselines.py; the
   sections 46/52 blame on the baselines docstring is backwards.
7. Two more record corrections from the critiques, verified against the
   files: the var-sine C3 prototype MSE 0.039 beats persistence 0.056 and
   beats C4 0.044, but the D7-var analyzer line (r18_analyze.py:288-295)
   only computed C2 minus C4 and dropped the C3 comparison. And the
   "C1_onehot" label is wrong on var-sine, where the winner uses
   smooth_sigma 2.0, so the input is smoothed bins, not one-hot.
8. Markov-2: both reviewers show the tasks are trivial for two-bin lookup
   (99.50 percent of Lorenz test 2-grams and 99.82 percent of var-sine test
   2-grams appear in training; two bins explain R squared 0.989 and 0.995
   of next-step variance). The wall that survives is narrower: the model
   still trails Markov-2 (Lorenz MSE 1.416, var-sine 0.0053) and a matched
   continuous-ridge ESN (Lorenz 2.097, var-sine 0.0072, reviewer-reported),
   and no linear W_dist1 on a frozen reservoir can exceed the linear
   decodability of y2_prev. A discriminating task needs multi-step memory
   or noise: NARMA-10 or Mackey-Glass with tau 17.

### 57.3 Bar and tally corrections

1. Lag-2 C1 is an analyzer invention, verified by me: the frozen protocol
   sets the 0.10 floor for C1, and r17_analyze.py applies it to lag-1 only
   and checks lag-2 on CI above zero alone. Under the frozen definition
   the lag-2 C1 pass (+0.004) fails by a factor of 24. The honest round-17
   tally is 1 of 9 (C5, a guard), not 2 of 9.
2. Section 52's "continuous wall survives correct scoring" is narrowed to:
   the wall survives only for the frozen Convention-B C1 configuration and
   for the Markov-2/ESN comparison. It does not survive Convention-A
   validation of round 15's own grid.
3. Section 53's "the wire carries no causal role" is narrowed to: the fixed
   random additive wire at gain 0.1 carries no useful causal role. Aligned
   subtractive feedback is untested at scale.
4. What survives of the thesis claim, per both reviewers: fixed random
   reservoir plus local signed-error learning can learn discrete sequential
   predictions and make continuous information accessible to a fixed
   prototype readout. What does not survive as demonstrated: causal
   top-down explanation away, precision weighting, burst confirmation, and
   any advantage over reservoir plus Widrow-Hoff NLMS. The accurate name
   stays local error-driven prediction learning on fixed random reservoirs.

### 57.4 Round 19 and the next experiment

1. Both reviewers warned round 19 repeats round 17's null: same fixed
   random wire, gains 0.1 and 0.3, validation on filler-dominated overall
   accuracy, no state-effect gate. Section 56 shows the warning came true
   (H8 +0.002, shuffle gap +0.001). The warning now applies to any
   follow-up, not to round 19, which is closed.
2. The reviewers disagree on the next experiment. Kimi: run precision-weighted
   causal feedback (B prime plus C); a generative rollout D alone cannot
   distinguish predictive coding from autoregressive NLMS. AGY: fix the
   feedback geometry first (subtractive minus W_prox transpose or locally
   learned wire), then run generative rollout D plus a revised B; precision
   C is premature without a causal wire. Common ground: fix the wire first
   with a per-level state-effect gate (p1_cos below 0.90) and selection on
   late-m1 or omission effects, not overall accuracy. Neither reviewer
   recommends an unchanged D or C on the current architecture.
3. AGY's proposed cheap pair (about 6 CPU-hours): aligned-feedback clamp
   plus multi-step omission rollout on noisy/occluded planted tasks with
   fresh seeds 302-321, then a Convention-A re-validation plus NARMA-10 /
   Mackey-Glass with standardized ridge probes. Kimi's bar for C: an
   explicit precision manipulation with an expectation-by-precision
   interaction on stimulus-specific error responses, with a live
   state-effective wire whose cut and shuffle remove the effect.

Files: r1718-kimi-critique.md, r1718-agy-critique.md in ~/workspace/thesis-rpc/.
Verification scripts and output:
~/workspace/goals/random-predictive-coding-simulation-proof-hunt/hidden_files/r1718_spotcheck.py,
r1718_d8check.py, r1718_spotcheck_out.txt. The r1718-critique-check cron is
removed; its job is done. AGY Opus was not attempted (quota-blocked to
about Dec 1; round-14 retry separately scheduled Dec 2).

Note (main agent, Nov 27, 2026, 01:35 PT): I re-ran both scripts myself
and reproduced every number exactly (A14 acc 0.4285 / MSE 5.073, C1_R18
0.3169 / 26.611, D8 norms 1.025 / 5.651, raw ridge 14.904 / 14.073,
standardized ridge 13.976 / 13.976). One script bug found: r1718_spotcheck.py
crashes in its D8 tail (line 44 passes bins=None to ridge_metrics, which
then fails on the within-1 computation). The A14/C1 comparison completes
before the crash, and the D8 numbers come from r1718_d8check.py, which
passes the bins correctly. No result changes; the bug is in the spot-check
script only.
## 58. Round 20: aligned-feedback clamp (revised experiment B) + multi-step omission rollout (experiment D) -- DESIGN + LAUNCH

(Nov 27, 2026. Design subagent report; protocol frozen in
PROTOCOL-R20-ALIGNED.md. Launched on Beast Nov 27, 2026,
02:12 PT; see the launch entry at the end of this section.)

### Why this round exists

Round 17 tested the wrong wire. It used a fixed random
additive wire (+W_fb @ d1, A26) and found it inert: cutting,
shuffling, or never building it changed omission accuracy by
at most half a point. Both external reviewers of rounds 17/18
(Kimi and AGY Gemini, section 57) agree the failure stands
only for that wire. Rao and Ballard (1999) and Chapter 3
specify ALIGNED SUBTRACTIVE feedback: minus the readout
transpose times the prediction. AGY's one-seed control
(reviewer-reported) reached omit_m2 1.000 with that wire at
gain 1.0. This round fixes the wire first (A29), then re-runs
the clamp (revised experiment B) and adds the multi-step
omission rollout (generative experiment D, first half).
Precision (experiment C) is NOT in this round: both reviewers
agree it is premature without a causal wire.

### Assumption A29 (rpc_v2.py)

Three parts, all default-off, all previous results reproduce
exactly (smoke S2: gain 0.0 reproduces the round-17 code path
to 6 decimals).

1. Aligned subtractive feedback: fb_wire="aligned" replaces
   W_fb at init with -W_prox.T, rescaled to the Frobenius norm
   of the random W_fb it replaces, so at the same fb_gain the
   drive magnitude matches the random wire and only the
   direction differs. Reproduces AGY's seed-241 control
   EXACTLY (normal_acc 0.2330, omit_m2 on 1.000 / cut 0.966,
   omit_m1 on 0.8034, p1_cos 0.8565, err_last 1.6162,
   cos_p1_omit_m2 on 0.036 / cut 0.401, fb_drive_ratio 0.383).
   At gain 0.3 the gate fails (p1_cos 0.9667, gap 0.000).
2. Learned back-projection feedback: fb_wire="learned" starts
   as the aligned map and is trained by a local composite
   delta rule (gradient descent on
   (1/2)||-d1 - W_prox @ W_fb @ d1||^2, NLMS form), so the
   feedback, passed through the readout, subtracts the
   prediction from the code. This is NOT AGY's literal
   formula (delta W_fb proportional to (r1 - W_fb d1) d1^T):
   that rule was implemented first and measured, and it is
   unusable. On seed 241 it collapses at gain 1.0
   (normal_acc 0.061, chance 0.0625, feedback 4.8x the total
   drive) and rotates AWAY from the aligned direction at
   every gain (fb_align -0.02 at 0.1, 0.11 at 0.3, 0.63 at
   1.0), because d1 barely predicts the reservoir state r1
   and the target has state scale while the drive has drive
   scale. The composite rule is stable at every gain
   (normal_acc 0.232 to 0.240), keeps its subtractive
   direction (fb_align 0.91), and grows its norm from 14.20
   to about 17.2. The deviation from AGY's literal formula
   is deliberate, measured, and documented in the protocol.
3. Blank token: _encode returns the zero vector for tok == K.
   No existing code passes token K, so no previous result
   changes. Used for occlusion and rollout blanks.
4. Corrected drive ratio: fb_drive_ratio = ||fb|| / ||total
   drive|| (round 17's fb_rms_ratio divided by the bottom-up
   norm only and overstated the feedback fraction by about
   1.7x).

### Design

Tasks: planted1-occ, planted2-occ (40% of filler positions
blanked in train AND test; motif positions and the filler
probe position never occluded; mask deterministic per
(task, seed, split); calibration and prototypes on CLEAN
train), sine (no occlusion). Seeds 302..321 (20 fresh).
Round-15 winner configs (planted alpha 1.0, smooth 0.0,
eta_d 0.05; sine eta_d 0.2), 5 epochs.

Validation: 3 wire types (aligned, learned, random) x 3
gains (0.1, 0.3, 1.0) on planted1-occ, val seeds 302..306.
Selection per wire type: largest val on-cut omit_m2 gap
among gains passing the state gate (val p1_cos < 0.90) and
stable (r1_max < 10); tie-break smallest gain. If no gain
passes the gate, the largest gap among stable gains wins
and the gate failure is flagged. The random wire's test gain
is the ALIGNED winner's gain (matched control).

Test: 5 nets per (task, seed), 11 lines per (task, seed),
660 test lines total. Conditions: aligned on/cut/shuffle,
learned on/cut/shuffle, random on/cut/shuffle (at the
aligned gain), wire-off, frozen-aligned.

Bars (exact Student-t, t(19) = 2.0930, correcting the
round-17/18 z=1.96 convention; lag-2 uses the SAME
0.10/0.05 floors as lag-1, fixing the round-17 analyzer
bug). B' (aligned wire is causal): B1 on-cut omit_m2 >=
+0.05 with lower t-CI > 0; B2 on-shuffle >= +0.05; B3
p1_cos mean < 0.90; B4 on-off CI excludes 0; B5 explaining
away (err_mag_last on < off, upper t-CI < 0); B6
cos_p1_omit_m2 on < cut with on mean < 0.20; B7 guard
(normal_acc >= 0.80x wire-off). B8 (secondary): aligned
gap exceeds random gap by >= +0.05. B9 (secondary):
learned on-cut gap >= +0.05. B10 (diagnostic): learned gap
minus aligned gap. B' passes iff B1-B6 pass and B7 passes.
D (rollout): D1-D3 sine rollout_acc on-cut / on-shuffle /
on-off >= +0.10; D4 sine late rollout_acc mean >= 0.30 with
lower t-CI > 0.20 (chance 0.031); D5-D6 planted rollout_comp
on-cut / on-shuffle >= +0.20; D7 (secondary) learned sine
rollout >= +0.10; D8 guard frozen sine rollout < 0.10. D
passes iff D1-D6 pass and D8 passes. Thesis-level rule
(AGY): predictive-coding causality is established iff B'
AND D both pass.

Pre-registered "causal but small" outcome: if B1 (or B2,
or B9) has mean > 0 with lower t-CI > 0 but the mean is
below the floor, the round reports "causal but small" as a
distinct outcome (neither pass nor fail).

Continuous re-validation is SPLIT into round 21
(Convention-A Lorenz/var-sine re-validation plus NARMA-10
and Mackey-Glass tau 17, column-standardized ridge probes,
local parameter-free continuous readout). It is sketched
in the protocol, not implemented in round 20.

### Smoke results (VM, seed 302, rpc_round20_smoke.py)

- S1 occlusion: deterministic=True, 1235 blanks in the
  200-sequence train split, 0 at motif positions, 0 at the
  probe position. PASS.
- S2 repro gain-0: driver normal_acc 0.195652, omit_m2
  0.875000, identical to the round-17 code path to 6
  decimals. PASS.
- S3 aligned gate: p1_cos 0.8033 (< 0.90), on-cut gap
  +0.0167, normal_acc 0.2383 (5 epochs, gain 1.0, occluded).
  PASS. Note: the occluded single-seed gap (+0.017) is
  SMALLER than the clean gap (+0.034 on seed 241), so the
  B1 floor (0.05) is above both measured single-seed gaps
  and the round may report "causal but small" rather than
  a pass.
- S4 shuffle RMS: on 0.2781, shuffle 0.2756, relative
  difference 0.0090 (within 5%). PASS.
- S5 clamp_frac: 0.0000. PASS.
- S6 learned stable: normal_acc 0.2296, fb_align 0.912,
  fb_norm 17.20, r1_max 0.97, err_last 1.6213 (5 epochs,
  gain 1.0, occluded). PASS.
- S7 rollout: sine rollout_acc 0.160, rollout_acc_late 0.030
  (n=20), planted rollout_comp 0.680 (n=50), frozen sine
  rollout_acc 0.043 (near chance 0.031). PASS. The sine
  rollout decays to chance by the late steps; the frozen
  rollout is near chance, so the D8 guard should hold.
- S8 CLI resumable: val lines 1 (expect 1), test lines 11
  (expect 11) after two runs each. No duplicate lines. PASS.

### Files

- PROTOCOL-R20-ALIGNED.md (frozen Nov 27, 2026; 11-point
  self-critique).
- rpc_round20_aligned.py (driver; val and test phases;
  resumable JSONL; one BLAS thread per process).
- r20/r20_analyze.py (analyzer; exact-t CIs; bar table and
  verdict; writes r20/r20_analysis.md).
- r20/monitor.py (val -> winner pick -> test -> analysis;
  9 val workers, 6 test workers, 660 test lines; detached
  workers, marker at launch time; Beast venv).
- rpc_round20_smoke.py (8 checks, seed 302).
- rpc_v2.py (A29: fb_wire, blank token, corrected drive
  ratio, composite learned rule).

### Two sharpest risks in this design

1. The B1 floor (0.05) is above every measured single-seed
   gap (+0.034 clean, +0.017 occluded). If the 20-seed mean
   follows the single-seed pattern, B1 fails and the round's
   headline is "causal but small" (if the CI excludes 0) or
   a null. The design handles this honestly (the
   pre-registered "causal but small" outcome), but the
   headline pass is genuinely uncertain, and D (the rollout)
   is the more sensitive test.
2. The learned wire is our fix, not AGY's formula. The
   composite rule is stable and aligned in smoke, but it has
   never been shown to beat the fixed aligned wire on any
   bar, and its fixed point (a map about 1.2x the aligned
   norm after 5 epochs) may still be growing when training
   stops. If the learned wire's bars (B9, D7) fail while the
   aligned wire's pass, the round cannot say whether a
   longer-trained learned wire would have won; that is a
   round-21 question, not a round-20 conclusion.

### Next step

Launch on Beast via r20/monitor.py (val -> winner -> test
-> analysis). Estimated about 6 CPU-hours, about 1 hour
wall with 6 workers. After completion: verify the analysis
by independent recomputation from the raw JSONL, write the
verdict as section 59, update README Status and MEMORY.md,
log a tracking entry on goal_a9a871e81255, remove the
r20-monitor cron, and deliver the verdict to Reza.

### 58.1 Launch (Nov 27, 2026, 02:12 PT)

Main-agent verification before launch: all 8 smoke checks
re-run independently on the VM, all PASS with numbers
identical to the design agent's report (S3 p1_cos 0.8033,
on-cut gap +0.0167; S6 fb_align 0.912, fb_norm 17.20; S7
sine rollout 0.160, late 0.030, planted comp 0.680, frozen
0.043). A29 code read in rpc_v2.py (aligned = -W_prox.T
norm-matched; learned composite rule with the AGY-formula
deviation documented; blank token; corrected drive ratio).
Files copied to Beast in /tmp/rpc-r20.tar.gz (7,911,654
bytes) and extracted at /home/reza/zfs-1tb/rpc-sim/thesis-rpc/.
Beast smoke re-run PASSED 8/8 with VM-identical numbers.
9 validation workers launched via r20/monitor.py (PIDs
542857-542865, 3 wire types x 3 gains, planted1-occ, seeds
302-306); val_launched marker written; 0 lines at launch.
Monitor cron r20-monitor created (every 30 min). Beast
load at launch 12.25 on 12 cores, driven by the connectome
chat's Allen binwidth runs (about 10 workers, untouched);
round-20 wall time may stretch past the about 1 hour
estimate. Verdict goes in section 59.

### 58.2 Validation winners (Nov 27, 2026, 02:44 PT)

45/45 val lines (5 per config), 45 unique keys, no
Traceback. Mean val on-cut omit_m2 gap (planted1-occ,
seeds 302-306): aligned 0.1 +0.005 (p1_cos 0.986), 0.3
+0.007 (0.949), 1.0 +0.017 (0.805); learned 0.1 +0.005
(0.983), 0.3 +0.014 (0.931), 1.0 -0.002 (0.751); random
0.1 -0.007 (0.990), 0.3 +0.004 (0.968), 1.0 -0.088
(0.754). Winners (per wire type, largest gap among gains
passing the state gate p1_cos < 0.90 and r1_max < 10):
aligned 1.0, learned 1.0, random 1.0; all gates PASS. The
gate does real work: for every wire type, gains 0.1 and
0.3 fail it (p1_cos 0.93 to 0.99), so only gain 1.0 moves
module 1's state enough to count as a live wire. The
learned winner looks odd (gain 1.0 gap -0.002 beats gain
0.3 gap +0.014) only until the gate is applied: gain 0.3
fails the gate at p1_cos 0.931, so gain 1.0 is the only
gated candidate. Random at gain 1.0 actively hurts: gap
-0.088 and val_acc 0.190 against aligned 0.236 and learned
0.232. The aligned val gap (+0.017) matches the smoke S3
single-seed gap (+0.0167) and sits well below the B1 floor
(0.05), so the test phase will likely land "causal but
small" if the 20-seed mean follows validation. Test phase
launched 02:44 PT: 6 detached workers (3 tasks x 2
seed-shards, seeds 302-321), test_launched marker written,
0 lines at launch.

## 59. Round 20 verdict: B' FAILS, D FAILS, aligned wire is "causal but small" on lag-1 only (Nov 27, 2026, 04:14 PT)

Test phase finished 04:14 PT: 660/660 test lines (20 seeds
302-321 x 11 configs x 3 tasks), 660 unique
(task, config, seed) keys, no Traceback in any worker log,
analysis marker written, r20_analyze.py rc=0. The analysis
(r20/r20_analysis.md) was verified by independent
recomputation from the raw JSONL
(r20/verify_r20.py, fresh code, exact Student-t CIs,
t(19) = 2.0930): every headline mean matches to 3
decimals. Two values sit on exact .0005 ties (sine
aligned-cut rollout_acc 0.1835, D1 mean -0.0365) and the
analysis rounds both half-up (0.184, -0.037); the
recomputation agrees under that rule.

### Bar table: 9 of 28 pass

Passes: B3, B4, B6, B7, B8 on lag-1; B3, B6, B7 on lag-2;
D8. Everything else fails.

1. B1 lag-1 FAIL: aligned on-cut omit_m2 +0.014, CI
   [0.005, 0.022]. Bar is >= +0.05 with lower CI > 0.
2. B2 lag-1 FAIL: aligned on-shuffle omit_m2 +0.033, CI
   [0.022, 0.044]. Same bar.
3. B3 lag-1 PASS: aligned p1_cos mean 0.804, CI
   [0.797, 0.811]. Bar is mean < 0.90.
4. B4 lag-1 PASS: aligned on-off omit_m2 +0.009, CI
   [0.003, 0.015]. Bar is CI excludes 0.
5. B5 lag-1 FAIL: err_mag_last on-off +0.0138, CI
   [0.008, 0.020]. Bar wants mean < 0 with upper CI < 0
   (explaining away). The sign is wrong: the wire ADDS
   error magnitude at the last step.
6. B6 lag-1 PASS: cos_p1_omit_m2 on-cut -0.391, CI
   [-0.408, -0.374]; on mean 0.033, CI [0.002, 0.063].
   Bar is mean < 0 with upper CI < 0 and on mean < 0.20.
7. B7 lag-1 guard PASS: aligned normal_acc 0.232 vs
   wire-off 0.221 (guard >= 0.80x).
8. B8 lag-1 PASS: aligned gap minus random gap +0.071, CI
   [0.048, 0.093]. Bar is >= +0.05 with lower CI > 0.
9. B9 lag-1 FAIL: learned on-cut omit_m2 +0.004, CI
   [-0.000, 0.009]. Bar is >= +0.05 with lower CI > 0.
10. B1 lag-2 FAIL: -0.006, CI [-0.016, 0.004].
11. B2 lag-2 FAIL: +0.015, CI [0.000, 0.030].
12. B3 lag-2 PASS: p1_cos mean 0.810, CI [0.802, 0.817].
13. B4 lag-2 FAIL: on-off omit_m2 -0.009, CI [-0.019, 0.000].
14. B5 lag-2 FAIL: err_mag_last on-off +0.0181, CI
    [0.013, 0.023]. Wrong sign again.
15. B6 lag-2 PASS: cos_p1_omit_m2 on-cut -0.381, CI
    [-0.402, -0.360]; on mean -0.038, CI [-0.075, -0.001].
16. B7 lag-2 guard PASS: aligned normal_acc 0.224 vs
    wire-off 0.220.
17. B8 lag-2 FAIL: aligned gap minus random gap +0.049, CI
    [0.028, 0.071]. One point under the +0.05 floor.
18. B9 lag-2 FAIL: learned on-cut omit_m2 -0.027, CI
    [-0.042, -0.013]. The learned wire hurts on lag-2.
19. B10 diagnostic: learned gap minus aligned gap -0.009,
    CI [-0.018, -0.001] on lag-1 and -0.021, CI
    [-0.032, -0.010] on lag-2. Learned is worse than
    aligned on both lags, and both CIs exclude 0.
20. D1 FAIL: sine rollout_acc aligned on-cut -0.037, CI
    [-0.051, -0.022]. Bar is >= +0.10.
21. D2 FAIL: sine rollout_acc on-shuffle -0.025, CI
    [-0.037, -0.014].
22. D3 FAIL: sine rollout_acc on-off -0.076, CI
    [-0.092, -0.061].
23. D4 FAIL: sine late rollout_acc (on) mean 0.051, CI
    [0.038, 0.064]. Bar is mean >= 0.30 with lower CI >
    0.20; chance is 0.031.
24. D5 lag-1 FAIL: rollout completion on-cut -0.146, CI
    [-0.190, -0.102]. Bar is >= +0.20.
25. D6 lag-1 FAIL: rollout completion on-shuffle -0.109, CI
    [-0.157, -0.061].
26. D5 lag-2 FAIL: rollout completion on-cut -0.225, CI
    [-0.265, -0.185].
27. D6 lag-2 FAIL: rollout completion on-shuffle -0.155, CI
    [-0.204, -0.106].
28. D7 FAIL: sine rollout_acc learned on-cut -0.035, CI
    [-0.047, -0.022].
29. D8 guard PASS: frozen sine rollout_acc mean 0.032, CI
    [0.025, 0.040]. Guard is mean < 0.10, so generation
    needs learning. But the learned net generates WORSE
    with the wire on.

### Verdict

1. Experiment B' (aligned wire is causal): FAILS. B1, B2,
   B5, B9 fail on lag-1; B1, B2, B4, B5, B8, B9 fail on
   lag-2. B' passes iff B1-B6 pass and B7 passes.
2. Experiment D (multi-step generation through the
   top-down pathway): FAILS. D1-D7 all fail, and every
   sign is negative: the wire harms rollout, it does not
   help it.
3. Thesis-level rule (AGY): predictive-coding causality
   is established iff B' AND D both pass. Not established.
4. Pre-registered "causal but small" outcome: APPLIES, on
   lag-1 only. B1 lag-1 (+0.014, CI [0.005, 0.022]) and
   B2 lag-1 (+0.033, CI [0.022, 0.044]) both have mean >
   0 with lower CI > 0 and mean below the +0.05 floor.
   It does NOT apply to B9 lag-1 (+0.004, CI
   [-0.000, 0.009], CI touches 0), B2 lag-2 (+0.015, CI
   [0.000, 0.030], lower bound is 0.000, not above 0),
   B1 lag-2 (-0.006), or B9 lag-2 (-0.027).

### What the numbers say

1. The aligned wire is genuinely causal on lag-1
   omission, but small: +1.4 points on-cut, +3.3 points
   on-shuffle. This is the outcome the design predicted
   as likely (section 58 risk 1): the B1 floor (0.05) sat
   above every measured single-seed gap (+0.034 clean,
   +0.017 occluded), and the 20-seed mean (+0.014)
   confirms it. The val gap (+0.017) predicted the test
   gap almost exactly.
2. The direction matters, not just the drive. B8 lag-1
   passes (+0.071): the aligned wire beats the random
   wire by more than the floor. The random wire at gain
   1.0 actively hurts (val gap -0.088; test random-on
   omit_m2 0.873 vs random-cut 0.931, so cutting the
   random wire helps). Lag-2 B8 (+0.049) misses the floor
   by one point.
3. The mechanism signature is present even where the
   accuracy gain is small. B6 passes on both lags: cutting
   the wire raises cos_p1_omit_m2 by about 0.39, so the
   wire does rotate module-1 state away from the
   prediction when module 2 is omitted. But B5 fails on
   both lags with the wrong sign: err_mag_last is HIGHER
   with the wire on (+0.0138, +0.0181), so there is no
   explaining-away signature. The wire changes the state
   in the predicted direction without reducing the error
   it is supposed to explain away.
4. The learned wire fails everywhere. B9 is a null on
   lag-1 and negative on lag-2; D7 is negative; B10 shows
   it loses to the fixed aligned wire on both lags. The
   composite rule stays stable and aligned (fb_align 0.91
   planted, 0.84 sine) but never beats the fixed wire on
   any bar. Whether a longer-trained learned wire would
   win is a round-21 question, not a round-20 conclusion.
5. Experiment D is a clean negative with the wrong sign.
   Sine rollout_acc: on 0.147, cut 0.184, wire-off 0.223.
   Cutting the wire, or never building it, generates
   better. Late rollout accuracy 0.051 is barely above
   chance (0.031) and far below the 0.30 bar. Planted
   rollout completion is higher with the wire cut
   (-0.146 lag-1, -0.225 lag-2). The D8 guard passes, so
   generation needs learning, but the learned net plus
   the aligned wire generates worse than the learned net
   alone.
6. The wire costs one-step accuracy on sine and nothing
   on planted. Sine normal_acc: aligned-on 0.666 vs
   wire-off 0.892, a 23-point cost. Planted normal_acc:
   aligned-on 0.232 vs wire-off 0.221, no cost (B7 guard
   passes on both lags). The wire's cost is
   task-dependent, which matters for round 21: the
   continuous tasks look more like sine than like
   planted, so carrying the aligned wire at gain 1.0 into
   round 21 carries a known one-step cost.
7. The gate did its job in validation (section 58.2) and
   the test confirms the choice: only gain 1.0 passes the
   state gate on any wire type, and the test winners are
   all gain 1.0. Nothing in the test phase contradicts
   the validation pick.

### Files

- r20/r20_analysis.md (analyzer output, 660 lines,
  660 unique keys).
- r20/r20_planted1-occ_test_302-321.jsonl,
  r20_planted2-occ_test_302-321.jsonl,
  r20_sine_test_302-321.jsonl (220 lines each).
- r20/r20_planted1-occ_val_*.jsonl (45 lines).
- r20/verify_r20.py (independent recomputation; all
  headline means match to 3 decimals).
- r20/winner.json (aligned 1.0, learned 1.0, random 1.0).

### Next step

Round 21 (continuous re-validation: Convention-A
Lorenz/var-sine plus NARMA-10 and Mackey-Glass tau 17,
column-standardized ridge probes, local parameter-free
continuous readout) is sketched in
PROTOCOL-R20-ALIGNED.md, not implemented. Open design
question for round 21: whether to carry the aligned wire
at gain 1.0 into the continuous tasks given the 23-point
sine one-step cost and the negative rollout sign, or to
test the wire on/off as a factor. The r20-monitor cron is
removed; its job is done.
## 60. Round 21: continuous re-validation (Convention-A + NARMA-10 / Mackey-Glass) -- DESIGN + LAUNCH

(Nov 27, 2026. Design subagent report; protocol frozen in
PROTOCOL-R21-CONTINUOUS.md. Launched on Beast Nov 27, 2026,
05:26 PT; see the launch entry at the end of this section.)

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

### 60.1 Launch (Nov 27, 2026, 05:26 PT)

Main-agent verification before launch: the VM smoke was
re-run independently (785 s, exit 0, zero failures) and
every number reproduced the design agent's report exactly
(Lorenz Convention-A acc 0.4285 / MSE 5.073 per-seed
[5.68, 4.62, 3.75, 5.83, 5.48]; var-sine 0.3230 / 0.0370
diff -0.0000 vs AGY; D8 standardized ridge ties 13.976 /
13.976, raw reproduces the frozen win 14.073 < 14.904;
NARMA-10 G1 seed 322 proto 0.0124 / centroid10 0.0081 /
ridge_std 0.0077; Mackey-Glass 0.0020 / 0.0021 / 0.0027;
CLI resumable 10 -> 10; frozen twin diff 0.00e+00;
centroid beta 5/10/20 = 10.5741 / 5.0133 / 4.4831 vs proto
6.6547; wire gate fb_drive_ratio 0.2982, wire-on proto
3.7809 vs off 6.6547 on lorenz seed 322). Protocol sections
1-10 and the wire decision (section 5) read in full; the
verdict rules and the 10-point self-critique stand.
Files copied to Beast in /tmp/rpc-r21.tar.gz (36,150 bytes):
rpc_round21_continuous.py, r21/r21_analyze.py, r21/monitor.py,
PROTOCOL-R21-CONTINUOUS.md, baselines.py (new make_narma10 /
make_mackeyglass verified present on Beast). Beast smoke
PASSED: lorenz val seeds 322-323 wrote 10 lines; lorenz
test seed 327 wrote 7 rows (token_delta acc 0.397 MSE
3.1756 ridge_std 5.0831; token_frozen 0.020 / 355.1006 /
5.1177; token_w_delta 0.390 / 3.9210 / 6.4896; bump_delta
0.423 / 3.0689 / 5.3452; baseline persist 14.9004,
Markov-2 1.2570, ESN scalar standardized 1.7694). Smoke
files removed after the check. One monitor run launched 4
detached token-val workers (PIDs 551476-551479, one per
task, seeds 322-326); token_val_launched marker written;
0 lines at launch. Monitor cron r21-monitor created
(every 30 min). Beast load at launch 24.20 on 12 cores
(connectome Allen binwidth runs, untouched); the protocol
says to give the round a full day, not an hour. Verdict
goes in section 61.

### 60.2 Token validation winners (Nov 27, 2026, 06:44 PT)

100/100 token val lines (25 per task), 25 unique keys per
task, no Traceback. Mean val prototype MSE per grid config
(seeds 322-326): lorenz G1 4.9682, G2 6.1658, G3 6.7281,
G4 4.6268, G5 8.1088; varsine G1 0.0373, G2 0.0391, G3
0.0320, G4 0.0240, G5 0.0539; narma10 G1 0.0138, G2 0.0133,
G3 0.0127, G4 0.0129, G5 0.0109; mackeyglass G1 0.0021, G2
0.0030, G3 0.0035, G4 0.0025, G5 0.0031. Winners
(winners_token.json): lorenz G4 (alpha 1.0, smooth 3.0,
eta_d 0.2, 4.6268); varsine G4 (alpha 1.0, smooth 3.0,
eta_d 0.2, 0.0240); narma10 G5 (alpha 1.0, smooth 2.0,
eta_d 0.02, 0.0109); mackeyglass G1 (alpha 1.0, smooth 2.0,
eta_d 0.2, 0.0021). The lorenz and varsine winners DIFFER
from the preregistered primaries (lorenz G1 smooth 2.0;
varsine alpha 0.7 / smooth 2.0), so the test phase runs
conditional primary arms on both tasks (W9/W10 logic):
the test target is 640 lines, not 560 (7 rows per
task-seed, plus 2 conditional rows per seed on lorenz and
varsine), and the
round tests the val winner against the primary head to
head on fresh seeds 327-346. Bump val launched (4 workers,
bump_val_launched marker, 0/40 lines at launch).

Bump val complete (Nov 27, 07:15 PT; 40/40 lines, 10 per
task, no duplicate keys, no Traceback; means recomputed
from raw JSONL, 5 seeds each): lorenz sigma 2.0 wins
(4.3965 vs 3.0 at 4.6281); var-sine sigma 3.0 wins
(0.0236 vs 2.0 at 0.0390); NARMA-10 sigma 3.0 wins
(0.0103 vs 2.0 at 0.0108); Mackey-Glass sigma 2.0 wins
(0.0022 vs 3.0 at 0.0025). Final winners in winner.json:
lorenz G4 + bump 2.0, var-sine G4 + bump 3.0, NARMA-10
G5 + bump 3.0, Mackey-Glass G1 + bump 2.0. Test phase
launched 07:15 PT (8 workers, PIDs 554081-554088, seeds
327-346, 0 lines at launch; monitor target 640 lines).
Beast load about 15.7.

## 61. Round 21 verdict: WALL FALLS for the continuous re-validation; the signed-error rule learns continuous trajectories (Nov 27, 2026, 09:14 PT)

Test complete 09:14 PT: 640 unique rows (180 lorenz, 180
var-sine, 140 NARMA-10, 140 Mackey-Glass), 0 duplicates in
the merged test files, 0 Tracebacks in any worker log.
Analysis r21/r21_analysis.md, verified by independent
recomputation from the raw JSONL (r21/verify_r21.py, fresh
code, not the analyzer): every per-task mean, every paired
difference and exact-t CI, all 54 bar outcomes, all
diagnostics, and the val winners reproduce to 3 decimals.
One sign-convention note in the verifier: F1 and the
wire-on-frozen diagnostic are computed as on-minus-off
there and off-minus-on in the analysis, so the signs flip
and the magnitudes and CI endpoints match exactly.
Main-agent second check (/tmp/r21_indep_check.py, written
from scratch): every headline mean and every checked bar
(W1/W2/W7/W8/W9/W10, N1/N2/N7/N8, R1/R4, F1/F2, B1,
W3/W4/W5/W6) reproduces to 4 decimals; 640 unique keys,
0 Tracebacks, r21-monitor cron confirmed removed.

### Val winners (seeds 322-326; restated from section 60.2)

1. Token: lorenz G4 (alpha 1.0, smooth 3.0, eta_d 0.2, val
   MSE 4.6268); var-sine G4 (same config, 0.0240); NARMA-10
   G5 (alpha 1.0, smooth 2.0, eta_d 0.02, 0.0109);
   Mackey-Glass G1 (alpha 1.0, smooth 2.0, eta_d 0.2,
   0.0021).
2. Bump sigma: lorenz 2.0 (4.3965 vs 4.6281 at 3.0);
   var-sine 3.0 (0.0236 vs 0.0390 at 2.0); NARMA-10 3.0
   (0.0103 vs 0.0108 at 2.0); Mackey-Glass 2.0 (0.0022 vs
   0.0025 at 3.0).
3. Lorenz and var-sine winners differ from the
   preregistered primaries, so the test ran the conditional
   primary arms (token_primary_delta/frozen, 20 seeds
   each); the new tasks take the val winner, no separate
   primary arm.

### Per-task test means (prototype MSE, 20 seeds 327-346)

1. Lorenz: delta 4.2875, frozen 236.8511, wire-on delta
   4.5180, bump delta 4.7361, persistence 12.5914,
   Markov-2 1.3961, ESN argmax 4.5633, ESN scalar-std
   1.7799.
2. Var-sine: delta 0.0270, frozen 0.9303, wire-on delta
   0.0448, bump delta 0.0274, persistence 0.0560,
   Markov-2 0.0052, ESN argmax 0.0450, ESN scalar-std
   0.0082.
3. NARMA-10: delta 0.0117, frozen 0.0915, wire-on delta
   0.0120, bump delta 0.0104, persistence 0.0091,
   Markov-2 0.0128, ESN argmax 0.0124, ESN scalar-std
   0.0070.
4. Mackey-Glass: delta 0.0020, frozen 0.1911, wire-on
   delta 0.0035, bump delta 0.0019, persistence 0.0012,
   Markov-2 0.0009, ESN argmax 0.0013, ESN scalar-std
   0.0004.

### The wall bars (protocol section 8: wall falls iff W1, W2, W7, W8 all pass)

1. W1 (lorenz primary beats persistence): paired diff
   +8.3039, 95% CI [7.5853, 9.0225], PASS.
2. W2 (var-sine primary beats persistence): +0.0290, CI
   [0.0219, 0.0360], PASS.
3. W7 (lorenz delta beats frozen): +232.5637, CI
   [188.6496, 276.4778], PASS.
4. W8 (var-sine delta beats frozen): +0.9033, CI
   [0.8092, 0.9974], PASS.
5. WALL FALLS for the re-validation tasks. The
   Convention-A prototype readout beats persistence on
   fresh seeds, and the win is due to learning: the
   trained net beats its frozen twin by a factor of 55
   on Lorenz (4.29 vs 236.85) and 34 on var-sine (0.027
   vs 0.930). Round 18's "continuous wall" does not
   survive valid validation on fresh seeds.
6. The preregistered primaries pass too (W9 +7.6045, CI
   [6.8184, 8.3906]; W10 +0.0187, CI [0.0131, 0.0244]),
   so the wall-fall does not depend on the val winners:
   the preregistered configs (lorenz primary MSE 4.9869,
   var-sine primary 0.0372) beat persistence on their
   own. Val selection is not doing the work.

### Bar table: 26 pass, 24 fail, 4 n/a (measurement-only F1), of 54 rows

1. W3 FAILS: lorenz primary does not beat ESN argmax
   (+0.2758, CI [-0.3146, 0.8663] covers 0; proto 4.2875
   vs ESN 4.5633, nominally better but not significant).
   W4 PASSES on var-sine (+0.0180, CI [0.0113, 0.0247]).
2. W5/W6 PASS: primary accuracy beats Markov-1 (lorenz
   +0.1920, CI [0.1754, 0.2086], acc 0.4047 vs 0.2127;
   var-sine +0.0442, CI [0.0330, 0.0555], acc 0.3254 vs
   0.2812).
3. N1/N2 FAIL: the prototype does NOT beat persistence
   on either new task (NARMA-10 -0.0026, CI
   [-0.0030, -0.0022]; Mackey-Glass -0.0008, CI
   [-0.0009, -0.0006]; both CIs entirely below 0, so
   persistence wins cleanly). One-step prediction on
   both new tasks is persistence-dominated, as the
   smoke test predicted.
4. N7 PASSES, N8 FAILS: beats Markov-2 on NARMA-10 only
   (+0.0011, CI [0.0007, 0.0015]; Mackey-Glass -0.0011,
   CI [-0.0013, -0.0009]). The tasks are not equally
   discriminating. N3 passes (beats ESN argmax on
   NARMA-10, +0.0007); N4 fails. N5/N6 pass (delta beats
   frozen on both new tasks, +0.0798 / +0.1892), so
   learning happens on the new tasks too, it just does
   not reach persistence.
5. R1 FAILS on all 4 tasks: the standardized trained
   ridge does not beat the standardized frozen ridge
   anywhere (lorenz +0.0185, CI [-0.0122, 0.0493];
   var-sine +0.0001, CI [-0.0001, 0.0004]; NARMA-10
   -0.0000; Mackey-Glass +0.0000). But it fails BY TIE,
   not by a frozen win: no CI sits below 0. Protocol
   section 8 item 5 ("delta actively harms... prototype
   wins are a readout artifact") does NOT trigger as
   written, because its condition (frozen beats trained)
   is not met. The pattern matches the round-14
   alignment story instead: the frozen drive already
   holds the information (the ridge reads it equally
   well from trained and frozen), and the delta rule
   aligns the drive into prototype space, which is
   exactly what W7/W8 measure. The ridge probe shows
   no learning signal; the prototype probe shows a
   large one. Both are true at once.
6. R4 FAILS everywhere: the ESN scalar-std ridge beats
   the trained standardized ridge on all 4 tasks
   (lorenz -2.7735; var-sine -0.0341; NARMA-10 -0.0009;
   Mackey-Glass -0.0029; all CIs below 0). The ESN
   scalar readout remains the strongest comparator, as
   expected. R3 passes on lorenz, var-sine, NARMA-10
   (trained ridge beats persistence) and fails on
   Mackey-Glass.
7. C1 passes ONLY on NARMA-10 (+0.0024, CI [0.0019,
   0.0029]; centroid beta=10 MSE 0.0093 vs prototype
   0.0117, a 20.5% cut, just over the protocol's 20%
   flag). It fails on lorenz (centroid 4.2912 vs proto
   4.2875, tied), var-sine (centroid 0.0391 vs proto
   0.0270, worse), and Mackey-Glass (0.0022 vs 0.0020,
   worse). The centroid is not the better continuous
   readout in general; the NARMA-10 cut is a
   task-specific readout effect, and it does not change
   the wall verdict because NARMA-10 still loses to
   persistence. C2 passes on lorenz, var-sine, NARMA-10
   (centroid beats persistence) and fails on
   Mackey-Glass.
8. F1 (wire effect, off-minus-on): NULL on lorenz
   (-0.2306, CI [-0.9934, 0.5322]) and NARMA-10
   (-0.0003, CI [-0.0008, 0.0002]); HURTS on var-sine
   (-0.0178, CI [-0.0337, -0.0020]) and Mackey-Glass
   (-0.0015, CI [-0.0028, -0.0003]). The wire never
   helps on continuous tasks, and on two tasks it
   significantly hurts. F2 passes on all 4 tasks (the
   wire is live: mean fb_drive_ratio 0.3161 lorenz,
   0.3127 var-sine, 0.3424 NARMA-10, 0.3103
   Mackey-Glass, all far above the 0.05 gate), so the
   F1 result is a live wire costing accuracy, not a
   dead wire doing nothing.
9. B1 passes ONLY on NARMA-10 (+0.0013, CI [0.0011,
   0.0016]; bump beats token); it fails on lorenz
   (-0.4487, CI covers 0), var-sine (-0.0005, covers 0),
   and Mackey-Glass (+0.0000, covers 0). B2 passes on
   lorenz (+7.8553) and var-sine (+0.0285), fails on
   NARMA-10 (-0.0013) and Mackey-Glass (-0.0007): the
   bump arm inherits the token arm's persistence losses
   on the new tasks. The bump input is not a general
   win.
10. D8 diagnostics (ridge std vs raw, trained vs frozen):
    lorenz trained std/raw 4.5533/4.7122, frozen std/raw
    4.5719/4.7118; var-sine 0.0423/0.0439 vs 0.0424/
    0.0435; NARMA-10 0.0079/0.0080 vs 0.0079/0.0079;
    Mackey-Glass 0.0032/0.0034 vs 0.0033/0.0034. The
    raw-frozen edge over raw-trained is 0.0004 at most
    (lorenz), i.e. the round-18 frozen-ridge win does
    not reproduce as a meaningful effect; under
    standardization the trained ridge is nominally
    better or tied everywhere.
11. Wire effect on the FROZEN net (frozen-off minus
    frozen-on proto MSE): lorenz -22.2907, var-sine
    -0.0282, NARMA-10 -0.0134, Mackey-Glass -0.0066, all
    negative. The wire hurts even without learning, so
    its cost is in the wiring itself, not an interaction
    with the delta rule.
12. Centroid beta sensitivity (beta 5/10/20 means):
    lorenz 8.4300/4.2912/3.5158; var-sine 0.1121/0.0391/
    0.0247; NARMA-10 0.0124/0.0093/0.0091; Mackey-Glass
    0.0071/0.0022/0.0018. Beta=20 beats beta=10 on every
    task; the best-beta gap trips the >20% flag on
    var-sine (37%) and sits at 18% on lorenz and
    Mackey-Glass. Beta=10 was not validated as the best
    beta, so the C1 verdicts carry a beta-choice caveat,
    worst on var-sine.

### Verdict

1. WALL FALLS for the re-validation tasks (W1, W2, W7,
   W8 all pass). The signed-error rule learns continuous
   trajectories: Lorenz and var-sine, Convention-A, fresh
   seeds, prototype readout, with the win due to learning
   (delta beats frozen by 55x and 34x).
2. This does NOT establish predictive coding. Round 20
   failed the revised clamp (B') and the generative
   rollout (D), and nothing in round 21 relitigates
   that. A wall-fall here establishes only that the
   signed-error rule learns continuous trajectories
   (protocol section 8, item 5).
3. The wall SURVIVES on the new tasks in the only sense
   that matters for the thesis claim: the prototype does
   not beat persistence on NARMA-10 or Mackey-Glass, and
   it beats Markov-2 on NARMA-10 only. The continuous
   tasks that need multi-step memory are still
   persistence-dominated at one step, and the ESN scalar
   ridge still wins everywhere. The re-validation tasks
   (Lorenz, var-sine) are the easy end of the continuous
   spectrum: Markov-2 nearly solves them, persistence is
   weak on them, and the prototype only has to clear a
   low bar.
4. The wire is a net tax on continuous one-step
   prediction: live (F2 passes everywhere) and never
   helpful (F1 NULL twice, HURTS twice), hurting even
   the frozen net. Combined with round 20, the aligned
   wire's only positive evidence remains the lag-1
   "causal but small" clamp effect; it buys nothing on
   continuous tasks.
5. Open: the wall-fall is one-step only (h = 1). No
   rollout was tested here, and round 20's rollout
   failed. The next decision is Reza's: whether a
   multi-step continuous rollout (h > 1) is the next
   round, or whether the one-step wall-fall plus the
   round-20 negatives close the continuous chapter and
   the hunt moves on.

## 62. Round 22: the precision test (experiment C) -- DESIGN + LAUNCH

Status: DESIGN + LAUNCHED Nov 27, 2026, 10:14 PT (see the
launch entry at the end of this section). Protocol frozen
Nov 27, 2026 (`PROTOCOL-R22-PRECISION.md`). Driver
`rpc_round22_precision.py`, analyzer `r22/r22_analyze.py`,
monitor `r22/monitor.py`, smoke `rpc_round22_smoke.py`.

## 62.1 What C is

Experiment C is the last unrun discriminator. Omission (A)
passed (round 16); the top-down clamp with the random wire
(B) failed (round 17); the aligned-wire clamp (B') failed
with a "causal but small" lag-1 effect (round 20); the
generative rollout (D) failed cleanly (round 20). What
remains is precision: in Rao-Ballard predictive coding,
prediction errors are weighted by precision (inverse
variance), so the expectation effect on the error is larger
when the sensory input is reliable than when it is noisy,
and attention reverses prediction-induced silencing by
up-weighting precision (Kok, Rahnev et al. 2012). The
model has no precision parameter, so the question is
whether it shows the expectation-by-precision interaction
anyway, and whether the live aligned feedback wire carries
it.

## 62.2 The precision operationalization

There is NO precision parameter in this model, and the
protocol does not pretend otherwise. Precision is
operationalized as test-time input reliability: at the
probe step only, the encoded probe token is corrupted
with additive Gaussian noise, x = normalize(encode(tok)
+ c*z), z ~ N(0, I), fed through `step_vec`. High
precision is c = 0; low precision is c = c_low, picked
per task in validation by the decode rule (smallest c
with mean decode(p1) < 0.75 on calibration trials: CTL
for planted, EXP for sine). Training is always clean,
so the learned prediction is identical across precision
conditions; only the sensory evidence at the probe
differs. Two alternatives were considered and rejected:
an error gain during training (scaling eta_d) changes
what is learned, not the test-time error response, and
confounds learning rate with precision; bump width
(smooth_sigma) as precision changes the calibration
basis and interacts with the delta rule in its own right
(round 15). The NLMS denominator ||y2||^2 + 1 is
explicitly NOT precision: it is input-norm
normalization of the learning rate, acting during
training only.

The core measurement is the expectation-by-precision
interaction on the stimulus-specific error response
e = p1 - d1: gap(c) = rel_e(VIO, c) - rel_e(EXP, c),
interaction = gap(0) - gap(c_low), with rel_e = ||e||
divided by the net's own mean ||p1|| on clean
calibration trials. Specificity is spec(c) = cos(e,
proto_presented) - cos(e, proto_predicted) on VIO
trials. Probe trials are matched within a seed (same
filler prefix across EXP/VIO/UNEXP/CTL) and the noise
vector is identical across wire modes for the same
trial. 20 prefixes per (task, seed). Nets per (task,
seed): aligned (gain 1.0), random (gain 1.0), off
(gain 0.0), frozen-aligned (gain 1.0, plasticity off);
8 config lines per (task, seed). Tasks: planted lag-1,
planted lag-2, sine. Round-15 winner configs, 5 epochs.
Val seeds 347-351; test seeds 352-371.

## 62.3 Design probe (seed 347, 20 prefixes)

Aligned wire, gain 1.0. Planted lag-1: VIO-EXP gap at
c = 0 is +0.157 raw; interaction (c = 0 minus c = 0.5)
is +0.066 to +0.098 raw (two noise draws), rel_e about
+0.048. VIO specificity spec(0) +0.54, spec(0.5) +0.20,
interaction +0.34. decode(e) on VIO at c = 0 is 1.00.
decode(p1) on CTL: 1.00 at c = 0, 0.85 at 0.25, 0.50
at 0.50. p1 decode EXP vs UNEXP 1.00 vs 1.00; ||p1||
1.64 vs 1.64 (no p1 suppression). Wire-cut interaction
+0.021 to +0.034; shuffle +0.002 to +0.023. Frozen:
gap0 -0.007, interaction about 0.00. Random wire:
interaction +0.072 raw, rel_e +0.030 (aligned +0.096 /
+0.056 in the same run); the random wire inflates
||p1|| (p1scale 2.37 vs 1.72 aligned).

Sine: gap0 +0.322 raw; interaction (c = 0 minus
c = 0.5) +0.238 to +0.277 raw, rel_e about 0.22.
spec(0) +0.64, spec(0.5) +0.26, interaction +0.38.
decode(e) on VIO at c = 0 is 1.00. decode(p1) on EXP:
1.00 at c = 0, 0.60 at 0.25, 0.20 at 0.50. Wire-cut
interaction +0.114; shuffle +0.102. Frozen: gap0
+0.017, interaction -0.010. Random wire: interaction
+0.317 raw (LARGER than aligned +0.285 in raw units)
but rel_e +0.164 (smaller than aligned +0.282)
because p1scale is 1.94 vs 1.01.

Headline of the probe: the interaction EXISTS wire-on
on both tasks, but it PERSISTS wire-cut at 30-45% of
the wire-on size, and the random wire carries 75% of
the aligned wire's interaction on planted. The wire
AMPLIFIES the interaction roughly 2-3x but does not
CREATE it. Kimi's strict bar (cut removes the effect)
is predicted to FAIL; the amplification (W1/W2) is the
testable wire claim. The interaction is mostly a
bottom-up sensory-noise effect.

A design-time bug: the probe used Python hash() on
trial-type strings for noise seeds; string hashing is
randomized per process, so noisy-condition numbers
varied between identical runs (planted gap at c = 0.5
was +0.091 in one run, +0.059 in another). The driver
uses integer-arithmetic seeds only (section 9.3 of the
protocol) and the smoke confirms determinism (S5).

## 62.4 Bars (frozen)

P-family (per task; sine primary, planted1
generalization, planted2 replication): P1 interaction
(aligned-on) >= floor_P1, lower CI > 0; P2 specificity
interaction >= floor_P2, lower CI > 0; P3 gap0 trained
minus frozen >= floor_P3, lower CI > 0; P4 guard:
spec(0) mean >= 0.25 with lower CI > 0.15 AND decode(e)
on VIO at c = 0 mean >= 0.50 with lower CI > 0.30.
Floors are set by the interaction validation phase as
50% of the val-measured value, minima 0.02 rel_e (P1,
P3) and 0.05 cosine (P2). W-family (per task): W1
interaction on minus cut >= floor_W1; W2 on minus
shuffle >= floor_W2; W3 aligned minus random >=
floor_W3; W4 on minus off >= floor_W4; W1s diagnostic
(strict Kimi: |interaction_cut| < floor_W1 and CI
covers 0); W5 gate (mean fb_drive_ratio > 0.05 AND
mean p1_cos < 0.90; if it fails the W-family is void
for that task); W6 guard (normal_acc aligned-on >=
0.80 x wire-off). S-family (planted only, wire-off,
equivalence): S1 |decP1 EXP-UNEXP| < 0.03 with CI
covering 0; S2 |p1n EXP-UNEXP| < 0.05 with CI covering
0; S3 measurement only wire-on. 34 pass/fail bars;
the verdict uses the preregistered subset (sine P1-P4,
planted1 P1, sine W1/W2).

Verdict: C STRONG PASS iff sine P1-P4 pass AND
planted1 P1 passes AND W1+W2 pass on sine. C PASS iff
sine P1-P4 pass AND planted1 P1 passes but W1/W2 fail
(interaction exists, wire does not carry it: a
bottom-up sensory-noise effect, not top-down
precision). C PARTIAL iff sine P1+P2 pass but planted1
P1 fails. C FAIL iff sine P1 or P2 fails, or sine P4
fails. Thesis-level: no outcome of round 22
establishes predictive coding. Round 20 failed B' and
D; a C pass establishes only that the model produces
precision-like behaviour from input noise plus error
selection.

## 62.5 VM smoke test (rpc_round22_smoke.py, seed 347)

All checks pass (run Nov 27, 2026, about 10 min on the
VM). Noise val: decP1 falls with c (planted1 CTL 1.00
at 0.10 -> 0.40 at 0.50; sine EXP 1.00 at 0.10 -> 0.20
at 0.25). c_low picked by the protocol rule: planted1
0.35, planted2 0.35, sine 0.25. Wire live: fbdr 0.251
planted1 / 0.302 sine, p1_cos 0.811 / 0.819 (round-20
values 0.31-0.38 / 0.804-0.810). Interaction on sine
0.362 rel_e, frozen -0.009 (null). Deterministic noise
confirmed (probe_pass twice on the same net: identical
aggregates). CLI resumable (noise val run twice: 5 lines
both times, 5 unique keys). Lag-2 machinery: 8 config
lines, key fields complete. Analyzer writes
r22_analysis.md with a bar tally (n = 1, so bars fail
on the n != 20 check, as designed; P4, W5, S1 pass on
the single seed).

Smoke floors (1 seed, 5 prefixes; the 5-seed val will
replace them): sine P1 0.181, P2 0.178, P3 0.235, W1
0.143, W2 0.154, W3 0.124, W4 0.020 (minimum); planted1
P1 0.031, P2 0.190, P3 0.042, W1-W4 0.020 (minimum);
planted2 P1 0.020 (minimum; lag-2 interaction -0.015
on the smoke seed), P2 0.149, W2 0.025.

Two findings from the smoke, recorded here because
they shape the verdict reading:
1. The shuffle is NOT magnitude-matched for the
   aligned wire: fb_drive_ratio on vs shuffle is 0.251
   vs 0.203 on planted1 (19% drop) and 0.302 vs 0.159
   on sine (47% drop). The aligned wire is a structured
   map (-W_prox.T); permuting d1 moves it out of the
   subspace where the map has large gain. A W2 pass
   therefore confounds feedback direction with feedback
   magnitude; the cut (W1) is the primary wire control.
   A rescaled shuffle would need an rpc_v2.py change
   and would break comparability with rounds 17/20; it
   is a follow-up, not part of this round.
2. Lag-2 is marginal at smoke scale: the magnitude
   interaction is -0.015 while the specificity
   interaction is +0.297, and the S2 p1n difference is
   -0.058, outside the 0.05 bound (||p1|| slightly
   SMALLER on expected trials, the direction the
   mapping report says the model should not show). The
   20-seed test decides whether either is real.

## 62.6 Sharpest risks (from protocol section 10)

1. The precision definition is a stipulation: the round
   tests sensory reliability, not precision weighting. A
   pass means the interaction exists WITHOUT precision
   weighting, which is the informative direction, but a
   critic can fairly say the round does not test
   precision at all.
2. The interaction is mostly bottom-up by construction
   (persists cut at 30-45%; random wire carries 75% on
   planted). If W1/W2 fail, the headline is "the
   interaction does not need the wire" -- a real result
   that does not advance the thesis claim.
3. The planted interaction is small (rel_e about 0.048
   vs 0.22 sine); the planted P1 floor (about 0.024-0.031)
   is close to measurement noise across seeds.
4. The floors are relative (50% of val); a weak val
   interaction produces a weak floor. The absolute minima
   are the only guard against a trivial bar.
5. rel_e normalization confounds wire comparisons: the
   random wire inflates ||p1||, so W3 is partly a test
   of the p1scale difference. The analyzer reports raw
   ||e|| interactions alongside rel_e.
6. The wire's effect on ||e|| is measurement-dependent:
   round 20 B5 found err_mag_last HIGHER with the wire
   on during training, while the design probe finds
   probe-step ||e|| LOWER with the wire on at test.
7. Sine has no UNEXP/CTL trials, so the S-family cannot
   run on sine and P3 (frozen) is the only sine control
   for "no learning".

## 62.7 Launch (not run)

Copy the driver, analyzer, monitor, and protocol to
Beast; Beast smoke (one noise val + one interaction val
+ one test invocation); monitor val launch; create the
r22-monitor cron (every 30 min, same pattern as
r21-monitor). Monitor phases: noise val (75 lines) ->
c_low -> interaction val (120 lines) -> floors ->
winner.json -> test (480 lines) -> concat + analysis.
Test phase is 240 nets, about 60-90 min on a loaded
Beast; total round about 3 h; give it a full day. Do NOT
edit simulation-results.md from this subagent; merge
this draft as section 62 when the round runs.

## 62.8 Launch (Nov 27, 2026, 10:14 PT)

Main-agent verification before launch: the VM smoke was
re-run independently (585 s, exit 0, "SMOKE PASSED: all
checks pass") and every number reproduced the design
agent's report exactly (fbdr 0.251 planted1 / 0.302 sine,
p1_cos 0.811 / 0.819; sine interaction 0.3622 rel_e,
frozen -0.0088; shuffle ratios 0.81 planted1 / 0.53 sine;
deterministic noise confirmed; CLI resumable 5 lines,
5 unique keys). Protocol sections 1-10 read in full; the
verdict rule, the 13-point self-critique, and the shuffle
caveat (W1 primary, W2 secondary) stand. Files copied to
Beast in /tmp/rpc-r22.tar.gz (32,285 bytes):
PROTOCOL-R22-PRECISION.md, rpc_round22_precision.py,
rpc_round22_smoke.py, r22/r22_analyze.py, r22/monitor.py,
r22/section_draft.md. Beast smoke PASSED (noise val 5
lines, interaction val 8 lines, test 8 lines, exit 0,
BEAST_SMOKE_OK; smoke files removed). One monitor run
launched 3 detached noise-val workers (PIDs 559227-
559229, one per task, seeds 347-351); noise_val_launched
marker written; 0/75 lines at launch. Monitor cron
r22-monitor created (every 30 min, first run 10:44:54
PT). Beast load at launch 10.13 on 12 cores (down from
24.20 at the round-21 launch); the protocol says the
round takes about 3 h, give it a full day. Verdict goes
in section 63. (Section 62.7 above is the design-time
launch checklist, now superseded by this entry.)

### 62.9 Noise validation (Nov 27, 2026, 10:44 PT)

75/75 noise val lines (25 per task), 25 unique keys per
task, no Traceback. Mean cal_decP1 by c: planted1 1.00 /
0.95 / 0.71 / 0.54 / 0.32 at c 0.10 / 0.15 / 0.25 / 0.35 /
0.50; planted2 1.00 / 0.99 / 0.79 / 0.61 / 0.39; sine
0.88 / 0.79 / 0.55 / 0.45 / 0.28. c_low picked by the
protocol rule (smallest c with mean decode < 0.75):
planted1 0.25, planted2 0.35, sine 0.25. No
over-corrupted flag (lowest picked decode 0.55 sine).
Planted1 c_low is 0.25, not the 1-seed smoke's 0.35:
the 5-seed val, not the smoke, sets the value. Interaction
val launched (3 workers, interaction_val_launched marker,
0/120 lines at launch).

Scope note (Nov 27, alignment synthesis): the Nov 26
23:03 no-stop order was read too broadly. The narrowed
scope covers parallel work on already-designed,
preregistered rounds only, never a new design or an
irreversible act. Round 22 is now designed, frozen, and
launched, so it continues to completion under that
narrowed scope; no new round design starts until Reza
lands and confirms the scope.

### 62.10 Interaction validation (Nov 27, 2026, 11:14 PT)

120/120 interaction val lines (40 per task), 40 unique
keys per task, no Traceback. Floors (winner.json):
planted1 P1 0.027 / P2 0.121 / P3 0.043 / W1-W4 0.020
(minima); planted2 P1 0.020 (minimum) / P2 0.146 / P3
0.020 / W1-W4 0.020; sine P1 0.079 / P2 0.136 / P3 0.108
/ W1 0.040 / W2 0.048 / W3 0.038 / W4 0.020. Val
interaction (aligned-on, rel_e, 5 seeds, recomputed
independently from the raw JSONL): sine 0.1583, t(4) CI
[0.0720, 0.2446]; planted1 0.0539, CI [0.0195, 0.0884];
planted2 0.0063, CI [-0.0372, 0.0497]. Specificity
interaction: sine 0.2715, planted1 0.2417, planted2
0.2927 (planted2 keeps a large specificity interaction
while its magnitude interaction is ~0). Frozen gap0 is
~0 on all tasks (-0.0073 / -0.0022 / +0.0005), so the
learning control is clean. Wire-cut interaction:
planted1 0.0322 (60% of wire-on persists), sine 0.0792
(50% persists), planted2 -0.0017. The wire carries 40%
of the interaction on planted1 and 50% on sine. The
monitor's "40% persists" phrasing for planted1 was the
wire-carried share, not the cut persistence; the cut
persistence is 60%, above the 30-45% range the design
probe showed on one seed. Planted2 P1 is expected to
fail (val interaction ~0 with CI covering 0); the
monitor logged the t(4) warning as designed. Test phase
launched 11:14 PT (6 workers, seeds 352-371, 2 shards
per task, 0/480 lines at launch).

## 63. Round 22: the precision test (experiment C) -- VERDICT (Nov 27, 2026, 12:15 PT)

Test phase complete: 480/480 lines (8 configs x 3 tasks x 20
seeds 352-371), 6 workers, launched 11:14 PT, analysis
written 12:15 PT, test phase about 61 min. The combined
352-371 files are the exact concatenation of the a/b shards
(identical rows, 0 keys with differing values); the 960-line
total across all test files is that duplication, not 960
nets. No Traceback in any .log. Analysis r22/r22_analysis.md,
analyzer r22/r22_analyze.py, monitor markers all present
(noise_val_launched, interaction_val_launched,
test_launched, analysis).

Independent verification (this run, script
r22/recompute_independent.py, written from the protocol
definitions without importing the analyzer): all 480 test
records reloaded from the raw JSONL; every headline mean,
median, CI, and paired difference recomputed with exact
Student-t (t(19) = 2.0930). All headline means match the
analysis to 3 decimals: 0 mismatches across 35 checked
values, including medians and CI bounds. The val means in
winner.json were recomputed from the raw interaction-val
JSONL and all seven floors per task re-derived as max(0.02,
0.5 x val): 0 floor mismatches. Bar tally recomputed: 24 of
34 pass, matching the analysis. c_low: sine 0.25, planted1
0.25, planted2 0.35 (from 62.9).

### 63.1 Sine (primary): 8 of 10 bars pass

P1 interaction (aligned-on) 0.1643, 95% CI [0.1416, 0.1871],
floor 0.0791, PASS. P2 spec interaction 0.2423, CI [0.2239,
0.2608], floor 0.1358, PASS. P3 gap0 trained minus frozen
0.2485, CI [0.2174, 0.2797], floor 0.1082, PASS. P4 spec0
0.5777, CI [0.5434, 0.6121]; decode(e) on VIO 0.8950, CI
[0.8403, 0.9497], PASS. W1 on minus cut 0.0805, CI [0.0624,
0.0986], floor 0.0395, PASS. W2 on minus shuffle 0.0955, CI
[0.0753, 0.1157], floor 0.0480, PASS. W3 aligned minus
random 0.0874, CI [0.0643, 0.1106], floor 0.0378, PASS. W4
on minus off -0.0605, CI [-0.0772, -0.0437], FAIL. W5 gate
fbdr 0.2792, p1_cos 0.7912, PASS. W6 normal_acc on minus
0.8x off -0.0687, CI [-0.0992, -0.0382], ratio on/off
0.720, FAIL. W1s diagnostic: interaction under cut 0.0838,
CI [0.0743, 0.0934], 51.0% of wire-on, PERSISTS. DIAG
interaction_raw aligned-on 0.1754 vs random-on 0.1563: in
raw units the random wire carries nearly as much as the
aligned wire, consistent with the rel_e / p1scale confound
(section 62.6 item 5).

### 63.2 Planted lag-1 (generalization): 10 of 12 bars pass

P1 0.0425, CI [0.0329, 0.0520], floor 0.0270, PASS. P2
0.2355, CI [0.2239, 0.2470], floor 0.1208, PASS. P3 0.0918,
CI [0.0779, 0.1058], floor 0.0431, PASS. P4 spec0 0.5487,
CI [0.5242, 0.5733]; decE_VIO 1.0000, PASS. W1 0.0246, CI
[0.0170, 0.0321], floor 0.0200, PASS. W2 0.0316, CI
[0.0236, 0.0396], floor 0.0200, PASS. W3 0.0198, CI
[0.0129, 0.0267], floor 0.0200, FAIL by 0.0002 (real but
below the floor). W4 0.0079, CI [0.0012, 0.0146], floor
0.0200, FAIL (real but small). W5 fbdr 0.2520, p1_cos
0.8410, PASS. W6 diff 0.0603, ratio 1.072, PASS. S1 decP1
EXP-UNEXP wire-off 0.0000, PASS. S2 p1n diff 0.0027, CI
[-0.0091, 0.0145], PASS. S3 (wire-on, measurement only)
p1n diff -0.0448. W1s diagnostic: interaction under cut
0.0179, 42.2% of wire-on, PERSISTS. DIAG interaction_raw
aligned-on 0.0723 vs random-on 0.0522.

### 63.3 Planted lag-2 (replication): 6 of 12 bars pass

P1 -0.0052, CI [-0.0159, 0.0054], floor 0.0200, FAIL. P2
0.3105, CI [0.2904, 0.3306], floor 0.1463, PASS. P3
0.0038, CI [-0.0071, 0.0147], floor 0.0200, FAIL. P4
spec0 0.5723, CI [0.5449, 0.5998]; decE_VIO 1.0000, PASS.
W1 -0.0053, W2 -0.0022, W3 -0.0045, W4 -0.0046, all FAIL.
W5 fbdr 0.2537, p1_cos 0.8448, PASS. W6 diff 0.0521,
ratio 1.032, PASS. S1 0.0000, S2 0.0032, CI [-0.0147,
0.0212], PASS. S3 (wire-on, measurement only) p1n diff
-0.0313. W1s diagnostic: interaction under cut 0.0001,
REMOVED by cut (there is nothing to remove: the
interaction itself is ~0). The magnitude interaction does
not survive 2-step memory; only the specificity
interaction does.

### 63.4 Verdict: C STRONG PASS (protocol section 8, rule 1)

Sine P1-P4 pass, planted1 P1 passes, sine W1 and W2 pass,
and the W5 gate passes on sine, so no demotion applies.
24 of 34 bars pass. The verdict rule is met exactly as
frozen.

### 63.5 What the verdict does and does not mean

1. The interaction is real, stimulus-specific, and learned:
   it is present on sine (0.1643) and planted lag-1
   (0.0425), the specificity interaction tracks it (0.2423
   / 0.2355), and the frozen control shows ~0 gap (P3
   0.2485 sine / 0.0918 planted1).
2. The wire carries a real increment over cut (W1 sine
   0.0805, planted1 0.0246), but the strict Kimi bar
   FAILS: the interaction persists under cut at 51.0% of
   wire-on on sine (0.0838 vs 0.1643) and 42.2% on
   planted1 (0.0179 vs 0.0425), inside / just above the
   30-45% the design probe predicted. Protocol section 8,
   "what would change the verdict" item 1 (W1s passes)
   did NOT happen, so the top-down precision reading is
   NOT strengthened. The wire amplifies the interaction
   roughly 2x; it does not create it.
3. The wire-on net does not exceed the wire-off net on
   sine: interaction wire-off 0.2248 vs wire-on 0.1643
   (W4 -0.0605, CI excludes 0). Ordering on sine is off >
   on > cut (0.0838) > shuffle (0.0688). W1 is true only
   against the cut control, not against no wire. On
   planted1 the ordering is on (0.0425) > off (0.0346) >
   cut (0.0179), and W4 is real but small (0.0079, below
   the 0.0200 floor).
4. The wire costs ordinary accuracy on sine: normal_acc
   ratio on/off 0.720 (W6 FAIL). On planted tasks the
   wire costs nothing (ratios 1.072 / 1.032, W6 PASS).
   The sine W6 fail is a real side effect of the aligned
   wire on the continuous task, not a measurement
   artifact: the same wire that carries the interaction
   degrades normal performance by 28%.
5. W2 passes on sine (0.0955) but the known confound
   stands (section 62.5 finding 1): the shuffle is not
   magnitude-matched, so W2 confounds direction with
   magnitude. The cut (W1) is the primary wire control
   and it passes.
6. S1 and S2 pass on both planted tasks wire-off (decP1
   diff 0.0000; p1n diff 0.0027 / 0.0032, CIs cover 0):
   p1 is NOT suppressed by expectation. The mapping
   report's falsifiable divergence is NOT falsified in
   the model's favour. S3 (wire-on, measurement only)
   shows a p1n diff of -0.0448 (planted1) and -0.0313
   (planted2): with the wire on, ||p1|| is slightly
   SMALLER on expected trials, the direction the mapping
   report says the model should not show, but this is a
   wire-on measurement, not a wire-off result, and it is
   not a bar.
7. Lag-2: the magnitude interaction is ~0 (-0.0052)
   while specificity stays large (0.3105). The
   interaction does not survive 2-step memory. Per
   protocol section 8 rule 5 this does not change the
   verdict, but it bounds the claim to 1-step expectation.

Thesis-level (protocol section 8 rule 6): no outcome of
round 22 establishes predictive coding. Round 20 failed
B' and D. A C STRONG PASS establishes only that the model
produces precision-like behaviour from input noise plus
error selection, with the live wire amplifying (not
creating) the effect. The discriminators now stand:
omission (A) passes, top-down clamp (B) fails, aligned
clamp (B') fails, generative rollout (D) fails,
precision (C) passes with the wire as amplifier. Next
queued: the multi-step (h > 1) continuous rollout, per
the Nov 27 00:00 redirection; no new design starts until
Reza lands and confirms scope. The r22-monitor cron is
removed; the analysis is delivered.

### 63.6 Main-agent second check (Nov 27, 2026, 12:18 PT)

Independent recomputation by the main agent from the
raw merged test JSONL (own script, protocol definitions
only, no analyzer import): every checked value matches
the analysis to 4 decimals -- sine P1 0.1643 CI
[0.1416, 0.1871], P2 0.2423, P3 0.2485, W1 0.0805, W2
0.0955, W3 0.0874, W4 -0.0605, W6 -0.0687 ratio 0.720,
cut 0.0838 (51.0%); planted1 P1 0.0425, P2 0.2355, P3
0.0918, W1 0.0246, W2 0.0316, W3 0.0198, W4 0.0079, W6
0.0603 ratio 1.072, cut 0.0179 (42.2%); planted2 P1
-0.0052, P2 0.3105, P3 0.0038, W1 -0.0053, W2 -0.0022,
W3 -0.0045, W4 -0.0046, W6 0.0521 ratio 1.032, cut
0.0001. Shard-vs-merged check: 0 mismatched rows on all
three tasks (a+b shards equal the merged files
exactly). 480 unique (task, config, seed) records: 60
(task, seed) pairs x 8 configs, from 240 nets (4 per
pair). Correction to the worker handoff and README
wording: "480 unique test nets" is wrong; it is 480
unique records from 240 nets. Section 63 above states
it correctly. Bar tally recount: sine 8/10, planted1
10/12, planted2 6/12, total 24/34. Verdict rule
confirmed: sine P1-P4 pass, planted1 P1 passes, sine
W1+W2 pass, W5 gate passes, so C STRONG PASS stands.
The r22-monitor cron is confirmed removed (cron.list,
Nov 27 12:18 PT). README Status updated. Next queued:
the multi-step (h > 1) continuous rollout; no new design
starts until Reza lands and confirms scope.
