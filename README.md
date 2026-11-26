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
