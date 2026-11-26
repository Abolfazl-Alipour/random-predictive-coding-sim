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
3. It fails in noise and on continuous signals: planted motifs
   0.113 at best (learned proximal projection), fixed sine 0.217
   at best (dot-product readout), variable sine 0.108, Lorenz
   0.093, 1D track 0.098. Every fix tried so far moves the
   number by 0.01 to 0.03 except the smooth input, and none
   passes the bar.
4. The bottleneck is the code, not the learning rule. The
   module-1 reservoir holds the current token perfectly (linear
   probe 1.000), but the module-2 code the distal drive reads
   identifies the previous token only 36% of the time. The loss
   happens at the random proximal projection and at the top-k
   sparsification of the module-2 output. A trained ridge readout
   on the same distal drive scores 0.556 to 0.673 on the fixed
   sine, so the information is in the drive and the prototype
   readout cannot extract it.

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

## How to run

1. Python 3 with PyTorch (CPU build is enough). See
   `requirements.txt`.
2. `python rpc_v2.py` runs the base model. Each `rpc_*.py`
   driver runs one experiment and prints one line per condition.
3. `simulation-results.md` is the full experiment record, one
   numbered section per test, with the exact reproduce command.

## Files

1. `rpc_v2.py`: the model. All assumptions are labelled A1 to
   A17 in the header.
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
   `rpc_sine_fix6b.py`, `rpc_sine_fix6c.py`, `rpc_sine_fix7.py`,
   `rpc_sine_fix8.py`: one driver per experiment. Each has its
   own docstring with the question, the design, and the result.
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
bars. Dot-product readout (A17) is the best 5-seed score at
0.217, still below the bar. The remaining decision is whether
to stop or accept a trained readout (0.556 on the fixed sine).
