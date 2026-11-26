"""Planted-motif fix round 13: clean rule tests, run by Muse
directly.

CORRECTION behind this round (section 38): the planted
stacking claim was confounded. A12 alone (0.113, section 23)
ran under learn_rule="coactivity", but A12+knn (0.074) ran
under the RPCNetwork default, learn_rule="burst" with
graded_conf off, because the planted fix drivers pass no
learn_rule. All planted A18/A19/A20/A21 numbers ran under
burst + one-hot + eta_b 0.05. This round runs the clean tests:
(a) A18 under coactivity, matching the sine A18 conditions'
rule (isolating the rule; the planted config otherwise keeps
its defaults: one-hot, eta_b 0.05, graded off); (b) A12+knn
under coactivity, the clean stacking test against A12's
0.113; (c) A18+proj under the default burst rule, matching
the other planted A18 tests, to test the misalignment on
planted motifs.

Planted motifs, alpha 1.0, 5 epochs, 5 seeds, full +
no-plasticity control. References (burst rule): cosine
0.072-0.073, A18 0.065, A18+read2 0.076, kNN 0.072.
Coactivity rule: A8 dense 0.072, A12 learned W_prox 0.113.
Bars: accuracy >= 0.2625, gain >= 0.10.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [
        ("A18+coact", {"full_code": True, "learn_rule": "coactivity"}),
        ("A12+knn+coact", {"learn_prox": True, "knn_readout": True,
                           "learn_rule": "coactivity"}),
        ("A18+proj", {"full_code": True, "proto_proj": True}),
    ]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_planted(seed, 1.0, **extra, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"planted {name} {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
                  f"mean_burst={np.mean(bursts):.3f} "
                  f"accs={[round(a, 3) for a in accs]}", flush=True)
