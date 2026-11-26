"""Round 14d: A24 delta rule on planted motifs, part 2 (rule controls).

Same settings as rpc_planted_fix14c.py (full_code=True,
norm_rows=False, use_confirmation=False, s_inh=0.0, alpha=1.0,
one-hot, graded_conf=False, n_out=256, 5 epochs, 5 seeds),
only the learning rule changes. This is AGY's 2x2 factorial:
delta vs coactivity vs burst vs centred covariance Hebbian at
n_out 256. The coactivity and burst controls run with
norm_rows=False (identical settings to the delta cell); without
row-norm preservation the Hebbian rules may saturate at the
clamp, which is part of the result. The cov control
(learn_rule="cov", eta_b=0.05) removes the DC mode without any
error signal. Readout is the unmodified p1-prototype cosine
only. Motif-only accuracy is scored by run_planted_motif from
rpc_planted_fix14c.py (mirrors run() exactly).
"""
import numpy as np
from rpc_planted_fix14c import run_planted_motif

BASE = dict(full_code=True, norm_rows=False, use_confirmation=False,
            s_inh=0.0, alpha=1.0, graded_conf=False, n_out=256)

if __name__ == "__main__":
    for name, extra in [
        ("coact-nout256", {"learn_rule": "coactivity", "eta_b": 0.05}),
        ("burst-nout256", {"learn_rule": "burst", "eta_b": 0.05}),
        ("cov-nout256", {"learn_rule": "cov", "eta_b": 0.05}),
    ]:
        accs, maccs = [], []
        for seed in range(5):
            r = run_planted_motif(seed, 1.0, **BASE, **extra)
            accs.append(r["acc"])
            maccs.append(r["motif_acc"])
        print(f"planted A24 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
              f"motif_acc mean={np.mean(maccs):.3f} "
              f"range=[{min(maccs):.3f},{max(maccs):.3f}] "
              f"accs={[round(a, 3) for a in accs]} "
              f"maccs={[round(a, 3) for a in maccs]}", flush=True)
