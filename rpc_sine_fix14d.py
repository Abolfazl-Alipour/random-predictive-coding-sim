"""Round 14d: A24 delta rule on sine-fixed, part 2 (rule controls).

Same settings as rpc_sine_fix14c.py (full_code=True,
norm_rows=False, use_confirmation=False, s_inh=0.0,
smooth_sigma=3.0, alpha=0.7, graded_conf=False, n_out=256,
5 epochs, 5 seeds), only the learning rule changes. This is
AGY's 2x2 factorial: delta vs coactivity vs burst vs centred
covariance Hebbian at n_out 256. The coactivity and burst
controls run with norm_rows=False (identical settings to the
delta cell); without row-norm preservation the Hebbian rules
may saturate at the clamp, which is part of the result and is
noted in the record. The cov control (learn_rule="cov",
eta_b=0.15) removes the DC mode without any error signal, so
it separates "centering fixes it" from "the error term fixes
it". Readout is the unmodified p1-prototype cosine only.
"""
import numpy as np
from rpc_simple import run_task

BASE = dict(full_code=True, norm_rows=False, use_confirmation=False,
            s_inh=0.0, smooth_sigma=3.0, alpha=0.7, graded_conf=False,
            n_out=256)

if __name__ == "__main__":
    for name, extra in [
        ("coact-nout256", {"learn_rule": "coactivity", "eta_b": 0.15}),
        ("burst-nout256", {"learn_rule": "burst", "eta_b": 0.15}),
        ("cov-nout256", {"learn_rule": "cov", "eta_b": 0.15}),
    ]:
        accs, bursts, y2s = [], [], []
        for seed in range(5):
            r = run_task("sine-fixed", seed, epochs=5, **BASE, **extra)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
            y2s.append(r["mean_y2"])
        print(f"sine-fixed A24 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
