"""Sine fix round 13, resume after the runtime restart killed
the first run (Nov 26 07:23 PT).

Already done and kept (rpc_sine_fix13_results.txt):
A18+burst full 0.022 (below chance, gain -0.001), A18+proj
full 0.105 with per-seed accs EXACTLY identical to A18 alone.
The proj identity is explained by rpc_proj_check.py: the
projection is a mathematical no-op for the cosine readout,
because (P d1).proto_k = d1.proto_k for every prototype in
the span, and cosine normalises away the scale change. The
projection removes about 31% of d1's energy (cos 0.689), and
that orthogonal part is exactly the information the ridge
probe sees but cosine cannot. So A22 is dead as a fix, and
A18+burst+proj below is expected to equal A18+burst; it is
run to confirm the no-op empirically.

This resume runs only the missing parts: A18+burst+proj
(full + control) and the per-seed burst probe. Appends to
the same results file. Same config: graded + eta_b 0.15,
alpha 0.7, smooth 3.0, 5 epochs, 5 seeds.
"""
import numpy as np
from rpc_simple import run_task
from rpc_sine_fix13 import probe_next_rule

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15)
    name, extra = ("A18+burst+proj", {"full_code": True,
                                       "learn_rule": "burst",
                                       "graded_conf": False,
                                       "proto_proj": True})
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs, bursts, y2s = [], [], []
        for seed in range(5):
            kw = {**base, **extra, **ex2}
            r = run_task("sine-fixed", seed, epochs=5, **kw)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
            y2s.append(r["mean_y2"])
        print(f"sine-fixed {name} {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    dec = [probe_next_rule(seed, full_code=True, learn_rule="burst",
                           graded_conf=False) for seed in range(5)]
    print(f"probe next-token from d1, A18+burst: "
          f"mean={np.mean(dec):.3f} per-seed="
          f"{[round(d, 3) for d in dec]}", flush=True)
