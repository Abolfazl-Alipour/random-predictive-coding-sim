"""Sine fix round 3, run by Muse directly.

Round 2 (rpc_sine_fix2_results.txt, 3 seeds): smooth sigma 2.0
scores 0.212 (gain +0.193), 0.019 short of the 0.231 accuracy bar;
sigma 3.0 scores 0.199 (gain +0.172); sigma 1.0 scores 0.132 (gain
+0.103). Smooth + delay is 0.150, no better than smooth alone, so
the delay line is dead for sine. CORRECTION: smooth 1.5 at 5 seeds
is 0.129 (gain +0.098), NOT the 3-seed 0.145 / +0.114, so it passes
neither bar at 5 seeds; 3-seed numbers overestimated it.
Round 3: 5-seed confirmations of sigma 2.0 / 2.5 / 3.0 (the peak
sits near 2.0-2.5), smooth 2.0 with 10 epochs (does more training
help now that the input is smooth?), and a mechanism probe: next-bin
and direction decode from r2_prev, y2_prev and d1 on a FROZEN
smooth-2.0 net, to test whether the smooth input raises the code's
direction information (one-hot input gave y2 direction 0.748 vs
r2 0.984).
All sine-fixed, alpha 1.0, coactivity, 5 epochs unless noted.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork
from rpc_sine_diag import collect, fit_ridge

if __name__ == "__main__":
    for sigma in [2.0, 2.5, 3.0]:
        for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_task("sine-fixed", seed, alpha=1.0, learn_rule="coactivity",
                             smooth_sigma=sigma, **ex2)
                accs.append(r["acc"])
                bursts.append(r["mean_burst"])
            print(f"sine-fixed smooth{sigma} 5seed {cond}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                  flush=True)
    for cond, ex2 in [("full", {}), ("no-plasticity", {"plasticity": False})]:
        accs = []
        for seed in range(3):
            r = run_task("sine-fixed", seed, alpha=1.0, learn_rule="coactivity",
                         smooth_sigma=2.0, epochs=10, **ex2)
            accs.append(r["acc"])
        print(f"sine-fixed smooth2.0 10ep {cond}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    # mechanism probe on a frozen smooth-2.0 net, 3 seeds
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = make_sine_fixed(n_seq=50, length=200, seed=seed)
        test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
        net = RPCNetwork(K=32, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                         n_res=200, n_out=64, alpha=1.0, learn_rule="coactivity",
                         smooth_sigma=2.0)
        net.calibrate(train)
        net.build_prototypes(train, proto_n=50)
        Ftr, ytr, dtr = collect(net, train)
        Fte, yte, dte = collect(net, test)
        Yn_tr = np.eye(32)[ytr]
        Yd_tr = np.eye(3)[dtr]
        line = f"smooth2.0 frozen probe seed {seed}: "
        for s in ["r2_prev", "y2_prev", "d1"]:
            W = fit_ridge(Ftr[s], Yn_tr)
            nxt = float(((Fte[s] @ W.T).argmax(axis=1) == yte).mean())
            Wd = fit_ridge(Ftr[s], Yd_tr)
            dr = float(((Fte[s] @ Wd.T).argmax(axis=1) == dte).mean())
            line += f"{s} next={nxt:.3f} dir={dr:.3f} | "
        print(line, flush=True)
