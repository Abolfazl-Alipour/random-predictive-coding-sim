"""Round 14c: A24 delta rule on sine-fixed, part 1 (delta + control).

The critics' joint recommendation (section 41.5): replace the
unsigned Hebbian distal update with a local signed
prediction-error rule, W_dist1 += eta_d / (||y2_prev||^2 + 1)
* outer(p1 - d1, y2_prev). AGY's 1-epoch checks: collapsed
seed 2 goes 0.025 -> 0.441 (n_out 64) and 0.507 (n_out 256);
planted 0.041 -> 0.141. This is the 5-seed 5-epoch verdict.

Config (explicit, AGY's spec): learn_rule="delta", eta_d=0.2,
full_code=True, norm_rows=False, use_confirmation=False,
s_inh=0.0, smooth_sigma=3.0, alpha=0.7, graded_conf=False,
5 epochs, 5 seeds. Conditions: delta at n_out 64 and 256, plus
the no-plasticity control on the delta-256 config (the gain
baseline). Readout is the unmodified p1-prototype cosine only.
Bars: sine accuracy >= 0.46 and gain >= +0.40 over the
no-plasticity control.

Also a ridge probe (next-token decode from d1, held-out test,
3 seeds, delta-256 trained vs frozen), using the SAME test data
as run_task (test seed+1000, proto_n=50) per the critics'
protocol fix. If the delta rule works, the trained drive
should decode well above the frozen drive.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_sine_fixed
from rpc_v2 import RPCNetwork

DELTA = dict(learn_rule="delta", eta_d=0.2, full_code=True,
             norm_rows=False, use_confirmation=False, s_inh=0.0,
             smooth_sigma=3.0, alpha=0.7, graded_conf=False)

def probe_delta(seed, plasticity, epochs=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(50, seed=seed)
    test = make_sine_fixed(20, seed=seed + 1000)
    net = RPCNetwork(K=32, n_out=256,
                     **{k: v for k, v in DELTA.items()
                        if k != "use_confirmation"})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=50)
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, plasticity=plasticity,
                         use_confirmation=False)
    def collect(data):
        F, toks = [], []
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    F.append(out["d1"].numpy().copy())
                    toks.append(tok)
        return np.stack(F), np.array(toks)
    Ftr, ytr = collect(train)
    Fte, yte = collect(test)
    Ytr = np.eye(32)[ytr]
    W = np.linalg.solve(Ftr.T @ Ftr + np.eye(Ftr.shape[1]),
                        Ftr.T @ Ytr).T
    return float(((Fte @ W.T).argmax(axis=1) == yte).mean())

if __name__ == "__main__":
    for name, extra in [
        ("delta-nout64", {"n_out": 64}),
        ("delta-nout256", {"n_out": 256}),
        ("delta256-no-plasticity", {"n_out": 256, "plasticity": False}),
    ]:
        accs, bursts, y2s = [], [], []
        for seed in range(5):
            r = run_task("sine-fixed", seed, epochs=5, **DELTA, **extra)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
            y2s.append(r["mean_y2"])
        print(f"sine-fixed A24 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
              f"mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    for cond, plast in [("trained", True), ("frozen", False)]:
        ds = [probe_delta(s, plast) for s in range(3)]
        print(f"probe delta256 d1 next-token {cond}: mean={np.mean(ds):.3f} "
              f"range=[{min(ds):.3f},{max(ds):.3f}] "
              f"accs={[round(d, 3) for d in ds]}", flush=True)
