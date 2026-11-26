"""Two jobs, run by Muse directly, no subagents.

1. Finish the pending TrackF no-confirmation condition (5 seeds, 3 epochs).
   Full and no-plasticity lines are already in rpc_track_F_results.txt.
2. Alpha test: does a faster reservoir let RPC learn the deterministic cycle?

Diagnosis (simulation-results.md section 13): cycle prototypes are
near-parallel (off-diag cosine mean 0.890) because the reservoir state
is set by shared history, not the current token. The distal drive DOES
learn (correct-prototype cosine 0.343 vs 0.089 control) but the wrong
prototype scores higher (0.418).

Fix tested here: raise the leak rate alpha (default 0.3) so the reservoir
tracks the current input and forgets history faster. Pure parameter
change, no new model machinery. alpha=1.0 removes the (1-alpha) decay
entirely. If prototypes separate and accuracy jumps, the readout was
starved of signal, not the learning rule. If accuracy stays at chance
with separated prototypes, the distal learning itself cannot point at
the right prototype, which tightens the falsification bound.

Task: cycle (K=8, chance 0.125), 5 seeds, full vs no-plasticity,
alpha in {0.7, 1.0}. Bars: acc >= 0.325 (chance + 0.20) and gain >= 0.10.
Also measures prototype off-diagonal cosine per alpha, pre-training.
"""
import numpy as np
import torch
from rpc_simple import run_task, make_cycle
from rpc_track_F import run_one as run_trackF
from rpc_v2 import RPCNetwork


def proto_overlap(alpha, seed=0, K=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_cycle(n_seq=100, length=96, K=K, seed=seed)
    net = RPCNetwork(K=K, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64, alpha=alpha)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=100)
    p = torch.nn.functional.normalize(net.prototypes, dim=1)
    cos = p @ p.T
    off = cos[~torch.eye(K, dtype=torch.bool)]
    print(f"alpha={alpha}: prototype off-diag cosine mean={float(off.mean()):.3f} "
          f"max={float(off.max()):.3f} min={float(off.min()):.3f}", flush=True)


if __name__ == "__main__":
    # 1. TrackF no-confirmation, 5 seeds
    accs, bursts = [], []
    for seed in range(5):
        r = run_trackF(seed, epochs=3, use_confirmation=False)
        accs.append(r["acc"]); bursts.append(r["mean_burst"])
    print(f"trackF no-confirmation: acc mean={np.mean(accs):.3f} "
          f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.031 "
          f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
          flush=True)

    # 2. Prototype separation vs alpha (pre-training diagnostic)
    for alpha in [0.3, 0.7, 1.0]:
        proto_overlap(alpha)

    # 3. Cycle at higher alpha, 5 seeds, full vs no-plasticity
    for alpha in [0.7, 1.0]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_task("cycle", seed, alpha=alpha, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"cycle alpha={alpha} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.125 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a, 3) for a in accs]}",
                  flush=True)
    print("DONE", flush=True)
