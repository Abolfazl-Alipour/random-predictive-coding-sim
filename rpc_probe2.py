"""Stage-by-stage probe of the mod1-to-mod2 pathway, run by Muse directly.

The section 20 probe found the code is the bottleneck: r1 decodes
the current token at 1.000, but y2_prev decodes the previous token
at only 0.361 (frozen) / 0.330 (trained). Where does the token code
degrade? Fit ridge regression (lambda=1.0, train only) from each
stage's activity to the CURRENT token, scored on held-out test.
Frozen network, planted motifs, alpha 1.0, 5 seeds 0-4.
Stages: u2 (mod2 input, tanh(W_12 @ r1)), e2 (mod2 expansion,
top-32 of 256), r2 (mod2 reservoir), p2 (mod2 proximal drive),
y2 (mod2 output code, top-6 of 64). By stationarity, y2 -> current
token here equals y2_prev -> previous token (0.361 frozen).
If the drop is at r2, the recurrence mixes history into the code.
If the drop is at y2, the top-k sparsification throws the
information away. Reference: r1 -> current 1.000, y2_prev -> prev
0.361 frozen.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork

STAGES = ["u2", "e2", "r2", "p2", "y2"]

def collect(net, data):
    feats = {s: [] for s in STAGES}
    toks = []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    for s in STAGES:
                        feats[s].append(out[s].numpy().copy())
                    toks.append(tok)
    return {s: np.stack(v) for s, v in feats.items()}, np.array(toks)

def fit_ridge(D, Y, lam=1.0):
    return np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ Y).T

if __name__ == "__main__":
    accs = {s: [] for s in STAGES}
    for seed in range(5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = make_data(n_seq=200)
        test = make_data(n_seq=50)
        net = RPCNetwork(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05,
                         eta_c=0.05, n_res=200, n_out=64, alpha=1.0,
                         learn_rule="coactivity")
        net.calibrate(train)
        net.build_prototypes(train, proto_n=200)
        Ftr, ytr = collect(net, train)
        Fte, yte = collect(net, test)
        Ytr = np.eye(16)[ytr]
        for s in STAGES:
            W = fit_ridge(Ftr[s], Ytr)
            accs[s].append(float(((Fte[s] @ W.T).argmax(axis=1) == yte).mean()))
    print("probe2 frozen stage->current-token, 5 seeds, chance=0.062:", flush=True)
    for s in STAGES:
        print(f"  {s}: mean={np.mean(accs[s]):.3f} "
              f"range=[{min(accs[s]):.3f},{max(accs[s]):.3f}] "
              f"accs={[round(a,3) for a in accs[s]]}", flush=True)
