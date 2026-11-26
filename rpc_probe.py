"""Linear probe of the code itself, run by Muse directly.

Question (section 20): is the bottleneck the CODE or the distal
mapping/learning? Fit ridge regression (lambda=1.0, train only)
from the code to the token it should encode, and score on held-out
test. Two probes, two networks, 5 seeds 0-4:
1. y2_prev -> previous token: the code the distal drive actually
   reads. If this decodes the previous token well (near 1.0), the
   code is clean and the bottleneck is the distal mapping/learning.
   If it decodes poorly, the code itself is noisy and that is the
   bottleneck.
2. r1 -> current token: the module-1 reservoir state after the
   current input. Sanity check on how much token information the
   reservoir holds at alpha 1.0.
Networks: frozen (calibrated, never trained) and trained
(coactivity, 5 epochs). Task: planted motifs, alpha 1.0.
Reference end-to-end numbers: coactivity-full proto 0.072, frozen
code + trained readout 0.163, Markov-1 0.278.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork

def collect(net, data):
    """Return (Y2prev, prev_tok) and (R1, cur_tok) for steps i > 0."""
    y2p, ptok, r1s, ctok = [], [], [], []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    y2p.append(out["y2_prev"].numpy().copy())
                    ptok.append(seq[i - 1])
                    r1s.append(out["r1"].numpy().copy())
                    ctok.append(tok)
    return (np.stack(y2p), np.array(ptok)), (np.stack(r1s), np.array(ctok))

def fit_ridge(D, Y, lam=1.0):
    return np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ Y).T

def run_probe(seed, train_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    net = RPCNetwork(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64, alpha=1.0,
                     learn_rule="coactivity")
    net.calibrate(train)
    net.build_prototypes(train, proto_n=200)
    for _ in range(train_epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True)
    (Dtr, ytr), (Rtr, ctr) = collect(net, train)
    (Dte, yte), (Rte, cte) = collect(net, test)
    W1 = fit_ridge(Dtr, np.eye(16)[ytr])
    W2 = fit_ridge(Rtr, np.eye(16)[ctr])
    acc_y2 = float(((Dte @ W1.T).argmax(axis=1) == yte).mean())
    acc_r1 = float(((Rte @ W2.T).argmax(axis=1) == cte).mean())
    return acc_y2, acc_r1

if __name__ == "__main__":
    for name, epochs in [("frozen", 0), ("trained-coactivity", 5)]:
        a_y2, a_r1 = [], []
        for seed in range(5):
            y2a, r1a = run_probe(seed, epochs)
            a_y2.append(y2a); a_r1.append(r1a)
        print(f"probe {name}: y2prev->prev mean={np.mean(a_y2):.3f} "
              f"range=[{min(a_y2):.3f},{max(a_y2):.3f}] "
              f"r1->cur mean={np.mean(a_r1):.3f} "
              f"range=[{min(a_r1):.3f},{max(a_r1):.3f}] chance=0.062 "
              f"y2={[round(a,3) for a in a_y2]} r1={[round(a,3) for a in a_r1]}",
              flush=True)
