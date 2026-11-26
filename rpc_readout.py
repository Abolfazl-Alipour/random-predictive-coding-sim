"""Learned readout on the distal drive, run by Muse directly.

Question (Nov 25): the fixed prototype-cosine readout cannot find the
cue inside a history-contaminated distal code. Can a TRAINED readout
find it? Fit ridge regression from the distal drive d1 (64-dim,
standardised by the calibration sd_d1) to the next token (one-hot,
K=16), on TRAIN data only, then score it on held-out test data.

No leakage: d1 at step t is computed from y2_prev only, before the
current token is fed in (see rpc_v2.py _step_core), so d1 carries no
information about the current token except through prediction.

Design (all planted motifs, alpha 1.0, 5 seeds 0-4, 5 epochs,
proto_n=200, ridge lambda=1.0, fixed, not tuned):
1. coactivity-full + learned readout: does the trained code contain
   the cue?
2. burst-full + learned readout: does the learning rule matter for
   the readout?
3. no-plasticity + learned readout: THE control. The network is
   frozen, so this is the fixed random code with a trained readout.
   If this scores as high as 1 and 2, the readout does all the work
   and RPC learning adds nothing.
Each condition also reports the prototype-readout accuracy on the
SAME network and test data, so the two readouts are compared
apples-to-apples. Reference prototype numbers (same task, seeds):
burst full 0.073, coactivity full 0.072, control 0.064.
Pass bars for the learned readout: acc >= 0.2625 (chance 0.0625 +
0.20) and gain >= 0.10 over the no-plasticity LEARNED-readout
control.

Departure, stated plainly: a trained readout is NOT in the thesis.
The thesis prediction is the cosine match against prototypes. This
test asks only whether the distal code CONTAINS the information,
not whether RPC as specified can use it. A pass here would mean the
information is present but the readout is the bottleneck; a fail
would mean the code itself does not carry the cue.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork

def collect_d1(net, data):
    """Run data with plasticity off; return standardised d1 (N, 64)
    and current token (N,) for steps i > 0 (step 0 has y2_prev = 0)."""
    feats, toks = [], []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    feats.append((out["d1"] / net.sd_d1).numpy().copy())
                    toks.append(tok)
    return np.stack(feats), np.array(toks)

def fit_ridge(D, Y, lam=1.0):
    """Ridge: W (K, 64) minimising ||Y - D W^T||^2 + lam ||W||^2."""
    n, d = D.shape
    A = D.T @ D + lam * np.eye(d)
    W = np.linalg.solve(A, D.T @ Y).T
    return W

def run_readout(seed, alpha=1.0, epochs=5, lam=1.0, **kw):
    step_kw = {k: kw.pop(k) for k in
               ("use_confirmation", "use_burst", "plasticity") if k in kw}
    proto_n = kw.pop("proto_n", 200)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, alpha=alpha)
    net = RPCNetwork(**{**defaults, **kw})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=proto_n)
    for ep in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **step_kw)
    # Fit the learned readout on TRAIN d1 (trained network).
    Dtr, ytr = collect_d1(net, train)
    Ytr = np.eye(net.K)[ytr]
    W = fit_ridge(Dtr, Ytr, lam=lam)
    # Score BOTH readouts on the same test steps.
    Dte, yte = collect_d1(net, test)
    ridge_pred = (Dte @ W.T).argmax(axis=1)
    ridge_acc = float((ridge_pred == yte).mean())
    proto_correct, total = 0, 0
    for seq in test:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            if i > 0:
                total += 1
                proto_correct += int(out["pred"] == tok)
    proto_acc = proto_correct / max(total, 1)
    return {"ridge_acc": ridge_acc, "proto_acc": proto_acc,
            "chance": 1 / net.K, "n_train": len(ytr)}

if __name__ == "__main__":
    for name, extra in [("coactivity-full", {"learn_rule": "coactivity"}),
                        ("burst-full", {"learn_rule": "burst"}),
                        ("no-plasticity", {"plasticity": False})]:
        ridge, proto = [], []
        for seed in range(5):
            r = run_readout(seed, **extra)
            ridge.append(r["ridge_acc"]); proto.append(r["proto_acc"])
        print(f"planted alpha1.0 readout {name}: ridge mean={np.mean(ridge):.3f} "
              f"range=[{min(ridge):.3f},{max(ridge):.3f}] "
              f"proto mean={np.mean(proto):.3f} chance=0.062 "
              f"ridge={[round(a,3) for a in ridge]}", flush=True)
