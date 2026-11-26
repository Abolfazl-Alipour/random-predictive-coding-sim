"""Trained-readout generalisation probe (round 9), run by Muse directly.

Question (Nov 26): the trained ridge readout rescues the FIXED sine
(0.556-0.673 on the distal drive, sections 25 and 27), but only the
fixed sine was measured. Does it rescue the VARIABLE sine and Lorenz
too, or only the fixed sine? That decides whether "accept the trained
readout" is a general result or a fixed-sine artifact.

Method: mirrors rpc_readout.py exactly. Smooth input sigma 3.0,
alpha 1.0, coactivity rule, 5 epochs, K=32, 3 seeds. For each task
(variable sine, Lorenz) and each condition (full, no-plasticity):
train the network, fit ridge regression (lambda 1.0) from the
standardised distal drive d1 to the next token (one-hot) on TRAIN
data only, score it on held-out test data, and score the prototype
readout on the same network and test data for the apples-to-apples
comparison. proto_n=20, matching run()/run_one, so the prototype
numbers compare directly with the published references (variable
sine 0.108, Lorenz 0.069 at 5 seeds; here 3 seeds).
No leakage: d1 at step t is computed from y2_prev only, before the
current token is fed in.
Bars (by analogy with the fixed sine): accuracy >= 0.231 (chance
0.031 + 0.20) and gain >= 0.10 over the no-plasticity LEARNED-readout
control.
Departure, stated plainly: a trained readout is NOT in the thesis.
This test asks only whether the distal code CONTAINS the information.
"""
import numpy as np
import torch
from rpc_v2 import RPCNetwork
from rpc_readout import collect_d1, fit_ridge
from rpc_spatial import make_sine, make_lorenz


def run_gen(task, seed, epochs=5, lam=1.0, **kw):
    step_kw = {k: kw.pop(k) for k in
               ("use_confirmation", "use_burst", "plasticity") if k in kw}
    torch.manual_seed(seed)
    np.random.seed(seed)
    maker = make_sine if task == "sine" else make_lorenz
    n_train = 100 if task == "sine" else 20
    n_test = 20 if task == "sine" else 10
    train = maker(n_seq=n_train, seed=seed)
    test = maker(n_seq=n_test, seed=seed + 1000)
    defaults = dict(K=32, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, alpha=1.0, smooth_sigma=3.0,
                    learn_rule="coactivity")
    net = RPCNetwork(**{**defaults, **kw})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=20)
    for ep in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **step_kw)
    Dtr, ytr = collect_d1(net, train)
    Ytr = np.eye(net.K)[ytr]
    W = fit_ridge(Dtr, Ytr, lam=lam)
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
    for task in ["sine", "lorenz"]:
        for name, extra in [("full", {}),
                            ("no-plasticity", {"plasticity": False})]:
            ridge, proto = [], []
            for seed in range(3):
                r = run_gen(task, seed, **extra)
                ridge.append(r["ridge_acc"])
                proto.append(r["proto_acc"])
            print(f"{task} readout {name}: ridge mean={np.mean(ridge):.3f} "
                  f"range=[{min(ridge):.3f},{max(ridge):.3f}] "
                  f"proto mean={np.mean(proto):.3f} chance=0.031 "
                  f"ridge={[round(a, 3) for a in ridge]} "
                  f"proto={[round(a, 3) for a in proto]}", flush=True)
