"""Learned proximal projection (A12), run by Muse directly.

Section 21 stage probe: r2 decodes the current token at 0.984 but
p2 only at 0.711, and y2 at 0.332. The fixed random W_prox mixes the
token signal with history noise. This test fits W_prox by ridge
regression from r2 onto a fixed random per-token code (train only),
then runs the planted task with the learned W_prox in place. This is
a departure from the thesis, where W_prox is fixed random and only
shaped by the confirmation rule.

Task: planted motifs, alpha 1.0, learn_rule="coactivity", prototype
readout, 5 seeds 0-4, 5 epochs, proto_n=200. Pass bars: acc >= 0.2625
and gain >= 0.10. Reference: coactivity-full 0.072, control 0.064,
rho0-full 0.090, wide256-full 0.067. Baselines: Markov-1 0.278.

Conditions:
  prox-full: learned W_prox, confirmation OFF (freezes the fit; the
    confirmation rule rewrites W_prox every step and would destroy
    it), coactivity distal learning.
  prox-confirm-on: learned W_prox, confirmation ON: does the thesis's
    own confirmation rule preserve or destroy the learned code?
  prox-no-plasticity: learned W_prox, no W_dist1 learning: how much
    does the clean code alone contribute with a random distal map?

Diagnostic first: p2 -> token decode (ridge probe, train only),
random W_prox vs learned W_prox, 5 seeds. Random reference 0.711.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork
from rpc_planted_alpha import run_planted


def p2_decode(seed, learn):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    net = RPCNetwork(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64, alpha=1.0, learn_rule="coactivity")
    resid = None
    if learn:
        resid = net.fit_prox(train)["fit_resid_mse"]

    def collect(data):
        F, Y = [], []
        with torch.no_grad():
            for seq in data:
                net.reset()
                for i, tok in enumerate(seq):
                    out = net.step(tok, train=False)
                    if i > 0:
                        F.append(out["p2"].numpy().copy())
                        Y.append(tok)
        return np.stack(F), np.array(Y)

    Ftr, ytr = collect(train)
    Fte, yte = collect(test)
    W = np.linalg.solve(Ftr.T @ Ftr + 1.0 * np.eye(Ftr.shape[1]),
                        Ftr.T @ np.eye(16)[ytr]).T
    acc = float(((Fte @ W.T).argmax(axis=1) == yte).mean())
    return acc, resid


if __name__ == "__main__":
    for learn in [False, True]:
        accs, resids = [], []
        for s in range(5):
            a, r = p2_decode(s, learn)
            accs.append(a)
            if r is not None:
                resids.append(r)
        extra = (f" fit_resid_mse={np.mean(resids):.4f}"
                 if resids else "")
        print(f"p2->token decode learn_prox={learn}: mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"accs={[round(a, 3) for a in accs]}{extra}", flush=True)
    for name, extra in [
            ("prox-full", {"learn_prox": True, "use_confirmation": False,
                           "learn_rule": "coactivity"}),
            ("prox-confirm-on", {"learn_prox": True,
                                 "learn_rule": "coactivity"}),
            ("prox-no-plasticity", {"learn_prox": True,
                                    "use_confirmation": False,
                                    "plasticity": False,
                                    "learn_rule": "coactivity"})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, **extra)
            accs.append(r["acc"])
            bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
