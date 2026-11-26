"""Why sine fails: ceilings and stage probes, run by Muse directly.

Sine-fixed: frequency 0.05 cycles/step, random phase per sequence,
length 200, K=32 bins, 50 train / 20 test. The next bin is NOT
determined by the previous bin alone: the same bin occurs rising and
falling with different successors. This driver measures (A) the
one-step, two-step and three-step ceilings (Markov fit on train,
scored on test) and the ambiguous fraction, (B) what each stage of a
frozen network decodes about the next bin and about the direction
(rising/falling/flat), (C) whether prototypes for the same bin split
by direction, and (D) the trained network's error profile by true
direction and by distance.
"""
import numpy as np
import torch
from collections import defaultdict, Counter
from rpc_simple import make_sine_fixed, run_task
from rpc_v2 import RPCNetwork


def markov_acc(train, test, order):
    counts = defaultdict(Counter)
    for seq in train:
        for i in range(order, len(seq)):
            counts[tuple(seq[i - order:i])][seq[i]] += 1
    table = {k: c.most_common(1)[0][0] for k, c in counts.items()}
    correct = total = 0
    for seq in test:
        for i in range(order, len(seq)):
            key = tuple(seq[i - order:i])
            if key in table:
                total += 1
                correct += int(table[key] == seq[i])
    return correct / max(total, 1)


def fit_ridge(D, Y, lam=1.0):
    return np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ Y).T


def collect(net, data):
    F = {"r2_prev": [], "y2_prev": [], "d1": [], "r1": [], "p1": []}
    nxt, direction = [], []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                prev_r2 = net.r2_prev.numpy().copy()
                prev_y2 = net.y2_prev.numpy().copy()
                out = net.step(tok, train=False)
                if i > 0:
                    F["r2_prev"].append(prev_r2)
                    F["y2_prev"].append(prev_y2)
                    F["d1"].append(out["d1"].numpy().copy())
                    F["r1"].append(out["r1"].numpy().copy())
                    F["p1"].append(out["p1"].numpy().copy())
                    nxt.append(tok)
                    d = tok - seq[i - 1]
                    direction.append(1 if d > 0 else (0 if d < 0 else 2))
    return {k: np.stack(v) for k, v in F.items()}, np.array(nxt), np.array(direction)


def make_net(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(n_seq=50, length=200, seed=seed)
    test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
    net = RPCNetwork(K=32, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64, alpha=1.0, learn_rule="coactivity")
    net.calibrate(train)
    net.build_prototypes(train, proto_n=50)
    return net, train, test


if __name__ == "__main__":
    # (A) ceilings
    for order in [1, 2, 3]:
        accs = []
        for seed in range(5):
            train = make_sine_fixed(n_seq=50, length=200, seed=seed)
            test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
            accs.append(markov_acc(train, test, order))
        print(f"sine-fixed Markov-{order}: mean={np.mean(accs):.3f} "
              f"accs={[round(a, 3) for a in accs]}", flush=True)
    amb = []
    for seed in range(5):
        train = make_sine_fixed(n_seq=50, length=200, seed=seed)
        test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
        succ = defaultdict(set)
        for seq in train:
            for i in range(1, len(seq)):
                succ[seq[i - 1]].add(seq[i])
        n = sum(len(seq) - 1 for seq in test)
        tot = sum(1 for seq in test for i in range(1, len(seq))
                  if len(succ[seq[i - 1]]) > 1)
        amb.append(tot / n)
    print(f"ambiguous-prev-bin fraction: mean={np.mean(amb):.3f}", flush=True)
    # (B) probes, frozen net, 5 seeds
    stages = ["r2_prev", "y2_prev", "d1", "r1", "p1"]
    nxt_acc = {s: [] for s in stages}
    dir_acc = {s: [] for s in stages}
    for seed in range(5):
        net, train, test = make_net(seed)
        Ftr, ytr, dtr = collect(net, train)
        Fte, yte, dte = collect(net, test)
        Yn_tr = np.eye(32)[ytr]
        Yd_tr = np.eye(3)[dtr]
        for s in stages:
            W = fit_ridge(Ftr[s], Yn_tr)
            nxt_acc[s].append(float(((Fte[s] @ W.T).argmax(axis=1) == yte).mean()))
            Wd = fit_ridge(Ftr[s], Yd_tr)
            dir_acc[s].append(float(((Fte[s] @ Wd.T).argmax(axis=1) == dte).mean()))
    print("frozen probe: next-bin decode (chance 0.031) / direction decode (chance 0.333):", flush=True)
    for s in stages:
        print(f"  {s}: next={np.mean(nxt_acc[s]):.3f} "
              f"[{min(nxt_acc[s]):.3f},{max(nxt_acc[s]):.3f}] "
              f"dir={np.mean(dir_acc[s]):.3f} "
              f"[{min(dir_acc[s]):.3f},{max(dir_acc[s]):.3f}]", flush=True)
    # (C) prototype split by direction, frozen net, 5 seeds
    split_cos, offdiag = [], []
    for seed in range(5):
        net, train, test = make_net(seed)
        Fte, yte, dte = collect(net, test)
        p1, bins, dirs = Fte["p1"], yte, dte
        cosines = []
        for b in range(32):
            up = p1[(bins == b) & (dirs == 1)]
            dn = p1[(bins == b) & (dirs == 0)]
            if len(up) >= 30 and len(dn) >= 30:
                a = up.mean(axis=0)
                c = dn.mean(axis=0)
                cosines.append(float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c))))
        split_cos.append(float(np.mean(cosines)) if cosines else float("nan"))
        # off-diagonal prototype cosine (mean p1 per bin)
        protos = []
        for b in range(32):
            m = p1[bins == b]
            if len(m):
                protos.append(m.mean(axis=0))
        P = np.stack(protos)
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        C = Pn @ Pn.T
        offdiag.append(float((C.sum() - np.trace(C)) / (C.size - C.shape[0])))
    print(f"same-bin rising-vs-falling prototype cosine: mean={np.nanmean(split_cos):.3f} "
          f"(1.0 = identical, direction invisible to the prototype)", flush=True)
    print(f"mean off-diagonal prototype cosine: {np.mean(offdiag):.3f}", flush=True)
    # (D) trained net error profile, 3 seeds, 5 epochs
    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = make_sine_fixed(n_seq=50, length=200, seed=seed)
        test = make_sine_fixed(n_seq=20, length=200, seed=seed + 1000)
        net = RPCNetwork(K=32, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                         n_res=200, n_out=64, alpha=1.0, learn_rule="coactivity")
        net.calibrate(train)
        net.build_prototypes(train, proto_n=50)
        for _ in range(5):
            for seq in train:
                net.reset()
                for tok in seq:
                    net.step(tok, train=True)
        errs = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # direction -> [wrong, total]
        dist = Counter()
        with torch.no_grad():
            for seq in test:
                net.reset()
                for i, tok in enumerate(seq):
                    out = net.step(tok, train=False)
                    if i > 0:
                        d = tok - seq[i - 1]
                        g = 1 if d > 0 else (0 if d < 0 else 2)
                        errs[g][1] += 1
                        if out["pred"] != tok:
                            errs[g][0] += 1
                            dist[min(abs(out["pred"] - tok), 6)] += 1
        acc = {g: 1 - errs[g][0] / max(errs[g][1], 1) for g in errs}
        tot = sum(dist.values())
        print(f"seed {seed} trained: acc rising={acc[1]:.3f} falling={acc[0]:.3f} "
              f"flat={acc[2]:.3f} | err distance: "
              + ", ".join(f"{k}:{dist[k] / max(tot, 1):.2f}" for k in range(7)), flush=True)
