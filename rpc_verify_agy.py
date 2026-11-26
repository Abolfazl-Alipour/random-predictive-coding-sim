"""Verification of AGY's critique claims (Muse, Nov 26, 2026).

Three 1-seed checks, each permitted by the critique brief (small checks only):
1. Frozen stage probe at n_out=64 vs 256 on planted motifs, seed 0.
   AGY claims a fixed random W_prox at n_out=256 loses ZERO information
   (p2 0.983, y2 full 0.867), so the 200->64 bottleneck, not fixed
   random wires, caused the 0.984->0.711 drop.
2. A18 sine-fixed seed 2 (collapsed), 1 epoch: d1 positivity, pairwise
   cosine, in-span energy fraction, unique bins predicted, net accuracy.
   AGY claims rank-1 positive-orthant DC collapse, NOT orthogonal hiding:
   collapsed seed 2 should have MORE in-span energy than working seed 0.
3. A23 frozen control, sine-fixed seed 0, 1 epoch, top-6 and A18:
   AGY claims the frozen no-plasticity control scores ~0.44 (top-6)
   and ~0.36 (A18) under drive prototypes, i.e. A23 hands the untrained
   network a supervised nearest-centroid classifier.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork
from rpc_simple import make_sine_fixed


def fit_ridge(D, Y, lam=1.0):
    return np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ Y).T


def collect_stages(net, data):
    feats = {"r2": [], "p2": [], "y2": []}
    toks = []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    for s in feats:
                        feats[s].append(out[s].numpy().copy())
                    toks.append(tok)
    return {s: np.stack(v) for s, v in feats.items()}, np.array(toks)


def check1():
    print("=== CHECK 1: frozen stage probe, planted, seed 0, n_out 64 vs 256 ===",
          flush=True)
    for n_out in (64, 256):
        for full in (False, True):
            torch.manual_seed(0)
            np.random.seed(0)
            train = make_data(n_seq=200)
            test = make_data(n_seq=50)
            net = RPCNetwork(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05,
                             eta_c=0.05, n_res=200, n_out=n_out, alpha=1.0,
                             learn_rule="coactivity", full_code=full)
            net.calibrate(train)
            net.build_prototypes(train, proto_n=200)
            Ftr, ytr = collect_stages(net, train)
            Fte, yte = collect_stages(net, test)
            Ytr = np.eye(16)[ytr]
            line = f"n_out={n_out} full_code={full}: "
            for s in ("r2", "p2", "y2"):
                W = fit_ridge(Ftr[s], Ytr)
                acc = float(((Fte[s] @ W.T).argmax(axis=1) == yte).mean())
                line += f"{s}={acc:.3f}  "
            print(line, flush=True)


def train_sine(seed, epochs, full_code, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_sine_fixed(50, seed=seed)
    test = make_sine_fixed(20, seed=seed + 1000)
    base = dict(K=32, smooth_sigma=3.0, alpha=0.7, graded_conf=True,
                eta_b=0.15, learn_rule="coactivity", full_code=full_code)
    net = RPCNetwork(**{**base, **kw})
    net.calibrate(train)
    net.build_prototypes(train, proto_n=50)
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True)
    return net, train, test


def d1_stats(net, test):
    D, preds, toks = [], [], []
    with torch.no_grad():
        for seq in test:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    D.append(out["d1"].numpy().copy())
                    preds.append(out["pred"])
                    toks.append(tok)
    D = np.stack(D)
    preds = np.array(preds)
    toks = np.array(toks)
    acc = float((preds == toks).mean())
    # in-span energy: P = pinv(M) @ M over the p1 prototypes M
    M = net.prototypes.numpy().astype(np.float64)
    P = np.linalg.pinv(M) @ M
    Pd = D @ P.T
    in_span = float(np.mean(np.sum(Pd ** 2, axis=1) / np.sum(D ** 2, axis=1)))
    pos_frac = float(np.mean(D > 0))
    # pairwise cosine on a subsample
    idx = np.random.RandomState(0).choice(len(D), size=min(300, len(D)),
                                          replace=False)
    Dn = D[idx] / np.linalg.norm(D[idx], axis=1, keepdims=True)
    cos = Dn @ Dn.T
    iu = np.triu_indices(len(idx), k=1)
    pair_cos = float(np.mean(cos[iu]))
    uniq, counts = np.unique(preds, return_counts=True)
    top1 = float(counts.max() / len(preds))
    wpos = float(np.mean(net.W_dist1.numpy() > 0))
    sv = np.linalg.svd(net.W_dist1.numpy(), compute_uv=False)
    sv1_e = float(sv[0] ** 2 / np.sum(sv ** 2))
    return dict(acc=acc, d1_pos=pos_frac, pair_cos=pair_cos,
                in_span=in_span, uniq_bins=len(uniq), top1=top1,
                w_pos=wpos, sv1_e=sv1_e)


def check2():
    print("=== CHECK 2: A18 sine seed 2 (collapsed) vs seed 0, 1 epoch ===",
          flush=True)
    for seed in (0, 2):
        net, train, test = train_sine(seed, 1, True)
        st = d1_stats(net, test)
        print(f"seed {seed} A18 1ep: acc={st['acc']:.3f} d1>0={st['d1_pos']:.3f} "
              f"pair_cos={st['pair_cos']:.3f} in_span={st['in_span']:.3f} "
              f"uniq_bins={st['uniq_bins']}/32 top1={st['top1']:.3f} "
              f"W>0={st['w_pos']:.3f} sv1_E={st['sv1_e']:.3f}", flush=True)
    net, train, test = train_sine(2, 1, False)
    st = d1_stats(net, test)
    print(f"seed 2 top6 1ep: acc={st['acc']:.3f} d1>0={st['d1_pos']:.3f} "
          f"pair_cos={st['pair_cos']:.3f} in_span={st['in_span']:.3f} "
          f"uniq_bins={st['uniq_bins']}/32 top1={st['top1']:.3f} "
          f"W>0={st['w_pos']:.3f} sv1_E={st['sv1_e']:.3f}", flush=True)


def check3():
    print("=== CHECK 3: A23 frozen control, sine seed 0, 1 epoch ===",
          flush=True)
    from rpc_v2 import run
    for full in (False, True):
        for plastic, label in ((True, "full"), (False, "no-plasticity")):
            torch.manual_seed(0)
            np.random.seed(0)
            train = make_sine_fixed(50, seed=0)
            test = make_sine_fixed(20, seed=1000)
            base = dict(K=32, smooth_sigma=3.0, alpha=0.7, graded_conf=True,
                        eta_b=0.15, learn_rule="coactivity", full_code=full,
                        drive_proto=True, proto_n=50)
            kw = dict(base)
            if not plastic:
                kw["plasticity"] = False
            r = run(train, test, epochs=1, **kw)
            print(f"A23 full_code={full} {label}: acc={r['acc']:.3f} "
                  f"mean_burst={r['mean_burst']:.3f}", flush=True)
        # p1-prototype baseline on the same config (no drive_proto)
        torch.manual_seed(0)
        np.random.seed(0)
        train = make_sine_fixed(50, seed=0)
        test = make_sine_fixed(20, seed=1000)
        base = dict(K=32, smooth_sigma=3.0, alpha=0.7, graded_conf=True,
                    eta_b=0.15, learn_rule="coactivity", full_code=full,
                    proto_n=50)
        r = run(train, test, epochs=1, plasticity=False, **base)
        print(f"p1-proto frozen full_code={full}: acc={r['acc']:.3f}",
              flush=True)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        check1()
    if which in ("all", "2"):
        check2()
    if which in ("all", "3"):
        check3()
    print("EXIT=0", flush=True)
