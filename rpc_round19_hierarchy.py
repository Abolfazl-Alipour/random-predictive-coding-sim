"""Round 19 driver: the hierarchical version (experiment E).

Protocol: PROTOCOL-R19-HIERARCHY.md (frozen Nov 27, 2026).
Seeds 282..301 (fresh; rounds 15-18 used 201..281).
Tasks: planted1, sine, hier (two-timescale).
Five nets per (task, seed):
  cand (val-winner n_levels, winner fb gains, delta)
    -> cand_on, cand_cut, cand_shuffle (one net, three passes)
  hier2_ladder (M=2, ladder alphas, fb_gain1=winner, delta)
  hier2_nofb (M=2, ladder alphas, fb_gain1=0.0, delta)
  flat (M=2, both alphas=task alpha, fb_gain1=0.0, delta:
    the round-15 winner config in the hierarchical class)
  cand_frozen (winner n_levels, winner fb gains, no plasticity)
plus one baseline line per (task, seed). If the val winner
is 2 levels, hier2_ladder duplicates cand-on and is skipped,
and bar H4 is not judged (protocol section 9).

Usage:
  python rpc_round19_hierarchy.py <task> val --seeds A-B --out FILE
      --n-levels M --fb-gain1 G [--fb-gain2 G]
  python rpc_round19_hierarchy.py <task> test --seeds A-B --out FILE
      --winners FILE [--epochs N]
Val phase: one config per worker (the round-18 fix: a worker
runs exactly its config, so 6 workers write 30 lines, no
duplication). Test phase needs --winners (winner.json from
the monitor: architecture winner + per-task base configs).
"""
import json
import os
import sys

# Threading fix (Nov 26, 18:24 PT, round 15): these runs are
# tiny-matrix (64x200, 200x200) and single-threaded by nature.
# With the default 12 BLAS threads per process on a loaded Beast
# (load 45), the threads spin-wait on barriers and burn CPU
# without progress. One thread per process restores
# near-single-threaded speed. Must be set before numpy/torch are
# imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import ESN, accuracy, markov_predict, persistence_preds
from rpc_hier import HIER_BLOCK, HierRPCNetwork, make_hier_data
from rpc_round15 import (K_OF, MOTIF_M1, MOTIF_M2, N_TRAIN,
                         compute_baselines, splits as splits15)
from rpc_simple import make_sine_fixed

TASKS = ["planted1", "sine", "hier"]
VAL_SEEDS = list(range(282, 287))
TEST_SEEDS = list(range(282, 302))
EPOCHS = 5
# Val grid (protocol section 6): 6 effective configs.
# (n_levels, fb_gain1, fb_gain2); fb_gain2 is None for M=2.
GRID = [(2, 0.1, None), (2, 0.3, None),
        (3, 0.1, 0.1), (3, 0.1, 0.3),
        (3, 0.3, 0.1), (3, 0.3, 0.3)]
K_OF19 = {"planted1": 16, "sine": 32, "hier": 16}
N_TRAIN19 = {"planted1": 200, "sine": 50, "hier": 100}
# Round-15 winners (r15/winners.json, verified Nov 26 20:44 PT),
# plus the hier base config (the planted winner config).
ROUND15_WINNER = {
    "planted1": {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.05},
    "sine": {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.2},
    "hier": {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.05},
}
CONDITIONS = ["cand_on", "cand_cut", "cand_shuffle", "hier2_ladder",
              "hier2_nofb", "flat", "cand_frozen"]


def ladder_alphas(task, n_levels):
    """Per-level leak rates, fixed by depth (protocol 3.4):
    level 1 = task winner alpha, level 2 = 0.3, level 3 = 0.1."""
    a1 = ROUND15_WINNER[task]["alpha"]
    return (a1, 0.3)[:n_levels] if n_levels == 2 else (a1, 0.3, 0.1)


def splits(task, seed):
    """(train, val, test, train_lab, val_lab, test_lab,
    train_meta, val_meta, test_meta, K). Labels/metas are
    None except for hier (generator positions)."""
    if task in ("planted1", "sine"):
        tr, va, te, _, _, _, K, _ = splits15(task, seed)
        return (tr, va, te, None, None, None,
                None, None, None, K)
    if task == "hier":
        np.random.seed(seed)
        tr, tr_lab, tr_meta = make_hier_data(100)
        va, va_lab, va_meta = make_hier_data(50)
        te, te_lab, te_meta = make_hier_data(50)
        return (tr, va, te, tr_lab, va_lab, te_lab,
                tr_meta, va_meta, te_meta, 16)
    raise ValueError(task)


def load_winners(path):
    """Architecture winner + per-task base configs (from the
    monitor). The base configs are asserted against the
    frozen round-15 winners, so a typo fails loudly."""
    assert path, "test phase needs --winners"
    with open(path) as f:
        w = json.load(f)
    for task in TASKS:
        got = {"alpha": w["tasks"][task]["alpha"],
               "smooth_sigma": w["tasks"][task]["smooth_sigma"],
               "eta_d": w["tasks"][task]["eta_d"]}
        assert got == ROUND15_WINNER[task], (
            f"winners.json {task} = {got}, protocol says "
            f"{ROUND15_WINNER[task]}; refusing to run")
    n_levels = int(w["n_levels"])
    assert n_levels in (2, 3), n_levels
    fb1 = float(w["fb_gain1"])
    assert fb1 in (0.1, 0.3), fb1
    fb2 = w["fb_gain2"]
    if n_levels == 2:
        assert fb2 is None, fb2
    else:
        fb2 = float(fb2)
        assert fb2 in (0.1, 0.3), fb2
    return {"n_levels": n_levels, "fb_gain1": fb1, "fb_gain2": fb2,
            "tasks": {t: ROUND15_WINNER[t] for t in TASKS}}


def net_kw(task, cfg, n_levels, fb_gains):
    """HierRPCNetwork kwargs. proto_n is a build_prototypes
    argument, not a constructor argument (round-14 gotcha)."""
    return dict(K=K_OF19[task], n_levels=n_levels,
                alphas=ladder_alphas(task, n_levels),
                fb_gains=tuple(fb_gains), eta_d=cfg["eta_d"],
                smooth_sigma=cfg["smooth_sigma"],
                n_out=256, n_res=200, norm_rows=False)


def train_net(task, train, cfg, n_levels, fb_gains, seed, epochs,
              plasticity=True):
    """Build, calibrate, prototype, and train a net; return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, n_levels, fb_gains)
    net = HierRPCNetwork(**kw)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=N_TRAIN19[task])
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, plasticity=plasticity)
    return net


def eval_net(net, test, labels, metas, task, fb_mode="on"):
    """One test pass over the test sequences (frozen weights).
    Convention A: pred(t) vs token t, positions i >= 1."""
    correct = total = 0
    m1c = m1n = m2c = m2n = 0
    early_c = early_n = late_c = late_n = 0
    top_c = top_m1c = top_m1n = 0
    bursts, r_maxs = [], 0.0
    fb_sums = [0.0] * (net.M - 1)
    fb_n = 0
    for si, seq in enumerate(test):
        net.reset()
        lab = labels[si] if labels is not None else None
        meta = metas[si] if metas is not None else None
        # starter_idx -> block position, for early/late m1.
        starter_block = {}
        if meta is not None:
            for s_idx, _ctx in meta:
                starter_block[s_idx] = s_idx % HIER_BLOCK
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False, fb_mode=fb_mode)
            bursts.append(out["burst_rate"])
            r_maxs = max(r_maxs, out["r_max"])
            for w in range(net.M - 1):
                fb_sums[w] += out["fb_ratios"][w]
            fb_n += 1
            if i < 1:
                continue
            total += 1
            ok = int(out["pred"] == tok)
            correct += ok
            if net.M >= 3:
                top_c += int(out["top_pred"] == tok)
            if task in ("planted1",):
                if (seq[i - 1], tok) in MOTIF_M1:
                    m1n += 1
                    m1c += ok
                    if net.M >= 3:
                        top_m1n += 1
                        top_m1c += int(out["top_pred"] == tok)
                if i >= 2 and (seq[i - 2], seq[i - 1], tok) in MOTIF_M2:
                    m2n += 1
                    m2c += ok
            elif task == "hier" and lab is not None:
                if lab[i] == "m1":
                    m1n += 1
                    m1c += ok
                    if net.M >= 3:
                        top_m1n += 1
                        top_m1c += int(out["top_pred"] == tok)
                    # Late = starter at block position >= 6.
                    blk = starter_block.get(i - 1, -1)
                    if blk >= 6:
                        late_n += 1
                        late_c += ok
                    elif blk >= 0:
                        early_n += 1
                        early_c += ok
                elif lab[i] == "m2":
                    m2n += 1
                    m2c += ok
    othn = total - m1n - m2n
    othc = correct - m1c - m2c
    cf = net.clamp_fracs()
    rec = {"acc": correct / max(total, 1),
           "m1": m1c / max(m1n, 1), "m1_n": m1n,
           "m2": m2c / max(m2n, 1), "m2_n": m2n,
           "oth": othc / max(othn, 1), "oth_n": othn,
           "all_motif": (m1c + m2c) / max(m1n + m2n, 1),
           "early_m1": early_c / max(early_n, 1), "early_m1_n": early_n,
           "late_m1": late_c / max(late_n, 1), "late_m1_n": late_n,
           "mean_burst": float(np.mean(bursts)) if bursts else 0.0,
           "clamp_frac": max(cf) if cf else 0.0,
           "clamp_fracs": cf,
           "fb_ratios": [s / max(fb_n, 1) for s in fb_sums],
           "r_max": r_maxs,
           "n_levels": net.M}
    if net.M >= 3:
        rec["top_acc"] = top_c / max(total, 1)
        rec["top_m1"] = top_m1c / max(top_m1n, 1)
        rec["top_m1_n"] = top_m1n
    else:
        rec["top_acc"] = None
        rec["top_m1"] = None
        rec["top_m1_n"] = 0
    return rec


def compute_baselines_hier(train, test, labels, metas, seed):
    """Markov-1/2, persistence, ESN on the identical hier test
    sequences, plus baseline m1/m2. Baseline predictions are
    one-step-ahead (p[t] predicts test[t+1]), so the motif
    position j is scored with p[j-1] (convention A)."""
    K = 16
    m1 = markov_predict(train, test, K, 1)
    m2 = markov_predict(train, test, K, 2)
    per = persistence_preds(test)
    esn = ESN(K, seed).fit(train).predict(test)
    out = {"markov1_acc": accuracy(m1, test),
           "markov2_acc": accuracy(m2, test),
           "persist_acc": accuracy(per, test),
           "esn_acc": accuracy(esn, test)}

    def motif_scores(preds):
        m1c = m1n = m2c = m2n = late_c = late_n = 0
        for si, seq in enumerate(test):
            lab = labels[si]
            meta = metas[si]
            starter_block = {s: s % HIER_BLOCK for s, _ in meta}
            p = preds[si]
            for j in range(1, len(seq)):
                if j - 1 >= len(p):
                    continue
                if lab[j] == "m1":
                    m1n += 1
                    ok = int(p[j - 1] == seq[j])
                    m1c += ok
                    if starter_block.get(j - 1, -1) >= 6:
                        late_n += 1
                        late_c += ok
                elif lab[j] == "m2":
                    m2n += 1
                    m2c += int(p[j - 1] == seq[j])
        return {"m1": m1c / max(m1n, 1), "m1_n": m1n,
                "m2": m2c / max(m2n, 1), "m2_n": m2n,
                "late_m1": late_c / max(late_n, 1),
                "late_m1_n": late_n}

    for name, preds in (("markov1", m1), ("markov2", m2),
                        ("persist", per), ("esn", esn)):
        s = motif_scores(preds)
        out[f"{name}_m1"] = s["m1"]
        out[f"{name}_m1_n"] = s["m1_n"]
        out[f"{name}_m2"] = s["m2"]
        out[f"{name}_late_m1"] = s["late_m1"]
    return out


def run_val(task, seed, n_levels, fb_gain1, fb_gain2, epochs):
    """One architecture-config validation run: train the
    candidate on train, score overall acc + m1 on VAL."""
    (tr, va, te, tr_lab, va_lab, te_lab,
     tr_meta, va_meta, te_meta, K) = splits(task, seed)
    cfg = ROUND15_WINNER[task]
    gains = [fb_gain1] if n_levels == 2 else [fb_gain1, fb_gain2]
    net = train_net(task, tr, cfg, n_levels, gains, seed, epochs)
    m = eval_net(net, va, va_lab, va_meta, task, fb_mode="on")
    return {"val_acc": m["acc"], "val_m1": m["m1"],
            "val_late_m1": m["late_m1"], "val_m2": m["m2"],
            "mean_burst": m["mean_burst"],
            "clamp_frac": m["clamp_frac"], "r_max": m["r_max"]}


def run_test_nets(task, seed, cfg, arch, epochs):
    """Train the five nets and yield (condition, metric dict)
    pairs in order. Yields as each net finishes, so the caller
    can write lines incrementally and a killed worker resumes
    per condition (retraining only the missing nets)."""
    (tr, va, te, tr_lab, va_lab, te_lab,
     tr_meta, va_meta, te_meta, K) = splits(task, seed)
    M = arch["n_levels"]
    g1, g2 = arch["fb_gain1"], arch["fb_gain2"]
    cand_gains = [g1] if M == 2 else [g1, g2]
    # 1. Candidate (winner architecture, delta).
    cand = train_net(task, tr, cfg, M, cand_gains, seed, epochs)
    yield "cand_on", eval_net(cand, te, te_lab, te_meta, task,
                              fb_mode="on")
    yield "cand_cut", eval_net(cand, te, te_lab, te_meta, task,
                               fb_mode="cut")
    yield "cand_shuffle", eval_net(cand, te, te_lab, te_meta, task,
                                   fb_mode="shuffle")
    # 2. Two-level ladder (separates the third level).
    if M == 3:
        lad = train_net(task, tr, cfg, 2, [g1], seed, epochs)
        yield "hier2_ladder", eval_net(lad, te, te_lab, te_meta,
                                       task, fb_mode="on")
    # 3. Two-level ladder, no feedback (separates feedback
    # from the slower level 2).
    nofb = train_net(task, tr, cfg, 2, [0.0], seed, epochs)
    yield "hier2_nofb", eval_net(nofb, te, te_lab, te_meta, task,
                                 fb_mode="on")
    # 4. Flat (round-15 winner config in the hierarchical class:
    # M=2, both alphas = task alpha, no feedback).
    a1 = cfg["alpha"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = net_kw(task, cfg, 2, [0.0])
    kw["alphas"] = (a1, a1)
    flat = HierRPCNetwork(**kw)
    flat.calibrate(tr)
    flat.build_prototypes(tr, proto_n=N_TRAIN19[task])
    for _ in range(epochs):
        for seq in tr:
            flat.reset()
            for tok in seq:
                flat.step(tok, train=True, plasticity=True)
    yield "flat", eval_net(flat, te, te_lab, te_meta, task,
                           fb_mode="on")
    # 5. Candidate frozen (reservoir control).
    frozen = train_net(task, tr, cfg, M, cand_gains, seed, epochs,
                       plasticity=False)
    yield "cand_frozen", eval_net(frozen, te, te_lab, te_meta, task,
                                  fb_mode="on")


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r.get("phase"), r.get("config"), r.get("seed")))
    return done


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    task, phase = sys.argv[1], sys.argv[2]
    assert task in TASKS and phase in ("val", "test"), (task, phase)
    seeds, out_path, winners_path, epochs = None, None, None, EPOCHS
    n_levels = fb_gain1 = fb_gain2 = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--seeds":
            a, b = sys.argv[i + 1].split("-")
            seeds = list(range(int(a), int(b) + 1))
            i += 2
        elif sys.argv[i] == "--out":
            out_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--winners":
            winners_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--epochs":
            epochs = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--n-levels":
            n_levels = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--fb-gain1":
            fb_gain1 = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--fb-gain2":
            fb_gain2 = float(sys.argv[i + 1])
            i += 2
        else:
            raise ValueError(sys.argv[i])
    assert seeds and out_path, "need --seeds and --out"
    if phase == "test":
        arch = load_winners(winners_path)
        cfg = arch["tasks"][task]
    else:
        assert task == "hier", "val runs on the hier task only"
        assert n_levels in (2, 3), "val needs --n-levels 2 or 3"
        assert fb_gain1 in (0.1, 0.3), "val needs --fb-gain1"
        if n_levels == 3:
            assert fb_gain2 in (0.1, 0.3), "val needs --fb-gain2"
        cfg = ROUND15_WINNER[task]
    done = load_done(out_path)
    with open(out_path, "a") as f:
        for seed in seeds:
            if phase == "val":
                key = ("val",
                       f"L{n_levels}|{fb_gain1}|{fb_gain2}", seed)
                if key in done:
                    continue
                r = run_val(task, seed, n_levels, fb_gain1,
                            fb_gain2, epochs)
                rec = {"task": task, "phase": "val",
                       "config": key[1], "seed": seed,
                       "n_levels": n_levels,
                       "fb_gain1": fb_gain1, "fb_gain2": fb_gain2,
                       "alpha": cfg["alpha"],
                       "smooth_sigma": cfg["smooth_sigma"],
                       "eta_d": cfg["eta_d"],
                       "epochs": epochs, **r}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{task} val L{n_levels} g1={fb_gain1} "
                      f"g2={fb_gain2} seed={seed}: "
                      f"val_acc={r['val_acc']:.3f} "
                      f"val_m1={r['val_m1']:.3f}", flush=True)
            else:
                for cond, r in run_test_nets(task, seed, cfg, arch,
                                             epochs):
                    key = ("test", cond, seed)
                    if key in done:
                        continue
                    rec = {"task": task, "phase": "test",
                           "config": cond, "seed": seed,
                           "n_levels": arch["n_levels"],
                           "fb_gain1": arch["fb_gain1"],
                           "fb_gain2": arch["fb_gain2"],
                           "alpha": cfg["alpha"],
                           "smooth_sigma": cfg["smooth_sigma"],
                           "eta_d": cfg["eta_d"],
                           "epochs": epochs, **r}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test {cond} seed={seed}: "
                          f"acc={r['acc']:.3f} m1={r['m1']:.3f}",
                          flush=True)
                bkey = ("test", "baseline", seed)
                if bkey not in done:
                    (tr, va, te, tr_lab, va_lab, te_lab,
                     tr_meta, va_meta, te_meta, K) = splits(task, seed)
                    if task == "hier":
                        b = compute_baselines_hier(
                            tr, te, te_lab, te_meta, seed)
                    else:
                        b = compute_baselines(task, tr, te, None,
                                              None, seed)
                    rec = {"task": task, "phase": "test",
                           "config": "baseline", "seed": seed, **b}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"{task} test baseline seed={seed}: "
                          f"markov1={b['markov1_acc']:.3f} "
                          f"persist={b['persist_acc']:.3f} "
                          f"esn={b['esn_acc']:.3f}", flush=True)
    print(f"done {task} {phase} seeds {seeds[0]}-{seeds[-1]}", flush=True)


if __name__ == "__main__":
    main()
