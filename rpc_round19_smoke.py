"""Round 19 smoke test (1 seed, 1 epoch unless noted).

Checks protocol section 10 before any Beast launch:
  1. Nested init: a 3-level net's levels 1-2 weights are EXACTLY
     a 2-level net's at the same seed.
  2. Flat reproduction (5 epochs): the hierarchical flat net
     (M=2, both alphas 1.0, fb 0.0) on seed 201 is within 0.05
     of the round-15 recorded delta acc (planted1 0.227,
     sine 0.872). Exact equality is impossible: the init draw
     order differs from RPCNetwork (W_12d, W_dist1b, W_conf
     are not drawn in the hierarchical class).
  3. Feedback changes the lower module's state: r1 cosine
     (on vs cut) below 1.0 and above 0.0; fb_rms_ratio above 0
     when on and 0.0 when cut; the shuffle drive measured on
     the SAME trajectory (same ds) matches the on drive within
     10 percent; the separate-pass fb RMS is reported with a
     25 percent bar because the wrong feedback changes the
     trajectory, so the gap is divergence, not a magnitude bug.
  4. Top readout finite and non-degenerate on M=3 (top_pred
     takes at least 2 distinct values).
  5. Timescale ladder: a ridge probe fit on train decodes the
     block context from each level's reservoir state at lag 11
     (context token at step 0, filler only after); level-3
     accuracy must exceed level-1 accuracy.
  6. Guardrails: all metrics finite, no NaN, r_max below 10,
     clamp_frac reported, driver CLI resumable, analyzer
     writes on 1-seed output.

Usage: python rpc_round19_smoke.py
Exit 0 only if every check passes.
"""
import json
import math
import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rpc_hier import HierRPCNetwork
from rpc_round19_hierarchy import (eval_net, ladder_alphas,
                                   load_winners, run_test_nets,
                                   splits)

BASE = os.path.dirname(os.path.abspath(__file__))
R19 = os.path.join(BASE, "r19")
PY = sys.executable
SEED = 282
FAIL = []


def check(name, cond, detail):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}: {detail}", flush=True)
    if not cond:
        FAIL.append(name)


def cos(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def train_hier(task, train, n_levels, fb_gains, seed, epochs=1,
               alphas=None, eta_d=None, plasticity=True):
    from rpc_round19_hierarchy import ROUND15_WINNER, N_TRAIN19
    cfg = ROUND15_WINNER[task]
    if alphas is None:
        alphas = ladder_alphas(task, n_levels)
    if eta_d is None:
        eta_d = cfg["eta_d"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = HierRPCNetwork(K={"planted1": 16, "sine": 32,
                            "hier": 16}[task],
                         n_levels=n_levels, alphas=alphas,
                         fb_gains=list(fb_gains), eta_d=eta_d,
                         smooth_sigma=cfg["smooth_sigma"],
                         n_out=256, n_res=200, norm_rows=False)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=N_TRAIN19[task])
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, plasticity=plasticity)
    return net


def main():
    # ---- 1. Nested init -------------------------------------
    torch.manual_seed(SEED)
    net2 = HierRPCNetwork(K=16, n_levels=2, alphas=(1.0, 0.3),
                          fb_gains=[0.3], eta_d=0.05)
    torch.manual_seed(SEED)
    net3 = HierRPCNetwork(K=16, n_levels=3, alphas=(1.0, 0.3, 0.1),
                          fb_gains=[0.3, 0.3], eta_d=0.05)
    same = True
    for i in range(1):  # M-1 = 1 wire for the 2-level net
        same &= bool(torch.equal(net2.W_proj[i], net3.W_proj[i]))
        same &= bool(torch.equal(net2.W_dist[i], net3.W_dist[i]))
        same &= bool(torch.equal(net2.W_fb[i], net3.W_fb[i]))
    for i in range(2):  # M = 2 modules
        same &= bool(torch.equal(net2.mods[i].W_in,
                                 net3.mods[i].W_in))
        same &= bool(torch.equal(net2.mods[i].W_res,
                                 net3.mods[i].W_res))
    check("1 nested-init", same,
          "levels 1-2 weights identical between M=2 and M=3 "
          f"at seed {SEED}")
    # ---- 2. Flat reproduction (5 epochs, seed 201) ---------
    for task, ref in (("planted1", 0.227), ("sine", 0.872)):
        tr, va, te, tr_lab, va_lab, te_lab, tr_meta, va_meta, \
            te_meta, K = splits(task, 201)
        flat = train_hier(task, tr, 2, [0.0], 201, epochs=5,
                          alphas=(1.0, 1.0))
        m = eval_net(flat, te, te_lab, te_meta, task, fb_mode="on")
        check(f"2 flat-{task}", abs(m["acc"] - ref) <= 0.05,
              f"hier flat acc {m['acc']:.3f} vs round-15 seed-201 "
              f"delta {ref} (diff {m['acc'] - ref:+.3f}, bar 0.05)")
    # ---- 3-6 share one trained hier3 net -------------------
    tr, va, te, tr_lab, va_lab, te_lab, tr_meta, va_meta, \
        te_meta, K = splits("hier", SEED)
    net = train_hier("hier", tr, 3, [0.3, 0.3], SEED, epochs=1)
    # ---- 3. Feedback changes the lower module's state ------
    # The shuffle drive is measured two ways: (a) same-trajectory,
    # the shuffle drive computed from the on-pass ds values, which
    # isolates the permutation effect; (b) separate-pass, which
    # also includes trajectory divergence (the wrong feedback
    # changes the state, so ds differs). Only (a) is a tight bar.
    r1_on, r1_cut, r1_shuf = [], [], []
    fb_on, fb_cut, fb_shuf = [], [], []
    on_drive, shuf_drive = [], []
    for seq in te[:5]:
        net.reset()
        for tok in seq:
            out = net.step(tok, train=False, fb_mode="on")
            r1_on.append(out["rs"][0].detach().numpy().astype(float))
            fb_on.append(out["fb_ratios"])
            for i in range(net.M - 1):
                ds = out["ds"][i]
                on_drive.append(float(
                    (net.fb_gains[i] * (net.W_fb[i] @ ds)).norm()))
                shuf_drive.append(float(
                    (net.fb_gains[i]
                     * (net.W_fb[i] @ ds[net.fb_perm[i]])).norm()))
        for mode, r1s, fbs in (("cut", r1_cut, fb_cut),
                               ("shuffle", r1_shuf, fb_shuf)):
            net.reset()
            for tok in seq:
                out = net.step(tok, train=False, fb_mode=mode)
                r1s.append(out["rs"][0].detach().numpy().astype(float))
                fbs.append(out["fb_ratios"])
    c = np.mean([cos(a, b) for a, b in zip(r1_on, r1_cut)])
    fb_on_m = float(np.mean(fb_on))
    fb_cut_m = float(np.mean(fb_cut))
    fb_shuf_m = float(np.mean(fb_shuf))
    drive_ratio = float(np.mean(shuf_drive) / np.mean(on_drive))
    check("3a fb-changes-state", 0.0 < c < 1.0,
          f"mean r1 cosine on vs cut = {c:.4f} "
          "(must be below 1.0 and above 0.0)")
    check("3b fb-ratios", fb_on_m > 0.0 and fb_cut_m == 0.0,
          f"fb_rms_ratio on = {fb_on_m:.4f}, cut = {fb_cut_m:.4f}")
    check("3c shuffle-drive-same-trajectory",
          0.90 <= drive_ratio <= 1.10,
          f"shuffle drive / on drive = {drive_ratio:.4f} "
          "(same ds, bar 0.90-1.10)")
    check("3d shuffle-rms-separate-pass",
          abs(fb_shuf_m - fb_on_m) <= 0.25 * max(fb_on_m, 1e-9),
          f"fb RMS shuffle = {fb_shuf_m:.4f} vs on = {fb_on_m:.4f} "
          f"(gap {100 * (fb_shuf_m - fb_on_m) / fb_on_m:+.1f} percent; "
          "bar 25 percent: the gap is trajectory divergence, "
          "not a magnitude bug, and it weakens the shuffle "
          "contrast as a diagnostic)")
    # ---- 4. Top readout finite and non-degenerate ----------
    tops = []
    for seq in te[:10]:
        net.reset()
        for tok in seq:
            out = net.step(tok, train=False, fb_mode="on")
            tops.append(out["top_pred"])
    tops = [t for t in tops if t >= 0]
    check("4 top-readout", len(set(tops)) >= 2,
          f"top_pred takes {len(set(tops))} distinct values "
          f"over {len(tops)} steps (must be >= 2)")
    # ---- 5. Timescale ladder probe -------------------------
    rng = np.random.RandomState(12345)
    probe_tr, probe_te = [], []
    for ctx in range(4):
        for _ in range(20):
            seq = [ctx] + list(rng.randint(4, 16, size=11))
            probe_tr.append((seq, ctx))
        for _ in range(10):
            seq = [ctx] + list(rng.randint(4, 16, size=11))
            probe_te.append((seq, ctx))
    # Fresh untrained net (plasticity off): the ladder is an
    # architecture property, not a learning effect.
    probe_net = train_hier("hier", tr, 3, [0.3, 0.3], SEED,
                           epochs=0)

    def state_at(seq, lag):
        probe_net.reset()
        r_at = None
        for i, tok in enumerate(seq):
            out = probe_net.step(tok, train=False, fb_mode="on")
            if i == lag:
                r_at = [r.detach().numpy().astype(float)
                        for r in out["rs"]]
        return r_at

    def probe_acc(lag):
        Xtr = {lv: [] for lv in range(3)}
        ytr = []
        Xte = {lv: [] for lv in range(3)}
        yte = []
        for seq, ctx in probe_tr:
            rs = state_at(seq, lag)
            for lv in range(3):
                Xtr[lv].append(rs[lv])
            ytr.append(ctx)
        for seq, ctx in probe_te:
            rs = state_at(seq, lag)
            for lv in range(3):
                Xte[lv].append(rs[lv])
            yte.append(ctx)
        accs = []
        for lv in range(3):
            A = np.array(Xtr[lv])
            Y = np.zeros((len(ytr), 4))
            Y[np.arange(len(ytr)), ytr] = 1.0
            W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]),
                                A.T @ Y)
            pred = np.argmax(np.array(Xte[lv]) @ W, axis=1)
            accs.append(float(np.mean(np.array(pred) == np.array(yte))))
        return accs

    acc11 = probe_acc(11)
    acc1 = probe_acc(1)
    check("5 timescale-ladder", acc11[2] > acc11[0],
          f"lag-11 context decode: L1 {acc11[0]:.2f}, "
          f"L2 {acc11[1]:.2f}, L3 {acc11[2]:.2f} "
          f"(chance 0.25; L3 must exceed L1). "
          f"Lag-1 sanity: L1 {acc1[0]:.2f}, L2 {acc1[1]:.2f}, "
          f"L3 {acc1[2]:.2f}")
    # ---- 6. Guardrails + driver CLI + analyzer ------------
    m = eval_net(net, te, te_lab, te_meta, "hier", fb_mode="on")
    finite = all(math.isfinite(v) for k, v in m.items()
                 if isinstance(v, float))
    check("6a finite", finite,
          f"all float metrics finite; r_max = {m['r_max']:.3f}, "
          f"clamp_frac = {m['clamp_frac']:.4f}")
    check("6b r-max", m["r_max"] < 10.0,
          f"r_max = {m['r_max']:.3f} (must be below 10)")
    check("6c clamp-reported", "clamp_frac" in m,
          f"clamp_frac = {m['clamp_frac']:.4f}, per-level "
          f"{[round(x, 4) for x in m['clamp_fracs']]}")
    # Driver CLI: 1 seed, 1 epoch, val, one config; resumable.
    tmp = os.path.join(R19, "smoke_val.jsonl")
    if os.path.exists(tmp):
        os.remove(tmp)
    cmd = [PY, os.path.join(BASE, "rpc_round19_hierarchy.py"),
           "hier", "val", "--seeds", "282-282", "--out", tmp,
           "--n-levels", "3", "--fb-gain1", "0.3",
           "--fb-gain2", "0.3", "--epochs", "1"]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    n1 = sum(1 for _ in open(tmp)) if os.path.exists(tmp) else 0
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    n2 = sum(1 for _ in open(tmp)) if os.path.exists(tmp) else 0
    check("6d driver-cli", r1.returncode == 0 and n1 == 1
          and r2.returncode == 0 and n2 == 1,
          f"val CLI rc {r1.returncode}/{r2.returncode}, "
          f"lines after run 1 = {n1}, after run 2 = {n2} "
          "(must stay 1: resumable)")
    if os.path.exists(tmp):
        os.remove(tmp)
    # Analyzer writes on 1-seed output (it exits 1 on coverage;
    # the check is that it writes r19_analysis.md).
    ana = os.path.join(R19, "r19_analysis.md")
    pre_exists = os.path.exists(ana)
    if not pre_exists:
        r3 = subprocess.run([PY, os.path.join(R19, "r19_analyze.py")],
                            capture_output=True, text=True)
        wrote = os.path.exists(ana)
        check("6e analyzer-writes", wrote,
              f"analyzer rc {r3.returncode} (1 expected: coverage "
              f"incomplete), r19_analysis.md written = {wrote}")
        if wrote:
            os.remove(ana)
    else:
        print("[SKIP] 6e analyzer-writes: a real r19_analysis.md "
              "already exists", flush=True)
    print("")
    if FAIL:
        print(f"SMOKE FAILED: {len(FAIL)} check(s): "
              + ", ".join(FAIL), flush=True)
        return 1
    print("SMOKE PASSED: all checks pass", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
