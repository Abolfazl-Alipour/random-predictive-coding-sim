"""Round 18 smoke test (VM only). 1 seed (262), 1 epoch, both tasks.

Checks (PROTOCOL-R18-CONTINUOUS.md section 10):
1. cont_bump at a bin centre == _encode(bin) for the same sigma.
2. cont_bin reproduces the generator's bins on the splits data.
3. Convention A acc > convention B acc, and convention A MSE <
   convention B MSE, on the smoke seed for both tasks (same
   predictions, scored both ways; convention B uses the exact
   run_cont indexing from rpc_round15).
4. Ridge fit runs, finite MSE on train and test; in-sample train
   ridge MSE < train prototype MSE.
5. All metrics finite, no NaN; clamp_frac reported; bump net's
   p1 differs from one-hot net's p1 on identical inputs.
6. Driver CLI resumability: run the driver twice with
   --seeds 262-262 --epochs 1 on a val file; second run writes
   nothing new.
7. Analyzer runs on the 1-seed test output and writes the
   analysis file (bars will mostly FAIL on 1 seed; the smoke
   checks that it runs, not the bars).
Exits 0 only if all checks pass.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import accuracy, mse_from_bin_preds
from rpc_round15 import (K_OF, compute_baselines, splits)
from rpc_round18_continuous import (
    LO_HI, ROUND15_WINNER, bump_pairs, collect, collect_train,
    eval_net, fit_ridge, proto_metrics, ridge_metrics, run_test_nets,
    train_bump, train_onehot)
from rpc_v2 import RPCNetwork, cont_bin, cont_bump

SEED = 262
EPOCHS = 1
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def finite(d):
    return all(isinstance(v, float) and math.isfinite(v)
               for k, v in d.items()
               if k in ("acc", "within1", "mse", "mean_burst",
                        "clamp_frac"))


def main():
    # 1. Bump at a bin centre == _encode(bin), same sigma.
    for task in ("lorenz", "varsine"):
        K = K_OF[task]
        lo, hi = LO_HI[task]
        for sigma in (1.0, 2.0, 3.0):
            net = RPCNetwork(K=K, smooth_sigma=sigma, n_res=200,
                             n_out=256)
            worst = 0.0
            for b in (0, 1, 7, 15, 16, 31):
                centre = lo + (b + 0.5) / K * (hi - lo)
                bump = cont_bump(centre, K, lo, hi, sigma)
                enc = net._encode(b)
                worst = max(worst, float((bump - enc).abs().max()))
            check(f"1 bump==_encode {task} sigma={sigma}",
                  worst < 1e-5, f"max|diff|={worst:.2e}")
    # 2. cont_bin reproduces the generator's bins.
    for task in ("lorenz", "varsine"):
        K = K_OF[task]
        lo, hi = LO_HI[task]
        tr, va, te, tr_c, va_c, te_c, _, _ = splits(task, SEED)
        bad = 0
        for bins, cont in zip(tr + va + te, tr_c + va_c + te_c):
            for b, v in zip(bins, cont):
                bad += int(cont_bin(v, K, lo, hi) != b)
        check(f"2 cont_bin==generator {task}", bad == 0,
              f"mismatches={bad}")
    # 3-5. Train all four nets (1 epoch, seed 262) and score.
    for task in ("lorenz", "varsine"):
        K = K_OF[task]
        lo, hi = LO_HI[task]
        tr, va, te, tr_c, va_c, te_c, _, decode = splits(task, SEED)
        cfg = ROUND15_WINNER[task]
        # Convention B (run_cont indexing, exact): preds for
        # i < len(seq)-1, scored against token t+1.
        net1 = train_onehot(task, tr, cfg, "delta", SEED, EPOCHS)
        preds_b, bursts = [], []
        for seq in te:
            net1.reset()
            p = []
            for i, tok in enumerate(seq):
                out = net1.step(tok, train=False,
                                use_confirmation=False)
                bursts.append(out["burst_rate"])
                if i < len(seq) - 1:
                    p.append(out["pred"])
            preds_b.append(p)
        acc_b = accuracy(preds_b, te)
        mse_b = mse_from_bin_preds(preds_b, te_c, decode)
        # Convention A (driver): pred(t) vs token t, t >= 1.
        r1 = eval_net(net1, te, te_c, decode, task)
        check(f"3 convA>convB acc {task}",
              r1["acc"] > acc_b,
              f"convA={r1['acc']:.3f} convB={acc_b:.3f}")
        check(f"3 convA<convB mse {task}",
              r1["mse"] < mse_b,
              f"convA={r1['mse']:.4f} convB={mse_b:.4f}")
        # 4. Ridge: in-sample train ridge MSE < train proto MSE.
        fit1 = fit_ridge(*collect_train(net1, tr, tr_c,
                                        use_vec=False))
        _, d1s_tr, vs_tr, bins_tr = collect(net1, tr, tr_c, False)
        r_tr = ridge_metrics(fit1, d1s_tr, vs_tr, bins_tr, task)
        # Train prototype MSE: re-collect preds on train.
        preds_tr, _, vs_tr2, bins_tr2 = collect(net1, tr, tr_c,
                                                False)
        m_tr = proto_metrics(preds_tr, vs_tr2, bins_tr2, decode)
        check(f"4 train ridge mse < train proto mse {task}",
              math.isfinite(r_tr["mse"]) and
              r_tr["mse"] < m_tr["mse"],
              f"ridge={r_tr['mse']:.4f} proto={m_tr['mse']:.4f}")
        # 5. Full test path: all 8 conditions + baseline, finite.
        winners = {t: {**ROUND15_WINNER[t], "bump_sigma": 2.0}
                   for t in ("lorenz", "varsine")}
        all_recs = {}
        for cond, r in run_test_nets(task, SEED, winners[task],
                                     2.0, EPOCHS):
            all_recs[cond] = r
            check(f"5 finite {task} {cond}", finite(r),
                  f"acc={r['acc']:.3f} mse={r['mse']:.4f}")
        b = compute_baselines(task, tr, te, te_c, decode, SEED)
        check(f"5 finite {task} baseline",
              all(math.isfinite(v) for v in b.values()),
              f"markov1={b['markov1_acc']:.3f} "
              f"persist={b['persist_acc']:.3f} "
              f"esn={b['esn_acc']:.3f}")
        # 5b. Bump net's p1 differs from one-hot net's p1.
        tr_xy, tr_v = bump_pairs(tr, tr_c, task, 2.0)
        te_xy, te_v = bump_pairs(te, te_c, task, 2.0)
        net2 = train_bump(task, tr_xy, cfg, "delta", SEED, EPOCHS)
        seq = te[0]
        seq_xy = te_xy[0]
        net1.reset()
        net2.reset()
        diffs = []
        for tok, (x, y) in zip(seq, seq_xy):
            o1 = net1.step(tok, train=False,
                           use_confirmation=False)
            o2 = net2.step_vec(x, train=False,
                               use_confirmation=False)
            diffs.append(float((o1["p1"] - o2["p1"]).abs().max()))
        check(f"5 bump p1 != one-hot p1 {task}",
              max(diffs) > 1e-6, f"max|dp1|={max(diffs):.4f}")
    # 6. Driver CLI resumability (val, seed 262, 1 epoch).
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "val.jsonl")
        cmd = [sys.executable, "rpc_round18_continuous.py",
               "lorenz", "val", "--seeds", "262-262", "--epochs",
               "1", "--out", out]
        r = subprocess.run(cmd, cwd=os.path.dirname(
            os.path.abspath(__file__)), capture_output=True,
            text=True)
        n1 = sum(1 for _ in open(out)) if os.path.exists(out) else 0
        r2 = subprocess.run(cmd, cwd=os.path.dirname(
            os.path.abspath(__file__)), capture_output=True,
            text=True)
        n2 = sum(1 for _ in open(out)) if os.path.exists(out) else 0
        check("6 CLI resumable (val)", r.returncode == 0 and
              r2.returncode == 0 and n1 == 3 and n2 == 3,
              f"first={n1} lines second={n2} lines "
              f"rc={r.returncode}/{r2.returncode}")
        # 7. Analyzer runs on 1-seed test output.
        wpath = os.path.join(tmp, "winner.json")
        with open(wpath, "w") as f:
            json.dump({t: {**ROUND15_WINNER[t], "bump_sigma": 2.0}
                       for t in ("lorenz", "varsine")}, f)
        tout = os.path.join(tmp, "test.jsonl")
        cmd = [sys.executable, "rpc_round18_continuous.py",
               "lorenz", "test", "--seeds", "262-262", "--epochs",
               "1", "--out", tout, "--winners", wpath]
        r = subprocess.run(cmd, cwd=os.path.dirname(
            os.path.abspath(__file__)), capture_output=True,
            text=True)
        n_test = sum(1 for _ in open(tout)) if os.path.exists(
            tout) else 0
        check("7 driver test CLI 1 seed", r.returncode == 0 and
              n_test == 9, f"lines={n_test} rc={r.returncode}")
        apath = os.path.join(tmp, "analysis.md")
        r = subprocess.run(
            [sys.executable, "r18/r18_analyze.py", "--in", tout,
             "--out", apath],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True)
        wrote = os.path.exists(apath) and os.path.getsize(apath) > 0
        check("7 analyzer writes on 1-seed output", wrote,
              f"rc={r.returncode} (1 is expected: incomplete "
              f"coverage) size="
              f"{os.path.getsize(apath) if wrote else 0}")
    if FAILURES:
        print(f"\nSMOKE FAILED: {len(FAILURES)} checks failed: "
              f"{FAILURES}", flush=True)
        return 1
    print("\nSMOKE PASSED: all checks pass.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
