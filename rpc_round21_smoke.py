"""Round-21 smoke test (VM only, no Beast launch).

Checks, in order:
  1. Convention-A Lorenz reproduces MSE ~5.07 on seeds 262-266
     (the section-57 spot check: acc 0.4285, MSE 5.073, per-seed
     [5.68, 4.62, 3.75, 5.83, 5.48]).
  2. Convention-A var-sine re-run on seeds 262-266 (a=0.7, s=2.0,
     eta=0.2): AGY reviewer-reported MSE 0.0370, never independently
     re-run before this script.
  3. D8 seed-262 check: standardized ridge ties (or beats raw) and
     the frozen-ridge win disappears under standardization (spot
     check: std 13.976 / 13.976, raw 14.904 trained / 14.073 frozen).
  4. NARMA-10 / Mackey-Glass data sanity: range, clipping, bin
     coverage, Markov-2 vs persistence (REPORTED, not gated:
     Markov-2 can lose to persistence on NARMA-10 because the
     driving input u is unobserved, which makes the task
     non-trivial for 2-gram lookup rather than unlearnable),
     and one delta run per task.
  5. Driver CLI resumable: running the same val command twice adds
     no lines.
  6. Frozen-twin exactness: the twin row equals a standalone frozen
     invocation row (proto MSE to 12 decimals).
  7. Centroid sanity: argmax(sims) == pred on every step; centroid
     beta=10 finite and within a factor of 2 of proto MSE (the
     centroid need NOT equal proto at high beta: on lorenz seed
     322 beta=20 beats proto by 33%, because the softmax still
     mixes neighbouring prototypes and corrects argmax errors
     near bin edges).
  8. Wire gate: mean fb_drive_ratio on the wire-on arm (lorenz,
     1 seed) -- the value the protocol gate is set from.
  9. Timing per invocation type (for the protocol cost estimate).

Run: ~/workspace/thesis-rpc/venv/bin/python rpc_round21_smoke.py
Exit 0 only if the gated checks pass. Check 2 reports the
measured var-sine number against AGY's 0.0370 (a mismatch is a
finding, not a smoke failure: the round exists to settle it on
fresh seeds). Check 4c is a report, not a gate.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rpc_round21_continuous as r21
from baselines import (markov_predict, mse_from_bin_preds,
                       persistence_mse)

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "venv", "bin", "python")
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "rpc_round21_continuous.py")
TMP = "/tmp/r21_smoke"

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def main():
    os.makedirs(TMP, exist_ok=True)
    t_start = time.time()

    # 1. Lorenz Convention-A reproduction (seeds 262-266).
    cfg_lor = {"alpha": 1.0, "smooth_sigma": 2.0, "eta_d": 0.2}
    accs, mses = [], []
    t0 = time.time()
    for seed in range(262, 267):
        rows = list(r21.run_test_arm("lorenz", seed, cfg_lor, "token",
                                     None, False, 5))
        m = dict(rows)["token_delta"]
        accs.append(m["acc"])
        mses.append(m["mse"])
        print(f"  lorenz A14 seed {seed}: acc={m['acc']:.4f} "
              f"mse={m['mse']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    print(f"  lorenz A14 MEAN acc={np.mean(accs):.4f} "
          f"mse={np.mean(mses):.3f} per-seed="
          f"{[round(x, 2) for x in mses]}", flush=True)
    check("1 lorenz Convention-A reproduces 5.073",
          abs(np.mean(mses) - 5.073) < 0.05
          and abs(np.mean(accs) - 0.4285) < 0.005,
          f"mean mse={np.mean(mses):.3f} acc={np.mean(accs):.4f}")

    # 2. Var-sine Convention-A re-run (seeds 262-266).
    cfg_var = {"alpha": 0.7, "smooth_sigma": 2.0, "eta_d": 0.2}
    accs, mses = [], []
    t0 = time.time()
    for seed in range(262, 267):
        rows = list(r21.run_test_arm("varsine", seed, cfg_var, "token",
                                     None, False, 5))
        m = dict(rows)["token_delta"]
        accs.append(m["acc"])
        mses.append(m["mse"])
        print(f"  varsine A14 seed {seed}: acc={m['acc']:.4f} "
              f"mse={m['mse']:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    print(f"  varsine A14 MEAN acc={np.mean(accs):.4f} "
          f"mse={np.mean(mses):.4f} per-seed="
          f"{[round(x, 4) for x in mses]}", flush=True)
    print(f"  varsine vs AGY reviewer-reported MSE 0.0370: "
          f"diff={np.mean(mses) - 0.0370:+.4f}", flush=True)
    check("2 varsine Convention-A re-run reported (not a gate)",
          np.isfinite(np.mean(mses)),
          f"mean mse={np.mean(mses):.4f} (AGY 0.0370)")

    # 3. D8 seed-262 check (C1 config: a=0.7, s=0.0, eta=0.2).
    cfg_c1 = {"alpha": 0.7, "smooth_sigma": 0.0, "eta_d": 0.2}
    tr, va, te, tr_c, va_c, te_c, K, decode = r21.splits21("lorenz",
                                                           262)
    net_tr = r21.train_token21("lorenz", tr, cfg_c1, "delta", 262, 5)
    net_fr = r21.train_token21("lorenz", tr, cfg_c1, "frozen", 262, 5)
    m_tr = r21.eval_net21(net_tr, tr, tr_c, te, te_c, decode,
                          "lorenz", use_vec=False)
    m_fr = r21.eval_net21(net_fr, tr, tr_c, te, te_c, decode,
                          "lorenz", use_vec=False)
    print(f"  D8 seed262 ridge trained std/raw: "
          f"{m_tr['ridge_std_mse']:.3f} / {m_tr['ridge_raw_mse']:.3f}; "
          f"frozen std/raw: {m_fr['ridge_std_mse']:.3f} / "
          f"{m_fr['ridge_raw_mse']:.3f}", flush=True)
    print(f"  D8 seed262 proto trained/frozen MSE: "
          f"{m_tr['mse']:.3f} / {m_fr['mse']:.3f}", flush=True)
    check("3a D8 standardized ridge ties (trained vs frozen within "
          "5%)",
          abs(m_tr["ridge_std_mse"] - m_fr["ridge_std_mse"])
          <= 0.05 * m_fr["ridge_std_mse"],
          f"std {m_tr['ridge_std_mse']:.3f} vs {m_fr['ridge_std_mse']:.3f}")
    check("3b D8 standardized trained ridge beats-or-ties raw "
          "trained ridge",
          m_tr["ridge_std_mse"] <= m_tr["ridge_raw_mse"] + 1e-9,
          f"std {m_tr['ridge_std_mse']:.3f} vs raw "
          f"{m_tr['ridge_raw_mse']:.3f}")
    check("3c D8 raw ridge reproduces the frozen win (artifact)",
          m_fr["ridge_raw_mse"] < m_tr["ridge_raw_mse"],
          f"raw frozen {m_fr['ridge_raw_mse']:.3f} < raw trained "
          f"{m_tr['ridge_raw_mse']:.3f}")

    # 4. New-task data sanity + one delta run per task.
    for task, lo, hi in (("narma10", 0.0, 1.0),
                         ("mackeyglass", 0.0, 1.5)):
        tr, va, te, tr_c, va_c, te_c, K, decode = r21.splits21(task,
                                                               322)
        all_c = [v for seq in tr_c + va_c + te_c for v in seq]
        all_b = [b for seq in tr + va + te for b in seq]
        lo_v, hi_v = min(all_c), max(all_c)
        n_bins = len(set(all_b))
        pm = persistence_mse(te_c)
        m2 = mse_from_bin_preds(
            markov_predict(tr, te, K, 2), te_c, decode)
        print(f"  {task}: value range [{lo_v:.3f}, {hi_v:.3f}] "
              f"(bins [{lo}, {hi}]); bins hit {n_bins}/32; "
              f"persist MSE {pm:.4f}; Markov-2 MSE {m2:.4f}",
              flush=True)
        check(f"4a {task} values inside bin range",
              lo_v >= lo and hi_v <= hi,
              f"[{lo_v:.3f}, {hi_v:.3f}] vs [{lo}, {hi}]")
        check(f"4b {task} bin coverage >= 20/32", n_bins >= 20,
              f"{n_bins}/32")
        # 4c is a REPORT, not a gate: Markov-2 need not beat
        # persistence for the task to be learnable (on NARMA-10
        # the driving input u is unobserved, so no bin-only
        # method can predict the u-driven part, and Markov-2 can
        # lose to persistence while a learned readout still wins).
        # The learnability signal is the delta run below (4d)
        # plus the centroid/ridge readouts reported there.
        print(f"  {task} 4c REPORT: Markov-2 {m2:.4f} vs persistence "
              f"{pm:.4f} -> "
              f"{'Markov-2 wins' if m2 < pm else 'persistence wins'} "
              f"(a Markov-2 loss means the task is non-trivial for "
              f"2-gram lookup, not that it is unlearnable)",
              flush=True)
        t0 = time.time()
        rows = list(r21.run_test_arm(task, 322,
                                     {"alpha": 1.0, "smooth_sigma": 2.0,
                                      "eta_d": 0.2},
                                     "token", None, False, 5))
        m = dict(rows)["token_delta"]
        print(f"  {task} delta G1 seed 322: acc={m['acc']:.3f} "
              f"mse={m['mse']:.4f} centroid10={m['centroid_mse_10']:.4f} "
              f"ridge_std={m['ridge_std_mse']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        check(f"4d {task} delta run finite + acc above chance",
              np.isfinite(m["mse"]) and m["acc"] > 1.0 / 32,
              f"acc={m['acc']:.3f} mse={m['mse']:.4f}")

    # 5. Driver CLI resumable.
    out = os.path.join(TMP, "r21_lorenz_val_token_smoke.jsonl")
    if os.path.exists(out):
        os.remove(out)
    cmd = [PY, DRIVER, "lorenz", "val", "--seeds", "322-323",
           "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    n1 = sum(1 for _ in open(out))
    subprocess.run(cmd, check=True, capture_output=True)
    n2 = sum(1 for _ in open(out))
    print(f"  CLI resumable: {n1} lines after run 1, {n2} after "
          f"run 2", flush=True)
    check("5 driver CLI resumable (no new lines on re-run)",
          n1 == 10 and n2 == n1, f"{n1} -> {n2} lines")

    # 6. Frozen-twin exactness (lorenz seed 322, token G1).
    rows = list(r21.run_test_arm(
        "lorenz", 322, {"alpha": 1.0, "smooth_sigma": 2.0,
                        "eta_d": 0.2},
        "token", None, False, 5))
    twin_mse = dict(rows)["token_frozen"]["mse"]
    tr, va, te, tr_c, va_c, te_c, K, decode = r21.splits21("lorenz",
                                                           322)
    net_sa = r21.train_token21(
        "lorenz", tr, {"alpha": 1.0, "smooth_sigma": 2.0,
                       "eta_d": 0.2},
        "frozen", 322, 5)
    m_sa = r21.eval_net21(net_sa, tr, tr_c, te, te_c, decode,
                          "lorenz", use_vec=False)
    print(f"  twin frozen proto MSE {twin_mse:.12f} vs standalone "
          f"{m_sa['mse']:.12f}", flush=True)
    check("6 frozen twin equals standalone frozen exactly",
          abs(twin_mse - m_sa["mse"]) < 1e-12,
          f"diff={abs(twin_mse - m_sa['mse']):.2e}")

    # 7. Centroid sanity (lorenz seed 322, token G1): argmax(sims)
    # == pred on every step; beta=20 close to proto.
    net = r21.train_token21(
        "lorenz", tr, {"alpha": 1.0, "smooth_sigma": 2.0,
                       "eta_d": 0.2},
        "delta", 322, 5)
    (preds, d1s, sims, vs, bins, mb, fbd, fbr) = r21.collect21(
        net, te, te_c, use_vec=False)
    argmax_ok = all(int(np.argmax(s)) == p
                    for s, p in zip(sims, preds))
    cen = r21.centroid_metrics(sims, vs, bins, decode, "lorenz")
    pm = r21.proto_metrics(preds, vs, bins, decode)
    print(f"  centroid: argmax(sims)==pred on all {len(preds)} steps: "
          f"{argmax_ok}; proto MSE {pm['mse']:.4f}; centroid "
          f"5/10/20: {cen[5.0]['mse']:.4f} / {cen[10.0]['mse']:.4f} / "
          f"{cen[20.0]['mse']:.4f}", flush=True)
    check("7a centroid argmax(sims) reproduces proto pred",
          argmax_ok, f"{len(preds)} steps")
    # 7b is a sanity gate, not an equality check: the centroid at
    # high beta need NOT equal the prototype readout (on lorenz
    # seed 322 beta=20 beats proto by 33%: the softmax still
    # mixes neighbouring prototypes, which corrects argmax errors
    # near bin edges). The gate is that beta=10 is finite and
    # within a factor of 2 of proto (same order of magnitude).
    check("7b centroid beta=10 finite and within 2x of proto MSE",
          np.isfinite(cen[10.0]["mse"])
          and 0.5 * pm["mse"] <= cen[10.0]["mse"] <= 2.0 * pm["mse"],
          f"beta10 {cen[10.0]['mse']:.4f} vs proto {pm['mse']:.4f}")

    # 8. Wire gate: mean fb_drive_ratio on the wire-on arm.
    # run_test_arm with wire_on=True yields token_w_* rows only,
    # so the wire-off comparison needs its own arm call.
    t0 = time.time()
    rows_w = list(r21.run_test_arm(
        "lorenz", 322, {"alpha": 1.0, "smooth_sigma": 2.0,
                        "eta_d": 0.2},
        "token", None, True, 5))
    rows_o = list(r21.run_test_arm(
        "lorenz", 322, {"alpha": 1.0, "smooth_sigma": 2.0,
                        "eta_d": 0.2},
        "token", None, False, 5))
    m_w = dict(rows_w)["token_w_delta"]
    m_o = dict(rows_o)["token_delta"]
    print(f"  wire-on lorenz seed 322: mean fb_drive_ratio="
          f"{m_w['fb_drive_ratio']:.4f} fb_rms_ratio="
          f"{m_w['fb_rms_ratio']:.4f}; proto MSE on "
          f"{m_w['mse']:.4f} vs off {m_o['mse']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    check("8 wire gate value reported (not a gate)",
          np.isfinite(m_w["fb_drive_ratio"]),
          f"fb_drive_ratio={m_w['fb_drive_ratio']:.4f}")

    # 9. Timing summary is the wall time of this script; the
    # per-invocation times are printed inline above.
    print(f"SMOKE DONE in {time.time() - t_start:.0f}s; failures: "
          f"{FAILURES if FAILURES else 'none'}", flush=True)
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
