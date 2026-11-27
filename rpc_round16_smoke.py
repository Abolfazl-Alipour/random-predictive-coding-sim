"""Round 16 smoke test (VM only, never a result).

Runs 1 seed (221), 1 epoch, planted lag-1 and planted lag-2, both
conditions, and checks:
1. pred(t) and d1(t) in the omission pass equal pred(t) and d1(t)
   in the normal pass at the same position (the blank cannot affect
   the drive formed before it is fed in; protocol section 1 item 6).
2. Motif detection finds about 2-3 m1 and 2-3 m2 positions per
   sequence on average.
3. All metrics are finite; no NaN, no exception.
4. The lag-2 detection path runs without error.

Usage: python rpc_round16_smoke.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from rpc_round16_omission import (K, WINNER, detect_motifs,
                                  run_omission, splits, train_net)


def alignment_check():
    """Direct check of protocol section 1 item 6 on one sequence."""
    train, test = splits("planted1", 221)
    net = train_net(train, WINNER, "delta", 221, epochs=1)
    seq = test[0]
    m1s, m2s, _ = detect_motifs(seq, 1)
    assert m2s, "no m2 position in the first test sequence"
    t = m2s[0]
    blank = torch.zeros(K)
    # Normal pass.
    net.reset()
    pred_n, d1_n = [], []
    for tok in seq:
        out = net.step(tok, train=False, use_confirmation=False)
        pred_n.append(out["pred"])
        d1_n.append(out["d1"])
    # Omission pass, blank at t.
    net.reset()
    pred_o, d1_o = [], []
    for i, tok in enumerate(seq):
        if i == t:
            out = net._step_core(blank, train=False,
                                 use_confirmation=False)
        else:
            out = net.step(tok, train=False, use_confirmation=False)
        pred_o.append(out["pred"])
        d1_o.append(out["d1"])
    same_pred = pred_n[t] == pred_o[t]
    same_d1 = torch.allclose(d1_n[t], d1_o[t])
    print(f"alignment: t={t} omitted={seq[t]} "
          f"pred_normal={pred_n[t]} pred_omit={pred_o[t]} "
          f"same_pred={same_pred} same_d1={same_d1}")
    assert same_pred and same_d1, "ALIGNMENT CHECK FAILED"
    # The blank must change the state AFTER the omission step:
    # d1 at t+1 must differ between the two passes.
    if t + 1 < len(seq):
        diff_t1 = not torch.allclose(d1_n[t + 1], d1_o[t + 1])
        print(f"alignment: d1 differs at t+1: {diff_t1}")
        assert diff_t1, "blank had no effect after the omission step"
    print("alignment check PASSED")


def fmt(v):
    """None means structurally empty; print it as n/a."""
    return "n/a" if v is None else f"{v:.3f}"


def main():
    alignment_check()
    for task in ("planted1", "planted2"):
        train, test = splits(task, 221)
        # Motif detection counts.
        lag = 1 if task == "planted1" else 2
        n_m1 = n_m2 = 0
        for seq in test:
            m1s, m2s, _ = detect_motifs(seq, lag)
            n_m1 += len(m1s)
            n_m2 += len(m2s)
        print(f"{task}: mean m1/seq={n_m1 / len(test):.2f} "
              f"mean m2/seq={n_m2 / len(test):.2f}")
        for cond in ("delta", "frozen"):
            r = run_omission(task, train, test, WINNER, cond, 221,
                             epochs=1)
            bad = [k for k, v in r.items()
                   if isinstance(v, float) and not math.isfinite(v)]
            # None means "structurally empty" (e.g. omit_acc_t1_m1
            # on lag-1); it is not a failed metric.
            print(f"{task} {cond} 1ep seed221: "
                  f"omit_m2={fmt(r['omit_acc_m2'])} "
                  f"omit_m1={fmt(r['omit_acc_m1'])} "
                  f"omit_filler={fmt(r['omit_acc_filler'])} "
                  f"cont_m1={fmt(r['omit_acc_cont_m1'])} "
                  f"normal_m2={fmt(r['normal_acc_m2'])} "
                  f"normal_m1={fmt(r['normal_acc_m1'])} "
                  f"omit_m2_t1={fmt(r['omit_acc_t1_m2'])} "
                  f"omit_m1_t1={fmt(r['omit_acc_t1_m1'])} "
                  f"omit_filler_t1={fmt(r['omit_acc_t1_filler'])} "
                  f"margin_m2={fmt(r['margin_m2'])} "
                  f"shuf_m2={fmt(r['omit_acc_shuf_m2'])} "
                  f"n_m2={r['n_m2']} n_m1={r['n_m1']} "
                  f"n_filler={r['n_filler']} "
                  f"nonfinite={bad}")
            assert not bad, f"non-finite metrics: {bad}"
    print("SMOKE DONE")


if __name__ == "__main__":
    main()
