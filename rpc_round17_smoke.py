"""Round 17 smoke test (VM only, never a result).

Runs 1 seed (241), 1 epoch, planted lag-1, all five conditions,
and checks (protocol section 7):
1. fb_gain=0.0 reproduces the round-16 config exactly: the
   wire-off run matches the round-16 driver's output on the
   same seed, 1 epoch (normal accuracy and omit_m2).
2. Wire-on changes p1: p1_cos (on vs cut) is below 1.0 and
   above 0.0.
3. All metrics finite; no NaN; r1_max bounded (< 10).
4. The shuffle mode's fb RMS matches the on-mode fb RMS within
   5% (same magnitude, different content).

Usage: python rpc_round17_smoke.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rpc_round16_omission import run_omission, splits
from rpc_round17_clamp import (run_omission_fb, splits3, train_net)

SEED = 241
EPOCHS = 1
FB_GAIN = 0.3


def check_finite(rec, label):
    bad = [k for k, v in rec.items()
           if isinstance(v, float) and not math.isfinite(v)]
    assert not bad, f"{label}: nonfinite {bad}"
    print(f"{label}: finite OK")


def main():
    train, val, test = splits3("planted1", SEED)
    cfg = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.05}
    # 1. fb_gain=0.0 reproduces round 16 exactly.
    train16, test16 = splits("planted1", SEED)
    assert [list(map(int, s)) for s in train16] == \
        [list(map(int, s)) for s in train], "splits differ"
    r16 = run_omission("planted1", train16, test16, cfg, "delta",
                       SEED, EPOCHS)
    r17off = run_omission_fb("planted1", train, test, cfg, "delta",
                             SEED, EPOCHS, 0.0, "on")
    for k in ("normal_acc_m2", "omit_acc_m2", "omit_acc_m1",
              "omit_acc_cont_m1"):
        a, b = r16[k], r17off[k]
        assert a is not None and b is not None, k
        assert abs(a - b) < 1e-9, f"{k}: round16 {a} vs r17-off {b}"
    print(f"check 1 PASSED: fb_gain=0.0 reproduces round 16 "
          f"(normal_acc_m2={r16['normal_acc_m2']:.3f} "
          f"omit_m2={r16['omit_acc_m2']:.3f})")
    # 2-4. Wire-on net, all three test modes, 1 epoch.
    net, ts = train_net(train, cfg, "delta", SEED, EPOCHS, FB_GAIN)
    from rpc_round17_clamp import paired_p1_cos
    p1_cos = paired_p1_cos(net, test)
    assert 0.0 < p1_cos < 1.0, f"p1_cos={p1_cos}"
    print(f"check 2 PASSED: p1_cos (on vs cut) = {p1_cos:.3f}")
    recs = {}
    for mode in ("on", "cut", "shuffle"):
        recs[mode] = run_omission_fb(
            "planted1", train, test, cfg, "delta", SEED, EPOCHS,
            FB_GAIN, mode, net=net, train_stats=ts)
        recs[mode]["p1_cos"] = p1_cos
        check_finite(recs[mode], f"wire-{mode}")
    assert recs["on"]["r1_max"] < 10, recs["on"]["r1_max"]
    print(f"check 3 PASSED: r1_max={recs['on']['r1_max']:.2f} "
          f"clamp_frac={recs['on']['clamp_frac']:.4f} "
          f"err {ts['err_mag_by_epoch'][0]:.3f}")
    fb_on = recs["on"]["fb_rms_ratio"]
    fb_sh = recs["shuffle"]["fb_rms_ratio"]
    assert fb_on > 0, "fb_rms_ratio is 0 with the wire on"
    assert abs(fb_sh - fb_on) / fb_on < 0.05, (fb_on, fb_sh)
    print(f"check 4 PASSED: fb_rms on={fb_on:.3f} "
          f"shuffle={fb_sh:.3f} cut={recs['cut']['fb_rms_ratio']:.3f}")
    for mode, r in recs.items():
        print(f"wire-{mode} 1ep seed241: "
              f"omit_m2={r['omit_acc_m2']:.3f} "
              f"omit_m1={r['omit_acc_m1']:.3f} "
              f"cont_m1={r['omit_acc_cont_m1']:.3f} "
              f"normal_acc={r['normal_acc']:.3f} "
              f"fb_rms={r['fb_rms_ratio']:.3f}")
    print("SMOKE DONE")


if __name__ == "__main__":
    main()
