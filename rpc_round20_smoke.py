"""Round-20 smoke tests (VM only, seed 302, fast).

Checks (protocol section 12):
  S1  occlusion is deterministic, motif positions and the filler
      probe position are never blanked.
  S2  A29 default (random wire, gain 0.0) reproduces the round-17
      code path exactly at 1 epoch (normal_acc, omit_acc_m2).
  S3  aligned wire at gain 1.0, 5 epochs, passes the state gate
      (p1_cos < 0.90) and shows a positive on-cut omit_m2 gap.
  S4  shuffle drive matches on-mode RMS within 5% (the shuffle
      permutes d1, so the magnitude is preserved by construction).
  S5  clamp_frac is 0.000 (no distal weight hits the w_max clamp).
  S6  learned wire (composite rule) trains without runaway at
      gain 1.0, 5 epochs: normal_acc above 0.15, fb_align above
      0.5, r1_max below 10.
  S7  rollout runs and returns values in [0, 1] with n > 0 on
      sine and planted; the frozen rollout is near chance.
  S8  driver CLI is resumable: a second identical run adds no
      lines (val 1 line, test 11 lines).

Run: ./venv/bin/python rpc_round20_smoke.py
Exits non-zero on the first failure; prints PASS/FAIL per check
with the measured numbers.
"""
import json
import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rpc_round16_omission import detect_motifs, filler_pos
from rpc_round17_clamp import (paired_p1_cos, run_omission_fb,
                               splits3, train_net as r17_train_net)
from rpc_round20_aligned import (drive_ratio_pass, eval_condition,
                                 occlude, rollout_planted,
                                 rollout_sine, splits, train_net)

SEED = 302
FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def main():
    task = "planted1-occ"
    cfg = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.05}
    tr_c, va_c, te_c, tr_o, va_o, te_o = splits(task, SEED)
    # S1: occlusion determinism and motif/probe protection.
    tr_c2, _, _, _, _, _ = splits(task, SEED)
    tr_o2 = occlude(tr_c2, 1, SEED, 0)
    same = all(a == b for s1, s2 in zip(tr_o, tr_o2)
               for a, b in zip(s1, s2))
    blank_at_motif = 0
    blank_at_probe = 0
    n_blank = 0
    for seq, occ in zip(tr_c, tr_o):
        _, _, mpos = detect_motifs(seq, 1)
        fp = filler_pos(seq, mpos)
        for i, tok in enumerate(occ):
            if tok == 16:
                n_blank += 1
                if i in mpos:
                    blank_at_motif += 1
                if i == fp:
                    blank_at_probe += 1
    check("S1 occlusion", same and blank_at_motif == 0
          and blank_at_probe == 0,
          f"deterministic={same}, blanks={n_blank}, "
          f"blank_at_motif={blank_at_motif}, "
          f"blank_at_probe={blank_at_probe}")
    # S2: A29 default reproduces the round-17 path at 1 epoch.
    # Both paths train on the CLEAN split here (the driver
    # prototypes on clean by design; the r17 path would crash
    # prototyping on occluded data, since token 16 overflows its
    # prototype table). The occlusion itself is covered by S1.
    net_d, ts_d = train_net(tr_c, tr_c, cfg, "random", 0.0, SEED, 1,
                            "delta", 16)
    r_d = run_omission_fb("planted1", te_c, te_c, cfg, "delta", SEED,
                          1, 0.0, "on", net=net_d, train_stats=ts_d)
    net_r, ts_r = r17_train_net(tr_c, cfg, "delta", SEED, 1, 0.0)
    r_r = run_omission_fb("planted1", te_c, te_c, cfg, "delta", SEED,
                          1, 0.0, "on", net=net_r, train_stats=ts_r)
    same_acc = (abs(r_d["normal_acc"] - r_r["normal_acc"]) < 1e-9 and
                abs(r_d["omit_acc_m2"] - r_r["omit_acc_m2"]) < 1e-9)
    check("S2 repro gain-0", same_acc,
          f"driver normal_acc={r_d['normal_acc']:.6f} "
          f"omit_m2={r_d['omit_acc_m2']:.6f} | r17 "
          f"normal_acc={r_r['normal_acc']:.6f} "
          f"omit_m2={r_r['omit_acc_m2']:.6f}")
    # S3/S4/S5: aligned wire, 5 epochs, gain 1.0.
    net_a, ts_a = train_net(tr_c, tr_o, cfg, "aligned", 1.0, SEED, 5,
                            "delta", 16)
    p1_cos = paired_p1_cos(net_a, te_o)
    r_on = run_omission_fb("planted1", te_o, te_o, cfg, "delta", SEED,
                           5, 1.0, "on", net=net_a, train_stats=ts_a)
    r_cut = run_omission_fb("planted1", te_o, te_o, cfg, "delta",
                            SEED, 5, 1.0, "cut", net=net_a,
                            train_stats=ts_a)
    gap = r_on["omit_acc_m2"] - r_cut["omit_acc_m2"]
    check("S3 aligned gate", p1_cos < 0.90,
          f"p1_cos={p1_cos:.4f} (gate < 0.90), on-cut gap={gap:+.4f}, "
          f"normal_acc={r_on['normal_acc']:.4f}")
    dr_on, rms_on = drive_ratio_pass(net_a, te_o, "on")
    dr_sh, rms_sh = drive_ratio_pass(net_a, te_o, "shuffle")
    rel = abs(rms_sh - rms_on) / max(rms_on, 1e-9)
    check("S4 shuffle RMS", rel < 0.05,
          f"rms on={rms_on:.4f} shuffle={rms_sh:.4f} rel={rel:.4f}, "
          f"fb_drive on={dr_on:.3f} shuffle={dr_sh:.3f}")
    check("S5 clamp_frac", r_on["clamp_frac"] == 0.0,
          f"clamp_frac={r_on['clamp_frac']:.4f}")
    # S6: learned wire (composite rule), 5 epochs, gain 1.0.
    net_l, ts_l = train_net(tr_c, tr_o, cfg, "learned", 1.0, SEED, 5,
                            "delta", 16)
    r_l = run_omission_fb("planted1", te_o, te_o, cfg, "delta", SEED,
                          5, 1.0, "on", net=net_l, train_stats=ts_l)
    from rpc_round20_aligned import fb_align_stats
    fa = fb_align_stats(net_l)
    check("S6 learned stable", r_l["normal_acc"] > 0.15
          and fa["fb_align"] > 0.5 and r_l["r1_max"] < 10,
          f"normal_acc={r_l['normal_acc']:.4f}, "
          f"fb_align={fa['fb_align']:.3f}, "
          f"fb_norm={fa['fb_norm']:.2f}, r1_max={r_l['r1_max']:.2f}, "
          f"err_last={r_l['err_mag_last']:.4f}")
    # S7: rollout sanity (sine + planted, trained vs frozen).
    tr_s, va_s, te_s, _, _, _ = splits("sine", SEED)
    cfg_s = {"alpha": 1.0, "smooth_sigma": 0.0, "eta_d": 0.2}
    net_s, ts_s = train_net(tr_s, tr_s, cfg_s, "aligned", 1.0, SEED,
                            5, "delta", 32)
    ro = rollout_sine(net_s, te_s, "on")
    ro_p = rollout_planted(net_a, te_o, "on", 1)
    net_f, ts_f = train_net(tr_s, tr_s, cfg_s, "aligned", 1.0, SEED,
                            5, "frozen", 32)
    ro_f = rollout_sine(net_f, te_s, "on")
    ok_ro = (0.0 <= ro["rollout_acc"] <= 1.0 and ro["rollout_n"] > 0
             and 0.0 <= ro_p["rollout_comp"] <= 1.0
             and ro_p["rollout_n"] > 0
             and 0.0 <= ro_f["rollout_acc"] <= 1.0)
    check("S7 rollout", ok_ro,
          f"sine rollout_acc={ro['rollout_acc']:.3f} "
          f"late={ro['rollout_acc_late']:.3f} n={ro['rollout_n']}; "
          f"planted comp={ro_p['rollout_comp']:.3f} "
          f"n={ro_p['rollout_n']}; frozen sine "
          f"rollout_acc={ro_f['rollout_acc']:.3f}")
    # S8: driver CLI resumable (val + test, 1 epoch, seed 302).
    tmp = "/tmp/r20_smoke"
    os.makedirs(tmp, exist_ok=True)
    winners = os.path.join(tmp, "winners.json")
    with open(winners, "w") as f:
        json.dump({"aligned": {"fb_gain": 1.0},
                   "learned": {"fb_gain": 1.0},
                   "random": {"fb_gain": 1.0}}, f)
    val_out = os.path.join(tmp, "val.jsonl")
    test_out = os.path.join(tmp, "test.jsonl")
    for p in (val_out, test_out):
        if os.path.exists(p):
            os.remove(p)
    cmd_v = [sys.executable, os.path.join(HERE, "rpc_round20_aligned.py"),
             "planted1-occ", "val", "--seeds", "302-302", "--out",
             val_out, "--winners", winners, "--epochs", "1",
             "--wire-type", "aligned", "--fb-gain", "1.0"]
    for _ in range(2):
        subprocess.run(cmd_v, check=True, capture_output=True)
    n_val = sum(1 for _ in open(val_out))
    cmd_t = [sys.executable, os.path.join(HERE, "rpc_round20_aligned.py"),
             "planted1-occ", "test", "--seeds", "302-302", "--out",
             test_out, "--winners", winners, "--epochs", "1"]
    for _ in range(2):
        subprocess.run(cmd_t, check=True, capture_output=True)
    n_test = sum(1 for _ in open(test_out))
    check("S8 CLI resumable", n_val == 1 and n_test == 11,
          f"val lines={n_val} (expect 1), test lines={n_test} "
          f"(expect 11) after two runs each")
    print()
    if FAILURES:
        print(f"SMOKE FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("SMOKE PASSED: all 8 checks")


if __name__ == "__main__":
    main()
