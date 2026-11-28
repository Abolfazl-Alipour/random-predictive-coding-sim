#!/usr/bin/env python3
"""Independent recomputation of round-20 headline numbers from raw JSONL.

Written from scratch for verification; does not import r20_analyze.py.
CI = mean +/- t * s / sqrt(n), sample std (ddof=1).
t(19) = 2.0930 (test, n=20); t(4) = 2.7760 (val, n=5).
"""
import json, glob, math, statistics

T19 = 2.0930
T4 = 2.7760

def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            rows.append(json.loads(line))
    return rows

def ci(vals, t):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    h = t * s / math.sqrt(len(vals))
    return m, m - h, m + h

def show(label, vals, t, expect=None):
    m, lo, hi = ci(vals, t)
    flag = ""
    if expect is not None:
        ok = abs(m - expect[0]) < 0.0005 and abs(lo - expect[1]) < 0.0005 and abs(hi - expect[2]) < 0.0005
        flag = " MATCH" if ok else f"  *** MISMATCH vs {expect}"
    print(f"{label}: mean={m:.3f} CI=[{lo:.3f},{hi:.3f}] n={len(vals)}{flag}")
    return m, lo, hi

# ---------- val ----------
val = load("r20_planted1-occ_val_*.jsonl")
print(f"val lines: {len(val)}, unique (task,config,seed): "
      f"{len({(r['task'], r['config'], r['seed']) for r in val})}")
for wire in ("aligned", "learned", "random"):
    for g in (0.1, 0.3, 1.0):
        rs = [r for r in val if r["wire_type"] == wire and r["fb_gain"] == g]
        assert len(rs) == 5, (wire, g, len(rs))
        show(f"val {wire} g{g} gap", [r["val_gap_on_cut"] for r in rs], T4)
        show(f"val {wire} g{g} p1_cos", [r["val_p1_cos"] for r in rs], T4)
        show(f"val {wire} g{g} acc", [r["val_acc"] for r in rs], T4)

# ---------- test ----------
test = load("r20_*_test_302-321.jsonl")
print(f"test lines: {len(test)}, unique (task,config,seed): "
      f"{len({(r['task'], r['config'], r['seed']) for r in test})}")

def by(task, config):
    return {r["seed"]: r for r in test if r["task"] == task and r["config"] == config}

def field(rows, k):
    return [r[k] for r in rows.values() if r.get(k) is not None]

def pdiff(task, c1, c2, k):
    a, b = by(task, c1), by(task, c2)
    assert set(a) == set(b) and len(a) == 20
    return [a[s][k] - b[s][k] for s in sorted(a)]

def cond_mean(task, config, k, expect=None):
    rows = by(task, config)
    return show(f"{task} {config} {k}", field(rows, k), T19, expect)

# condition means (spot-check headline values)
cond_mean("planted1-occ", "aligned-on", "omit_acc_m2", (0.993, 0.990, 0.996))
cond_mean("planted1-occ", "aligned-cut", "omit_acc_m2", (0.979, 0.971, 0.988))
cond_mean("planted1-occ", "aligned-on", "normal_acc", (0.232, 0.227, 0.237))
cond_mean("planted1-occ", "wire-off", "normal_acc", (0.221, 0.215, 0.227))
cond_mean("planted1-occ", "aligned-on", "p1_cos", (0.804, 0.797, 0.811))
cond_mean("planted1-occ", "random-on", "omit_acc_m2", (0.873, 0.845, 0.902))
cond_mean("planted2-occ", "aligned-on", "omit_acc_m2", (0.961, 0.948, 0.975))
cond_mean("sine", "aligned-on", "normal_acc", (0.666, 0.640, 0.691))
cond_mean("sine", "wire-off", "normal_acc", (0.892, 0.881, 0.904))
cond_mean("sine", "aligned-on", "rollout_acc", (0.147, 0.136, 0.158))
cond_mean("sine", "aligned-cut", "rollout_acc", (0.184, 0.167, 0.200))
cond_mean("sine", "aligned-on", "rollout_acc_late", (0.051, 0.038, 0.064))
cond_mean("sine", "frozen-aligned", "rollout_acc", (0.032, 0.025, 0.040))

# ---------- bars B' ----------
for lag in ("planted1-occ", "planted2-occ"):
    show(f"B1 {lag} aligned on-cut omit_m2",
         pdiff(lag, "aligned-on", "aligned-cut", "omit_acc_m2"), T19,
         (0.014, 0.005, 0.022) if lag == "planted1-occ" else (-0.006, -0.016, 0.004))
    show(f"B2 {lag} aligned on-shuffle omit_m2",
         pdiff(lag, "aligned-on", "aligned-shuffle", "omit_acc_m2"), T19,
         (0.033, 0.022, 0.044) if lag == "planted1-occ" else (0.015, 0.000, 0.030))
    show(f"B4 {lag} aligned on-off omit_m2",
         pdiff(lag, "aligned-on", "wire-off", "omit_acc_m2"), T19,
         (0.009, 0.003, 0.015) if lag == "planted1-occ" else (-0.009, -0.019, 0.000))
    show(f"B5 {lag} aligned err_mag_last on-off",
         pdiff(lag, "aligned-on", "wire-off", "err_mag_last"), T19,
         (0.0138, 0.008, 0.020) if lag == "planted1-occ" else (0.0181, 0.013, 0.023))
    show(f"B6 {lag} aligned cos_p1_omit_m2 on-cut",
         pdiff(lag, "aligned-on", "aligned-cut", "cos_p1_omit_m2"), T19,
         (-0.391, -0.408, -0.374) if lag == "planted1-occ" else (-0.381, -0.402, -0.360))
    show(f"B6 {lag} cos_p1_omit_m2 on mean",
         field(by(lag, "aligned-on"), "cos_p1_omit_m2"), T19,
         (0.033, 0.002, 0.063) if lag == "planted1-occ" else (-0.038, -0.075, -0.001))
    show(f"B9 {lag} learned on-cut omit_m2",
         pdiff(lag, "learned-on", "learned-cut", "omit_acc_m2"), T19,
         (0.004, -0.000, 0.009) if lag == "planted1-occ" else (-0.027, -0.042, -0.013))
    # B8: aligned gap minus random gap, per seed
    ag = pdiff(lag, "aligned-on", "aligned-cut", "omit_acc_m2")
    rg = pdiff(lag, "random-on", "random-cut", "omit_acc_m2")
    show(f"B8 {lag} aligned gap minus random gap",
         [a - b for a, b in zip(ag, rg)], T19,
         (0.071, 0.048, 0.093) if lag == "planted1-occ" else (0.049, 0.028, 0.071))
    # B10 diagnostic
    lg = pdiff(lag, "learned-on", "learned-cut", "omit_acc_m2")
    show(f"B10 {lag} learned gap minus aligned gap",
         [a - b for a, b in zip(lg, ag)], T19,
         (-0.009, -0.018, -0.001) if lag == "planted1-occ" else (-0.021, -0.032, -0.010))
    # B3 p1_cos
    show(f"B3 {lag} aligned p1_cos mean",
         field(by(lag, "aligned-on"), "p1_cos"), T19,
         (0.804, 0.797, 0.811) if lag == "planted1-occ" else (0.810, 0.802, 0.817))

# ---------- bars D ----------
show("D1 sine rollout aligned on-cut",
     pdiff("sine", "aligned-on", "aligned-cut", "rollout_acc"), T19, (-0.037, -0.051, -0.022))
show("D2 sine rollout aligned on-shuffle",
     pdiff("sine", "aligned-on", "aligned-shuffle", "rollout_acc"), T19, (-0.025, -0.037, -0.014))
show("D3 sine rollout aligned on-off",
     pdiff("sine", "aligned-on", "wire-off", "rollout_acc"), T19, (-0.076, -0.092, -0.061))
show("D7 sine rollout learned on-cut",
     pdiff("sine", "learned-on", "learned-cut", "rollout_acc"), T19, (-0.035, -0.047, -0.022))
for lag in ("planted1-occ", "planted2-occ"):
    show(f"D5 {lag} rollout_comp aligned on-cut",
         pdiff(lag, "aligned-on", "aligned-cut", "rollout_comp"), T19,
         (-0.146, -0.190, -0.102) if lag == "planted1-occ" else (-0.225, -0.265, -0.185))
    show(f"D6 {lag} rollout_comp aligned on-shuffle",
         pdiff(lag, "aligned-on", "aligned-shuffle", "rollout_comp"), T19,
         (-0.109, -0.157, -0.061) if lag == "planted1-occ" else (-0.155, -0.204, -0.106))

# diagnostics: fb_align / fb_norm means
for task in ("planted1-occ", "planted2-occ", "sine"):
    for wire in ("aligned", "learned", "random"):
        show(f"diag {task} {wire}-on fb_align",
             field(by(task, f"{wire}-on"), "fb_align"), T19)
        show(f"diag {task} {wire}-on fb_norm",
             field(by(task, f"{wire}-on"), "fb_norm"), T19)
print("DONE")
