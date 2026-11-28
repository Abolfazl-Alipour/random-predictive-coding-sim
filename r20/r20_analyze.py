"""Round-20 analysis: aligned-feedback clamp (revised experiment B)
+ multi-step omission rollout (experiment D).

Protocol: PROTOCOL-R20-ALIGNED.md. Seeds 302..321 (20 fresh).
Reads r20/r20_{task}_test_302-321.jsonl and the val files,
computes per-task / per-condition means, medians, IQRs, EXACT
Student-t 95% CIs, paired differences, and checks every protocol
bar. Writes r20/r20_analysis.md and prints a summary.

CI convention (protocol section 9, correcting rounds 17/18):
exact Student-t, NOT z=1.96. t(19) = 2.0930 for the 20-seed test
bars; t(4) = 2.7760 for 5-seed val summaries. scipy is absent
from the venv, so the constants are hardcoded from the standard
t table. None values are structurally empty measures (e.g.
omission fields on sine) and are skipped, never treated as zero.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["planted1-occ", "planted2-occ", "sine"]
LABELS = {"planted1-occ": "Planted lag-1, 40% occluded",
          "planted2-occ": "Planted lag-2, 40% occluded",
          "sine": "Sine-fixed"}
CONDITIONS = ["aligned-on", "aligned-cut", "aligned-shuffle",
              "learned-on", "learned-cut", "learned-shuffle",
              "random-on", "random-cut", "random-shuffle",
              "wire-off", "frozen-aligned"]
T19 = 2.0930  # exact Student-t, 19 df, two-sided 95%.
T4 = 2.7760   # exact Student-t, 4 df, two-sided 95%.
CHANCE_PLANTED = 1.0 / 16.0
CHANCE_SINE = 1.0 / 32.0


def ci(vals, t=T19):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    half = t * s / math.sqrt(len(v))
    return m, m - half, m + half


def stats(vals, t=T19):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return {"mean": None, "median": None, "iqr": [None, None],
                "ci_lo": None, "ci_hi": None, "n": 0}
    m, lo, hi = ci(vals, t)
    return {"mean": float(m), "median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)),
                    float(np.percentile(v, 75))],
            "ci_lo": float(lo), "ci_hi": float(hi), "n": len(v)}


def fmt(s):
    if s["mean"] is None:
        return "n/a (structurally empty)"
    return (f"mean={s['mean']:.3f} median={s['median']:.3f} "
            f"IQR=[{s['iqr'][0]:.3f},{s['iqr'][1]:.3f}] "
            f"95% CI=[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] n={s['n']}")


def paired(vals_a, vals_b):
    """Paired per-seed differences, skipping seeds where either
    side is None. Returns (mean, ci_lo, ci_hi, n) with exact t."""
    d = [a - b for a, b in zip(vals_a, vals_b)
         if a is not None and b is not None]
    m, lo, hi = ci(d)
    return m, lo, hi, len(d)


def gap_vec(by, task, cond_on, cond_cut, field):
    """Per-seed (on - cut) gap dict, keyed by seed. Pairing by
    seed (not by list position) so a missing value on one side
    never misaligns the other side."""
    ro = by.get((task, cond_on), {})
    rc = by.get((task, cond_cut), {})
    out = {}
    for s in sorted(set(ro) & set(rc)):
        a, b = ro[s].get(field), rc[s].get(field)
        if a is not None and b is not None:
            out[s] = a - b
    return out


def paired_dict(d1, d2):
    """Paired difference of two seed-keyed dicts."""
    keys = sorted(set(d1) & set(d2))
    return paired([d1[k] for k in keys], [d2[k] for k in keys])


def load_test():
    recs = []
    for task in TASKS:
        p = os.path.join(HERE, f"r20_{task}_test_302-321.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def load_val():
    recs = []
    for p in glob.glob(os.path.join(HERE, "r20_*_val_*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def check_bar(name, cond, detail):
    return {"name": name, "pass": bool(cond), "detail": detail}


def fci(ci_tuple):
    return f"[{ci_tuple[1]:.3f},{ci_tuple[2]:.3f}]"


def main():
    recs = load_test()
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    keys = {(r["task"], r["config"], r["seed"]) for r in recs}
    lines = ["# Round 20 analysis: aligned-feedback clamp + rollout",
             "",
             "Protocol: PROTOCOL-R20-ALIGNED.md. Seeds 302..321 "
             "(20 fresh seeds). Chance: planted 1/16 = 0.0625, "
             "sine 1/32 = 0.03125.",
             "CI = mean +/- exact Student-t * sample-std / sqrt(n): "
             "t(19) = 2.0930 for test bars, t(4) = 2.7760 for val.",
             f"Test lines: {len(recs)} (expected 660), "
             f"unique keys: {len(keys)}.",
             ""]
    # Validation: winner pick per wire type.
    val = load_val()
    vg = {}
    for r in val:
        vg.setdefault((r["wire_type"], r["fb_gain"]), []).append(r)
    lines.append("## Validation (planted1-occ, val seeds 302..306)")
    lines.append("")
    lines.append("Winner rule (protocol section 7): largest val "
                 "on-cut omit_m2 gap among gains passing the state "
                 "gate (val p1_cos < 0.90) and stable (r1_max < 10, "
                 "no NaN); tie-break smallest gain. If no gain "
                 "passes the gate, the largest gap among stable "
                 "gains wins and the gate failure is flagged. The "
                 "val-phase guard (val normal_acc >= 0.80x wire-off) "
                 "is computed from the test phase (B7); the val "
                 "phase has no wire-off condition.")
    winners = {}
    for wt in ("aligned", "learned", "random"):
        lines.append(f"### Wire {wt}")
        for g in (0.1, 0.3, 1.0):
            rows = vg.get((wt, g), [])
            if not rows:
                lines.append(f"- fb_gain={g}: no val rows")
                continue
            s_acc = stats([r["val_acc"] for r in rows], t=T4)
            s_gap = stats([r["val_gap_on_cut"] for r in rows], t=T4)
            s_cos = stats([r["val_p1_cos"] for r in rows], t=T4)
            lines.append(f"- fb_gain={g}: val_acc {fmt(s_acc)}; "
                         f"val on-cut gap {fmt(s_gap)}; "
                         f"val p1_cos {fmt(s_cos)}")
        # Winner: largest val gap among gate-passing gains
        # (p1_cos < 0.90, stable, val_acc >= 0.80x off); fallback
        # is the largest gap among stable gains, flagged.
        cands = []
        for g in (0.1, 0.3, 1.0):
            rows = vg.get((wt, g), [])
            if len(rows) != 5:
                continue
            gap = float(np.mean([r["val_gap_on_cut"] for r in rows]))
            cos = float(np.mean([r["val_p1_cos"] for r in rows]))
            acc = float(np.mean([r["val_acc"] for r in rows]))
            rmax = max(r["r1_max"] for r in rows)
            cands.append((g, gap, cos, acc, rmax))
        gated = [c for c in cands if c[2] < 0.90 and c[4] < 10.0]
        pool = gated if gated else [c for c in cands if c[4] < 10.0]
        pool = pool if pool else cands
        # Largest gap; tie-break smallest gain.
        pool.sort(key=lambda c: (-c[1], c[0]))
        if pool:
            g, gap, cos, acc, rmax = pool[0]
            winners[wt] = {"fb_gain": g, "val_gap": gap,
                           "val_p1_cos": cos, "val_acc": acc,
                           "gate": cos < 0.90}
            lines.append(f"- winner {wt}: fb_gain={g} "
                         f"(val gap {gap:+.3f}, p1_cos {cos:.3f}, "
                         f"val_acc {acc:.3f}, "
                         f"gate {'PASS' if cos < 0.90 else 'FAIL'})")
        lines.append("")
    lines.append("## Test: condition means")
    lines.append("")
    for task in TASKS:
        lines.append(f"### {LABELS[task]}")
        lines.append("")
        for cond in CONDITIONS:
            rows = by.get((task, cond), {})
            if not rows:
                lines.append(f"- {cond}: no rows")
                continue
            seeds = sorted(rows)
            vals = lambda f: [rows[s].get(f) for s in seeds]
            s_acc = stats(vals("normal_acc"))
            s_m2 = stats(vals("omit_acc_m2"))
            s_m1 = stats(vals("omit_acc_m1"))
            s_ro = stats(vals("rollout_acc"))
            s_rc = stats(vals("rollout_comp"))
            s_cos = stats(vals("p1_cos"))
            s_dr = stats(vals("fb_drive_ratio"))
            lines.append(f"- {cond}: normal_acc {fmt(s_acc)}; "
                         f"omit_m2 {fmt(s_m2)}; omit_m1 {fmt(s_m1)}; "
                         f"rollout_acc {fmt(s_ro)}; "
                         f"rollout_comp {fmt(s_rc)}; "
                         f"p1_cos {fmt(s_cos)}; "
                         f"fb_drive_ratio {fmt(s_dr)}")
        lines.append("")
    # Bars. Helper to pull per-seed vectors.
    def vec(task, cond, field):
        rows = by.get((task, cond), {})
        return [rows[s].get(field) for s in sorted(rows)]

    def bar_gap(task, a, b, field, floor):
        m, lo, hi, n = paired(vec(task, a, field), vec(task, b, field))
        return m, lo, hi, n, (m >= floor and lo > 0)

    verdicts = []
    lines.append("## Bars: experiment B' (planted, occluded)")
    lines.append("")
    for task in ("planted1-occ", "planted2-occ"):
        lag = "lag-1" if task == "planted1-occ" else "lag-2"
        lines.append(f"### {lag} ({task})")
        m, lo, hi, n, ok = bar_gap(task, "aligned-on", "aligned-cut",
                                   "omit_acc_m2", 0.05)
        verdicts.append(check_bar(
            f"B1-{lag}", ok,
            f"aligned on-cut omit_m2 {m:+.3f} CI {fci((m, lo, hi))} "
            f"(bar >= +0.05, lower CI > 0)"))
        m2, lo2, hi2, n2, ok2 = bar_gap(
            task, "aligned-on", "aligned-shuffle", "omit_acc_m2", 0.05)
        verdicts.append(check_bar(
            f"B2-{lag}", ok2,
            f"aligned on-shuffle omit_m2 {m2:+.3f} CI {fci((m2, lo2, hi2))} "
            f"(bar >= +0.05, lower CI > 0)"))
        s_cos = stats(vec(task, "aligned-on", "p1_cos"))
        ok3 = s_cos["mean"] is not None and s_cos["mean"] < 0.90
        verdicts.append(check_bar(
            f"B3-{lag}", ok3,
            f"aligned p1_cos {fmt(s_cos)} (bar mean < 0.90)"))
        m4, lo4, hi4, _ = paired(vec(task, "aligned-on", "omit_acc_m2"),
                                 vec(task, "wire-off", "omit_acc_m2"))
        ok4 = lo4 > 0 or hi4 < 0
        verdicts.append(check_bar(
            f"B4-{lag}", ok4,
            f"aligned on-off omit_m2 {m4:+.3f} CI {fci((m4, lo4, hi4))} "
            f"(bar CI excludes 0)"))
        m5, lo5, hi5, _ = paired(vec(task, "aligned-on", "err_mag_last"),
                                 vec(task, "wire-off", "err_mag_last"))
        ok5 = m5 < 0 and hi5 < 0
        verdicts.append(check_bar(
            f"B5-{lag}", ok5,
            f"aligned err_mag_last on-off {m5:+.4f} CI {fci((m5, lo5, hi5))} "
            f"(bar mean < 0, upper CI < 0: explaining away)"))
        m6, lo6, hi6, _ = paired(vec(task, "aligned-on",
                                      "cos_p1_omit_m2"),
                                 vec(task, "aligned-cut",
                                      "cos_p1_omit_m2"))
        s6 = stats(vec(task, "aligned-on", "cos_p1_omit_m2"))
        ok6 = (m6 < 0 and hi6 < 0 and s6["mean"] is not None
               and s6["mean"] < 0.20)
        verdicts.append(check_bar(
            f"B6-{lag}", ok6,
            f"aligned cos_p1_omit_m2 on-cut {m6:+.3f} CI "
            f"{fci((m6, lo6, hi6))}; on mean {fmt(s6)} "
            f"(bar mean < 0 with upper CI < 0, on mean < 0.20)"))
        s7a = stats(vec(task, "aligned-on", "normal_acc"))
        s7b = stats(vec(task, "wire-off", "normal_acc"))
        ok7 = (s7a["mean"] is not None and s7b["mean"] is not None
               and s7a["mean"] >= 0.80 * s7b["mean"])
        verdicts.append(check_bar(
            f"B7-{lag} (guard)", ok7,
            f"aligned normal_acc {fmt(s7a)} vs wire-off {fmt(s7b)} "
            f"(guard >= 0.80x)"))
        # B8: aligned gap exceeds random gap (paired by seed).
        ga = gap_vec(by, task, "aligned-on", "aligned-cut",
                     "omit_acc_m2")
        gr = gap_vec(by, task, "random-on", "random-cut",
                     "omit_acc_m2")
        m8, lo8, hi8, n8 = paired_dict(ga, gr)
        ok8 = m8 >= 0.05 and lo8 > 0
        verdicts.append(check_bar(
            f"B8-{lag}", ok8,
            f"aligned gap minus random gap {m8:+.3f} CI "
            f"{fci((m8, lo8, hi8))} (bar >= +0.05, lower CI > 0)"))
        m9, lo9, hi9, _, ok9 = bar_gap(task, "learned-on", "learned-cut",
                                       "omit_acc_m2", 0.05)
        verdicts.append(check_bar(
            f"B9-{lag}", ok9,
            f"learned on-cut omit_m2 {m9:+.3f} CI {fci((m9, lo9, hi9))} "
            f"(bar >= +0.05, lower CI > 0)"))
        gl = gap_vec(by, task, "learned-on", "learned-cut",
                     "omit_acc_m2")
        m10, lo10, hi10, _ = paired_dict(gl, ga)
        lines.append(f"- B10-{lag} (diagnostic, not a bar): learned gap "
                     f"minus aligned gap {m10:+.3f} CI "
                     f"{fci((m10, lo10, hi10))}")
        lines.append("")
    lines.append("## Bars: experiment D (rollout)")
    lines.append("")
    # Sine rollout.
    m, lo, hi, _, ok = bar_gap("sine", "aligned-on", "aligned-cut",
                               "rollout_acc", 0.10)
    verdicts.append(check_bar(
        "D1", ok, f"sine rollout acc aligned on-cut {m:+.3f} CI "
        f"{fci((m, lo, hi))} (bar >= +0.10, lower CI > 0)"))
    m, lo, hi, _, ok = bar_gap("sine", "aligned-on", "aligned-shuffle",
                               "rollout_acc", 0.10)
    verdicts.append(check_bar(
        "D2", ok, f"sine rollout acc aligned on-shuffle {m:+.3f} CI "
        f"{fci((m, lo, hi))} (bar >= +0.10, lower CI > 0)"))
    m, lo, hi, _, ok = bar_gap("sine", "aligned-on", "wire-off",
                               "rollout_acc", 0.10)
    verdicts.append(check_bar(
        "D3", ok, f"sine rollout acc aligned on-off {m:+.3f} CI "
        f"{fci((m, lo, hi))} (bar >= +0.10, lower CI > 0)"))
    s = stats(vec("sine", "aligned-on", "rollout_acc_late"))
    ok4 = s["mean"] is not None and s["mean"] >= 0.30 and s["ci_lo"] > 0.20
    verdicts.append(check_bar(
        "D4", ok4, f"sine rollout late acc (on) {fmt(s)} "
        f"(bar mean >= 0.30, lower CI > 0.20; chance 0.031)"))
    # Planted rollout (lag-1 primary, lag-2 reported).
    for task in ("planted1-occ", "planted2-occ"):
        lag = "lag-1" if task == "planted1-occ" else "lag-2"
        m, lo, hi, _, ok = bar_gap(task, "aligned-on", "aligned-cut",
                                   "rollout_comp", 0.20)
        verdicts.append(check_bar(
            f"D5-{lag}", ok,
            f"{lag} rollout completion aligned on-cut {m:+.3f} CI "
            f"{fci((m, lo, hi))} (bar >= +0.20, lower CI > 0)"))
        m, lo, hi, _, ok = bar_gap(task, "aligned-on", "aligned-shuffle",
                                   "rollout_comp", 0.20)
        verdicts.append(check_bar(
            f"D6-{lag}", ok,
            f"{lag} rollout completion aligned on-shuffle {m:+.3f} CI "
            f"{fci((m, lo, hi))} (bar >= +0.20, lower CI > 0)"))
    m, lo, hi, _, ok = bar_gap("sine", "learned-on", "learned-cut",
                               "rollout_acc", 0.10)
    verdicts.append(check_bar(
        "D7", ok, f"sine rollout acc learned on-cut {m:+.3f} CI "
        f"{fci((m, lo, hi))} (bar >= +0.10, lower CI > 0)"))
    s8 = stats(vec("sine", "frozen-aligned", "rollout_acc"))
    ok8 = s8["mean"] is not None and s8["mean"] < 0.10
    verdicts.append(check_bar(
        "D8 (guard)", ok8,
        f"sine rollout acc frozen (on) {fmt(s8)} "
        f"(guard mean < 0.10: generation needs learning)"))
    lines.append("## Bar verdicts")
    lines.append("")
    npass = 0
    for v in verdicts:
        lines.append(f"- **{v['name']} {'PASS' if v['pass'] else 'FAIL'}**: "
                     f"{v['detail']}")
        npass += int(v["pass"])
    lines.append("")
    lines.append(f"Bars: {npass} / {len(verdicts)} pass.")
    lines.append("")
    # Verdict rules.
    b1 = {v["name"]: v["pass"] for v in verdicts}
    bprime = all(b1.get(f"B{i}-lag-1", False) for i in (1, 2, 3, 4, 5, 6)) \
        and b1.get("B7-lag-1 (guard)", False)
    dpass = all(b1.get(k, False) for k in
                ("D1", "D2", "D3", "D4", "D5-lag-1", "D6-lag-1")) \
        and b1.get("D8 (guard)", False)
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Experiment B' (aligned wire is causal): "
                 f"{'PASSES' if bprime else 'FAILS'} "
                 f"(B1-B6 + B7 guard on lag-1).")
    lines.append(f"- Experiment D (multi-step generation through the "
                 f"top-down pathway): {'PASSES' if dpass else 'FAILS'} "
                 f"(D1-D6 + D8 guard).")
    if bprime and dpass:
        lines.append("- Predictive-coding causality is genuinely "
                     "established on this architecture (AGY decision "
                     "rule): the aligned wire is causal AND sustains "
                     "multi-step generation.")
    elif bprime:
        lines.append("- The wire is causal for one-step prediction but "
                     "does not sustain multi-step generation.")
    elif dpass:
        lines.append("- Generation works but the one-step clamp does "
                     "not: report both plainly; this combination was "
                     "not predicted.")
    else:
        lines.append("- Neither half passes: the aligned wire does not "
                     "establish predictive-coding causality on this "
                     "architecture.")
    # Diagnostics: learned-wire alignment, drive ratios.
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    for task in TASKS:
        for cond in ("aligned-on", "learned-on", "random-on"):
            rows = by.get((task, cond), {})
            if not rows:
                continue
            seeds = sorted(rows)
            sa = stats([rows[s].get("fb_align") for s in seeds])
            sn = stats([rows[s].get("fb_norm") for s in seeds])
            lines.append(f"- {task} {cond}: fb_align {fmt(sa)}; "
                         f"fb_norm {fmt(sn)}")
    with open(os.path.join(HERE, "r20_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"bars: {npass} / {len(verdicts)} pass; "
          f"B' {'PASS' if bprime else 'FAIL'}, D {'PASS' if dpass else 'FAIL'}")


if __name__ == "__main__":
    main()
