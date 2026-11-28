"""Round-17 analysis: reads validation JSONLs (gain pick with the
guard) and test JSONLs, computes per-task / per-condition means,
medians, IQRs, 95% CIs, paired differences, and checks every
protocol bar. Writes r17/r17_analysis.md and prints a summary.
Run when the test phase completes (20 seeds).

None values are structurally empty measures (e.g. omit_acc_t1_m1
on lag-1) and are skipped, never treated as zero.
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["planted1", "planted2"]
CONDITIONS = ["wire-on", "wire-cut", "wire-shuffle", "wire-off",
              "frozen"]
LABELS = {"planted1": "Planted lag-1", "planted2": "Planted lag-2"}
CHANCE = 1.0 / 16.0


def ci(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    half = 1.96 * s / math.sqrt(len(v))
    return m, m - half, m + half


def stats(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return {"mean": None, "median": None, "iqr": [None, None],
                "ci_lo": None, "ci_hi": None, "n": 0}
    m, lo, hi = ci(vals)
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


def load_test():
    recs = []
    for task in TASKS:
        for shard in ("a", "b"):
            p = os.path.join(HERE, f"r17_{task}_test_{shard}.jsonl")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
    return recs


def load_val():
    recs = []
    for p in glob.glob(os.path.join(HERE, "r17_planted1_val_*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def paired(vals_a, vals_b):
    """Paired per-seed differences, skipping seeds where either
    side is None. Returns (mean, ci_lo, ci_hi, n)."""
    d = [a - b for a, b in zip(vals_a, vals_b)
         if a is not None and b is not None]
    m, lo, hi = ci(d)
    return m, lo, hi, len(d)


def main():
    recs = load_test()
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    lines = ["# Round 17 analysis (top-down clamp, experiment B)",
             "",
             "Protocol: PROTOCOL-R17-CLAMP.md. Seeds 241..260 "
             "(20 fresh seeds). Chance = 1/16 = 0.0625.",
             "CI = mean +/- 1.96 * sample-std / sqrt(n).",
             ""]
    # Validation: gain pick with the guard.
    val = load_val()
    vg = {}
    for r in val:
        vg.setdefault(r["fb_gain"], []).append(r["val_acc"])
    lines.append("## Validation (planted lag-1, val seeds 241..245)")
    lines.append("")
    winner, winner_acc = None, -1.0
    for g in sorted(vg):
        s = stats(vg[g])
        lines.append(f"- fb_gain={g}: val_acc {fmt(s)}")
        if g > 0 and s["mean"] is not None and s["mean"] > winner_acc:
            winner, winner_acc = g, s["mean"]
    g0 = stats(vg.get(0.0, []))
    guard_ok = (g0["mean"] is not None and winner is not None
                and winner_acc >= 0.80 * g0["mean"])
    lines.append(f"- winner fb_gain={winner} (val_acc={winner_acc:.3f}); "
                 f"fb_gain=0.0 val_acc={g0['mean']:.3f}; "
                 f"guard (>= 0.80x): {'PASS' if guard_ok else 'FAIL'}")
    lines.append("")
    verdicts = []
    for task in TASKS:
        lines.append(f"## {LABELS[task]}")
        lines.append("")
        w = by.get((task, "wire-on"), {})
        c = by.get((task, "wire-cut"), {})
        s = by.get((task, "wire-shuffle"), {})
        o = by.get((task, "wire-off"), {})
        f = by.get((task, "frozen"), {})
        seeds = sorted(w)
        for cond, dd in (("wire-on", w), ("wire-cut", c),
                         ("wire-shuffle", s), ("wire-off", o),
                         ("frozen", f)):
            s_m2 = stats([dd[x]["omit_acc_m2"] for x in seeds])
            s_m1 = stats([dd[x]["omit_acc_m1"] for x in seeds])
            s_co = stats([dd[x]["omit_acc_cont_m1"] for x in seeds])
            s_na = stats([dd[x]["normal_acc"] for x in seeds])
            s_fb = stats([dd[x]["fb_rms_ratio"] for x in seeds])
            lines.append(f"- {cond} omit_m2: {fmt(s_m2)}")
            lines.append(f"- {cond} omit_m1: {fmt(s_m1)}")
            lines.append(f"- {cond} cont_m1: {fmt(s_co)}")
            lines.append(f"- {cond} normal_acc: {fmt(s_na)}")
            lines.append(f"- {cond} fb_rms_ratio: {fmt(s_fb)}")
        # p1_cos (per-net, on the wire-on records) and err_mag.
        s_pc = stats([w[x]["p1_cos"] for x in seeds])
        s_e0 = stats([w[x]["err_mag_first"] for x in seeds])
        s_e1 = stats([w[x]["err_mag_last"] for x in seeds])
        s_eo = stats([o[x]["err_mag_last"] for x in seeds])
        lines.append(f"- wire-on p1_cos (on vs cut): {fmt(s_pc)}")
        lines.append(f"- wire-on err_mag first epoch: {fmt(s_e0)}")
        lines.append(f"- wire-on err_mag last epoch: {fmt(s_e1)}")
        lines.append(f"- wire-off err_mag last epoch: {fmt(s_eo)}")
        # Paired contrasts (lag-1 bars; lag-2 reported the same way).
        c1, c1lo, c1hi, c1n = paired(
            [w[x]["omit_acc_m2"] for x in seeds],
            [c[x]["omit_acc_m2"] for x in seeds])
        c2, c2lo, c2hi, c2n = paired(
            [w[x]["omit_acc_m2"] for x in seeds],
            [s[x]["omit_acc_m2"] for x in seeds])
        c4, c4lo, c4hi, c4n = paired(
            [w[x]["omit_acc_m2"] for x in seeds],
            [o[x]["omit_acc_m2"] for x in seeds])
        lines.append(f"- paired wire-on - wire-cut omit_m2: "
                     f"mean diff={c1:+.3f} 95% CI=[{c1lo:+.3f},{c1hi:+.3f}] n={c1n}")
        lines.append(f"- paired wire-on - wire-shuffle omit_m2: "
                     f"mean diff={c2:+.3f} 95% CI=[{c2lo:+.3f},{c2hi:+.3f}] n={c2n}")
        lines.append(f"- paired wire-on - wire-off omit_m2: "
                     f"mean diff={c4:+.3f} 95% CI=[{c4lo:+.3f},{c4hi:+.3f}] n={c4n}")
        lines.append("")
        s_wna = stats([w[x]["normal_acc"] for x in seeds])
        s_ona = stats([o[x]["normal_acc"] for x in seeds])
        if task == "planted1":
            checks = [
                ("C1 cut degrades omit_m2 by >= 0.10, CI above 0",
                 c1 >= 0.10 and c1lo > 0),
                ("C2 shuffle degrades omit_m2 by >= 0.10, CI above 0",
                 c2 >= 0.10 and c2lo > 0),
                ("C3 p1_cos mean < 0.90",
                 s_pc["mean"] is not None and s_pc["mean"] < 0.90),
                ("C4 wire-off omit_m2 differs (paired CI excludes 0)",
                 c4lo > 0 or c4hi < 0),
                ("C5 guard: wire-on normal_acc >= 0.80x wire-off",
                 s_wna["mean"] is not None and s_ona["mean"] is not None
                 and s_wna["mean"] >= 0.80 * s_ona["mean"]),
            ]
        else:
            checks = [
                ("C1-lag2 cut degrades omit_m2, CI above 0", c1lo > 0),
                ("C2-lag2 shuffle degrades omit_m2, CI above 0", c2lo > 0),
                ("C3-lag2 p1_cos mean < 0.90",
                 s_pc["mean"] is not None and s_pc["mean"] < 0.90),
                ("C4-lag2 wire-off omit_m2 differs (CI excludes 0)",
                 c4lo > 0 or c4hi < 0),
            ]
        for name, ok in checks:
            lines.append(f"- BAR {name}: {'PASS' if ok else 'FAIL'}")
            verdicts.append((task, name, ok))
        lines.append("")
    n_pass = sum(1 for _, _, ok in verdicts if ok)
    lines.append(f"## Summary: {n_pass}/{len(verdicts)} bars pass")
    lines.append("")
    lines.append("Experiment B passes if C1-C4 pass on lag-1. C5 is "
                 "a guard, not a pass bar. Lag-2 bars are reported "
                 "the same way.")
    with open(os.path.join(HERE, "r17_analysis.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
