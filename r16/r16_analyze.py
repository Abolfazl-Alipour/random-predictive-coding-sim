"""Round-16 analysis: reads all test JSONLs, computes per-task /
per-condition means, medians, IQRs, 95% CIs, paired differences,
and checks every protocol bar. Writes r16/r16_analysis.md and
prints a summary. Run when the test phase completes (20 seeds).

None values are structurally empty measures (e.g. omit_acc_t1_m1
on lag-1) and are skipped, never treated as zero.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["planted1", "planted2"]
CONDITIONS = ["delta", "frozen"]
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


def load():
    recs = []
    for task in TASKS:
        for shard in ("a", "b"):
            p = os.path.join(HERE, f"r16_{task}_test_{shard}.jsonl")
            if not os.path.exists(p):
                continue
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
    recs = load()
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    lines = ["# Round 16 analysis (omission test, frozen weights)",
             "",
             "Protocol: PROTOCOL-R16-OMISSION.md. Seeds 221..240 "
             "(20 fresh seeds). Chance = 1/16 = 0.0625.",
             "CI = mean +/- 1.96 * sample-std / sqrt(n).",
             ""]
    verdicts = []
    for task in TASKS:
        lines.append(f"## {LABELS[task]}")
        lines.append("")
        d = by.get((task, "delta"), {})
        f = by.get((task, "frozen"), {})
        seeds = sorted(d)
        for cond, dd in (("delta", d), ("frozen", f)):
            s_m2 = stats([dd[x]["omit_acc_m2"] for x in seeds])
            s_m1 = stats([dd[x]["omit_acc_m1"] for x in seeds])
            s_fi = stats([dd[x]["omit_acc_filler"] for x in seeds])
            s_co = stats([dd[x]["omit_acc_cont_m1"] for x in seeds])
            s_mg = stats([dd[x]["margin_m2"] for x in seeds])
            s_sh = stats([dd[x]["omit_acc_shuf_m2"] for x in seeds])
            s_nm2 = stats([dd[x]["normal_acc_m2"] for x in seeds])
            lines.append(f"- {cond} omit_m2: {fmt(s_m2)}")
            lines.append(f"- {cond} omit_m1: {fmt(s_m1)}")
            lines.append(f"- {cond} omit_filler: {fmt(s_fi)}")
            lines.append(f"- {cond} cont_m1: {fmt(s_co)}")
            lines.append(f"- {cond} margin_m2: {fmt(s_mg)}")
            lines.append(f"- {cond} shuf_m2: {fmt(s_sh)}")
            lines.append(f"- {cond} normal_m2: {fmt(s_nm2)}")
        # Paired contrasts.
        dm, dlo, dhi, dn = paired(
            [d[x]["omit_acc_m2"] for x in seeds],
            [d[x]["omit_acc_filler"] for x in seeds])
        lines.append(f"- paired delta omit_m2 - omit_filler: "
                     f"mean diff={dm:+.3f} 95% CI=[{dlo:+.3f},{dhi:+.3f}] n={dn}")
        cm, clo, chi, cn = paired(
            [d[x]["omit_acc_cont_m1"] for x in seeds],
            [d[x]["omit_acc_t1_filler"] for x in seeds])
        lines.append(f"- paired delta cont_m1 - t1_filler: "
                     f"mean diff={cm:+.3f} 95% CI=[{clo:+.3f},{chi:+.3f}] n={cn}")
        fm, flo, fhi, fn = paired(
            [d[x]["omit_acc_cont_m1"] for x in seeds],
            [f[x]["omit_acc_cont_m1"] for x in seeds])
        lines.append(f"- paired delta - frozen cont_m1: "
                     f"mean diff={fm:+.3f} 95% CI=[{flo:+.3f},{fhi:+.3f}] n={fn}")
        lines.append("")
        # Bars.
        s_m2 = stats([d[x]["omit_acc_m2"] for x in seeds])
        s_m1 = stats([d[x]["omit_acc_m1"] for x in seeds])
        s_co = stats([d[x]["omit_acc_cont_m1"] for x in seeds])
        s_fm2 = stats([f[x]["omit_acc_m2"] for x in seeds])
        s_fmg = stats([f[x]["margin_m2"] for x in seeds])
        s_sh = stats([d[x]["omit_acc_shuf_m2"] for x in seeds])
        if task == "planted1":
            checks = [
                ("A1 omit_m2 mean >= 0.80, lower CI > 0.50",
                 s_m2["mean"] is not None and s_m2["mean"] >= 0.80
                 and s_m2["ci_lo"] > 0.50),
                ("A2 omit_m2 - filler paired mean > 0.30, CI above 0",
                 dm > 0.30 and dlo > 0),
                ("A3 frozen omit_m2 mean < 0.15, margin < 0.05",
                 s_fm2["mean"] is not None and s_fm2["mean"] < 0.15
                 and s_fmg["mean"] is not None and s_fmg["mean"] < 0.05),
                ("A4 shuf_m2 mean < 0.15",
                 s_sh["mean"] is not None and s_sh["mean"] < 0.15),
                ("A5 omit_m1 mean >= 0.50, lower CI > 0.25",
                 s_m1["mean"] is not None and s_m1["mean"] >= 0.50
                 and s_m1["ci_lo"] > 0.25),
                ("A6 cont_m1 lower CI > chance (0.0625)",
                 s_co["ci_lo"] is not None and s_co["ci_lo"] > CHANCE),
                ("A7 cont_m1 - t1_filler paired CI above 0",
                 clo > 0),
            ]
        else:
            checks = [
                ("B1 omit_m2 lower CI > chance (0.0625)",
                 s_m2["ci_lo"] is not None and s_m2["ci_lo"] > CHANCE),
                ("B2 omit_m2 - filler paired CI above 0",
                 dlo > 0),
                ("B3 frozen omit_m2 mean < 0.15",
                 s_fm2["mean"] is not None and s_fm2["mean"] < 0.15),
                ("B4 cont_m1 lower CI > chance (0.0625)",
                 s_co["ci_lo"] is not None and s_co["ci_lo"] > CHANCE),
            ]
        for name, ok in checks:
            lines.append(f"- BAR {name}: {'PASS' if ok else 'FAIL'}")
            verdicts.append((task, name, ok))
        lines.append("")
    n_pass = sum(1 for _, _, ok in verdicts if ok)
    lines.append(f"## Summary: {n_pass}/{len(verdicts)} bars pass")
    lines.append("")
    lines.append("Experiment A passes if A1-A5 pass on lag-1. A6, A7, "
                 "and the lag-2 bars are reported separately.")
    with open(os.path.join(HERE, "r16_analysis.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
