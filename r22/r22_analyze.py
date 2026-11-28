"""Round-22 analysis: the precision test (experiment C).

Protocol: PROTOCOL-R22-PRECISION.md. Seeds 352..371 (20 fresh).
Reads r22/r22_{task}_test_352-371.jsonl (one per task) and
r22/winner.json (c_low per task and the val-set floors),
computes per-task / per-config means, medians, IQRs, EXACT
Student-t 95% CIs, paired differences, and checks every
protocol bar. Writes r22/r22_analysis.md and prints a
summary.

CI convention: exact Student-t, NOT z=1.96. t(19) = 2.0930 for
the 20-seed test bars; t(4) = 2.7760 for 5-seed val summaries.
scipy is absent from the venv, so the constants are hardcoded
from the standard t table. A bar passes when the paired
per-seed difference favours the first-named side with its
lower 95% CI above 0, unless the bar says otherwise (P4 is
a two-part absolute bar, W5 is a gate, W6 is a guard ratio,
S1/S2 are equivalence bars, W1s is a diagnostic).
"""
import argparse
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
TASKS = ["planted1", "planted2", "sine"]
LABELS = {"planted1": "Planted lag-1 (K=16)",
          "planted2": "Planted lag-2 (K=16)",
          "sine": "Sine (K=32)"}
CONFIGS = ["aligned-on", "aligned-cut", "aligned-shuffle",
           "random-on", "random-cut", "random-shuffle",
           "wire-off", "frozen-aligned"]
T19 = 2.0930  # exact Student-t, 19 df, two-sided 95%.
T4 = 2.7760   # exact Student-t, 4 df, two-sided 95%.
N_TEST = 20
# Absolute bars (protocol section 7).
P4_SPEC_FLOOR = 0.25
P4_SPEC_CI = 0.15
P4_DECE_FLOOR = 0.50
P4_DECE_CI = 0.30
W5_FBDR = 0.05
W5_P1COS = 0.90
W6_RATIO = 0.80
S1_BOUND = 0.03
S2_BOUND = 0.05


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
        return "n/a"
    return (f"mean={s['mean']:.4f} median={s['median']:.4f} "
            f"IQR=[{s['iqr'][0]:.4f},{s['iqr'][1]:.4f}] "
            f"95% CI=[{s['ci_lo']:.4f},{s['ci_hi']:.4f}] n={s['n']}")


def paired(d1, d2):
    """Paired per-seed differences (d1 - d2), exact-t CI.
    d1, d2 are {seed: value} dicts."""
    keys = sorted(set(d1) & set(d2))
    d = [d1[k] - d2[k] for k in keys
         if d1[k] is not None and d2[k] is not None]
    m, lo, hi = ci(d)
    return m, lo, hi, len(d)


def load_jsonl(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def load_test():
    recs = []
    for task in TASKS:
        p = os.path.join(ROOT, f"r22_{task}_test_352-371.jsonl")
        recs.extend(load_jsonl(p))
    return recs


def load_winner():
    p = os.path.join(ROOT, "winner.json")
    with open(p) as f:
        return json.load(f)


def derived(by, task, config):
    """Per-seed derived quantities for one (task, config):
    gap0, gapL, interaction (rel_e and raw), spec0, specL,
    spec_interaction, decE_VIO, fbdr (mean of EXP/VIO c0),
    p1n diffs (planted only)."""
    rows = by.get((task, config), {})
    out = {}
    for s, r in rows.items():
        ps = r["p1scale"]
        gap0 = (r["vio_c0_err"] - r["exp_c0_err"]) / ps
        gapL = (r["vio_cl_err"] - r["exp_cl_err"]) / ps
        gap0_raw = r["vio_c0_err"] - r["exp_c0_err"]
        gapL_raw = r["vio_cl_err"] - r["exp_cl_err"]
        spec0 = r["vio_c0_cePres"] - r["vio_c0_cePred"]
        specL = r["vio_cl_cePres"] - r["vio_cl_cePred"]
        d = {
            "gap0": gap0, "gapL": gapL,
            "interaction": gap0 - gapL,
            "gap0_raw": gap0_raw, "gapL_raw": gapL_raw,
            "interaction_raw": gap0_raw - gapL_raw,
            "spec0": spec0, "specL": specL,
            "spec_interaction": spec0 - specL,
            "decE_vio": r["vio_c0_decE"],
            "decP1_exp": r["exp_c0_decP1"],
            "decP1_unexp": r["unexp_c0_decP1"],
            "p1n_exp": r["exp_c0_p1n"],
            "p1n_unexp": r["unexp_c0_p1n"],
            "fbdr": (r["exp_c0_fbdr"] + r["vio_c0_fbdr"]) / 2.0,
            "p1_cos": r["p1_cos"],
            "normal_acc": r["normal_acc"],
        }
        if d["decP1_unexp"] is not None:
            d["decP1_diff"] = d["decP1_exp"] - d["decP1_unexp"]
            d["p1n_diff"] = (d["p1n_exp"] - d["p1n_unexp"]) / ps
        out[s] = d
    return out


def vec(der, key):
    return {s: d[key] for s, d in der.items()}


def bar(name, ok, detail):
    return {"name": name, "pass": bool(ok), "detail": detail}


def main():
    recs = load_test()
    winner = load_winner()
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    der = {(t, c): derived(by, t, c)
           for t in TASKS for c in CONFIGS}
    bars = []
    lines = []
    lines.append("# Round 22 analysis: the precision test "
                 "(experiment C)")
    lines.append("")
    lines.append(f"Test records: {len(recs)} "
                 f"(expect 480: 8 configs x 3 tasks x 20 seeds).")
    for task in TASKS:
        n = {c: len(by.get((task, c), {})) for c in CONFIGS}
        lines.append(f"{task}: " + ", ".join(
            f"{c}={n[c]}" for c in CONFIGS))
    lines.append("")
    lines.append("## Floors (from winner.json, set by the "
                 "interaction validation phase)")
    for task in TASKS:
        w = winner[task]
        lines.append(f"{task}: c_low={w['c_low']:.2f} "
                     f"floor_P1={w['floor_P1']:.4f} "
                     f"floor_P2={w['floor_P2']:.4f} "
                     f"floor_P3={w['floor_P3']:.4f} "
                     f"floor_W1={w['floor_W1']:.4f} "
                     f"floor_W2={w['floor_W2']:.4f} "
                     f"floor_W3={w['floor_W3']:.4f} "
                     f"floor_W4={w['floor_W4']:.4f}")
    lines.append("")

    for task in TASKS:
        w = winner[task]
        lines.append(f"## {LABELS[task]}")
        ao = der[(task, "aligned-on")]
        ac = der[(task, "aligned-cut")]
        ash = der[(task, "aligned-shuffle")]
        ro = der[(task, "random-on")]
        off = der[(task, "wire-off")]
        fr = der[(task, "frozen-aligned")]
        # P1: interaction, aligned-on.
        s = stats(list(vec(ao, "interaction").values()))
        m, lo, hi, n = paired(vec(ao, "interaction"),
                              {k: 0.0 for k in ao})
        ok = (m >= w["floor_P1"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"P1 {task}", ok,
                        f"interaction {fmt(s)} floor="
                        f"{w['floor_P1']:.4f}"))
        lines.append(f"P1 interaction (aligned-on): {fmt(s)} "
                     f"floor={w['floor_P1']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # P2: specificity interaction, aligned-on.
        s = stats(list(vec(ao, "spec_interaction").values()))
        m, lo, hi, n = paired(vec(ao, "spec_interaction"),
                              {k: 0.0 for k in ao})
        ok = (m >= w["floor_P2"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"P2 {task}", ok,
                        f"spec_interaction {fmt(s)} floor="
                        f"{w['floor_P2']:.4f}"))
        lines.append(f"P2 spec interaction (aligned-on): {fmt(s)} "
                     f"floor={w['floor_P2']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # P3: learning control, gap0 trained - frozen.
        m, lo, hi, n = paired(vec(ao, "gap0"), vec(fr, "gap0"))
        ok = (m >= w["floor_P3"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"P3 {task}", ok,
                        f"gap0 trained-frozen {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n} floor="
                        f"{w['floor_P3']:.4f}"))
        lines.append(f"P3 gap0 trained - frozen: mean={m:.4f} "
                     f"CI=[{lo:.4f},{hi:.4f}] n={n} "
                     f"floor={w['floor_P3']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # P4: specificity guard (absolute).
        s1 = stats(list(vec(ao, "spec0").values()))
        s2 = stats(list(vec(ao, "decE_vio").values()))
        ok = (s1["mean"] >= P4_SPEC_FLOOR
              and s1["ci_lo"] > P4_SPEC_CI
              and s2["mean"] >= P4_DECE_FLOOR
              and s2["ci_lo"] > P4_DECE_CI)
        bars.append(bar(f"P4 {task}", ok,
                        f"spec0 {fmt(s1)}; decE_VIO {fmt(s2)}"))
        lines.append(f"P4 spec0: {fmt(s1)} (floor "
                     f"{P4_SPEC_FLOOR}, CI > {P4_SPEC_CI}); "
                     f"decE_VIO: {fmt(s2)} (floor "
                     f"{P4_DECE_FLOOR}, CI > {P4_DECE_CI}) "
                     f"{'PASS' if ok else 'FAIL'}")
        # W1: on - cut.
        m, lo, hi, n = paired(vec(ao, "interaction"),
                              vec(ac, "interaction"))
        ok = (m >= w["floor_W1"]) and (lo > 0) and (n == N_TEST)
        small = (m > 0) and (lo > 0) and (m < w["floor_W1"])
        bars.append(bar(f"W1 {task}", ok,
                        f"interaction on-cut {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n} floor="
                        f"{w['floor_W1']:.4f} "
                        f"causal-but-small={small}"))
        lines.append(f"W1 interaction on - cut: mean={m:.4f} "
                     f"CI=[{lo:.4f},{hi:.4f}] n={n} "
                     f"floor={w['floor_W1']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}"
                     f"{' (causal but small)' if small else ''}")
        # W2: on - shuffle.
        m, lo, hi, n = paired(vec(ao, "interaction"),
                              vec(ash, "interaction"))
        ok = (m >= w["floor_W2"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"W2 {task}", ok,
                        f"interaction on-shuffle {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n} floor="
                        f"{w['floor_W2']:.4f}"))
        lines.append(f"W2 interaction on - shuffle: mean={m:.4f} "
                     f"CI=[{lo:.4f},{hi:.4f}] n={n} "
                     f"floor={w['floor_W2']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # W3: aligned - random.
        m, lo, hi, n = paired(vec(ao, "interaction"),
                              vec(ro, "interaction"))
        ok = (m >= w["floor_W3"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"W3 {task}", ok,
                        f"interaction aligned-random {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n} floor="
                        f"{w['floor_W3']:.4f}"))
        lines.append(f"W3 interaction aligned - random: "
                     f"mean={m:.4f} CI=[{lo:.4f},{hi:.4f}] n={n} "
                     f"floor={w['floor_W3']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # W4: on - off.
        m, lo, hi, n = paired(vec(ao, "interaction"),
                              vec(off, "interaction"))
        ok = (m >= w["floor_W4"]) and (lo > 0) and (n == N_TEST)
        bars.append(bar(f"W4 {task}", ok,
                        f"interaction on-off {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n} floor="
                        f"{w['floor_W4']:.4f}"))
        lines.append(f"W4 interaction on - off: mean={m:.4f} "
                     f"CI=[{lo:.4f},{hi:.4f}] n={n} "
                     f"floor={w['floor_W4']:.4f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # W1s: diagnostic, strict Kimi version.
        s = stats(list(vec(ac, "interaction").values()))
        ok = (abs(s["mean"]) < w["floor_W1"]
              and s["ci_lo"] <= 0 <= s["ci_hi"])
        lines.append(f"W1s (diagnostic) interaction_cut: {fmt(s)} "
                     f"floor={w['floor_W1']:.4f} "
                     f"{'REMOVED by cut' if ok else 'PERSISTS cut'}")
        # W5: gate.
        s_fb = stats(list(vec(ao, "fbdr").values()))
        s_pc = stats(list(vec(ao, "p1_cos").values()))
        ok = (s_fb["mean"] > W5_FBDR
              and s_pc["mean"] < W5_P1COS)
        bars.append(bar(f"W5 {task}", ok,
                        f"fbdr {fmt(s_fb)} (> {W5_FBDR}); "
                        f"p1_cos {fmt(s_pc)} (< {W5_P1COS})"))
        lines.append(f"W5 gate fbdr: {fmt(s_fb)}; p1_cos: "
                     f"{fmt(s_pc)} {'PASS' if ok else 'FAIL'}")
        # W6: guard.
        m, lo, hi, n = paired(
            vec(ao, "normal_acc"),
            {k: W6_RATIO * v for k, v in
             vec(off, "normal_acc").items()})
        ratio = (np.mean(list(vec(ao, "normal_acc").values()))
                 / np.mean(list(vec(off, "normal_acc").values())))
        ok = (lo > 0) and (n == N_TEST)
        bars.append(bar(f"W6 {task}", ok,
                        f"normal_acc on - 0.8*off {m:.4f} "
                        f"CI=[{lo:.4f},{hi:.4f}] n={n}; "
                        f"mean ratio on/off={ratio:.3f}"))
        lines.append(f"W6 normal_acc on vs 0.8*off: mean diff="
                     f"{m:.4f} CI=[{lo:.4f},{hi:.4f}] n={n}; "
                     f"mean ratio={ratio:.3f} "
                     f"{'PASS' if ok else 'FAIL'}")
        # Raw-unit interaction (confound check, section 10.6).
        s = stats(list(vec(ao, "interaction_raw").values()))
        lines.append(f"DIAG interaction_raw (aligned-on): {fmt(s)}")
        s = stats(list(vec(ro, "interaction_raw").values()))
        lines.append(f"DIAG interaction_raw (random-on): {fmt(s)}")
        # S-family (planted only).
        if task != "sine":
            s = stats(list(vec(off, "decP1_diff").values()))
            ok = (abs(s["mean"]) < S1_BOUND
                  and s["ci_lo"] <= 0 <= s["ci_hi"])
            bars.append(bar(f"S1 {task}", ok,
                            f"decP1 EXP-UNEXP (wire-off) {fmt(s)} "
                            f"bound={S1_BOUND}"))
            lines.append(f"S1 decP1 EXP-UNEXP wire-off: {fmt(s)} "
                         f"bound={S1_BOUND} "
                         f"{'PASS' if ok else 'FAIL'}")
            s = stats(list(vec(off, "p1n_diff").values()))
            ok = (abs(s["mean"]) < S2_BOUND
                  and s["ci_lo"] <= 0 <= s["ci_hi"])
            bars.append(bar(f"S2 {task}", ok,
                            f"p1n EXP-UNEXP (wire-off) {fmt(s)} "
                            f"bound={S2_BOUND}"))
            lines.append(f"S2 p1n EXP-UNEXP wire-off: {fmt(s)} "
                         f"bound={S2_BOUND} "
                         f"{'PASS' if ok else 'FAIL'}")
            s3a = stats(list(vec(ao, "decP1_diff").values()))
            s3b = stats(list(vec(ao, "p1n_diff").values()))
            lines.append(f"S3 (measurement only) wire-on: "
                         f"decP1 diff {fmt(s3a)}; p1n diff "
                         f"{fmt(s3b)}")
        lines.append("")

    lines.append("## Bar tally")
    passed = [b for b in bars if b["pass"]]
    lines.append(f"{len(passed)} of {len(bars)} bars pass.")
    for b in bars:
        lines.append(f"{'PASS' if b['pass'] else 'FAIL'} "
                     f"{b['name']}: {b['detail']}")
    lines.append("")
    lines.append("## Verdict (protocol section 8)")
    bp = {b["name"]: b["pass"] for b in bars}
    w5_sine = bp.get("W5 sine", False)
    if (bp.get("P1 sine") and bp.get("P2 sine")
            and bp.get("P3 sine") and bp.get("P4 sine")
            and bp.get("P1 planted1")
            and bp.get("W1 sine") and bp.get("W2 sine")):
        verdict = "C STRONG PASS"
    elif (bp.get("P1 sine") and bp.get("P2 sine")
          and bp.get("P3 sine") and bp.get("P4 sine")
          and bp.get("P1 planted1")):
        verdict = "C PASS (interaction only; wire does not carry it)"
    elif bp.get("P1 sine") and bp.get("P2 sine"):
        verdict = "C PARTIAL (interaction under deterministic " \
                  "expectation only)"
    else:
        verdict = "C FAIL"
    if not w5_sine and verdict == "C STRONG PASS":
        verdict = "C PASS (W5 gate failed on sine; W-family void)"
    lines.append(f"Verdict: {verdict}")
    lines.append("")
    lines.append("Thesis-level: no outcome of round 22 establishes "
                 "predictive coding. Round 20 failed B' and D; a C "
                 "pass establishes only that the model produces "
                 "precision-like behaviour from input noise plus "
                 "error selection.")

    text = "\n".join(lines) + "\n"
    out = os.path.join(ROOT, "r22_analysis.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE,
                    help="directory holding winner.json and the "
                         "r22_{task}_test_352-371.jsonl files")
    args = ap.parse_args()
    ROOT = args.root
    main()
