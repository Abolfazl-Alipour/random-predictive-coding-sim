"""Round 18 analyzer: checks bars D1-D8 (PROTOCOL-R18-CONTINUOUS.md)
and writes r18/r18_analysis.md.

Usage: python r18/r18_analyze.py [--in FILES...] [--out FILE]
Default input: r18/r18_lorenz_test_262-281.jsonl and
r18/r18_varsine_test_262-281.jsonl. Default output:
r18/r18_analysis.md. Exits 0 if the input files cover all 20
test seeds for both tasks, else 1 (so the monitor can tell a
complete run from a partial one).
"""
import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SEEDS = list(range(262, 282))
MODEL_CONDS = ["C1_onehot_delta_proto", "C2_onehot_delta_ridge",
               "C3_bump_delta_proto", "C4_bump_delta_ridge",
               "C5_onehot_frozen_proto", "C6_bump_frozen_proto",
               "C2f_onehot_frozen_ridge", "C4f_bump_frozen_ridge"]


def load(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def stats(vals):
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n == 0:
        # No data for this (task, condition): report NaN, never
        # crash. Callers that need a value check n first.
        nan = float("nan")
        return {"n": 0, "mean": nan, "median": nan, "q1": nan,
                "q3": nan, "ci_lo": nan, "ci_hi": nan}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    return {"n": n, "mean": mean, "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)),
            "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se}


def paired(a, b):
    """Paired difference a - b per seed (dicts seed -> value).
    Returns (mean, ci_lo, ci_hi, n)."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    n = len(d)
    mean = float(d.mean()) if n else 0.0
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def fmt(x):
    return f"{x:.3f}"


def check_bar(name, mean, lo, hi, want_positive, note=""):
    """want_positive: True if a positive paired mean is a pass
    (e.g. C1 acc - persist acc); the caller sets the sign so
    that positive means 'the named condition wins'."""
    ok = (lo > 0) if want_positive else (hi < 0)
    return {"name": name, "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "pass": bool(ok), "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.inputs:
        inputs = args.inputs
    else:
        inputs = [os.path.join(BASE, "r18", f"r18_{t}_test_262-281.jsonl")
                  for t in ("lorenz", "varsine")]
    out = args.out or os.path.join(BASE, "r18", "r18_analysis.md")
    rows = load(inputs)
    # task -> config -> seed -> rec
    by_task = {}
    for r in rows:
        if r.get("phase") != "test":
            continue
        by_task.setdefault(r["task"], {}).setdefault(
            r["config"], {})[r["seed"]] = r
    complete = all(
        len(by_task.get(t, {}).get(c, {})) == len(TEST_SEEDS)
        for t in ("lorenz", "varsine")
        for c in MODEL_CONDS + ["baseline"])
    lines = []
    lines.append("# Round 18 analysis: the continuous-task wall")
    lines.append("")
    lines.append(f"Seeds: {TEST_SEEDS[0]}..{TEST_SEEDS[-1]} "
                 f"(n = {len(TEST_SEEDS)}). All metrics convention A "
                 "(pred(t) vs token t; MSE decode(pred(t)) vs cont[t]). "
                 "Bars are paired per-seed differences, 95% CI = "
                 "mean +/- 1.96 * std / sqrt(20).")
    lines.append("")
    lines.append(f"Input files: {', '.join(inputs)}")
    lines.append(f"Coverage complete: {complete} "
                 f"({len(rows)} test lines read).")
    lines.append("")
    # 1. Per-condition summary table.
    lines.append("## 1. Per-condition summary (mean [95% CI], median, IQR)")
    lines.append("")
    for task in ("lorenz", "varsine"):
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| condition | acc | within1 | mse |")
        lines.append("|---|---|---|---|")
        for cond in MODEL_CONDS:
            seeds = by_task.get(task, {}).get(cond, {})
            if not seeds:
                lines.append(f"| {cond} | no data | | |")
                continue
            a = stats([seeds[s]["acc"] for s in sorted(seeds)])
            w = stats([seeds[s]["within1"] for s in sorted(seeds)])
            m = stats([seeds[s]["mse"] for s in sorted(seeds)])
            lines.append(
                f"| {cond} | {fmt(a['mean'])} "
                f"[{fmt(a['ci_lo'])}, {fmt(a['ci_hi'])}] "
                f"med {fmt(a['median'])} IQR "
                f"[{fmt(a['q1'])}, {fmt(a['q3'])}] | "
                f"{fmt(w['mean'])} | {fmt(m['mean'])} "
                f"[{fmt(m['ci_lo'])}, {fmt(m['ci_hi'])}] "
                f"med {fmt(m['median'])} IQR "
                f"[{fmt(m['q1'])}, {fmt(m['q3'])}] |")
        b = by_task.get(task, {}).get("baseline", {})
        if b:
            s = stats([b[x]["markov1_acc"] for x in sorted(b)])
            lines.append(
                f"| baseline Markov-1 | {fmt(s['mean'])} | | |")
            s = stats([b[x]["persist_acc"] for x in sorted(b)])
            lines.append(
                f"| baseline persistence | {fmt(s['mean'])} | | |")
            s = stats([b[x]["esn_acc"] for x in sorted(b)])
            lines.append(f"| baseline ESN | {fmt(s['mean'])} | | |")
            s = stats([b[x]["persist_mse"] for x in sorted(b)])
            lines.append(
                f"| baseline persistence MSE | | | {fmt(s['mean'])} "
                f"[{fmt(s['ci_lo'])}, {fmt(s['ci_hi'])}] |")
            s = stats([b[x]["markov1_mse"] for x in sorted(b)])
            lines.append(
                f"| baseline Markov-1 MSE | | | {fmt(s['mean'])} |")
            s = stats([b[x]["esn_mse"] for x in sorted(b)])
            lines.append(
                f"| baseline ESN MSE | | | {fmt(s['mean'])} "
                f"[{fmt(s['ci_lo'])}, {fmt(s['ci_hi'])}] |")
        lines.append("")
    # 2. Bars.
    def acc_map(task, cond):
        return {s: r["acc"] for s, r in
                by_task.get(task, {}).get(cond, {}).items()}

    def mse_map(task, cond):
        return {s: r["mse"] for s, r in
                by_task.get(task, {}).get(cond, {}).items()}

    def base_acc(task, key):
        return {s: r[key] for s, r in
                by_task.get(task, {}).get("baseline", {}).items()}

    def base_mse(task, key):
        return {s: r[key] for s, r in
                by_task.get(task, {}).get("baseline", {}).items()}

    bars = []
    # D1: Lorenz C1 acc beats persistence AND Markov-1.
    m, lo, hi, n = paired(acc_map("lorenz", "C1_onehot_delta_proto"),
                          base_acc("lorenz", "persist_acc"))
    d1a = check_bar("D1a Lorenz C1 acc - persistence acc", m, lo, hi,
                    True)
    m, lo, hi, n = paired(acc_map("lorenz", "C1_onehot_delta_proto"),
                          base_acc("lorenz", "markov1_acc"))
    d1b = check_bar("D1b Lorenz C1 acc - Markov-1 acc", m, lo, hi,
                    True)
    bars.append({"name": "D1 Lorenz C1 acc beats persistence AND "
                 "Markov-1", "pass": d1a["pass"] and d1b["pass"],
                 "parts": [d1a, d1b]})
    # D2: var-sine C1 acc beats persistence AND Markov-1.
    m, lo, hi, n = paired(acc_map("varsine", "C1_onehot_delta_proto"),
                          base_acc("varsine", "persist_acc"))
    d2a = check_bar("D2a var-sine C1 acc - persistence acc", m, lo,
                    hi, True)
    m, lo, hi, n = paired(acc_map("varsine", "C1_onehot_delta_proto"),
                          base_acc("varsine", "markov1_acc"))
    d2b = check_bar("D2b var-sine C1 acc - Markov-1 acc", m, lo,
                    hi, True)
    bars.append({"name": "D2 var-sine C1 acc beats persistence AND "
                 "Markov-1", "pass": d2a["pass"] and d2b["pass"],
                 "parts": [d2a, d2b]})
    # D3: var-sine C1 mse < persistence mse.
    m, lo, hi, n = paired(base_mse("varsine", "persist_mse"),
                          mse_map("varsine", "C1_onehot_delta_proto"))
    bars.append({"name": "D3 var-sine C1 mse < persistence mse",
                 "pass": bool(lo > 0),
                 "parts": [check_bar("D3 var-sine persistence mse - "
                                     "C1 mse", m, lo, hi, True)]})
    # D4: Lorenz C1 mse < persistence mse.
    m, lo, hi, n = paired(base_mse("lorenz", "persist_mse"),
                          mse_map("lorenz", "C1_onehot_delta_proto"))
    bars.append({"name": "D4 Lorenz C1 mse < persistence mse",
                 "pass": bool(lo > 0),
                 "parts": [check_bar("D4 Lorenz persistence mse - "
                                     "C1 mse", m, lo, hi, True)]})
    # D5: Lorenz C2 mse < C1 mse.
    m, lo, hi, n = paired(mse_map("lorenz", "C1_onehot_delta_proto"),
                          mse_map("lorenz", "C2_onehot_delta_ridge"))
    d5 = check_bar("D5 Lorenz C1 mse - C2 ridge mse", m, lo, hi,
                   True)
    bars.append({"name": "D5 Lorenz C2 ridge mse < C1 mse",
                 "pass": d5["pass"], "parts": [d5]})
    # D6: Lorenz C3 mse < C1 mse.
    m, lo, hi, n = paired(mse_map("lorenz", "C1_onehot_delta_proto"),
                          mse_map("lorenz", "C3_bump_delta_proto"))
    d6 = check_bar("D6 Lorenz C1 mse - C3 bump mse", m, lo, hi,
                   True)
    bars.append({"name": "D6 Lorenz C3 bump mse < C1 mse",
                 "pass": d6["pass"], "parts": [d6]})
    # D7: Lorenz C4 mse < C2 mse AND C4 mse < C3 mse.
    m1, lo1, hi1, _ = paired(mse_map("lorenz",
                                     "C2_onehot_delta_ridge"),
                             mse_map("lorenz",
                                     "C4_bump_delta_ridge"))
    d7a = check_bar("D7a Lorenz C2 mse - C4 mse", m1, lo1, hi1,
                    True)
    m2, lo2, hi2, _ = paired(mse_map("lorenz",
                                     "C3_bump_delta_proto"),
                             mse_map("lorenz",
                                     "C4_bump_delta_ridge"))
    d7b = check_bar("D7b Lorenz C3 mse - C4 mse", m2, lo2, hi2,
                    True)
    judged = d5["pass"] or d6["pass"]
    bars.append({"name": "D7 Lorenz C4 mse < C2 mse AND C4 mse < "
                 "C3 mse (judged only if D5 or D6 passes)",
                 "pass": (d7a["pass"] and d7b["pass"]) if judged
                 else False,
                 "parts": [d7a, d7b],
                 "note": "" if judged else
                 "not judged: D5 and D6 both failed"})
    # D8: Lorenz C2 mse < C2f mse.
    m, lo, hi, n = paired(mse_map("lorenz",
                                  "C2f_onehot_frozen_ridge"),
                          mse_map("lorenz",
                                  "C2_onehot_delta_ridge"))
    d8 = check_bar("D8 Lorenz C2f frozen-ridge mse - C2 mse",
                   m, lo, hi, True)
    bars.append({"name": "D8 Lorenz C2 ridge mse < C2f "
                 "frozen-ridge mse",
                 "pass": d8["pass"], "parts": [d8]})
    lines.append("## 2. Bars D1-D8 (paired per-seed differences)")
    lines.append("")
    for b in bars:
        lines.append(f"### {b['name']}: "
                     f"{'PASS' if b['pass'] else 'FAIL'}")
        if b.get("note"):
            lines.append(f"Note: {b['note']}")
        for p in b["parts"]:
            lines.append(f"- {p['name']}: mean paired diff "
                         f"{fmt(p['mean'])}, 95% CI "
                         f"[{fmt(p['ci_lo'])}, {fmt(p['ci_hi'])}], "
                         f"{'PASS' if p['pass'] else 'FAIL'}")
        lines.append("")
    # 3. Var-sine secondary D5-D8 analogues.
    lines.append("## 3. Var-sine secondary analogues (no bars)")
    lines.append("")
    for label, a_cond, b_cond in (
            ("D5-var: C2 ridge mse < C1 mse",
             "C1_onehot_delta_proto", "C2_onehot_delta_ridge"),
            ("D6-var: C3 bump mse < C1 mse",
             "C1_onehot_delta_proto", "C3_bump_delta_proto"),
            ("D7-var: C4 mse < C2 mse and C4 mse < C3 mse",
             "C2_onehot_delta_ridge", "C4_bump_delta_ridge"),
            ("D8-var: C2 ridge mse < C2f frozen-ridge mse",
             "C2f_onehot_frozen_ridge", "C2_onehot_delta_ridge")):
        m, lo, hi, n = paired(mse_map("varsine", a_cond),
                              mse_map("varsine", b_cond))
        lines.append(f"- {label}: mean paired diff {fmt(m)}, 95% "
                     f"CI [{fmt(lo)}, {fmt(hi)}] (n={n})")
    lines.append("")
    # 4. Verdict.
    h0 = bars[0]["pass"] and bars[1]["pass"] and bars[2]["pass"]
    d4 = bars[3]["pass"]
    d5p, d6p, d7p = bars[4]["pass"], bars[5]["pass"], bars[6]["pass"]
    d8p = bars[7]["pass"]
    lines.append("## 4. Verdict (protocol section 9)")
    lines.append("")
    lines.append(f"- H0 (D1, D2, D3 pass): "
                 f"{'PASS' if h0 else 'FAIL'}")
    lines.append(f"- D4 (Lorenz C1 MSE beats persistence): "
                 f"{'PASS' if d4 else 'FAIL'}")
    if h0 and d4:
        lines.append("- Verdict: the wall was entirely scoring.")
    elif h0 and not d4:
        if d5p and not d6p:
            cause = "readout"
        elif d6p and not d5p:
            cause = "input"
        elif d7p:
            cause = "both/interacting"
        else:
            cause = "target (c) by elimination"
        lines.append(f"- Verdict: H0 passes but the Lorenz MSE wall "
                     f"is real. Residual cause: {cause}.")
    elif not h0:
        lines.append("- Verdict: H0 fails. The continuous wall "
                     "survives correct scoring; the pilot was not "
                     "the whole story. Read the per-condition "
                     "table and the paired differences before "
                     "designing the next round.")
    lines.append(f"- D8 (delta adds continuous information): "
                 f"{'PASS' if d8p else 'FAIL'}")
    lines.append("")
    # 5. Frozen controls (sanity).
    lines.append("## 5. Frozen controls (sanity)")
    lines.append("")
    for task in ("lorenz", "varsine"):
        c5 = stats([r["acc"] for r in
                    by_task.get(task, {}).get(
                        "C5_onehot_frozen_proto", {}).values()])
        c1 = stats([r["acc"] for r in
                    by_task.get(task, {}).get(
                        "C1_onehot_delta_proto", {}).values()])
        if c5["n"] and c1["n"]:
            lines.append(f"- {task}: C1 acc {fmt(c1['mean'])} vs "
                         f"C5 frozen acc {fmt(c5['mean'])} "
                         f"(learning gap).")
    lines.append("")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out} (complete={complete})")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
