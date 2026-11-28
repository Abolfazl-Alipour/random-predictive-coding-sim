"""Round 19 analyzer: checks bars H1-H9 from PROTOCOL-R19-HIERARCHY.md.

Reads r19/r19_{task}_test_282-301.jsonl (one file per task,
concatenated by the monitor) plus r19/winner.json (the val
winner; H4 applies only if the winner is 3 levels). Writes
r19/r19_analysis.md. Exits 1 if coverage is incomplete
(480 test lines expected) or if a frozen sanity check fails.
"""
import json
import math
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R19 = os.path.join(BASE, "r19")
TASKS = ["planted1", "sine", "hier"]
TEST_SEEDS = list(range(282, 302))
N_TEST = len(TEST_SEEDS)
MODEL_CONDS = ["cand_on", "cand_cut", "cand_shuffle",
               "hier2_ladder", "hier2_nofb", "flat", "cand_frozen"]
EXPECTED_LINES = 480


def load(task):
    path = os.path.join(R19, f"r19_{task}_test_282-301.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def stats(vals):
    a = np.array(vals, dtype=float)
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a))
    return {"mean": m, "median": float(np.median(a)),
            "iqr": (float(np.percentile(a, 25)),
                    float(np.percentile(a, 75))),
            "ci": (m - 1.96 * se, m + 1.96 * se),
            "min": float(np.min(a)), "max": float(np.max(a))}


def paired(a, b):
    """Paired per-seed differences a - b with a 95 percent CI."""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    se = sd / math.sqrt(len(d))
    return {"mean": m, "ci": (m - 1.96 * se, m + 1.96 * se)}


def check_bar(name, cond, detail):
    return {"name": name, "passed": bool(cond), "detail": detail}


def f(x):
    return f"{x:.3f}"


def fci(ci):
    return f"({f(ci[0])}, {f(ci[1])})"


def main():
    all_rows = {t: load(t) for t in TASKS}
    n_lines = sum(len(v) for v in all_rows.values())
    winner = None
    wpath = os.path.join(R19, "winner.json")
    if os.path.exists(wpath):
        with open(wpath) as fh:
            winner = json.load(fh)
    out = []
    out.append("# Round 19 analysis: hierarchy (experiment E)")
    out.append("")
    out.append(f"Coverage: {n_lines} / {EXPECTED_LINES} test lines.")
    if winner is not None:
        g2 = winner["fb_gain2"]
        out.append(f"Val winner: n_levels={winner['n_levels']}, "
                   f"fb_gain1={winner['fb_gain1']}, "
                   f"fb_gain2={g2}, val_acc={winner['val_acc']:.3f}.")
    out.append("")
    # Coverage check.
    missing = []
    for task in TASKS:
        rows = all_rows[task]
        have = {}
        for r in rows:
            have[(r["config"], r["seed"])] = have.get(
                (r["config"], r["seed"]), 0) + 1
        for cond in MODEL_CONDS + ["baseline"]:
            for s in TEST_SEEDS:
                if cond == "hier2_ladder" and winner is not None \
                        and winner["n_levels"] == 2:
                    continue
                if (cond, s) not in have:
                    missing.append(f"{task}/{cond}/{s}")
        dupes = [k for k, v in have.items() if v > 1]
        if dupes:
            missing.append(f"{task} duplicates: {dupes[:5]}")
    ok = n_lines == EXPECTED_LINES or (
        winner is not None and winner["n_levels"] == 2
        and n_lines == EXPECTED_LINES - 60)
    if missing:
        out.append("## Coverage gaps")
        out.append("")
        for m in missing[:40]:
            out.append(f"1. {m}")
        out.append("")
    # Per-task condition tables.
    data = {}
    for task in TASKS:
        rows = all_rows[task]
        conds = {}
        for r in rows:
            conds.setdefault(r["config"], {})[r["seed"]] = r
        data[task] = conds
        out.append(f"## {task}")
        out.append("")
        out.append("| condition | acc | m1 | late_m1 | top_acc | "
                   "mean_burst | clamp |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for cond in MODEL_CONDS:
            if cond not in conds:
                continue
            seeds = sorted(conds[cond])
            acc = stats([conds[cond][s]["acc"] for s in seeds])
            m1 = stats([conds[cond][s]["m1"] for s in seeds])
            lm = stats([conds[cond][s]["late_m1"] for s in seeds])
            tp = [conds[cond][s]["top_acc"] for s in seeds
                  if conds[cond][s]["top_acc"] is not None]
            tp_s = stats(tp) if tp else None
            mb = stats([conds[cond][s]["mean_burst"] for s in seeds])
            cf = stats([conds[cond][s]["clamp_frac"] for s in seeds])
            tp_txt = f(tp_s["mean"]) if tp_s else "-"
            out.append(f"| {cond} | {f(acc['mean'])} {fci(acc['ci'])} | "
                       f"{f(m1['mean'])} {fci(m1['ci'])} | "
                       f"{f(lm['mean'])} {fci(lm['ci'])} | "
                       f"{tp_txt} | {f(mb['mean'])} | "
                       f"{f(cf['mean'])} |")
        if "baseline" in conds:
            b = [conds["baseline"][s] for s in sorted(conds["baseline"])]
            out.append("| baseline (markov1) | "
                       f"{f(stats([x['markov1_acc'] for x in b])['mean'])} | "
                       f"{f(stats([x.get('markov1_m1', 0) for x in b])['mean']) if 'markov1_m1' in b[0] else '-'} | - | - | - | - |")
            out.append("| baseline (markov2) | "
                       f"{f(stats([x['markov2_acc'] for x in b])['mean'])} | - | - | - | - | - |")
            out.append("| baseline (persist) | "
                       f"{f(stats([x['persist_acc'] for x in b])['mean'])} | - | - | - | - | - |")
            out.append("| baseline (esn) | "
                       f"{f(stats([x['esn_acc'] for x in b])['mean'])} | - | - | - | - | - |")
        out.append("")
    # Bars (only when coverage is complete; otherwise write
    # the coverage section and stop, so a partial run still
    # produces r19_analysis.md instead of crashing).
    bars = []
    h = data["hier"]
    p = data["planted1"]
    s = data["sine"]
    if missing:
        out.append("## Bars")
        out.append("")
        out.append("1. Not computed: coverage incomplete, so no bar "
                   "is judged.")
        out.append("")
        text = "\n".join(out)
        with open(os.path.join(R19, "r19_analysis.md"), "w") as fh:
            fh.write(text + "\n")
        print(text)
        print(f"COVERAGE INCOMPLETE: {n_lines} lines, "
              f"{len(missing)} gaps", file=sys.stderr)
        return 1

    def vals(task_data, cond, field):
        return [task_data[cond][sd][field] for sd in TEST_SEEDS]

    # H1: hier overall acc, cand_on - flat >= +0.10.
    d = paired(vals(h, "cand_on", "acc"), vals(h, "flat", "acc"))
    bars.append(check_bar(
        "H1", d["mean"] >= 0.10 and d["ci"][0] > 0,
        f"hier acc cand_on - flat = {f(d['mean'])} {fci(d['ci'])} "
        f"(bar +0.10, lower CI > 0)"))
    # H2: hier late m1, cand_on - flat >= +0.20.
    d = paired(vals(h, "cand_on", "late_m1"),
               vals(h, "flat", "late_m1"))
    bars.append(check_bar(
        "H2", d["mean"] >= 0.20 and d["ci"][0] > 0,
        f"hier late m1 cand_on - flat = {f(d['mean'])} {fci(d['ci'])} "
        f"(bar +0.20, lower CI > 0)"))
    # H3: hier late m1, cand_on mean >= 0.45, lower CI > 0.35.
    st = stats(vals(h, "cand_on", "late_m1"))
    bars.append(check_bar(
        "H3", st["mean"] >= 0.45 and st["ci"][0] > 0.35,
        f"hier late m1 cand_on = {f(st['mean'])} {fci(st['ci'])} "
        f"(bar mean >= 0.45, lower CI > 0.35)"))
    # H4: hier late m1, cand_on - hier2_ladder >= +0.10
    # (only if the winner is 3 levels).
    if "hier2_ladder" in h:
        d = paired(vals(h, "cand_on", "late_m1"),
                   vals(h, "hier2_ladder", "late_m1"))
        bars.append(check_bar(
            "H4", d["mean"] >= 0.10 and d["ci"][0] > 0,
            f"hier late m1 cand_on - hier2_ladder = "
            f"{f(d['mean'])} {fci(d['ci'])} (bar +0.10, lower CI > 0)"))
    else:
        bars.append(check_bar(
            "H4", None,
            "not judged: val winner is 2 levels, so hier2_ladder "
            "duplicates cand_on (protocol section 9)"))
    # H5: planted1 m1, cand_on - flat >= +0.05.
    d = paired(vals(p, "cand_on", "m1"), vals(p, "flat", "m1"))
    bars.append(check_bar(
        "H5", d["mean"] >= 0.05 and d["ci"][0] > 0,
        f"planted1 m1 cand_on - flat = {f(d['mean'])} {fci(d['ci'])} "
        f"(bar +0.05, lower CI > 0)"))
    # H6: planted1 m1, cand_on mean >= 0.80.
    st = stats(vals(p, "cand_on", "m1"))
    bars.append(check_bar(
        "H6", st["mean"] >= 0.80,
        f"planted1 m1 cand_on = {f(st['mean'])} {fci(st['ci'])} "
        f"(bar mean >= 0.80)"))
    # H7: sine acc, cand_on - flat >= +0.03.
    d = paired(vals(s, "cand_on", "acc"), vals(s, "flat", "acc"))
    bars.append(check_bar(
        "H7", d["mean"] >= 0.03 and d["ci"][0] > 0,
        f"sine acc cand_on - flat = {f(d['mean'])} {fci(d['ci'])} "
        f"(bar +0.03, lower CI > 0)"))
    # H8: hier overall acc, cand_on - cand_cut >= +0.05.
    d = paired(vals(h, "cand_on", "acc"), vals(h, "cand_cut", "acc"))
    bars.append(check_bar(
        "H8", d["mean"] >= 0.05 and d["ci"][0] > 0,
        f"hier acc cand_on - cand_cut = {f(d['mean'])} {fci(d['ci'])} "
        f"(bar +0.05, lower CI > 0)"))
    # H9: hier overall acc, cand_on - cand_frozen >= +0.10.
    d = paired(vals(h, "cand_on", "acc"),
               vals(h, "cand_frozen", "acc"))
    bars.append(check_bar(
        "H9", d["mean"] >= 0.10 and d["ci"][0] > 0,
        f"hier acc cand_on - cand_frozen = {f(d['mean'])} "
        f"{fci(d['ci'])} (bar +0.10, lower CI > 0)"))
    # Shuffle diagnostic (not a bar).
    d = paired(vals(h, "cand_on", "acc"),
               vals(h, "cand_shuffle", "acc"))
    shuffle_txt = (f"hier acc cand_on - cand_shuffle = "
                   f"{f(d['mean'])} {fci(d['ci'])}")
    # Frozen sanity: frozen must stay near chance on hier.
    fr = stats(vals(h, "cand_frozen", "acc"))
    frozen_ok = fr["mean"] < 0.25
    out.append("## Bars")
    out.append("")
    for b in bars:
        mark = "PASS" if b["passed"] else (
            "N/A" if b["passed"] is None else "FAIL")
        out.append(f"1. **{b['name']} {mark}**: {b['detail']}")
    out.append("")
    out.append("## Shuffle diagnostic (not a bar)")
    out.append("")
    out.append(f"1. {shuffle_txt}. If H8 passes but the shuffle gap "
               "is near zero, the effect is drive magnitude, not "
               "feedback content (protocol section 9).")
    out.append("")
    # Verdict.
    h1, h2, h3 = bars[0]["passed"], bars[1]["passed"], bars[2]["passed"]
    h4 = bars[3]["passed"]
    h5, h6, h7 = bars[4]["passed"], bars[5]["passed"], bars[6]["passed"]
    h8, h9 = bars[7]["passed"], bars[8]["passed"]
    out.append("## Verdict")
    out.append("")
    if h1 and h2 and h3:
        out.append("1. H0 PASSES: the hierarchy wins the hierarchical "
                   "task (H1, H2, H3 all pass).")
    else:
        out.append("1. H0 FAILS: at least one of H1, H2, H3 fails, so "
                   "the hierarchy does not win the hierarchical task.")
    if h4 is None:
        out.append("1. H4 not judged (2-level winner).")
    elif h4:
        out.append("1. H4 passes: the third level beats the "
                   "two-level ladder on late m1.")
    else:
        out.append("1. H4 fails: the third level does not beat the "
                   "two-level ladder, so the win (if any) comes from "
                   "a slower level 2, not from depth.")
    out.append(f"1. Old tasks: H5 {'passes' if h5 else 'fails'}, "
               f"H6 {'passes' if h6 else 'fails'}, "
               f"H7 {'passes' if h7 else 'fails'}.")
    out.append(f"1. Mechanism: H8 {'passes' if h8 else 'fails'} "
               f"(feedback used at test time), "
               f"H9 {'passes' if h9 else 'fails'} (learning matters).")
    out.append("")
    out.append("## Frozen sanity")
    out.append("")
    out.append(f"1. hier frozen acc = {f(fr['mean'])} "
               f"{fci(fr['ci'])} (must stay below 0.25): "
               f"{'PASS' if frozen_ok else 'FAIL'}.")
    out.append("")
    out.append("## Diagnostics")
    out.append("")
    # Top readout (M=3 only), flat vs cand on hier.
    if "hier2_ladder" in h or (winner and winner["n_levels"] == 3):
        tp = stats(vals(h, "cand_on", "top_acc"))
        out.append(f"1. hier top readout (level 3) acc = "
                   f"{f(tp['mean'])} {fci(tp['ci'])} "
                   "(secondary readout, no bar).")
    fl = stats(vals(h, "flat", "acc"))
    co = stats(vals(h, "cand_on", "acc"))
    out.append(f"1. hier flat acc = {f(fl['mean'])} "
               f"{fci(fl['ci'])}; cand_on acc = {f(co['mean'])} "
               f"{fci(co['ci'])}.")
    n_pass = sum(1 for b in bars if b["passed"])
    out.append("")
    out.append(f"Bars: {n_pass} / 9 pass "
               f"({sum(1 for b in bars if b['passed'] is None)} N/A).")
    text = "\n".join(out)
    with open(os.path.join(R19, "r19_analysis.md"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    if missing or not ok:
        print(f"COVERAGE INCOMPLETE: {n_lines} lines, "
              f"{len(missing)} gaps", file=sys.stderr)
        return 1
    if not frozen_ok:
        print("FROZEN SANITY FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
