"""Round-21 analysis: continuous re-validation (Convention-A +
NARMA-10 / Mackey-Glass), standardized ridge probes, centroid
readout, aligned-wire on/off factor.

Protocol: PROTOCOL-R21-CONTINUOUS.md. Seeds 327..346 (20 fresh).
Reads r21/r21_{task}_test_327-346.jsonl and the val files,
computes per-task / per-condition means, medians, IQRs, EXACT
Student-t 95% CIs, paired differences, and checks every protocol
bar. Writes r21/r21_analysis.md and prints a summary.

CI convention: exact Student-t, NOT z=1.96. t(19) = 2.0930 for
the 20-seed test bars; t(4) = 2.7760 for 5-seed val summaries.
scipy is absent from the venv, so the constants are hardcoded
from the standard t table. A bar passes when the paired
per-seed difference has its lower 95% CI above 0, unless the
bar says otherwise (F1 is a three-way measurement, F2 is a
gate, the beta check is a fragility flag).
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["lorenz", "varsine", "narma10", "mackeyglass"]
LABELS = {"lorenz": "Lorenz x (K=32, len 24)",
          "varsine": "Variable-frequency sine (K=32, len 100)",
          "narma10": "NARMA-10 (K=32, len 100)",
          "mackeyglass": "Mackey-Glass D=17 (K=32, len 100)"}
T19 = 2.0930  # exact Student-t, 19 df, two-sided 95%.
T4 = 2.7760   # exact Student-t, 4 df, two-sided 95%.
N_TEST = 20


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


def paired(vals_a, vals_b):
    """Paired per-seed differences (a - b), exact-t CI."""
    d = [a - b for a, b in zip(vals_a, vals_b)
         if a is not None and b is not None]
    m, lo, hi = ci(d)
    return m, lo, hi, len(d)


def field_vec(by, task, config, field):
    """Per-seed values of `field`, keyed by seed."""
    rows = by.get((task, config), {})
    return {s: r.get(field) for s, r in rows.items()}


def paired_dict(d1, d2):
    keys = sorted(set(d1) & set(d2))
    return paired([d1[k] for k in keys], [d2[k] for k in keys])


def load_test():
    recs = []
    for task in TASKS:
        p = os.path.join(HERE, f"r21_{task}_test_327-346.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def load_val():
    recs = []
    for p in glob.glob(os.path.join(HERE, "r21_*_val_*.jsonl")):
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def check_bar(name, cond, detail):
    return {"name": name, "pass": bool(cond), "detail": detail}


def fci(tup):
    return f"[{tup[1]:.4f},{tup[2]:.4f}]"


def main():
    recs = load_test()
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    keys = {(r["task"], r["config"], r["seed"]) for r in recs}
    lines = ["# Round 21 analysis: continuous re-validation",
             "",
             "Protocol: PROTOCOL-R21-CONTINUOUS.md. Seeds 327..346 "
             "(20 fresh seeds). CI = mean +/- exact Student-t * "
             "sample-std / sqrt(n): t(19) = 2.0930 for test bars, "
             "t(4) = 2.7760 for val.",
             ""]
    # Data integrity: duplicates and seed counts.
    dup = len(recs) - len(keys)
    lines.append(f"Test rows: {len(recs)} (unique keys {len(keys)}, "
                 f"duplicates {dup}).")
    for task in TASKS:
        for cfg in ("token_delta", "token_frozen", "token_w_delta",
                    "token_w_frozen", "bump_delta", "bump_frozen",
                    "baseline"):
            n = len(by.get((task, cfg), {}))
            flag = "" if n == N_TEST else "  <-- EXPECTED 20"
            lines.append(f"  {task} {cfg}: {n} seeds{flag}")
        for cfg in ("token_primary_delta", "token_primary_frozen"):
            n = len(by.get((task, cfg), {}))
            if n:
                flag = "" if n == N_TEST else "  <-- EXPECTED 20"
                lines.append(f"  {task} {cfg}: {n} seeds{flag}")
    lines.append("")

    # Per-task summary table (prototype MSE, centroid, ridge).
    lines.append("## Per-task means (prototype MSE, 20 seeds)")
    lines.append("")
    lines.append("| task | token_delta proto | token_frozen proto | "
                 "token_w_delta proto | bump_delta proto | "
                 "persistence | Markov-2 | ESN argmax | "
                 "ESN scalar-std |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        def m(cfg, field="mse"):
            return stats([r.get(field)
                          for r in by.get((task, cfg), {}).values()])
        row = [LABELS[task]]
        for cfg in ("token_delta", "token_frozen", "token_w_delta",
                    "bump_delta"):
            row.append(f"{m(cfg)['mean']:.4f}" if m(cfg)["mean"] is not None
                       else "n/a")
        bl = by.get((task, "baseline"), {})
        for f in ("persist_mse", "markov2_mse", "esn_mse",
                  "esn_scalar_std_mse"):
            s = stats([r.get(f) for r in bl.values()])
            row.append(f"{s['mean']:.4f}" if s["mean"] is not None
                       else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    bars = []

    def diff_bar(name, task, cfg_a, field_a, cfg_b, field_b,
                 direction="a_lt_b"):
        """Bar: paired (b - a) for a_lt_b (pass when the lower CI
        of b - a is above 0, i.e. a beats b), or (a - b) for
        a_gt_b. Returns the bar dict."""
        va = field_vec(by, task, cfg_a, field_a)
        vb = field_vec(by, task, cfg_b, field_b)
        if direction == "a_lt_b":
            m, lo, hi, n = paired_dict(vb, va)
        else:
            m, lo, hi, n = paired_dict(va, vb)
        ok = bool(n == N_TEST and lo > 0)
        detail = (f"paired diff mean={m:+.4f} 95% CI={fci((m, lo, hi))} "
                  f"n={n}")
        return check_bar(name, ok, detail)

    # --- W bars: Convention-A re-validation (lorenz, varsine). ---
    for task, w1, w3, w5, w7, w9 in (
            ("lorenz", "W1", "W3", "W5", "W7", "W9"),
            ("varsine", "W2", "W4", "W6", "W8", "W10")):
        bars.append(diff_bar(
            f"{w1} ({task} primary proto MSE beats persistence)",
            task, "token_delta", "mse", "baseline", "persist_mse"))
        bars.append(diff_bar(
            f"{w3} ({task} primary proto MSE beats ESN argmax)",
            task, "token_delta", "mse", "baseline", "esn_mse"))
        bars.append(diff_bar(
            f"{w5} ({task} primary proto acc beats Markov-1)",
            task, "token_delta", "acc", "baseline", "markov1_acc",
            direction="a_gt_b"))
        bars.append(diff_bar(
            f"{w7} ({task} delta proto MSE beats frozen proto MSE)",
            task, "token_delta", "mse", "token_frozen", "mse"))
        if ("token_primary_delta" in
                [c for (t, c) in by if t == task]):
            bars.append(diff_bar(
                f"{w9} ({task} preregistered primary proto MSE beats "
                "persistence)",
                task, "token_primary_delta", "mse", "baseline",
                "persist_mse"))
        else:
            bars.append({"name": f"{w9} ({task} preregistered primary "
                         "proto MSE beats persistence)",
                         "pass": None,
                         "detail": "n/a: val winner == primary, no "
                         "separate primary arm"})

    # --- N bars: new discriminating tasks. ---
    for task, n1, n3, n5, n7 in (
            ("narma10", "N1", "N3", "N5", "N7"),
            ("mackeyglass", "N2", "N4", "N6", "N8")):
        bars.append(diff_bar(
            f"{n1} ({task} proto MSE beats persistence)",
            task, "token_delta", "mse", "baseline", "persist_mse"))
        bars.append(diff_bar(
            f"{n3} ({task} proto MSE beats ESN argmax)",
            task, "token_delta", "mse", "baseline", "esn_mse"))
        bars.append(diff_bar(
            f"{n5} ({task} delta proto MSE beats frozen proto MSE)",
            task, "token_delta", "mse", "token_frozen", "mse"))
        bars.append(diff_bar(
            f"{n7} ({task} proto MSE beats Markov-2)",
            task, "token_delta", "mse", "baseline", "markov2_mse"))

    # --- R / C / F / B bars: all four tasks. ---
    for task in TASKS:
        bars.append(diff_bar(
            f"R1 ({task} standardized trained ridge beats "
            "standardized frozen ridge)",
            task, "token_delta", "ridge_std_mse",
            "token_frozen", "ridge_std_mse"))
        bars.append(diff_bar(
            f"R3 ({task} standardized trained ridge beats persistence)",
            task, "token_delta", "ridge_std_mse",
            "baseline", "persist_mse"))
        bars.append(diff_bar(
            f"R4 ({task} standardized trained ridge beats ESN "
            "scalar-std ridge)",
            task, "token_delta", "ridge_std_mse",
            "baseline", "esn_scalar_std_mse"))
        bars.append(diff_bar(
            f"C1 ({task} centroid beta=10 beats prototype)",
            task, "token_delta", "centroid_mse_10",
            "token_delta", "mse"))
        bars.append(diff_bar(
            f"C2 ({task} centroid beta=10 beats persistence)",
            task, "token_delta", "centroid_mse_10",
            "baseline", "persist_mse"))
        # F1: three-way wire measurement (off - on), not pass/fail.
        va = field_vec(by, task, "token_delta", "mse")
        vw = field_vec(by, task, "token_w_delta", "mse")
        m, lo, hi, n = paired_dict(va, vw)
        if n == N_TEST and lo > 0:
            verdict = "HELPS (wire-on MSE lower, CI above 0)"
        elif n == N_TEST and hi < 0:
            verdict = "HURTS (wire-on MSE higher, CI below 0)"
        else:
            verdict = "NULL (CI covers 0)"
        bars.append({"name": f"F1 ({task} wire effect on proto MSE: "
                     "off - on)",
                     "pass": None,
                     "detail": f"mean={m:+.4f} 95% CI={fci((m, lo, hi))} "
                     f"n={n} -> {verdict}"})
        # F2: wire-live gate (mean fb_drive_ratio on the wire arm).
        wrows = by.get((task, "token_w_delta"), {})
        fb = stats([r.get("fb_drive_ratio") for r in wrows.values()])
        gate = (fb["mean"] is not None and fb["mean"] > 0.05)
        bars.append(check_bar(
            f"F2 ({task} wire-live gate: mean fb_drive_ratio > 0.05)",
            gate,
            f"mean fb_drive_ratio={fb['mean']:.4f} "
            f"95% CI=[{fb['ci_lo']:.4f},{fb['ci_hi']:.4f}] n={fb['n']}"
            if fb["mean"] is not None else "no wire rows"))
        bars.append(diff_bar(
            f"B1 ({task} bump proto MSE beats token proto MSE)",
            task, "bump_delta", "mse", "token_delta", "mse"))
        bars.append(diff_bar(
            f"B2 ({task} bump proto MSE beats persistence)",
            task, "bump_delta", "mse", "baseline", "persist_mse"))

    lines.append("## Bar table")
    lines.append("")
    npass = sum(1 for b in bars if b["pass"] is True)
    nfail = sum(1 for b in bars if b["pass"] is False)
    nna = sum(1 for b in bars if b["pass"] is None)
    lines.append(f"Pass/fail bars: {npass} pass, {nfail} fail, "
                 f"{nna} n/a or measurement-only, of {len(bars)} rows.")
    lines.append("")
    for b in bars:
        mark = ("PASS" if b["pass"] is True else
                "FAIL" if b["pass"] is False else "n/a")
        lines.append(f"1. [{mark}] {b['name']}: {b['detail']}")
    lines.append("")

    # Verdict rules (protocol section 8).
    def bar_pass(prefix):
        for b in bars:
            if b["name"].startswith(prefix + " "):
                return b["pass"] is True
        return False

    wall = []
    if bar_pass("W1") and bar_pass("W2"):
        if bar_pass("W7") and bar_pass("W8"):
            wall.append("WALL FALLS for the re-validation tasks: the "
                        "Convention-A prototype readout beats persistence "
                        "on fresh seeds AND the win is due to learning "
                        "(delta beats frozen).")
        else:
            wall.append("WALL PARTLY DOWN: the prototype readout beats "
                        "persistence, but delta does not beat frozen, so "
                        "the credit goes to the frozen reservoir, not to "
                        "learning.")
    else:
        wall.append("WALL SURVIVES: the Convention-A prototype readout "
                    "does not beat persistence on fresh seeds.")
    n7, n8 = bar_pass("N7"), bar_pass("N8")
    n1, n2 = bar_pass("N1"), bar_pass("N2")
    if n7 and n8:
        wall.append("DISCRIMINATION: RPC beats Markov-2 on both new "
                    "tasks; NARMA-10 and Mackey-Glass are genuine "
                    "multi-step-memory tasks and RPC competes on them.")
    elif n7 or n8:
        won = [t for t, p in (("narma10", n7), ("mackeyglass", n8))
               if p]
        lost = [t for t, p in (("narma10", n7), ("mackeyglass", n8))
                if not p]
        wall.append(f"BEATS MARKOV-2 ON ONE TASK ONLY: {', '.join(won)} "
                    f"passes N7/N8; {', '.join(lost)} does not. The "
                    f"tasks are not equally discriminating.")
    elif n1 and n2:
        wall.append("PARTIAL DISCRIMINATION: RPC beats persistence on "
                    "both new tasks but not Markov-2; the tasks need "
                    "memory but a 2-gram lookup still wins.")
    elif n1 or n2:
        won = [t for t, p in (("narma10", n1), ("mackeyglass", n2))
               if p]
        lost = [t for t, p in (("narma10", n1), ("mackeyglass", n2))
                if not p]
        wall.append(f"MIXED: RPC beats persistence on {', '.join(won)} "
                    f"only, not on {', '.join(lost)}.")
    else:
        wall.append("NO DISCRIMINATION: RPC does not beat persistence "
                    "on either new task.")
    lines.append("## Verdict")
    lines.append("")
    for w in wall:
        lines.append(f"1. {w}")
    lines.append("")
    lines.append("Round 21 does NOT establish predictive coding even if "
                 "the wall falls: round 20 failed the revised clamp (B') "
                 "and the generative rollout (D). A wall-fall here "
                 "establishes only that the signed-error rule learns "
                 "continuous trajectories. See protocol section 8.")
    lines.append("")

    # Diagnostics: centroid beta sensitivity, raw vs std ridge (D8),
    # wire effect on the frozen net.
    lines.append("## Diagnostics (not bars)")
    lines.append("")
    for task in TASKS:
        d = by.get((task, "token_delta"), {})
        f = by.get((task, "token_frozen"), {})
        c5 = stats([r.get("centroid_mse_5") for r in d.values()])
        c10 = stats([r.get("centroid_mse_10") for r in d.values()])
        c20 = stats([r.get("centroid_mse_20") for r in d.values()])
        lines.append(
            f"1. {task} centroid MSE beta 5/10/20: "
            f"{c5['mean']:.4f} / {c10['mean']:.4f} / {c20['mean']:.4f} "
            f"(beta=10 is primary; a >20% gap between the best beta "
            f"and beta=10 flags beta-sensitivity).")
        rs = stats([r.get("ridge_std_mse") for r in d.values()])
        rr = stats([r.get("ridge_raw_mse") for r in d.values()])
        fs = stats([r.get("ridge_std_mse") for r in f.values()])
        fr = stats([r.get("ridge_raw_mse") for r in f.values()])
        lines.append(
            f"1. {task} ridge MSE trained std/raw: {rs['mean']:.4f} / "
            f"{rr['mean']:.4f}; frozen std/raw: {fs['mean']:.4f} / "
            f"{fr['mean']:.4f} (D8: if raw frozen beats raw trained "
            f"while std trained beats std frozen, the round-18 "
            f"frozen-ridge win was a regularization artifact).")
        wf = by.get((task, "token_w_frozen"), {})
        wm, _, _, _ = paired_dict(
            field_vec(by, task, "token_frozen", "mse"),
            field_vec(by, task, "token_w_frozen", "mse"))
        lines.append(
            f"1. {task} wire effect on the FROZEN net (frozen-off minus "
            f"frozen-on proto MSE): {wm:+.4f} (positive means the wire "
            f"helps even without learning).")
    lines.append("")

    # Val summary.
    val = load_val()
    if val:
        lines.append("## Val summary (5 seeds 322-326)")
        lines.append("")
        for task in TASKS:
            for kind in ("grid", "bumpsigma"):
                rows = [r for r in val if r["task"] == task and
                        r["config"].startswith(kind)]
                if not rows:
                    continue
                cfgs = sorted({r["config"] for r in rows})
                for cfg in cfgs:
                    s = stats([r["val_mse"] for r in rows
                               if r["config"] == cfg], t=T4)
                    lines.append(f"1. {task} {cfg}: val proto MSE "
                                 f"{fmt(s)}")
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(HERE, "r21_analysis.md"), "w") as fo:
        fo.write(text)
    print(text)


if __name__ == "__main__":
    main()
