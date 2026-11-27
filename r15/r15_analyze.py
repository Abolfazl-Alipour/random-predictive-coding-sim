"""Round-15 analysis: reads all test JSONLs, computes per-task /
per-condition means, medians, IQRs, 95% CIs, paired differences, and
checks every protocol bar. Writes r15/r15_analysis.md and prints a
summary. Run by r15/monitor.py when the test phase completes."""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["planted1", "planted2", "sine", "lorenz", "varsine"]
CONDITIONS = ["delta", "tempcov", "burst", "frozen"]
LABELS = {"planted1": "Planted lag-1", "planted2": "Planted lag-2",
          "sine": "Sine-fixed", "lorenz": "Lorenz", "varsine": "Var sine"}


def ci(vals):
    v = np.array(vals, dtype=float)
    m, s = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
    half = 1.96 * s / math.sqrt(len(v))
    return m, m - half, m + half


def stats(vals):
    v = np.array(vals, dtype=float)
    m, lo, hi = ci(vals)
    return {"mean": float(m), "median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)),
                    float(np.percentile(v, 75))],
            "ci_lo": float(lo), "ci_hi": float(hi), "n": len(v)}


def load():
    recs = []
    for task in TASKS:
        for shard in ("a", "b"):
            p = os.path.join(HERE, f"r15_{task}_test_{shard}.jsonl")
            with open(p) as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
    return recs


def main():
    recs = load()
    by = {}
    base = {}
    for r in recs:
        if r["config"] == "baseline":
            base.setdefault(r["task"], {})[r["seed"]] = r
        else:
            by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r
    lines = ["# Round 15 analysis (preregistered replication)",
             "",
             "Protocol: PROTOCOL-R15.md. Seeds 201..220 (20 seeds).",
             "CI = mean +/- 1.96 * sample-std / sqrt(20).",
             ""]
    verdicts = []
    for task in TASKS:
        lines.append(f"## {LABELS[task]}")
        lines.append("")
        for cond in CONDITIONS:
            d = by.get((task, cond), {})
            accs = [d[s]["acc"] for s in sorted(d)]
            s = stats(accs)
            extra = ""
            if task in ("planted1", "planted2"):
                m1 = stats([d[x]["m1"] for x in sorted(d)])
                m2 = stats([d[x]["m2"] for x in sorted(d)])
                extra = (f" | m1 mean={m1['mean']:.3f} "
                         f"m2 mean={m2['mean']:.3f} "
                         f"all-motif mean="
                         f"{np.mean([d[x]['all_motif'] for x in sorted(d)]):.3f}")
            if task in ("lorenz", "varsine"):
                w1 = stats([d[x]["within1"] for x in sorted(d)])
                ms = stats([d[x]["mse"] for x in sorted(d)])
                extra = (f" | within1 mean={w1['mean']:.3f} "
                         f"mse mean={ms['mean']:.3f}")
            cf = float(np.mean([d[x]["clamp_frac"] for x in sorted(d)]))
            lines.append(f"- {cond}: acc mean={s['mean']:.3f} "
                         f"median={s['median']:.3f} "
                         f"IQR=[{s['iqr'][0]:.3f},{s['iqr'][1]:.3f}] "
                         f"95% CI=[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] "
                         f"n={s['n']} clamp_frac={cf:.4f}{extra}")
        # paired diffs (delta vs tempcov, delta vs frozen)
        for other in ("tempcov", "frozen"):
            diffs = [by[(task, "delta")][s]["acc"]
                     - by[(task, other)][s]["acc"]
                     for s in sorted(by[(task, "delta")])]
            m, lo, hi = ci(diffs)
            lines.append(f"- paired delta-{other}: mean diff={m:+.3f} "
                         f"95% CI=[{lo:+.3f},{hi:+.3f}]")
        # baselines
        b = base.get(task, {})
        if b:
            lines.append(f"- baselines (identical test seqs): "
                         f"Markov-1={np.mean([b[s]['markov1_acc'] for s in sorted(b)]):.3f} "
                         f"Markov-2={np.mean([b[s]['markov2_acc'] for s in sorted(b)]):.3f} "
                         f"persistence={np.mean([b[s]['persist_acc'] for s in sorted(b)]):.3f} "
                         f"ESN={np.mean([b[s]['esn_acc'] for s in sorted(b)]):.3f}")
            if task in ("lorenz", "varsine"):
                lines.append(f"- baseline MSE: Markov-1="
                             f"{np.mean([b[s]['markov1_mse'] for s in sorted(b)]):.3f} "
                             f"persistence="
                             f"{np.mean([b[s]['persist_mse'] for s in sorted(b)]):.3f} "
                             f"ESN={np.mean([b[s]['esn_mse'] for s in sorted(b)]):.3f}")
        lines.append("")
        # bars
        d = by[(task, "delta")]
        s = stats([d[x]["acc"] for x in sorted(d)])
        bb = base.get(task, {})
        m1acc = float(np.mean([bb[x]["markov1_acc"] for x in sorted(bb)])) if bb else float("nan")
        pacc = float(np.mean([bb[x]["persist_acc"] for x in sorted(bb)])) if bb else float("nan")
        if task == "planted1":
            m1 = stats([d[x]["m1"] for x in sorted(d)])
            m2 = stats([d[x]["m2"] for x in sorted(d)])
            diffs = [d[x]["acc"] - by[(task, "tempcov")][x]["acc"]
                     for x in sorted(d)]
            _, dlo, _ = ci(diffs)
            checks = [
                ("lower 95% CI > 0.22", s["ci_lo"] > 0.22),
                ("target mean >= 0.25", s["mean"] >= 0.25),
                ("m1 mean >= 0.80", m1["mean"] >= 0.80),
                ("m2 mean >= 0.90", m2["mean"] >= 0.90),
                ("beats tempcov paired (CI above 0)", dlo > 0),
            ]
        elif task == "planted2":
            checks = [
                ("mean >= 0.20", s["mean"] >= 0.20),
                ("mean >= Markov-1 + 0.10", s["mean"] >= m1acc + 0.10),
            ]
        elif task == "sine":
            checks = [("mean >= 0.80", s["mean"] >= 0.80)]
        else:
            f = stats([by[(task, "frozen")][x]["acc"] for x in sorted(d)])
            ms = stats([d[x]["mse"] for x in sorted(d)])
            bm = float(np.mean([bb[x]["markov1_mse"] for x in sorted(bb)])) if bb else float("nan")
            pm = float(np.mean([bb[x]["persist_mse"] for x in sorted(bb)])) if bb else float("nan")
            checks = [
                ("beats persistence on acc", s["mean"] > pacc),
                ("beats Markov-1 on acc", s["mean"] > m1acc),
                ("beats persistence on MSE", ms["mean"] < pm),
                ("beats Markov-1 on MSE", ms["mean"] < bm),
                ("gain >= +0.15 over frozen", s["mean"] - f["mean"] >= 0.15),
            ]
        for name, ok in checks:
            lines.append(f"- BAR {name}: {'PASS' if ok else 'FAIL'}")
            verdicts.append((task, name, ok))
        lines.append("")
    n_pass = sum(1 for _, _, ok in verdicts if ok)
    lines.append(f"## Summary: {n_pass}/{len(verdicts)} bars pass")
    with open(os.path.join(HERE, "r15_analysis.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
