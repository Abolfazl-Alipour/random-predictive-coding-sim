"""Independent recomputation of round-22 headline numbers.

Written from the protocol definitions (PROTOCOL-R22-PRECISION.md
sections 6-8) and the raw JSONL fields only. Does NOT import
r22_analyze.py. Uses the stdlib statistics module; t(19)=2.0930.
Compares every headline mean/CI in r22_analysis.md to 3 decimals
and re-derives every bar verdict and the C verdict.
"""
import json, math, statistics

T19 = 2.0930
TASKS = ["planted1", "planted2", "sine"]
CONFIGS = ["aligned-on", "aligned-cut", "aligned-shuffle",
           "random-on", "random-cut", "random-shuffle",
           "wire-off", "frozen-aligned"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def mean_ci(vals):
    v = [x for x in vals if x is not None]
    m = statistics.fmean(v)
    s = statistics.stdev(v) if len(v) > 1 else 0.0
    half = T19 * s / math.sqrt(len(v))
    return m, m - half, m + half, len(v)


def med_iqr(vals):
    v = sorted(x for x in vals if x is not None)
    n = len(v)
    med = statistics.median(v)
    q1 = v[(n - 1) // 4] if n % 4 else statistics.median(v[:n // 2])
    # use linear-interpolation quantiles like numpy default
    def quant(p):
        h = (n - 1) * p
        lo = int(math.floor(h)); hi = int(math.ceil(h))
        return v[lo] + (v[hi] - v[lo]) * (h - lo)
    return med, quant(0.25), quant(0.75)


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys
         if a[k] is not None and b[k] is not None]
    return mean_ci(d)


def derive(r):
    ps = r["p1scale"]
    gap0 = (r["vio_c0_err"] - r["exp_c0_err"]) / ps
    gapL = (r["vio_cl_err"] - r["exp_cl_err"]) / ps
    spec0 = r["vio_c0_cePres"] - r["vio_c0_cePred"]
    specL = r["vio_cl_cePres"] - r["vio_cl_cePred"]
    d = {
        "gap0": gap0, "gapL": gapL,
        "interaction": gap0 - gapL,
        "interaction_raw": (r["vio_c0_err"] - r["exp_c0_err"])
                           - (r["vio_cl_err"] - r["exp_cl_err"]),
        "spec0": spec0, "specL": specL,
        "spec_interaction": spec0 - specL,
        "decE_vio": r["vio_c0_decE"],
        "fbdr": (r["exp_c0_fbdr"] + r["vio_c0_fbdr"]) / 2.0,
        "p1_cos": r["p1_cos"],
        "normal_acc": r["normal_acc"],
    }
    if r["unexp_c0_decP1"] is not None:
        d["decP1_diff"] = r["exp_c0_decP1"] - r["unexp_c0_decP1"]
        d["p1n_diff"] = (r["exp_c0_p1n"] - r["unexp_c0_p1n"]) / ps
    return d


def main():
    # ---- load test records (352-371 files only: 480 unique nets) ----
    by = {}
    n_test = 0
    for t in TASKS:
        for r in load(f"r22_{t}_test_352-371.jsonl"):
            by.setdefault((t, r["config"]), {})[r["seed"]] = derive(r)
            n_test += 1
    print(f"test records loaded: {n_test} (expect 480)")
    for t in TASKS:
        print(f"  {t}: " + ", ".join(
            f"{c}={len(by[(t, c)])}" for c in CONFIGS))

    winner = json.load(open("winner.json"))

    # ---- verify floors against val interaction files ----
    print("\n-- floor check: recompute val means, floor = max(0.02, 0.5*val) --")
    floor_bad = 0
    for t in TASKS:
        vby = {}
        for r in load(f"r22_{t}_val_interaction.jsonl"):
            vby.setdefault((t, r["config"]), {})[r["seed"]] = derive(r)
        ao, ac, ash, ro, off, fr = (vby[(t, c)] for c in
            ["aligned-on", "aligned-cut", "aligned-shuffle",
             "random-on", "wire-off", "frozen-aligned"])
        val = {
            "interaction": mean_ci([d["interaction"] for d in ao.values()])[0],
            "spec_interaction": mean_ci([d["spec_interaction"] for d in ao.values()])[0],
            "gap_diff": paired({s: d["gap0"] for s, d in ao.items()},
                               {s: d["gap0"] for s, d in fr.items()})[0],
            "on_cut": paired({s: d["interaction"] for s, d in ao.items()},
                             {s: d["interaction"] for s, d in ac.items()})[0],
            "on_shuffle": paired({s: d["interaction"] for s, d in ao.items()},
                                 {s: d["interaction"] for s, d in ash.items()})[0],
            "aligned_random": paired({s: d["interaction"] for s, d in ao.items()},
                                     {s: d["interaction"] for s, d in ro.items()})[0],
            "on_off": paired({s: d["interaction"] for s, d in ao.items()},
                             {s: d["interaction"] for s, d in off.items()})[0],
        }
        w = winner[t]
        for k, v in val.items():
            stored = w["val"][k]
            flag = "" if abs(v - stored) < 5e-4 else "  <-- MISMATCH"
            if flag:
                floor_bad += 1
            print(f"  {t} val[{k}] recomputed={v:.4f} stored={stored:.4f}{flag}")
        for fk, vk in [("floor_P1", "interaction"), ("floor_P2", "spec_interaction"),
                       ("floor_P3", "gap_diff"), ("floor_W1", "on_cut"),
                       ("floor_W2", "on_shuffle"), ("floor_W3", "aligned_random"),
                       ("floor_W4", "on_off")]:
            expect = max(0.02, 0.5 * val[vk])
            got = w[fk]
            flag = "" if abs(expect - got) < 5e-4 else "  <-- MISMATCH"
            if flag:
                floor_bad += 1
            print(f"  {t} {fk} recomputed={expect:.4f} stored={got:.4f}{flag}")
        print(f"  {t} c_low={w['c_low']}")

    # ---- recompute every bar ----
    print("\n-- bar recomputation vs r22_analysis.md (means to 3 decimals) --")
    analysis = open("r22_analysis.md").read()
    bars = {}
    mismatches = []

    def check(label, mine, stored_str):
        # stored_str like "0.0425" parsed from analysis text
        ok = abs(mine - float(stored_str)) < 5e-4
        if not ok:
            mismatches.append((label, mine, stored_str))
        print(f"  {label}: mine={mine:.4f} analysis={stored_str} "
              f"{'OK' if ok else 'MISMATCH'}")

    for t in TASKS:
        w = winner[t]
        ao, ac, ash, ro, off, fr = (by[(t, c)] for c in
            ["aligned-on", "aligned-cut", "aligned-shuffle",
             "random-on", "wire-off", "frozen-aligned"])
        inter = lambda d: {s: x["interaction"] for s, x in d.items()}
        spec = lambda d: {s: x["spec_interaction"] for s, x in d.items()}
        gap0 = lambda d: {s: x["gap0"] for s, x in d.items()}

        m, lo, hi, n = mean_ci([x["interaction"] for x in ao.values()])
        bars[f"P1 {t}"] = (m >= w["floor_P1"]) and (lo > 0) and n == 20
        check(f"P1 {t} interaction mean", m,
              _find(analysis, t, "P1 interaction", "mean="))
        med, q1, q3 = med_iqr([x["interaction"] for x in ao.values()])
        check(f"P1 {t} interaction median", med,
              _find(analysis, t, "P1 interaction", "median="))
        check(f"P1 {t} interaction CI lo", lo,
              _find_ci(analysis, t, "P1 interaction")[0])
        check(f"P1 {t} interaction CI hi", hi,
              _find_ci(analysis, t, "P1 interaction")[1])

        m2, lo2, hi2, _ = mean_ci([x["spec_interaction"] for x in ao.values()])
        bars[f"P2 {t}"] = (m2 >= w["floor_P2"]) and (lo2 > 0)
        check(f"P2 {t} spec_interaction mean", m2,
              _find(analysis, t, "P2 spec interaction", "mean="))

        m3, lo3, hi3, n3 = paired(gap0(ao), gap0(fr))
        bars[f"P3 {t}"] = (m3 >= w["floor_P3"]) and (lo3 > 0) and n3 == 20
        check(f"P3 {t} gap0 diff mean", m3,
              _find(analysis, t, "P3 gap0 trained", "mean="))

        s0m, s0lo, s0hi, _ = mean_ci([x["spec0"] for x in ao.values()])
        dvm, dvlo, dvhi, _ = mean_ci([x["decE_vio"] for x in ao.values()])
        bars[f"P4 {t}"] = (s0m >= 0.25 and s0lo > 0.15
                           and dvm >= 0.50 and dvlo > 0.30)
        check(f"P4 {t} spec0 mean", s0m,
              _find(analysis, t, "P4 spec0", "mean="))
        check(f"P4 {t} decE_vio mean", dvm,
              _find(analysis, t, "P4 spec0", "decE_VIO: mean="))

        mW1, loW1, hiW1, _ = paired(inter(ao), inter(ac))
        bars[f"W1 {t}"] = (mW1 >= w["floor_W1"]) and (loW1 > 0)
        check(f"W1 {t} on-cut mean", mW1,
              _find(analysis, t, "W1 interaction on - cut", "mean="))

        mW2, loW2, hiW2, _ = paired(inter(ao), inter(ash))
        bars[f"W2 {t}"] = (mW2 >= w["floor_W2"]) and (loW2 > 0)
        check(f"W2 {t} on-shuffle mean", mW2,
              _find(analysis, t, "W2 interaction on - shuffle", "mean="))

        mW3, loW3, hiW3, _ = paired(inter(ao), inter(ro))
        bars[f"W3 {t}"] = (mW3 >= w["floor_W3"]) and (loW3 > 0)
        check(f"W3 {t} aligned-random mean", mW3,
              _find(analysis, t, "W3 interaction aligned - random", "mean="))

        mW4, loW4, hiW4, _ = paired(inter(ao), inter(off))
        bars[f"W4 {t}"] = (mW4 >= w["floor_W4"]) and (loW4 > 0)
        check(f"W4 {t} on-off mean", mW4,
              _find(analysis, t, "W4 interaction on - off", "mean="))

        # W1s diagnostic
        mcut, locut, hicut, _ = mean_ci([x["interaction"] for x in ac.values()])
        removed = abs(mcut) < w["floor_W1"] and locut <= 0 <= hicut
        check(f"W1s {t} interaction_cut mean", mcut,
              _find(analysis, t, "W1s (diagnostic) interaction_cut", "mean="))
        print(f"  W1s {t}: cut interaction={mcut:.4f} vs on="
              f"{m:.4f} -> persists {mcut / m * 100:.1f}% of on"
              if m else "")
        print(f"  W1s {t}: {'REMOVED' if removed else 'PERSISTS'} by cut")

        fbm, _, _, _ = mean_ci([x["fbdr"] for x in ao.values()])
        pcm, _, _, _ = mean_ci([x["p1_cos"] for x in ao.values()])
        bars[f"W5 {t}"] = (fbm > 0.05) and (pcm < 0.90)
        check(f"W5 {t} fbdr mean", fbm,
              _find(analysis, t, "W5 gate fbdr", "mean="))
        check(f"W5 {t} p1_cos mean", pcm,
              _find(analysis, t, "W5 gate fbdr", "p1_cos: mean="))

        mW6, loW6, hiW6, n6 = paired(
            {s: x["normal_acc"] for s, x in ao.items()},
            {s: 0.8 * x["normal_acc"] for s, x in off.items()})
        ratio = (statistics.fmean([x["normal_acc"] for x in ao.values()])
                 / statistics.fmean([x["normal_acc"] for x in off.values()]))
        bars[f"W6 {t}"] = (loW6 > 0) and n6 == 20
        check(f"W6 {t} on-0.8off mean diff", mW6,
              _find(analysis, t, "W6 normal_acc on vs 0.8*off", "mean diff="))
        print(f"  W6 {t}: ratio on/off={ratio:.3f}")

        raw_ao, _, _, _ = mean_ci([x["interaction_raw"] for x in ao.values()])
        check(f"DIAG {t} interaction_raw aligned-on mean", raw_ao,
              _find(analysis, t, "DIAG interaction_raw (aligned-on)", "mean="))

        if t != "sine":
            s1m, s1lo, s1hi, _ = mean_ci([x["decP1_diff"] for x in off.values()])
            bars[f"S1 {t}"] = abs(s1m) < 0.03 and s1lo <= 0 <= s1hi
            check(f"S1 {t} decP1_diff mean", s1m,
                  _find(analysis, t, "S1 decP1 EXP-UNEXP wire-off", "mean="))
            s2m, s2lo, s2hi, _ = mean_ci([x["p1n_diff"] for x in off.values()])
            bars[f"S2 {t}"] = abs(s2m) < 0.05 and s2lo <= 0 <= s2hi
            check(f"S2 {t} p1n_diff mean", s2m,
                  _find(analysis, t, "S2 p1n EXP-UNEXP wire-off", "mean="))
            s3m, _, _, _ = mean_ci([x["p1n_diff"] for x in ao.values()])
            check(f"S3 {t} p1n_diff wire-on mean", s3m,
                  _find(analysis, t, "S3 (measurement only) wire-on",
                        "p1n diff mean="))

    print("\n-- bar verdicts (mine vs analysis) --")
    npass = 0
    for name, ok in bars.items():
        want = f"{'PASS' if ok else 'FAIL'} {name}:"
        found = want in analysis
        if not found:
            mismatches.append((f"bar verdict {name}", ok, "analysis disagrees"))
        npass += ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}"
              f"  {'matches analysis' if found else 'MISMATCH vs analysis'}")
    print(f"\ntally: {npass} of {len(bars)} bars pass "
          f"(analysis says 24 of 34)")

    v = ("C STRONG PASS" if (bars["P1 sine"] and bars["P2 sine"]
                             and bars["P3 sine"] and bars["P4 sine"]
                             and bars["P1 planted1"]
                             and bars["W1 sine"] and bars["W2 sine"]
                             and bars["W5 sine"])
         else "NOT C STRONG PASS")
    print(f"verdict (protocol s8 rule 1 + W5-sine gate): {v}")
    print(f"floor mismatches: {floor_bad}, value mismatches: {len(mismatches)}")
    if mismatches:
        print("MISMATCHES:", mismatches)


LABELS = {"planted1": "## Planted lag-1 (K=16)",
          "planted2": "## Planted lag-2 (K=16)",
          "sine": "## Sine (K=32)"}


def _section_lines(text, task):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l == LABELS[task])
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def _find(text, task, line_start, field):
    """Find, within the task's own section, the line starting with
    line_start and return its field= numeric string."""
    import re
    for line in _section_lines(text, task):
        if line.startswith(line_start):
            m = re.search(re.escape(field) + r"(-?[\d.]+)", line)
            if m:
                return m.group(1)
    raise KeyError(f"not found: {task} / {line_start} / {field}")


def _find_ci(text, task, line_start):
    import re
    for line in _section_lines(text, task):
        if line.startswith(line_start):
            m = re.search(r"95% CI=\[(-?[\d.]+),(-?[\d.]+)\]", line)
            if m:
                return m.group(1), m.group(2)
    raise KeyError(f"CI not found: {task} / {line_start}")


if __name__ == "__main__":
    main()
