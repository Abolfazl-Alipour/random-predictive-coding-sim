"""Independent recomputation of round-21 results from raw JSONL.
Written fresh for verification; does not import r21_analyze.py.
Compares every headline number against r21_analysis.md to 3 decimals."""
import json, glob, math, statistics

TASKS = ["lorenz", "varsine", "narma10", "mackeyglass"]
T19 = 2.0930
T4 = 2.7760
DIR = "/home/hatch/workspace/thesis-rpc/r21"

def load(pat):
    recs = []
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            if line.strip():
                recs.append(json.loads(line))
    return recs

def mean(v):
    return sum(v) / len(v)

def ci(vals, t=T19):
    m = mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    h = t * s / math.sqrt(len(vals))
    return m, m - h, m + h

def paired(a, b):
    # a, b: dicts seed -> value; diff = b - a (a beats b when diff > 0)
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    m, lo, hi = ci(d)
    return m, lo, hi, len(d)

# --- integrity: merged files vs shards ---
for task in TASKS:
    merged = [json.loads(l) for l in open(f"{DIR}/r21_{task}_test_327-346.jsonl")]
    keys = {(r["task"], r["config"], r["seed"]) for r in merged}
    print(f"{task}: merged lines={len(merged)} unique={len(keys)}")

test = load(f"{DIR}/r21_*_test_327-346.jsonl")
by = {}
for r in test:
    by.setdefault((r["task"], r["config"]), {})[r["seed"]] = r

def vec(task, cfg, field):
    return {s: r[field] for s, r in by.get((task, cfg), {}).items()}

# --- per-task means table ---
print("\nPER-TASK MEANS (mse unless noted)")
for task in TASKS:
    row = []
    for cfg in ("token_delta", "token_frozen", "token_w_delta", "bump_delta"):
        row.append(f"{mean(list(vec(task, cfg, 'mse').values())):.4f}")
    bl = by.get((task, "baseline"), {})
    for f in ("persist_mse", "markov2_mse", "esn_mse", "esn_scalar_std_mse"):
        row.append(f"{mean([r[f] for r in bl.values()]):.4f}")
    print(task, " | ".join(row))

# --- W bars (wall condition) ---
print("\nWALL BARS")
for task, w1, w7 in (("lorenz", "W1", "W7"), ("varsine", "W2", "W8")):
    m, lo, hi, n = paired(vec(task, "token_delta", "mse"), vec(task, "baseline", "persist_mse"))
    print(f"{w1} {task} delta-vs-persist: mean={m:+.4f} CI=[{lo:.4f},{hi:.4f}] n={n} pass={lo > 0}")
    m, lo, hi, n = paired(vec(task, "token_delta", "mse"), vec(task, "token_frozen", "mse"))
    print(f"{w7} {task} delta-vs-frozen: mean={m:+.4f} CI=[{lo:.4f},{hi:.4f}] n={n} pass={lo > 0}")

# --- all 54 bars, count pass/fail/n/a ---
bars = []
def diff_bar(name, task, ca, fa, cb, fb, direction="a_lt_b"):
    va, vb = vec(task, ca, fa), vec(task, cb, fb)
    if direction == "a_lt_b":
        m, lo, hi, n = paired(va, vb)
    else:
        m, lo, hi, n = paired(vb, va)
    ok = (n == 20 and lo > 0)
    bars.append((name, ok, m, lo, hi, n))

for task, w1, w3, w5, w7, w9 in (("lorenz", "W1", "W3", "W5", "W7", "W9"),
                                  ("varsine", "W2", "W4", "W6", "W8", "W10")):
    diff_bar(w1, task, "token_delta", "mse", "baseline", "persist_mse")
    diff_bar(w3, task, "token_delta", "mse", "baseline", "esn_mse")
    diff_bar(w5, task, "token_delta", "acc", "baseline", "markov1_acc", "a_gt_b")
    diff_bar(w7, task, "token_delta", "mse", "token_frozen", "mse")
    if (task, "token_primary_delta") in by:
        diff_bar(w9, task, "token_primary_delta", "mse", "baseline", "persist_mse")
    else:
        bars.append((w9, None, None, None, None, 0))

for task, n1, n3, n5, n7 in (("narma10", "N1", "N3", "N5", "N7"),
                              ("mackeyglass", "N2", "N4", "N6", "N8")):
    diff_bar(n1, task, "token_delta", "mse", "baseline", "persist_mse")
    diff_bar(n3, task, "token_delta", "mse", "baseline", "esn_mse")
    diff_bar(n5, task, "token_delta", "mse", "token_frozen", "mse")
    diff_bar(n7, task, "token_delta", "mse", "baseline", "markov2_mse")

for task in TASKS:
    diff_bar("R1", task, "token_delta", "ridge_std_mse", "token_frozen", "ridge_std_mse")
    diff_bar("R3", task, "token_delta", "ridge_std_mse", "baseline", "persist_mse")
    diff_bar("R4", task, "token_delta", "ridge_std_mse", "baseline", "esn_scalar_std_mse")
    diff_bar("C1", task, "token_delta", "centroid_mse_10", "token_delta", "mse")
    diff_bar("C2", task, "token_delta", "centroid_mse_10", "baseline", "persist_mse")
    va, vw = vec(task, "token_delta", "mse"), vec(task, "token_w_delta", "mse")
    m, lo, hi, n = paired(va, vw)
    bars.append((f"F1-{task}", None, m, lo, hi, n))
    fb = [r["fb_drive_ratio"] for r in by.get((task, "token_w_delta"), {}).values()]
    fbm, flo, fhi = ci(fb)
    bars.append((f"F2-{task}", fbm > 0.05, fbm, flo, fhi, len(fb)))
    diff_bar("B1", task, "bump_delta", "mse", "token_delta", "mse")
    diff_bar("B2", task, "bump_delta", "mse", "baseline", "persist_mse")

npass = sum(1 for b in bars if b[1] is True)
nfail = sum(1 for b in bars if b[1] is False)
nna = sum(1 for b in bars if b[1] is None)
print(f"\nBARS: {npass} pass, {nfail} fail, {nna} n/a, total {len(bars)}")
for name, ok, m, lo, hi, n in bars:
    mark = "PASS" if ok is True else ("FAIL" if ok is False else "n/a")
    if m is None:
        print(f"  [{mark}] {name}: no data")
    else:
        print(f"  [{mark}] {name}: mean={m:+.4f} CI=[{lo:.4f},{hi:.4f}] n={n}")

# --- diagnostics spot check ---
print("\nDIAGNOSTICS")
for task in TASKS:
    d = by.get((task, "token_delta"), {})
    f = by.get((task, "token_frozen"), {})
    c5 = mean([r["centroid_mse_5"] for r in d.values()])
    c10 = mean([r["centroid_mse_10"] for r in d.values()])
    c20 = mean([r["centroid_mse_20"] for r in d.values()])
    rs = mean([r["ridge_std_mse"] for r in d.values()])
    rr = mean([r["ridge_raw_mse"] for r in d.values()])
    fs = mean([r["ridge_std_mse"] for r in f.values()])
    fr = mean([r["ridge_raw_mse"] for r in f.values()])
    wm, _, _, _ = paired(vec(task, "token_frozen", "mse"), vec(task, "token_w_frozen", "mse"))
    print(f"{task} centroid 5/10/20: {c5:.4f}/{c10:.4f}/{c20:.4f} | ridge trained std/raw {rs:.4f}/{rr:.4f} frozen std/raw {fs:.4f}/{fr:.4f} | wire-on-frozen {wm:+.4f}")

# --- val winners ---
print("\nVAL WINNERS")
val = load(f"{DIR}/r21_*_val_*.jsonl")
for task in TASKS:
    grids = {}
    for r in val:
        if r["task"] == task and r["config"].startswith("grid|"):
            grids.setdefault(r["config"], []).append(r["val_mse"])
    best = min(grids.items(), key=lambda kv: mean(kv[1]))
    bumps = {}
    for r in val:
        if r["task"] == task and r["config"].startswith("bumpsigma|"):
            bumps.setdefault(r["config"], []).append(r["val_mse"])
    bb = min(bumps.items(), key=lambda kv: mean(kv[1]))
    print(f"{task}: token winner {best[0]} mean={mean(best[1]):.4f} | bump winner {bb[0]} mean={mean(bb[1]):.4f}")
