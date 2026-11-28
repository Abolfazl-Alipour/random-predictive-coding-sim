"""Round 22 smoke test (VM only, no Beast).

Runs the driver on ONE val seed (347) with a small probe
(--n-prefix 5) and checks the machinery the protocol rests on:
  S1 wire-live gate: aligned gain 1.0 has mean fb_drive_ratio
     > 0.05 and p1_cos < 0.90 on the probe pass.
  S2 shuffle control: fb_drive_ratio on vs shuffle is reported;
     the shuffle is NOT magnitude-matched for the aligned wire
     (fbdr drops 19-47%), so the check is that the shuffle is
     live feedback (fbdr > 0.05) and smaller than on-mode, with
     the ratio reported. The magnitude confound on W2 is recorded
     in the protocol (self-critique point 13); the cut (W1) is
     the primary wire control.
  S3 the precision manipulation changes the error response:
     the sine interaction (gap0 - gap at c=0.5) > 0.10 raw on
     the aligned wire-on net, and |frozen interaction| < 0.05.
  S4 CLI resumable: running the noise val twice leaves the
     line count unchanged and produces no duplicate keys.
  S5 deterministic noise: probe_pass twice on the same trained
     net gives identical aggregates (integer-arithmetic seeds,
     never hash()).
  S6 lag-2 probe machinery: the planted2 interaction val runs
     and writes 8 complete config lines.
  S7 analyzer writes: the analyzer runs on the smoke outputs
     (scratch dir) and writes r22_analysis.md with a bar tally.
  S8 noise calibration: decP1 on the calibration trials falls
     with c (clean 1.00 -> <= 0.75 at c = 0.50 on planted1).

Usage: ./venv/bin/python rpc_round22_smoke.py [--keep]
--keep leaves /tmp/r22_smoke/ in place for inspection.
Exits 0 only if every check passes.
"""
import argparse
import json
import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

BASE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(BASE, "venv", "bin", "python")
DRIVER = os.path.join(BASE, "rpc_round22_precision.py")
ANALYZER = os.path.join(BASE, "r22", "r22_analyze.py")
SCRATCH = "/tmp/r22_smoke"
SEED = 347
N_PREFIX = 5

FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"command failed rc={r.returncode}: "
                         f"{' '.join(cmd)}")
    return r


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)

    # Noise val on planted1 + sine, one seed, small probe.
    noise_files = {}
    for task in ("planted1", "sine"):
        out = os.path.join(SCRATCH, f"noise_{task}.jsonl")
        noise_files[task] = out
        if os.path.exists(out):
            os.remove(out)
        run([PY, DRIVER, task, "val", "--seeds", f"{SEED}-{SEED}",
             "--out", out, "--val-kind", "noise",
             "--n-prefix", str(N_PREFIX)])
    # S4: resumability (run planted1 noise val again).
    n0 = len(load(noise_files["planted1"]))
    run([PY, DRIVER, "planted1", "val", "--seeds", f"{SEED}-{SEED}",
         "--out", noise_files["planted1"], "--val-kind", "noise",
         "--n-prefix", str(N_PREFIX)])
    n1 = len(load(noise_files["planted1"]))
    keys = [(r["task"], r["config"], r["seed"], r["c_low"])
            for r in load(noise_files["planted1"])]
    check("S4 resumable noise val", n0 == n1 == 5
          and len(set(keys)) == len(keys),
          f"lines before={n0} after={n1} unique keys={len(set(keys))}")

    # S8: decP1 falls with c on planted1 calibration (CTL).
    rows = load(noise_files["planted1"])
    dec = {r["c_low"]: r["cal_decP1"] for r in rows}
    check("S8 decP1 falls with c (planted1 CTL)",
          dec[0.10] >= dec[0.50] and dec[0.50] <= 0.75
          and dec[0.10] > 0.50,
          f"decP1 by c: {dec}")
    rows_s = load(noise_files["sine"])
    dec_s = {r["c_low"]: r["cal_decP1"] for r in rows_s}
    check("S8 decP1 falls with c (sine EXP)",
          dec_s[0.10] >= dec_s[0.25] and dec_s[0.25] <= 0.75,
          f"decP1 by c: {dec_s}")

    # c_low by the protocol rule (smallest c with decP1 < 0.75).
    def pick_c(dec):
        for c in (0.10, 0.15, 0.25, 0.35, 0.50):
            if dec[c] < 0.75:
                return c
        return 0.50
    c_low = {"planted1": pick_c(dec), "sine": pick_c(dec_s),
             "planted2": pick_c(dec)}
    winners_noise = {t: {"c_low": c_low[t]} for t in c_low}
    with open(os.path.join(SCRATCH, "winners_noise.json"),
              "w") as f:
        json.dump(winners_noise, f, indent=2)

    # Interaction val on planted1 + sine + planted2, one seed.
    inter_files = {}
    for task in ("planted1", "planted2", "sine"):
        out = os.path.join(SCRATCH, f"inter_{task}.jsonl")
        inter_files[task] = out
        if os.path.exists(out):
            os.remove(out)
        run([PY, DRIVER, task, "val", "--seeds", f"{SEED}-{SEED}",
             "--out", out, "--val-kind", "interaction",
             "--winners", os.path.join(SCRATCH,
                                       "winners_noise.json"),
             "--n-prefix", str(N_PREFIX)])

    # S6: lag-2 machinery.
    rows2 = load(inter_files["planted2"])
    cfgs = {r["config"] for r in rows2}
    key_fields = ["vio_c0_err", "exp_c0_err", "vio_cl_err",
                  "exp_cl_err", "vio_c0_cePres", "vio_c0_cePred"]
    complete = all(all(r[k] is not None for k in key_fields)
                   for r in rows2)
    check("S6 lag-2 probe machinery", len(rows2) == 8
          and len(cfgs) == 8 and complete,
          f"lines={len(rows2)} configs={sorted(cfgs)} "
          f"key fields complete={complete}")

    def by_cfg(task, cfg):
        for r in load(inter_files[task]):
            if r["config"] == cfg:
                return r
        return None

    def interaction(rec):
        ps = rec["p1scale"]
        gap0 = (rec["vio_c0_err"] - rec["exp_c0_err"]) / ps
        gapL = (rec["vio_cl_err"] - rec["exp_cl_err"]) / ps
        return gap0 - gapL, gap0

    # S1: wire-live gate (aligned-on, sine + planted1).
    for task in ("planted1", "sine"):
        rec = by_cfg(task, "aligned-on")
        fbdr = (rec["exp_c0_fbdr"] + rec["vio_c0_fbdr"]) / 2.0
        check(f"S1 wire live ({task})",
              fbdr > 0.05 and rec["p1_cos"] < 0.90,
              f"fbdr={fbdr:.3f} p1_cos={rec['p1_cos']:.3f}")

    # S2: shuffle control. FINDING (smoke run 1): the shuffle is
    # NOT magnitude-matched for the ALIGNED wire: fbdr drops
    # 19% on planted1 (0.251 -> 0.203) and 47% on sine
    # (0.302 -> 0.159). The aligned wire is a structured map
    # (-W_prox.T); permuting d1 moves it out of the subspace
    # where the map has large gain, so the shuffled feedback is
    # smaller. The check is therefore that the shuffle is LIVE
    # feedback (fbdr > 0.05) and smaller than on-mode, with the
    # ratio reported; the magnitude confound on W2 is recorded
    # in the protocol (self-critique point 13) and the cut (W1)
    # is the primary wire control.
    for task in ("planted1", "sine"):
        on = by_cfg(task, "aligned-on")
        sh = by_cfg(task, "aligned-shuffle")
        f_on = (on["exp_c0_fbdr"] + on["vio_c0_fbdr"]) / 2.0
        f_sh = (sh["exp_c0_fbdr"] + sh["vio_c0_fbdr"]) / 2.0
        check(f"S2 shuffle is live but smaller ({task})",
              f_sh > 0.05 and f_sh < f_on,
              f"fbdr on={f_on:.3f} shuffle={f_sh:.3f} "
              f"ratio={f_sh / f_on:.2f} (shuffle is NOT "
              f"magnitude-matched; see protocol 10.13)")

    # S3: the precision manipulation changes the error response.
    rec = by_cfg("sine", "aligned-on")
    inter, _ = interaction(rec)
    rec_f = by_cfg("sine", "frozen-aligned")
    inter_f, _ = interaction(rec_f)
    check("S3 interaction on sine, null on frozen",
          inter > 0.10 and abs(inter_f) < 0.05,
          f"aligned-on interaction={inter:.4f} rel_e; "
          f"frozen interaction={inter_f:.4f} rel_e")

    # S5: deterministic noise (in-process, same trained net).
    sys.path.insert(0, BASE)
    import numpy as np
    import rpc_round22_precision as d
    tr, _, te = d.splits("planted1", SEED)
    trials = d.build_probes("planted1", SEED, te, N_PREFIX)
    net, _ = d.train_net(tr, d.cfg_for("planted1"), "aligned",
                         1.0, SEED, 5, "trained", 16)
    a1, _ = d.probe_pass(net, trials, [0.0, 0.5], ["on"], SEED, 16)
    a2, _ = d.probe_pass(net, trials, [0.0, 0.5], ["on"], SEED, 16)
    same = all(
        abs(a1[m][c][t][k] - a2[m][c][t][k]) < 1e-12
        for m in a1 for c in a1[m] for t in a1[m][c]
        for k in a1[m][c][t])
    check("S5 deterministic noise", same,
          "probe_pass twice on the same net: identical" if same
          else "probe_pass differs between identical runs")

    # S7: analyzer writes on the smoke outputs.
    winner = {}
    for task in ("planted1", "planted2", "sine"):
        rows_t = load(inter_files[task])
        byc = {r["config"]: r for r in rows_t}
        def inter_of(rec):
            ps = rec["p1scale"]
            return ((rec["vio_c0_err"] - rec["exp_c0_err"]) / ps
                    - (rec["vio_cl_err"] - rec["exp_cl_err"]) / ps)
        def spec_inter_of(rec):
            return ((rec["vio_c0_cePres"] - rec["vio_c0_cePred"])
                    - (rec["vio_cl_cePres"] - rec["vio_cl_cePred"]))
        v_p1 = inter_of(byc["aligned-on"])
        v_p2 = spec_inter_of(byc["aligned-on"])
        v_p3 = ((byc["aligned-on"]["vio_c0_err"]
                 - byc["aligned-on"]["exp_c0_err"])
                / byc["aligned-on"]["p1scale"]
                - (byc["frozen-aligned"]["vio_c0_err"]
                   - byc["frozen-aligned"]["exp_c0_err"])
                / byc["frozen-aligned"]["p1scale"])
        winner[task] = {
            "c_low": c_low[task],
            "floor_P1": max(0.5 * v_p1, 0.02),
            "floor_P2": max(0.5 * v_p2, 0.05),
            "floor_P3": max(0.5 * v_p3, 0.02),
            "floor_W1": max(0.5 * (v_p1 - inter_of(
                byc["aligned-cut"])), 0.02),
            "floor_W2": max(0.5 * (v_p1 - inter_of(
                byc["aligned-shuffle"])), 0.02),
            "floor_W3": max(0.5 * (v_p1 - inter_of(
                byc["random-on"])), 0.02),
            "floor_W4": max(0.5 * (v_p1 - inter_of(
                byc["wire-off"])), 0.02),
        }
        # Analyzer expects test files named
        # r22_{task}_test_352-371.jsonl in the root.
        with open(os.path.join(
                SCRATCH, f"r22_{task}_test_352-371.jsonl"), "w") as f:
            for r in rows_t:
                f.write(json.dumps(r) + "\n")
    with open(os.path.join(SCRATCH, "winner.json"), "w") as f:
        json.dump(winner, f, indent=2)
    run([PY, ANALYZER, "--root", SCRATCH])
    ana = os.path.join(SCRATCH, "r22_analysis.md")
    ok = os.path.exists(ana) and "Bar tally" in open(ana).read()
    check("S7 analyzer writes", ok,
          f"{ana} exists with a bar tally" if ok
          else "analyzer did not write the analysis file")

    print()
    if FAILURES:
        print(f"SMOKE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SMOKE PASSED: all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
