"""Full smoke test for rpc_round15.py on Beast: one 5-epoch run per
code path (planted delta/tempcov/burst, sine delta, lorenz delta,
varsine delta), seed 201, val split. Usage:
  python rpc_round15_smoke_full.py <task-group>
Groups: planted (delta+tempcov+burst on planted1), sine, cont
(lorenz+varsine delta)."""
import sys

from rpc_round15 import compute_baselines, run_one, splits

GROUP = sys.argv[1]
CFG_PLANTED = dict(alpha=1.0, smooth_sigma=0.0, eta_d=0.05)
CFG_SINE = dict(alpha=0.7, smooth_sigma=3.0, eta_d=0.2)
CFG_CONT = dict(alpha=1.0, smooth_sigma=0.0, eta_d=0.2)

if GROUP == "planted":
    tr, va, te, _, _, _, K, dec = splits("planted1", 201)
    for cond in ("delta", "tempcov", "burst"):
        r = run_one("planted1", (tr, va, None, dec), CFG_PLANTED, cond, 201)
        print(f"planted1 val {cond} seed201: "
              f"acc={r['acc']:.3f} m1={r['m1']:.3f} m2={r['m2']:.3f} "
              f"oth={r['oth']:.3f} burst={r['mean_burst']:.3f} "
              f"clamp={r['clamp_frac']:.4f}", flush=True)
elif GROUP == "sine":
    tr, va, te, _, _, _, K, dec = splits("sine", 201)
    r = run_one("sine", (tr, va, None, dec), CFG_SINE, "delta", 201)
    print(f"sine val delta seed201: acc={r['acc']:.3f} "
          f"burst={r['mean_burst']:.3f} clamp={r['clamp_frac']:.4f}",
          flush=True)
elif GROUP == "cont":
    for task in ("lorenz", "varsine"):
        tr, va, te, tr_c, va_c, te_c, K, dec = splits(task, 201)
        r = run_one(task, (tr, va, va_c, dec), CFG_CONT, "delta", 201)
        print(f"{task} val delta seed201: acc={r['acc']:.3f} "
              f"within1={r['within1']:.3f} mse={r['mse']:.3f} "
              f"burst={r['mean_burst']:.3f} clamp={r['clamp_frac']:.4f}",
              flush=True)
        b = compute_baselines(task, tr, te, te_c, dec, 201)
        print(f"{task} baselines seed201:",
              {k: round(v, 3) for k, v in b.items()}, flush=True)
else:
    raise ValueError(GROUP)
print(f"SMOKE {GROUP} DONE", flush=True)
