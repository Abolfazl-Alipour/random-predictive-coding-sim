"""Staged sweep for rpc_v2.py (sequential, same code path as rpc_v2.py main).
Metric: mean over seeds of (acc_full - acc_noplasticity), paired per seed.
The no-plasticity control does not depend on theta_p/theta_d (test
predictions use untrained weights; theta only gates training updates), so
one control run per (seed, N_r, N_o) is reused across theta/eta configs.
Theta grid is in z-scores on standardised drives: {0.5, 1.0, 2.0, 3.0}.
This spans the v1 raw grid {0.05, 0.1, 0.2, 0.3}: raw 0.05 is z~0.9 on p1
(sd 0.054) and z~0.5 on d1 (sd ~0.10-0.14); raw 0.1 is z~1.9/1.0; raw 0.2
is z~3.7/2.0; raw 0.3 is z~5.5/3.0.
Stages: A theta 4x4 x 5 seeds (eta 0.05, 200/64); B eta {0.01,0.05,0.2}
x 5 seeds at best theta; C sizes {(200,64),(200,128),(500,64),(500,128)}
x 5 seeds at best theta+eta. Results appended to sweep_results.csv.
"""
import csv, itertools, os, sys, numpy as np
sys.path.insert(0, "/home/hatch/workspace/thesis-rpc")
from rpc_v2 import run_experiment

OUT = "/home/hatch/workspace/thesis-rpc/sweep_results.csv"
SEEDS = [0, 1, 2, 3, 4]
_controls = {}

def control(seed, n_res, n_out):
    key = (seed, n_res, n_out)
    if key not in _controls:
        r = run_experiment(seed, plasticity=False, n_res=n_res, n_out=n_out)
        _controls[key] = r["acc"]
    return _controls[key]

def run_config(tag, seeds, **kw):
    n_res = kw.get("n_res", 200); n_out = kw.get("n_out", 64)
    accs, gains, bursts, y2s = [], [], [], []
    for s in seeds:
        r = run_experiment(s, **kw)
        c = control(s, n_res, n_out)
        accs.append(r["acc"]); gains.append(r["acc"] - c)
        bursts.append(r["mean_burst"]); y2s.append(r["mean_y2"])
    row = dict(tag=tag, theta_p=kw.get("theta_p", 1.0), theta_d=kw.get("theta_d", 1.0),
               eta_b=kw.get("eta_b", 0.05), eta_c=kw.get("eta_c", 0.05),
               n_res=n_res, n_out=n_out,
               acc_mean=float(np.mean(accs)), acc_min=float(np.min(accs)),
               acc_max=float(np.max(accs)),
               gain_mean=float(np.mean(gains)), gain_min=float(np.min(gains)),
               gain_max=float(np.max(gains)),
               burst_mean=float(np.mean(bursts)), y2_mean=float(np.mean(y2s)),
               accs=",".join(f"{a:.3f}" for a in accs))
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"{tag}: gain mean={row['gain_mean']:+.3f} range=[{row['gain_min']:+.3f},{row['gain_max']:+.3f}] "
          f"acc mean={row['acc_mean']:.3f} range=[{row['acc_min']:.3f},{row['acc_max']:.3f}] "
          f"burst={row['burst_mean']:.3f} y2={row['y2_mean']:.3f}", flush=True)
    return row

def main():
    # Stage A: theta grid
    results_a = []
    for tp, td in itertools.product([0.5, 1.0, 2.0, 3.0], [0.5, 1.0, 2.0, 3.0]):
        results_a.append(run_config(f"A_tp{tp}_td{td}", SEEDS, theta_p=tp, theta_d=td,
                                    eta_b=0.05, eta_c=0.05, n_res=200, n_out=64))
    best_a = max(results_a, key=lambda r: r["gain_mean"])
    print(f"BEST A: theta_p={best_a['theta_p']} theta_d={best_a['theta_d']} gain={best_a['gain_mean']:+.3f}", flush=True)
    # Stage B: eta at best theta
    results_b = []
    for eta in [0.01, 0.05, 0.2]:
        results_b.append(run_config(f"B_eta{eta}", SEEDS, theta_p=best_a["theta_p"],
                                    theta_d=best_a["theta_d"], eta_b=eta, eta_c=eta,
                                    n_res=200, n_out=64))
    best_b = max(results_b, key=lambda r: r["gain_mean"])
    print(f"BEST B: eta={best_b['eta_b']} gain={best_b['gain_mean']:+.3f}", flush=True)
    # Stage C: sizes at best theta+eta
    results_c = []
    for nr, no in [(200, 64), (200, 128), (500, 64), (500, 128)]:
        results_c.append(run_config(f"C_nr{nr}_no{no}", SEEDS, theta_p=best_b["theta_p"],
                                    theta_d=best_b["theta_d"], eta_b=best_b["eta_b"],
                                    eta_c=best_b["eta_c"], n_res=nr, n_out=no))
    best_c = max(results_c, key=lambda r: r["gain_mean"])
    print(f"BEST C: N_r={best_c['n_res']} N_o={best_c['n_out']} gain={best_c['gain_mean']:+.3f}", flush=True)
    print(f"OVERALL BEST: theta_p={best_c['theta_p']} theta_d={best_c['theta_d']} "
          f"eta_b=eta_c={best_c['eta_b']} N_r={best_c['n_res']} N_o={best_c['n_out']} "
          f"gain mean={best_c['gain_mean']:+.3f} acc mean={best_c['acc_mean']:.3f}", flush=True)

if __name__ == "__main__":
    main()
