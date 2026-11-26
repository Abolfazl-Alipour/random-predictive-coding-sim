"""Smoke test for A18 (full graded code) and A19 (kNN readout).
Run by Muse directly. 1 epoch, seed 0, sine-fixed, coactivity,
smooth 3.0, alpha 0.7, graded + eta_b 0.15. Also measures what
drives the p2 scale drift that broke the naive relu(p2n): the
post-training y2 mean with confirmation on vs off, and with
plasticity off.
"""
from rpc_simple import run_task

if __name__ == "__main__":
    base = dict(learn_rule="coactivity", smooth_sigma=3.0, alpha=0.7,
                graded_conf=True, eta_b=0.15)
    for name, kw in [
        ("baseline cosine top6", {}),
        ("A18 full code", {"full_code": True}),
        ("A18 noconf", {"full_code": True, "use_confirmation": False}),
        ("A18 noplast", {"full_code": True, "plasticity": False}),
        ("A19 knn readout", {"knn_readout": True}),
        ("A18+A19", {"full_code": True, "knn_readout": True}),
    ]:
        r = run_task("sine-fixed", 0, epochs=1, **base, **kw)
        print(f"{name}: acc={r['acc']:.3f} burst={r['mean_burst']:.3f} "
              f"y2={r['mean_y2']:.3f} g={r['mean_g']:.3f}", flush=True)
