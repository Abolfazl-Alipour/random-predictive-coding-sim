"""Center-surround and normalization variants, run by Muse directly.

Reza's direction (Nov 25, 17:57 PT): keep trying from different
perspectives and from the literature; noise is handled in brain
circuits by center-surround inhibition and normalization.

Literature (all verified by search Nov 25):
- Srinivasan, Laughlin & Dubs 1982, Proc R Soc B: the retinal
  surround predicts the centre signal and subtracts it, protecting
  fine detail against intrinsic noise. Center-surround IS predictive
  coding, and its purpose is noise protection.
- "Sensory noise predicts divisive reshaping of receptive fields",
  PLoS Comput Biol 2017: optimal competition under signal-dependent
  noise takes the form of divisive inhibition from the surround.
- "Relating sparse and predictive coding to divisive normalization",
  PLOS Comput Biol 2025: sparse and predictive coding are unified
  through divisive normalization; the normalization signal acts as a
  contextual expectation.
- Carandini & Heeger 2012: normalization is the canonical cortical
  computation, giving robustness to input gain.
- "Divisive Normalization, Line Attractor Networks and Ideal
  Observers": divisive normalization approximates maximum likelihood
  estimation and plays a critical role in noise filtering.

Variants (A9 in rpc_v2.py), each a single change, planted motifs,
alpha 1.0, learn_rule="coactivity", prototype readout, 5 seeds 0-4,
5 epochs, proto_n=200. Screen is full-only; a variant's own
no-plasticity control runs only if its full accuracy jumps.
Reference: coactivity-full proto 0.072, control 0.064.
Pass bars: acc >= 0.2625 and gain >= 0.10.
1. surround: surround_beta=1.0, subtract the mean expansion activity
   before top-k, both modules. Diffuse background drive is suppressed
   before it reaches the reservoir.
2. d1-sharpen: d1_sharpen=True, sign-preserving square of d1 for the
   readout only (divisive normalization, global pool, exponent 2, up
   to a per-step scalar the cosine ignores). Burst gate untouched.
3. post-topk: post_topk=True, the coactivity post term is the top 10%
   of p1n instead of every cell above threshold. Winners suppress
   the rest; only winning cells learn.
"""
import numpy as np
from rpc_planted_alpha import run_planted

if __name__ == "__main__":
    for name, extra in [("surround", {"surround_beta": 1.0}),
                        ("d1-sharpen", {"d1_sharpen": True}),
                        ("post-topk", {"post_topk": True})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 norm {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
