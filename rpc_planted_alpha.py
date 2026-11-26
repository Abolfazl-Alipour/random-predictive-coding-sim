"""Planted motifs at alpha 1.0 (and 0.7), run by Muse directly.

The remaining open question from section 14: does the alpha fix work
on the REAL thesis case, motifs planted in random background noise,
not pure repetition? At alpha 0.3 the planted task scored full 0.075
vs control 0.062 (gain +0.013, 5 seeds). Pure repetition at alpha 1.0
scored cycle 0.843 and motif 0.934, so the mechanism can learn a
deterministic sequence when the code separates. The planted task adds
random background tokens, which may or may not re-confuse the code.

Task: baselines.make_data (200 train / 50 test, length 24, K=16,
motifs (1,2,3), (5,6,7), (9,10,11) planted in random background),
5 seeds 0-4, 5 epochs, prototypes from all train sequences.
Alpha 1.0 and 0.7, conditions full / no-plasticity / no-confirmation.
Pass bars: acc >= 0.2625 (chance 0.0625 + 0.20) and gain >= 0.10.
Baselines at alpha 0.3: Markov-1 0.278, ESN 0.274, LSTM 0.283.
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import run

def run_planted(seed, alpha, epochs=5, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64, proto_n=200, alpha=alpha)
    return run(train, test, epochs=epochs, **{**defaults, **kw})

if __name__ == "__main__":
    for alpha in [1.0, 0.7]:
        for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                            ("no-confirmation", {"use_confirmation": False})]:
            accs, bursts = [], []
            for seed in range(5):
                r = run_planted(seed, alpha, **extra)
                accs.append(r["acc"]); bursts.append(r["mean_burst"])
            print(f"planted alpha{alpha} {name}: acc mean={np.mean(accs):.3f} "
                  f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
                  f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
                  flush=True)
