"""Reza's "hit and solidify" rule on planted motifs, run by Muse directly.

Reza's proposal (Nov 25, 17:34 PT): module 2 is connected all-to-all to
module 1 at first, so its outputs are bound to hit the activity in
module 1 no matter what they are; that solidifies the connection; repeat
to learn the pattern.

What is actually new here (A8 in rpc_v2.py): the dense random W_dist1 is
NOT new, v2 already initialises it dense all-to-all (A2). The new part is
the learning rule. v2's rule is burst-gated: W_dist1 learns only when the
burst fires, and the burst needs the distal drive to already match the
proximal input. In noise the cue token's drive rarely matches, so the cue
mapping never gets built (cold start). A8 drops the gate on the learning:
W_dist1 += eta * outer(post, y2_prev) where post is whichever module-1
cells are active for the current input (p1n > theta_p). The first
co-occurrence already makes the connection; repetition solidifies it. The
burst still fires and still drives the confirmation rule and inhibition;
only the distal learning is ungated.

Task: planted motifs (baselines.make_data, 200 train / 50 test, length
24, K=16), alpha 1.0, 5 seeds 0-4, 5 epochs, proto_n=200, conditions
full / no-plasticity / no-confirmation, learn_rule="coactivity".
Reference numbers (burst rule, rpc_planted_alpha_results.txt, same task
and seeds): alpha 1.0 full 0.073, control 0.064, no-confirmation 0.072.
Pass bars: acc >= 0.2625 (chance 0.0625 + 0.20) and gain >= 0.10.
Sanity check: pure repeating motif at alpha 1.0 with the new rule
(burst rule: full 0.934, control 0.322).
"""
import numpy as np
from rpc_planted_alpha import run_planted
from rpc_simple import run_task

if __name__ == "__main__":
    for name, extra in [("full", {}), ("no-plasticity", {"plasticity": False}),
                        ("no-confirmation", {"use_confirmation": False})]:
        accs, bursts = [], []
        for seed in range(5):
            r = run_planted(seed, 1.0, learn_rule="coactivity", **extra)
            accs.append(r["acc"]); bursts.append(r["mean_burst"])
        print(f"planted alpha1.0 coactivity {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.062 "
              f"mean_burst={np.mean(bursts):.3f} accs={[round(a,3) for a in accs]}",
              flush=True)
    accs = []
    for seed in range(5):
        r = run_task("motif", seed, alpha=1.0, learn_rule="coactivity")
        accs.append(r["acc"])
    print(f"motif alpha1.0 coactivity full: acc mean={np.mean(accs):.3f} "
          f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.250 "
          f"accs={[round(a,3) for a in accs]}", flush=True)
