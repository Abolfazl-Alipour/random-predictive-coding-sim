"""Round 14c: A24 delta rule on planted motifs, part 1 (delta + control).

Same design as rpc_sine_fix14c.py, on planted motifs. AGY's
1-epoch check: planted seed 0 A18 n_out 256 goes 0.041 -> 0.141
in one epoch. This is the 5-seed 5-epoch verdict.

Config (explicit, AGY's spec): learn_rule="delta", eta_d=0.05,
full_code=True, norm_rows=False, use_confirmation=False,
s_inh=0.0, alpha=1.0, one-hot, graded_conf=False, 5 epochs,
5 seeds. Conditions: delta at n_out 64 and 256, plus the
no-plasticity control on the delta-256 config (the gain
baseline). Readout is the unmodified p1-prototype cosine only.
Bars: planted accuracy >= 0.22 (0.2625 if the original strict
bar is kept) and gain >= +0.15, plus motif-only accuracy
>= 0.75.

Motif-only accuracy is scored by run_planted_motif, which
mirrors run() exactly (same data order, same defaults, same
training loop) and additionally counts only the steps where
the previous two tokens are a motif prefix (1,2)/(5,6)/(9,10)
and the current token is the motif successor (3/7/11).
"""
import numpy as np
import torch
from baselines import make_data
from rpc_v2 import RPCNetwork, run

DELTA = dict(learn_rule="delta", eta_d=0.05, full_code=True,
             norm_rows=False, use_confirmation=False, s_inh=0.0,
             alpha=1.0, graded_conf=False)

def run_planted_motif(seed, alpha, epochs=5, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    defaults = dict(K=16, theta_p=1.0, theta_d=1.0, eta_b=0.05,
                    eta_c=0.05, n_res=200, n_out=64, proto_n=200,
                    alpha=alpha)
    merged = {**defaults, **kw}
    step_kw = {k: merged.pop(k) for k in
               ("use_confirmation", "use_burst", "plasticity") if k in merged}
    proto_n = merged.pop("proto_n", 20)
    net = RPCNetwork(**merged)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=proto_n)
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **step_kw)
    if getattr(net, "drive_proto", False):
        net.build_drive_prototypes(train, proto_n=proto_n)
    correct = total = mcorrect = mtotal = 0
    for seq in test:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            if i > 0:
                total += 1
                correct += int(out["pred"] == tok)
                if i >= 2:
                    p = (seq[i - 2], seq[i - 1])
                    if ((p == (1, 2) and tok == 3) or
                            (p == (5, 6) and tok == 7) or
                            (p == (9, 10) and tok == 11)):
                        mtotal += 1
                        mcorrect += int(out["pred"] == tok)
    return {"acc": correct / max(total, 1),
            "motif_acc": mcorrect / max(mtotal, 1), "motif_n": mtotal}

if __name__ == "__main__":
    for name, extra in [
        ("delta-nout64", {"n_out": 64}),
        ("delta-nout256", {"n_out": 256}),
        ("delta256-no-plasticity", {"n_out": 256, "plasticity": False}),
    ]:
        accs, maccs, bursts = [], [], []
        for seed in range(5):
            r = run_planted_motif(seed, **DELTA, **extra)
            accs.append(r["acc"])
            maccs.append(r["motif_acc"])
            bursts.append(r["motif_n"])
        print(f"planted A24 {name}: acc mean={np.mean(accs):.3f} "
              f"range=[{min(accs):.3f},{max(accs):.3f}] chance=0.0625 "
              f"motif_acc mean={np.mean(maccs):.3f} "
              f"range=[{min(maccs):.3f},{max(maccs):.3f}] "
              f"motif_n={int(np.mean(bursts))} "
              f"accs={[round(a, 3) for a in accs]} "
              f"maccs={[round(a, 3) for a in maccs]}", flush=True)
