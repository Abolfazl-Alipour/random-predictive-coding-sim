"""Diagnostic: does d1 actually learn to point at the right prototype?

Train RPC v2 on the pure repeating cycle (0..7, K=8, seed 0), then on
held-out test sequences measure, at each step, the cosine of the distal
drive d1 with the CORRECT next-token prototype and with the best WRONG
prototype. If learning works, correct cosine rises above wrong cosine
after training. If it does not, the failure is in the distal learning
itself, not in the burst gate or the readout threshold.

Same measurement for the no-plasticity control.
"""
import numpy as np
import torch
from rpc_v2 import RPCNetwork
from rpc_simple import make_cycle

def measure(seed=0, plasticity=True, epochs=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_cycle(n_seq=100, length=96, K=8, seed=seed)
    test = make_cycle(n_seq=20, length=96, K=8, seed=seed + 1000)
    net = RPCNetwork(K=8, theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                     n_res=200, n_out=64)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=100)
    for _ in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, plasticity=plasticity)
    # Evaluate: cosine of d1 against prototypes, per step.
    proto = torch.nn.functional.normalize(net.prototypes, dim=1)
    correct_cos, wrong_cos, preds = [], [], []
    with torch.no_grad():
        for seq in test:
            net.reset()
            for i, tok in enumerate(seq):
                x = torch.nn.functional.one_hot(torch.tensor(tok), 8).float()
                d1 = net.W_dist1 @ net.y2_prev
                p1, r1, _ = net.mod1.forward(x)
                u2 = torch.tanh(net.W_12 @ r1)
                p2, r2, _ = net.mod2.forward(u2)
                y2 = net._y2_from_p2n(p2 / net.sd_p2)
                if i > 0:
                    dn = torch.nn.functional.normalize(d1[None, :], dim=1)
                    cos = (dn @ proto.T)[0]
                    correct_cos.append(float(cos[tok]))
                    wrong = cos.clone()
                    wrong[tok] = -10
                    wrong_cos.append(float(wrong.max()))
                    preds.append(int(torch.argmax(cos).item()) == tok)
                net.y2_prev.copy_(y2)
                net.r2_prev.copy_(r2)
                net.p2_prev.copy_(p2 / net.sd_p2)
    return {"acc": float(np.mean(preds)), "correct_cos": float(np.mean(correct_cos)),
            "wrong_cos": float(np.mean(wrong_cos))}

if __name__ == "__main__":
    for plasticity in [True, False]:
        r = measure(plasticity=plasticity)
        print(f"cycle diag plasticity={plasticity}: acc={r['acc']:.3f} "
              f"correct_cos={r['correct_cos']:.3f} wrong_cos={r['wrong_cos']:.3f}",
              flush=True)
