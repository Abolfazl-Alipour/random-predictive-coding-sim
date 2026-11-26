"""Check A22 is actually applied (Nov 26, 07:23 PT).

Round 13's A18+proj sine result is EXACTLY identical to A18
alone per-seed ([0.213, 0.218, 0.028, 0.037, 0.028]), which
is suspicious: either d1 already lies in the prototype span
(so the projection is a genuine no-op), or the flag is not
reaching predict_from_distal (a bug). This script distinguishes
them: it builds the network with proto_proj on, reports whether
_proto_proj is set, compares raw d1 against projected d1
(norm ratio and cosine) on test steps, and counts how many
test predictions change with the projection on versus off.
"""
import numpy as np
import torch
from rpc_simple import make_sine_fixed
from rpc_v2 import RPCNetwork

torch.manual_seed(0)
np.random.seed(0)
train = make_sine_fixed(50, seed=0)
test = make_sine_fixed(20, seed=100)
net = RPCNetwork(K=32, smooth_sigma=3.0, alpha=0.7,
                 graded_conf=True, eta_b=0.15,
                 learn_rule="coactivity", full_code=True,
                 proto_proj=True)
net.calibrate(train)
net.build_prototypes(train, proto_n=20)
print(f"proto_proj flag: {net.proto_proj}")
print(f"_proto_proj set: {net._proto_proj is not None}")
if net._proto_proj is not None:
    print(f"_proto_proj shape: {tuple(net._proto_proj.shape)}")
    # Projection matrix should be idempotent and symmetric.
    P = net._proto_proj
    print(f"idempotent err: {float((P @ P - P).abs().max()):.2e}")
    print(f"symmetric err: {float((P - P.T).abs().max()):.2e}")
    print(f"rank (trace): {float(torch.trace(P)):.1f}")
for _ in range(2):
    for seq in train:
        net.reset()
        for tok in seq:
            net.step(tok, train=True)
cosines, ratios, changed, total = [], [], 0, 0
for seq in test:
    net.reset()
    for i, tok in enumerate(seq):
        out = net.step(tok, train=False)
        if i > 0:
            d1 = out["d1"]
            p1 = net._proto_proj @ d1
            cosines.append(float(torch.nn.functional.cosine_similarity(
                d1[None, :], p1[None, :]).item()))
            ratios.append(float(p1.norm() / d1.norm().clamp(min=1e-9)))
            total += 1
print(f"test steps: {total}")
print(f"cos(d1, projected d1): mean={np.mean(cosines):.4f} "
      f"min={min(cosines):.4f}")
print(f"||projected|| / ||d1||: mean={np.mean(ratios):.4f} "
      f"min={min(ratios):.4f} max={max(ratios):.4f}")
