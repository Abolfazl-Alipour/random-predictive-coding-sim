"""Diagnosis B: why d1 is identically zero, and what unblocks it.
p2 sequence is independent of theta_f (y2_prev only enters via d1), so we can
replay y2 = (p2 > theta_f) offline for candidate theta_f values using the
saved P2 array and the v1 W_dist1 (recreated with the same seed).
d1 at step t+1 = W_dist1 @ y2_t, d1 at first step of each sequence = 0.
"""
import numpy as np, torch, sys
sys.path.insert(0, "/home/hatch/workspace/thesis-rpc")
from rpc_toy import RPCNetwork

torch.manual_seed(0); np.random.seed(0)
z = np.load("/home/hatch/workspace/thesis-rpc/diag_arrays.npz")
P1, P2, is_motif = z["P1"], z["P2"], z["is_motif"]
net = RPCNetwork()
W = net.W_dist1.numpy()  # (64,64), init std 0.05
print("P(p2 > t):")
for t in [0.02,0.03,0.05,0.08,0.10,0.12,0.15]:
    print(f"  t={t}: {float((P2>t).mean()):.4f} per unit; steps with any y2: {float((P2>t).any(axis=1).mean()):.3f}")
print("\ntheta_f | y2 rate | d1 sd | d1 p90 | d1 p99 | d1 max")
for tf in [0.03,0.05,0.08,0.10]:
    Y2 = (P2 > tf).astype(np.float32)
    D1 = np.zeros_like(P2)
    # sequences are 24 steps, in order
    for s in range(200):
        for j in range(1,24):
            D1[s*24+j] = W @ Y2[s*24+j-1]
    print(f"{tf:.2f} | {Y2.mean():.4f} | {D1.std():.4f} | {np.percentile(D1,90):.4f} | {np.percentile(D1,99):.4f} | {D1.max():.4f}")
    # burst sweep on this D1 with P1
    print("   theta_p theta_d -> burst/step, hit, false-hit")
    for tp in [0.05,0.1,0.2]:
        for td in [0.02,0.05,0.08,0.10]:
            b = ((P1>tp)&(D1>td))
            bs = b.any(axis=1)
            print(f"   {tp:.2f} {td:.2f} -> {b.mean():.4f}, {bs[is_motif].mean():.3f}, {bs[~is_motif].mean():.3f}")
