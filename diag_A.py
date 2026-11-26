"""Diagnosis A: plasticity-OFF distributions of p1, d1, p2, y2 in v1 RPCNetwork.
Replicates RPCNetwork.step() logic with train=False (no W updates, no inhib),
but records p1/d1/p2/y2 per step. Threshold sweep is then done offline on the
recorded arrays, which is exact for plasticity OFF because y2_prev depends
only on theta_f (fixed 0.5), not on theta_p/theta_d.
Motif positions: regenerated with position tracking (same generator as v1
make_data, but recording where each motif was planted). Hit = burst step
(any unit bursts, i.e. mean burst rate > 0) on a motif token at motif offset
1 or 2 (2nd/3rd token, where a prediction from the previous token could fire).
False hit = burst step elsewhere.
"""
import math, numpy as np, torch
import sys
sys.path.insert(0, "/home/hatch/workspace/thesis-rpc")
from rpc_toy import RPCNetwork, make_data

torch.manual_seed(0); np.random.seed(0)

def make_data_pos(n_seq=200, length=24, K=16, motifs=((1,2,3),(5,6,7),(9,10,11))):
    data, pos = [], []
    for _ in range(n_seq):
        seq = np.random.randint(0, K, size=length).tolist()
        mpos = set()
        for motif in motifs:
            p = np.random.randint(0, length - len(motif))
            seq[p:p+len(motif)] = list(motif)
            mpos.update([p+1, p+2])  # 2nd/3rd tokens of motif
        data.append(seq); pos.append(mpos)
    return data, pos

def collect(net, data):
    P1, D1, P2, Y2 = [], [], [], []
    with torch.no_grad():
        for seq in data:
            net.reset()
            for tok in seq:
                x = torch.nn.functional.one_hot(torch.tensor(tok), net.K).float()
                d1 = net.W_dist1 @ net.y2_prev
                p1, r1, _ = net.mod1.forward(x)
                u2 = torch.tanh(net.W_12 @ r1)
                p2, r2, _ = net.mod2.forward(u2)
                y2 = (net.g2 * p2 > net.theta_f).float()
                P1.append(p1.numpy().copy()); D1.append(d1.numpy().copy())
                P2.append(p2.numpy().copy()); Y2.append(y2.numpy().copy())
                net.y2_prev = y2.clone(); net.r2_prev = r2.clone(); net.p2_prev = p2.clone()
    return (np.stack(P1), np.stack(D1), np.stack(P2), np.stack(Y2))

def stats(a):
    return dict(mean=float(a.mean()), sd=float(a.std()), p50=float(np.percentile(a,50)),
                p90=float(np.percentile(a,90)), p99=float(np.percentile(a,99)),
                max=float(a.max()), frac_pos=float((a>0).mean()))

if __name__ == "__main__":
    data, pos = make_data_pos(200)
    net = RPCNetwork()  # defaults theta_p=theta_d=theta_f=0.5
    net.build_prototypes(make_data(n_seq=200))  # as in v1 run(); prototypes unused here except predict
    P1, D1, P2, Y2 = collect(net, data)
    print(f"steps={P1.shape[0]} N_o={P1.shape[1]}")
    for name, a in [("p1",P1),("d1",D1),("p2",P2),("y2",Y2)]:
        s = stats(a); print(name, {k: round(v,4) for k,v in s.items()})
    wn = torch.linalg.norm(net.W_dist1, dim=1).numpy()
    print("W_dist1 row norms:", {k: round(float(v),4) for k,v in
          [("mean",wn.mean()),("sd",wn.std()),("p50",np.percentile(wn,50)),
           ("p90",np.percentile(wn,90)),("p99",np.percentile(wn,99)),("max",wn.max())]})
    print("W_dist1 entries: mean %.4f sd %.4f (init std 0.05)" % (float(net.W_dist1.mean()), float(net.W_dist1.std())))
    print("y2 rate (mean over units/steps) = %.4f ; frac steps with any y2>0 = %.4f" % (Y2.mean(), (Y2.sum(1)>0).mean()))
    # motif mask per step
    is_motif = np.zeros(P1.shape[0], dtype=bool)
    i = 0
    for mpos in pos:
        for j in range(24):
            if j in mpos: is_motif[i] = True
            i += 1
    print(f"motif steps (offset 1-2): {is_motif.sum()} / {len(is_motif)} = {is_motif.mean():.3f}")
    print("\ntheta_p theta_d | burst/step | hit | false-hit | hit-false")
    for tp in [0.05,0.1,0.2,0.3,0.5,0.8]:
        for td in [0.05,0.1,0.2,0.3,0.5,0.8]:
            b = ((P1>tp)&(D1>td))  # (steps, units) bool
            burst_rate = b.mean()  # per unit per step
            burst_step = b.any(axis=1)
            hit = burst_step[is_motif].mean()
            fh = burst_step[~is_motif].mean()
            print(f"{tp:.2f} {td:.2f} | {burst_rate:.4f} | {hit:.3f} | {fh:.3f} | {hit-fh:+.3f}")
    # also: what fraction of (p1,d1) pairs pass each single threshold
    print("\nSingle-threshold pass fractions:")
    for t in [0.05,0.1,0.2,0.3,0.5,0.8]:
        print(f" t={t}: P(p1>t)={float((P1>t).mean()):.4f}  P(d1>t)={float((D1>t).mean()):.4f}  P(p1>t & d1>t)={float(((P1>t)&(D1>t)).mean()):.4f}")
    # correlation between p1 and d1 (same unit, same step)
    print("\ncorr(p1,d1) same-unit = %.4f ; corr(|p1|,|d1|) = %.4f" % (
        float(np.corrcoef(P1.ravel(), D1.ravel())[0,1]),
        float(np.corrcoef(np.abs(P1.ravel()), np.abs(D1.ravel()))[0,1])))
    np.savez("/home/hatch/workspace/thesis-rpc/diag_arrays.npz", P1=P1, D1=D1, P2=P2, Y2=Y2, is_motif=is_motif)
