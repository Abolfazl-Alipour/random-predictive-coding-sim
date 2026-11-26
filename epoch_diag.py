"""Per-epoch diagnosis: does accuracy track learning, or y2 runaway?
Runs one seed, evaluates held-out test after each train epoch, reports
acc / mean burst / mean y2 per epoch. Used to decide whether the
confirmation rule (W_prox growth) inflates drives without bound.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hatch/workspace/thesis-rpc")
from rpc_v2 import RPCNetwork, make_data

def run_epochs(seed, epochs=5, **kw):
    step_kw = {k: kw.pop(k) for k in ("use_confirmation", "use_burst", "plasticity") if k in kw}
    torch.manual_seed(seed); np.random.seed(seed)
    train = make_data(n_seq=200); test = make_data(n_seq=50)
    net = RPCNetwork(**kw)
    net.calibrate(train); net.build_prototypes(train)
    rows = []
    for ep in range(epochs):
        for seq in train:
            net.reset()
            for tok in seq:
                net.step(tok, train=True, **step_kw)
        correct, total, bursts, y2s = 0, 0, [], []
        for seq in test:
            net.reset()
            for i, tok in enumerate(seq):
                out = net.step(tok, train=False)
                if i > 0:
                    total += 1; correct += int(out["pred"] == tok)
                bursts.append(out["burst_rate"]); y2s.append(out["y2_rate"])
        rows.append((correct/total, float(np.mean(bursts)), float(np.mean(y2s))))
    return rows

if __name__ == "__main__":
    base = dict(theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05, n_res=200, n_out=64)
    for name, extra in [
        ("full", {}),
        ("no-confirmation", {"use_confirmation": False}),
        ("full eta_c=0.01", {"eta_c": 0.01}),
        ("full eta_c=0.2", {"eta_c": 0.2}),
        ("full + topk y2 (A6)", {"y2_topk": True}),
        ("no-confirmation + topk y2 (A6)", {"use_confirmation": False, "y2_topk": True}),
    ]:
        rows = run_epochs(0, **{**base, **extra})
        print(name, flush=True)
        for ep, (a, b, y) in enumerate(rows):
            print(f"  ep{ep+1}: acc={a:.3f} burst={b:.3f} y2={y:.3f}", flush=True)
