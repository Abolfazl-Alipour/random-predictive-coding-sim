"""Fast smoke test for rpc_round15.py (no full training runs)."""
import numpy as np
import torch

from baselines import make_data
from rpc_round15 import (compute_baselines, make_var_sine_cont, splits)
from rpc_spatial import make_sine as make_sine_spatial
from rpc_v2 import RPCNetwork

# 1. var-sine cont generator reproduces make_sine bins exactly.
for seed in (201, 202, 203):
    b1, c1 = make_var_sine_cont(20, seed=seed)
    b2 = make_sine_spatial(20, seed=seed)
    assert b1 == b2, f"bins differ at seed {seed}"
    assert len(c1) == 20 and len(c1[0]) == 100
print("var-sine bins reproduce make_sine exactly: OK")

# 2. Splits: sizes and disjointness.
for task in ("planted1", "planted2", "sine", "lorenz", "varsine"):
    tr, va, te, tr_c, va_c, te_c, K, dec = splits(task, 201)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    assert tr != va and tr != te and va != te, f"{task} splits collide"
    print(f"{task}: train={len(tr)} val={len(va)} test={len(te)} K={K} "
          f"cont={'yes' if tr_c else 'no'}")

# 3. tempcov + burst branches on a tiny net (2 seqs, 1 epoch).
torch.manual_seed(0)
np.random.seed(0)
train = make_data(n_seq=2)
for rule, extra in [("tempcov", dict(eta_d=0.05)),
                    ("burst", dict(norm_rows=True))]:
    net = RPCNetwork(K=16, n_out=16, learn_rule=rule, full_code=True,
                     s_inh=0.0, **extra)
    net.calibrate(train)
    net.build_prototypes(train, proto_n=2)
    assert net.p1_ema is None and net.y2_ema is None
    for seq in train:
        net.reset()
        for tok in seq:
            net.step(tok, train=True, use_confirmation=False)
    if rule == "tempcov":
        assert net.p1_ema is not None and net.y2_ema is not None
    else:
        # burst never touches the EMA state; it stays None by design.
        assert net.p1_ema is None and net.y2_ema is None
    cf = float((net.W_dist1.abs() >= net.w_max - 1e-6).float().mean())
    print(f"{rule}: tiny run OK, clamp_frac={cf:.3f}, "
          f"W_dist1 finite={bool(torch.isfinite(net.W_dist1).all())}")

# 4. Baselines compute on a small planted split.
tr, va, te, _, _, _, K, dec = splits("planted1", 201)
b = compute_baselines("planted1", tr, te, None, None, 201)
print("baselines planted1 seed201:", {k: round(v, 3) for k, v in b.items()})
assert 0.0 <= b["markov1_acc"] <= 1.0 and 0.0 <= b["esn_acc"] <= 1.0
print("ALL FAST SMOKE TESTS PASS")
