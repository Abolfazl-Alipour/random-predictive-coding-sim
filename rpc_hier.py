"""Hierarchical random predictive coding (assumption A28, round 19).

A28 generalises the two-module network (rpc_v2.RPCNetwork) to M
levels. Each level is a fixed random reservoir (rpc_v2.
ReservoirModule). Each level l < M has one distal wire W_dist_l
trained ONLY by the delta rule (A24): d_l(t) predicts the code
p_l(t) of level l from the code y_{l+1} of the level above at
the previous step. Each level l < M also has a fixed random
feedback wire W_fb_l into level l's reservoir (the A26 form),
never trained. The top level has no distal wire and no delta
update: it only supplies the slow code. The bottom readout is
the fixed p1-prototype cosine (round-15 winner config); the
secondary readout is the top prediction d_{M-1} against
mean-p_{M-1} prototypes.

This file is additive: rpc_v2.py is NOT edited (rounds 17 and
18 run on Beast against that file). The A28 spec lives in
PROTOCOL-R19-HIERARCHY.md. The class reuses ReservoirModule
and make_data from rpc_v2, and the same _encode, full-code,
calibration, and prototype machinery, generalised per level.

Design choices (see the protocol for the reasoning):
1. Level alphas are fixed by depth: 1.0, 0.3, 0.1 (fast to
   slow). rho 0.9 everywhere. No confirmation, no inhibition
   (use_confirmation False, s_inh 0.0 always): the round-15
   winner config, generalised.
2. Weights are drawn level by level in order, from one seeded
   RNG stream, so a 3-level net's level-1 and level-2 weights
   are EXACTLY a 2-level net's weights at the same seed
   (the nested-init property, smoke-tested).
3. Calibration is two passes, mirroring rpc_v2.calibrate
   (pass 1: p scales; pass 2: d scales with y live, fb off).
   There is no pass 3: no confirmation drive exists here.
"""
import math

import numpy as np
import torch

from rpc_v2 import ReservoirModule, make_data

# Hierarchical task motifs (PROTOCOL-R19-HIERARCHY.md section 5).
# Context C has two motifs; both starters (4 and 5) are shared
# across contexts, so the continuation depends on the context.
HIER_MOTIFS = {
    0: [(4, 6, 7), (5, 8, 9)],
    1: [(4, 8, 9), (5, 10, 11)],
    2: [(4, 10, 11), (5, 12, 13)],
    3: [(4, 12, 13), (5, 14, 15)],
}
HIER_BLOCK = 12
HIER_N_BLOCKS = 4


def make_hier_data(n_seq=100, length=48, K=16):
    """Two-timescale sequences (uses the GLOBAL numpy RNG, like
    baselines.make_data, so the round-15 split scheme
    np.random.seed(s); train, val, test in order works unchanged).

    Length 48: 4 blocks of 12 steps. Block b starts with a
    context token C_b in {0, 1, 2, 3}. The other 11 positions
    hold 2 planted motifs (one starting with 4, one with 5,
    from HIER_MOTIFS[C_b]) and filler drawn uniformly from
    4..15, so context tokens appear only at block starts.

    Returns (seqs, labels, metas). labels[i] is a per-position
    tag: "ctx" (block start), "s" (motif starter), "m1" (2nd
    motif token), "m2" (3rd motif token), "filler". metas[i]
    is a list of (starter_idx, context) for the 8 planted
    motifs, used for the early/late m1 split.
    """
    assert length == HIER_BLOCK * HIER_N_BLOCKS
    seqs, labels, metas = [], [], []
    for _ in range(n_seq):
        seq = [0] * length
        lab = ["filler"] * length
        meta = []
        for b in range(HIER_N_BLOCKS):
            base = b * HIER_BLOCK
            ctx = int(np.random.randint(0, 4))
            seq[base] = ctx
            lab[base] = "ctx"
            motifs = HIER_MOTIFS[ctx]
            # Possible motif starts: block positions 1..9 (a
            # 3-long motif must end by position 11). Place the
            # two motifs at non-overlapping starts.
            starts = list(range(1, 10))
            s1 = int(np.random.choice(starts))
            starts = [s for s in starts if abs(s - s1) > 2]
            s2 = int(np.random.choice(starts))
            order = [0, 1] if np.random.rand() < 0.5 else [1, 0]
            for s, mi in zip((s1, s2), order):
                motif = motifs[mi]
                for k, tok in enumerate(motif):
                    seq[base + s + k] = tok
                    lab[base + s + k] = ("s", "m1", "m2")[k]
                meta.append((base + s, ctx))
            for i in range(base + 1, base + HIER_BLOCK):
                if lab[i] == "filler":
                    seq[i] = int(np.random.randint(4, K))
        seqs.append(seq)
        labels.append(lab)
        metas.append(meta)
    return seqs, labels, metas


class HierRPCNetwork(torch.nn.Module):
    """M-level hierarchical RPC network (assumption A28).

    Args (all mirror RPCNetwork where they overlap):
      K: token classes. n_levels M: 2 or 3. alphas: per-level
        leak rates, length M, fixed by depth (1.0, 0.3, 0.1).
      fb_gains: per-wire feedback gains, length M - 1
        (fb_gains[i] feeds level i+1, 0-based). eta_d: delta
        step size. w_max: distal clamp. n_exp, n_res, n_out:
        reservoir sizes (256, 200, 256, the round-15 winner).
      theta_p, theta_d: z-thresholds for the burst-rate
        diagnostic only (no learning uses them here).
      smooth_sigma: A14 smooth input (0.0 = one-hot).
      norm_rows: False (round-15 winner); True restores A5
        row-norm preservation on the delta updates.
    """

    def __init__(self, K=16, n_levels=3, alphas=(1.0, 0.3, 0.1),
                 fb_gains=(0.1, 0.1), eta_d=0.05, w_max=2.0,
                 n_exp=256, n_res=200, n_out=256,
                 theta_p=1.0, theta_d=1.0, w_dist_std=0.5,
                 smooth_sigma=0.0, norm_rows=False):
        super().__init__()
        assert n_levels in (2, 3), n_levels
        assert len(alphas) == n_levels, alphas
        assert len(fb_gains) == n_levels - 1, fb_gains
        self.K = K
        self.M = n_levels
        self.alphas = tuple(alphas)
        self.fb_gains = tuple(float(g) for g in fb_gains)
        self.eta_d = eta_d
        self.w_max = w_max
        self.theta_p, self.theta_d = theta_p, theta_d
        self.smooth_sigma = smooth_sigma
        self.norm_rows = norm_rows
        self.n_out = n_out
        self.n_res = n_res
        # Draw order is load-bearing (nested init, protocol 3.5):
        # all of level l's matrices are drawn before level l+1's,
        # so a 3-level net's levels 1-2 equal a 2-level net's.
        self.mods = torch.nn.ModuleList()
        self.W_proj = []   # W_proj[i]: into level i+2 from r_{i+1}
        self.W_dist = []   # W_dist[i]: predicts level i+1 from i+2
        self.W_fb = []     # W_fb[i]: feedback into level i+1
        self.fb_perm = []  # fixed shuffle permutation per wire
        self.W_dist_row0 = []
        for lv in range(n_levels):
            d_in = K if lv == 0 else n_res
            self.mods.append(ReservoirModule(
                d_in, n_exp, n_res, n_out,
                alpha=self.alphas[lv]))
            if lv > 0:
                self.W_proj.append(torch.randn(n_res, n_res)
                                   / math.sqrt(n_res))
            if lv < n_levels - 1:
                self.W_dist.append(torch.randn(n_out, n_out)
                                   * w_dist_std / math.sqrt(n_out))
                self.W_fb.append(torch.randn(n_res, n_out)
                                 / math.sqrt(n_out))
                self.fb_perm.append(torch.randperm(n_out))
        for name, lst in (("W_proj", self.W_proj),
                          ("W_dist", self.W_dist),
                          ("W_fb", self.W_fb),
                          ("fb_perm", self.fb_perm)):
            for i, t in enumerate(lst):
                self.register_buffer(f"{name}_{i}", t)
        # Rebind the lists to the registered buffers so the
        # tensors live on the module (same pattern as rpc_v2).
        self.W_proj = [getattr(self, f"W_proj_{i}")
                       for i in range(len(self.W_proj))]
        self.W_dist = [getattr(self, f"W_dist_{i}")
                       for i in range(len(self.W_dist))]
        self.W_fb = [getattr(self, f"W_fb_{i}")
                     for i in range(len(self.W_fb))]
        self.fb_perm = [getattr(self, f"fb_perm_{i}")
                        for i in range(len(self.fb_perm))]
        self.W_dist_row0 = [torch.linalg.norm(w, dim=1)
                            for w in self.W_dist]
        self.y_prev = [torch.zeros(n_out) for _ in range(n_levels)]
        self.sd_p = [1.0] * n_levels
        self.sd_d = [1.0] * (n_levels - 1)
        self.prototypes = None
        self._proto_norm = None

    def reset(self):
        for m in self.mods:
            m.reset()
        for i in range(self.M):
            self.y_prev[i].zero_()

    def _encode(self, tok):
        """Token -> input vector (identical to RPCNetwork._encode)."""
        if self.smooth_sigma > 0:
            j = torch.arange(self.K, dtype=torch.float32)
            x = torch.exp(-0.5 * ((j - float(tok))
                                  / self.smooth_sigma) ** 2)
            return x / x.norm()
        return torch.nn.functional.one_hot(torch.tensor(tok),
                                           self.K).float()

    @staticmethod
    def _code(pn):
        """Full graded code (A18): per-step standardised drive,
        relu of the z-score. Identical to _y2_from_p2n with
        full_code=True."""
        z = (pn - pn.mean()) / (pn.std() + 1e-6)
        return torch.relu(z)

    def _forward_chain(self, x):
        """Run the input chain with no y, no d, no fb. Returns
        (ps, rs): per-level output codes and reservoir states.
        Used by calibration pass 1 and build_prototypes."""
        ps, rs = [], []
        u = x
        for lv in range(self.M):
            if lv > 0:
                u = torch.tanh(self.W_proj[lv - 1] @ rs[-1])
            p, r, _ = self.mods[lv].forward(u)
            ps.append(p)
            rs.append(r)
        return ps, rs

    def calibrate(self, data, n_seq=50):
        """Two passes, mirroring rpc_v2.calibrate. Pass 1 records
        p_l scales (y held at 0; p does not depend on y).
        Pass 2 replays with y live (fb off, fb_mode=cut) and
        records d_l scales."""
        with torch.no_grad():
            # Pass 1: p scales.
            pss = [[] for _ in range(self.M)]
            for seq in data[:n_seq]:
                self.reset()
                for tok in seq:
                    ps, _ = self._forward_chain(self._encode(tok))
                    for lv in range(self.M):
                        pss[lv].append(ps[lv].numpy().copy())
            self.sd_p = [float(np.std(np.stack(v))) or 1.0
                         for v in pss]
            # Pass 2: d scales, y live, fb off.
            dss = [[] for _ in range(self.M - 1)]
            for seq in data[:n_seq]:
                self.reset()
                for tok in seq:
                    out = self.step(tok, train=False, fb_mode="cut")
                    for i in range(self.M - 1):
                        dss[i].append(out["ds"][i].numpy().copy())
            self.sd_d = [float(np.std(np.stack(v))) or 1.0
                         for v in dss]
        self.reset()
        return {"sd_p": self.sd_p, "sd_d": self.sd_d}

    def build_prototypes(self, data, proto_n=20):
        """Mean p_l per token, per level (like build_prototypes,
        generalised). proto[0] is the bottom readout; proto[M-2]
        is the top readout."""
        sums = [torch.zeros(self.K, self.n_out)
                for _ in range(self.M)]
        counts = torch.zeros(self.K)
        self.reset()
        with torch.no_grad():
            for seq in data[:proto_n]:
                self.reset()
                for tok in seq:
                    ps, _ = self._forward_chain(self._encode(tok))
                    for lv in range(self.M):
                        sums[lv][tok] += ps[lv]
                    counts[tok] += 1
        self.prototypes = [s / counts[:, None].clamp(min=1)
                           for s in sums]
        self._proto_norm = [torch.nn.functional.normalize(p, dim=1)
                            for p in self.prototypes]
        self.reset()

    def _predict(self, d, level):
        """Cosine readout of a distal drive against the level's
        prototypes. Returns -1 before prototypes are built."""
        if self.prototypes is None:
            return -1
        a = torch.nn.functional.normalize(d[None, :], dim=1)
        return int(torch.argmax(a @ self._proto_norm[level].T).item())

    def step(self, tok, train=True, plasticity=True, fb_mode="on"):
        x = self._encode(tok)
        return self._step_core(x, train=train, plasticity=plasticity,
                               fb_mode=fb_mode)

    def _step_core(self, x, train=True, plasticity=True, fb_mode="on"):
        with torch.no_grad():
            # d[i] predicts level i+1 (0-based) from level i+2's
            # code at the previous step, before any module runs.
            ds = [self.W_dist[i] @ self.y_prev[i + 1]
                  for i in range(self.M - 1)]
            # Feedback per wire (A26 form), all wires share one
            # mode in a call: "on", "cut", or "shuffle".
            fbs = []
            for i in range(self.M - 1):
                if self.fb_gains[i] == 0.0 or fb_mode == "cut":
                    fbs.append(None)
                elif fb_mode == "shuffle":
                    fbs.append(self.fb_gains[i] *
                               (self.W_fb[i] @ ds[i][self.fb_perm[i]]))
                elif fb_mode == "on":
                    fbs.append(self.fb_gains[i] *
                               (self.W_fb[i] @ ds[i]))
                else:
                    raise ValueError(fb_mode)
            # Forward chain: level 1 reads the input, each higher
            # level reads the reservoir state below it.
            ps, rs = [], []
            u = x
            for lv in range(self.M):
                if lv > 0:
                    u = torch.tanh(self.W_proj[lv - 1] @ rs[-1])
                fb = fbs[lv] if lv < self.M - 1 else None
                p, r, _ = self.mods[lv].forward(u, fb=fb)
                ps.append(p)
                rs.append(r)
            # Codes (full graded code, A18) and readouts.
            ys = [self._code(ps[lv] / self.sd_p[lv])
                  for lv in range(self.M)]
            pred = self._predict(ds[0], 0)
            top_pred = (self._predict(ds[self.M - 2], self.M - 2)
                        if self.M >= 3 else pred)
            p1n = ps[0] / self.sd_p[0]
            d1n = ds[0] / self.sd_d[0]
            b1 = ((p1n > self.theta_p) & (d1n > self.theta_d)).float()
            # Delta updates, one per wire: the error (p_l - d_l)
            # against the code above, NLMS denominator (A24).
            if train and plasticity:
                for i in range(self.M - 1):
                    err = ps[i] - ds[i]
                    denom = float(self.y_prev[i + 1]
                                  @ self.y_prev[i + 1]) + 1.0
                    self.W_dist[i] += ((self.eta_d / denom) *
                                       torch.outer(err,
                                                   self.y_prev[i + 1]))
                    self.W_dist[i].clamp_(-self.w_max, self.w_max)
                    if self.norm_rows:
                        n = self.W_dist[i].norm(dim=1,
                                                keepdim=True).clamp(min=1e-9)
                        self.W_dist[i].mul_(
                            self.W_dist_row0[i][:, None] / n)
            for lv in range(self.M):
                self.y_prev[lv].copy_(ys[lv])
        fb_ratios = []
        for i in range(self.M - 1):
            bu = self.mods[i].last_bu_norm
            fb_ratios.append(self.mods[i].last_fb_norm / bu
                             if bu > 0 else 0.0)
        return {"pred": pred, "top_pred": top_pred,
                "burst_rate": float(b1.mean()),
                "d1": ds[0],
                "ds": ds,
                "top_d": ds[self.M - 2] if self.M >= 3 else ds[0],
                "fb_ratios": fb_ratios,
                "r_max": float(max(r.abs().max().item()
                                   for r in rs)),
                "p1": ps[0], "r1": rs[0], "ps": ps, "rs": rs,
                "ys": ys}

    def clamp_fracs(self):
        """Fraction of W_dist entries at the w_max clamp, per wire."""
        return [float((w.abs() >= self.w_max - 1e-6).float().mean())
                for w in self.W_dist]
