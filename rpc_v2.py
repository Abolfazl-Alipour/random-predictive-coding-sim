"""RPC toy v2: honest-scale fix of rpc_toy.py (v1).

v1 failure (measured, see diag_A.md): burst rate exactly 0.000, accuracy
0.049 vs chance 0.062. Root cause is TWO stacked scale failures, not one:
  (1) p2 max is 0.162 but theta_f is 0.5, so y2 (module-2 output) NEVER
      fires, so y2_prev is always 0, so distal drive d1 = W_dist1 @ y2_prev
      is IDENTICALLY zero (mean 0, sd 0, max 0 over 4800 steps). No
      theta_p/theta_d in any grid can fix this: d1 > theta_d is false for
      every theta_d > 0.
  (2) p1 max is 0.282 but theta_p is 0.5, so the proximal leg never passes
      either, even if d1 were nonzero.
The thesis (Ch.3 pp.89-98) gives no numbers for any threshold, drive scale,
or learning rate, so every number below that the thesis does not supply is
labelled ASSUMPTION A# here and in diag_A.md.

What v2 keeps from v1 / the thesis (no new machinery):
  - same two-module reservoir structure, same make_data motifs
  - same burst rule shape: burst = proximal AND distal coincidence
  - learning is local burst-gated Hebbian only (W_dist1) plus the
    confirmation rule (W_prox of module 2), as in v1
  - no backprop through the reservoir, no trained readout doing the
    learning; prediction is still cosine match of d1 against mean-p1
    prototypes built from train data, exactly as in v1
  - thresholds are fit ONLY on train data (calibration pass, plasticity
    OFF, before any learning), never on the test set

ASSUMPTION A1 (thresholds from pre-learning train quantiles, standardised
  drives): p1, p2, d1, d2 are divided by their standard deviations measured
  on train data before learning (frozen thereafter), and theta_p, theta_d,
  theta_f are z-thresholds on those standardised drives (defaults
  theta_p=1.0, theta_d=1.0, theta_f=1.28). Justification: the thesis gives
  no threshold values; v1's 0.5 is arbitrary and off-scale by 3-10x
  (p1 sd 0.054, p2 sd 0.033). A z-threshold is scale-free, so the same
  numbers work at any N_r/N_o, and fitting the scales on train data only
  keeps the test set untouched.
ASSUMPTION A2 (W_dist1 fan-in scaling): W_dist1 init std 0.5/sqrt(N_o)
  instead of v1's raw 0.05. Justification: every other matrix in v1 uses
  1/sqrt(fan-in) scaling; 0.05 was unscaled. At N_o=64 this is 0.0625,
  barely above v1's 0.05, so this is a consistency fix, not a tuning knob.
  (With A1, the burst rate is set by the z-thresholds, not by this scale.)
ASSUMPTION A3 (graded ungated burst ONLY as the no-burst-gate ablation):
  b1 = sigmoid(p1n * d1n) on standardised drives. v1's ablation used
  sigmoid(p1*d1) on raw drives, which is ~0.5 constant because the raw
  product is ~0. Standardising keeps the ablation meaningful (graded
  coincidence with no hard gate). The full model still uses the hard AND,
  faithful to the thesis ("burst" is a discrete event).
ASSUMPTION A4 (theta_f fixed at z=1.28, i.e. the 90th percentile of a
  standard normal): y2 fires on the top ~10% of p2 values. Justification:
  y2 must be sparse but nonzero for d1 to exist at all; 10% is the
  standard sparse-code rate and matches the p2 quantile measured in
  diagnosis (P(p2>0.05)=0.056 at raw theta_f=0.05). Not swept in v2.
ASSUMPTION A6 (k-winners-take-all y2, fixed sparsity; ADOPTED as
  default): the fixed-threshold y2 = (p2n > theta_f) lets the
  confirmation rule destroy sparsity: W_prox rows rotate toward frequent
  r2 patterns (alignment growth, not norm growth, so A5 cannot stop it),
  p2 grows, y2 rate goes 0.08 -> 0.77 by epoch 1 and accuracy freezes.
  k-WTA (y2 = 1 for the top-k p2 units each step, k = 10% of N_o) holds
  the code sparse by construction, as lateral inhibition does in a sparse
  code, and breaks the feedback loop because ranking is scale-invariant.
  The thesis specifies neither a threshold nor a k for y2. ADOPTED
  because the epoch test (seed 0, diag_A.md section 6.3) shows it works:
  y2 rate 0.094 stable across all 5 epochs, full accuracy 0.095/0.084/
  0.085/0.085/0.085 instead of 0.071/0.077/0.077/0.070/0.070, and full
  no longer loses to no-confirmation. With A6 on, theta_f is unused
  (y2 is rank-based); set y2_topk=False to recover the threshold path
  (A4) and the runaway with it.
ASSUMPTION A8 (learn_rule="coactivity", Reza's "hit and solidify",
  Nov 25): the distal learning is ungated co-activity. The post term
  is the set of module-1 cells active for the current input
  (p1n > theta_p), wired to the previous module-2 code y2_prev, with
  no burst gate on the learning. Rationale: all-to-all dense wiring
  guarantees the previous code reaches whatever module-1 cells are
  active, so the first co-occurrence already makes the connection and
  repetition solidifies it. The burst still fires and still drives the
  confirmation rule and the inhibition; only the distal learning is
  ungated. Default learn_rule="burst" keeps the thesis-faithful
  burst-gated rule. NOTE: dense random W_dist1 is NOT new here (A2
  already initialises it dense all-to-all); what A8 changes is only
  which post term the Hebbian update uses.
ASSUMPTION A9 (normalization / center-surround, Reza's direction,
  Nov 25): the brain handles noise with center-surround inhibition
  and divisive normalization, and the literature ties both directly
  to predictive coding: Srinivasan, Laughlin & Dubs 1982 (Proc R Soc
  B) show the retinal surround predicts the centre signal and
  subtracts it, protecting fine detail against intrinsic noise;
  "Sensory noise predicts divisive reshaping of receptive fields"
  (PLoS Comput Biol 2017) shows optimal competition under
  signal-dependent noise takes the form of divisive inhibition from
  the surround; "Relating sparse and predictive coding to divisive
  normalization" (PLOS Comput Biol 2025) unifies sparse and
  predictive coding through divisive normalization; Carandini &
  Heeger 2012 establish normalization as the canonical cortical
  computation, giving robustness to input gain. Three single-change
  variants, each testable alone:
  (a) surround_beta: subtract the mean expansion activity before
  the top-k (center-surround at the input stage); only expansion
  units standing out above the pool mean survive, so diffuse
  background drive is suppressed before it reaches the reservoir.
  (b) d1_sharpen: sign-preserving square of d1 (d1 * |d1|) for the
  READOUT only; this is divisive normalization with a global pool
  and exponent 2, up to a per-step scalar the cosine ignores, so it
  sharpens contrast inside the distal pattern. The burst gate is
  untouched, so the attribution is clean.
  (c) post_topk: the coactivity post term is the top 10% of p1n
  instead of every cell above threshold (winners suppress the rest,
  center-surround in module 1); only the winning cells learn.
  A plain global rescaling of d1 or r is deliberately NOT a variant:
  the readout is cosine and the gates use z-scores, so a global
  scale change is a no-op.
ASSUMPTION A10 (two-step distal drive / delay line, Nov 25):
CORRECTED Nov 25 18:05 (see simulation-results.md 19b): the planted
task does NOT require two-step memory. It uses three motifs, and a
one-step model scores about 0.36 (Markov-1 proves 0.278); a
second-order model raises the ceiling only to about 0.39. The delay
line adds a second distal matrix W_dist1b reading y2_prev2, the
module-2 code from two steps ago, so d1 = W_dist1 @ y2_prev +
W_dist1b @ y2_prev2. Both matrices learn with the same rule (burst
gate or coactivity post), the same rate, and the same row-norm
preservation. Expected gain is small (about 0.03 of ceiling), so
this is a check, not the bottleneck. The bottleneck is code quality
plus learning dilution (19b). Sharper tests: a linear probe of the
code (rpc_probe.py) and a mod2-direct ablation (A11).
ASSUMPTION A11 (mod2-direct ablation, Nov 25): feed the CLEAN token
straight into module 2 instead of the reservoir state r1, through a
FIXED random projection u2 = W_12d @ x (W_12d is n_res by d_in, so
each token maps to one distinct fixed vector; the raw one-hot is
16-dim and module 2 expects 200-dim input, so a direct wire is a
dimension error). Module 2's reservoir still runs, but its input is
the token itself, so y2_prev becomes a clean code of the previous
token and the distal mapping becomes a clean token-to-token map
learned by RPC's own rules. This is a clean ablation of the
architecture: if mod2-direct passes the planted bar while the
reservoir-fed version fails at 0.072, the reservoir is the
bottleneck and the distal learning rule is fine; if it still fails,
the learning rule itself cannot concentrate the signal, and the
bottleneck is the rule. Biologically this is a parallel direct
pathway alongside the recurrent one, not a readout change.
ASSUMPTION A12 (learned proximal projection, Nov 25): W_prox, the
  projection from the module-2 reservoir r2 to the proximal drive p2,
  is FIT by ridge regression instead of left random. The stage probe
  (section 21) shows r2 decodes the current token at 0.984 but p2
  only at 0.711: the fixed random W_prox mixes the token signal with
  history noise, and the top-k y2 then drops it to 0.332. The fit
  maps r2 onto a fixed random 64-dim code E for each token (one code
  per token, unit norm), using train data only, before calibration,
  so all calibrated scales are measured with the learned W_prox in
  place. This is a DEPARTURE from the thesis, where W_prox is fixed
  random and only shaped by the confirmation rule. The confirmation
  rule is OFF by default in this test (use_confirmation=False),
  because it rewrites W_prox every step and would destroy the fit;
  a separate condition leaves it on to test whether the thesis's own
  rule preserves or destroys a learned proximal code. If the learned
  W_prox passes the planted bar, the fixed random projection is the
  bottleneck and the theory needs a plastic proximal path; if it
  fails, the bottleneck sits in the post term or the learning rule,
  not in W_prox.
ASSUMPTION A13 (learned module-1 proximal projection, Nov 25): the
  A12 result (0.113) cleaned only the prediction side (y2_prev). The
  post side of the Hebbian pair, p1, is still a fixed random
  projection of r1, so the learned rows pair a clean previous code
  with a noisy current code. fit_prox1 fits mod1.W_prox by ridge
  from r1 onto the SAME per-token code E as A12, train only, before
  calibration, so prediction and post term share one code space.
  The confirmation rule never touches mod1.W_prox, so the fit is
  stable through training. Combined with A12 this is the "both
  sides clean" test: if the planted and sine scores still stall,
  the bottleneck is the learning rule's dilution, not the codes.
ASSUMPTION A14 (smooth input code / population coding, Nov 25): the
  one-hot token makes every pair of bins orthogonal, so the network
  gets no hint that neighbouring sine bins are near in value. Real
  neurons encode a continuous value with overlapping tuning curves.
  With smooth_sigma > 0 the input is a Gaussian bump centred on the
  bin (unit norm), so neighbouring bins share input. Single change:
  only the input vector changes; classes, prototypes, and rules are
  untouched. If the sine needs metric structure to learn phase,
  this should move it; if it does not, the failure is not the input
  code.
ASSUMPTION A15 (direction-split prototypes, Nov 25): the readout
  gap is the biggest remaining loss (section 24: prototype readout
  0.078 vs ridge readout 0.556 on the same d1). One prototype per
  bin averages the rising and falling versions of that bin, so a
  distal drive that leans toward one direction still has to match
  the blended mean. With split_proto on, build_prototypes stores
  three prototypes per bin (rising, falling, flat, by the sign of
  tok minus the previous tok); combos with no train occurrences fall
  back to the plain prototype. Prediction is still argmax cosine
  over the prototype set, and the token is idx % K. This is a
  departure from the single-prototype thesis readout, but it stays
  inside the thesis idea (match the prediction against stored
  patterns): the stored pattern is now context-specific, as a
  brain's prediction of a value on the way up differs from the
  same value on the way down. It only helps if d1 itself carries
  direction, which the smooth input gives it (section 27 probe:
  d1 direction decode 0.892 at sigma 2.0 vs 0.740 one-hot).
ASSUMPTION A16 (graded confirmation, Nov 26): the confirmation
  signal is a strength, not a switch. The binary burst gate
  decides WHERE learning happens (which units), but a well-matched
  prediction should teach more than a poorly matched one. With
  graded_conf on, each step computes g, the cosine match between
  the prediction (the distal drive d1, read from the previous
  module-2 code) and the actual input's code (the proximal drive
  p1), clipped to [0, 1], and the distal Hebbian update is scaled
  by g. The confirmation rule and the inhibition keep the binary
  burst, so this test isolates one mechanism: learning in
  proportion to match quality. Rationale: with a binary gate, a
  row learns equally from a rising and a falling successor, so it
  averages them and matches neither; with graded confirmation, the
  better-matching direction teaches more and the row should
  concentrate on it. This is still inside the thesis idea (the
  burst marks a confirmed prediction); it only says the
  confirmation has a strength. It differs from A3, which grades
  the burst per unit and removes the threshold; A16 keeps the
  threshold and grades the update by the step-level match. The
  diagnostic is the mean and spread of g during training: if g
  barely varies across steps, grading is just a smaller learning
  rate and should not help.
ASSUMPTION A17 (dot-product readout, Nov 26): the last untested
  tweak to the prototype readout. The cosine readout normalises
  d1 and every prototype, so only the angle matters and a strong
  prediction counts the same as a weak one. The dot-product
  readout scores each prototype by the raw dot product d1 .
  proto_k, which is ||d1|| * ||proto_k|| * cos, so each prototype
  is weighted by its stored strength (its mean p1 norm). Tokens
  whose stored pattern fires more strongly win ties. Note the
  honest limit: a dot product against UNIT prototypes would be
  identical to cosine, because ||d1|| is the same for every k, so
  the only real dot-product variant is the raw-prototype one, and
  it only changes the answer if the prototype norms actually
  differ across tokens. The diagnostic is the spread of the
  prototype norms: if they are nearly equal, this test is vacuous
  and the readout is not the lever.
ASSUMPTION A5 (row-norm preservation after each Hebbian update):
  after every W_dist1 / W_prox update, each row is rescaled to its
  initial L2 norm. Justification: the v1 rules as written are purely
  potentiating (b1 >= 0, y2 >= 0, so W_dist1 only grows) with no
  depression, normalisation, or bound except the +-2.0 clamp. Measured
  (seed 0): after 1 epoch 30% of W_dist1 entries and 29% of W_prox2
  entries are pinned at the clamp, mean |W_dist1| grows 0.05 -> 0.73,
  y2 rate goes 0.08 -> 0.77, and accuracy freezes at the same value
  for epochs 1-5. Row-norm preservation is standard synaptic scaling /
  homeostatic normalisation: learning changes each row's DIRECTION
  (which is all the cosine prediction reads), not its magnitude, so
  drive scales stay at their calibrated values and y2 stays sparse.
  It is local (per row) and uses no gradient. Set norm_rows=False to
  recover the saturating v1 behaviour.

Not added (deliberately): no graded coincidence in the full model, no
bigger N_o/N_r by default, no shrink of D, no learned readout, no
test-set threshold fitting. Vector inputs (step_vec / run_vec, added
2026-11-25 for the track integrator test) reuse the identical step core,
calibration, and prototype machinery; only the input encoding differs
(see ASSUMPTION A7 in rpc_track_F.py).
"""
import math
import numpy as np
import torch

# ---------- data: random tokens with planted motifs (identical to v1) ----------
def make_data(n_seq=200, length=24, K=16, motifs=((1, 2, 3), (5, 6, 7), (9, 10, 11))):
    data = []
    for _ in range(n_seq):
        seq = np.random.randint(0, K, size=length).tolist()
        for motif in motifs:
            pos = np.random.randint(0, length - len(motif))
            seq[pos:pos + len(motif)] = list(motif)
        data.append(seq)
    return data

# ---------- fixed random reservoir module (identical to v1) ----------
class ReservoirModule(torch.nn.Module):
    def __init__(self, d_in, n_exp=256, n_res=200, n_out=64,
                 k_e=32, alpha=0.3, rho=0.9, learn_prox=False,
                 surround_beta=0.0):
        super().__init__()
        self.n_res, self.n_out = n_res, n_out
        self.k_e, self.alpha = k_e, alpha
        self.learn_prox = learn_prox
        # A9a: center-surround at the input stage (0.0 = off).
        self.surround_beta = surround_beta
        W_exp = torch.randn(n_exp, d_in) / math.sqrt(d_in)
        self.register_buffer("W_exp", W_exp)
        W_in = torch.randn(n_res, n_exp) / math.sqrt(n_exp)
        self.register_buffer("W_in", W_in)
        W_res = torch.randn(n_res, n_res) * 0.1
        ev = torch.linalg.eigvals(W_res).abs().max()
        W_res = W_res / ev * rho
        self.register_buffer("W_res", W_res)
        W_prox = torch.randn(n_out, n_res) / math.sqrt(n_res) * 0.5
        self.register_buffer("W_prox", W_prox)
        self.register_buffer("r", torch.zeros(n_res))
        self.register_buffer("inhib", torch.zeros(n_res))

    def reset(self):
        self.r.zero_()
        self.inhib.zero_()

    def forward(self, u):
        e = torch.relu(self.W_exp @ u)
        if self.surround_beta > 0:  # A9a: subtract the pool mean.
            e = torch.relu(e - self.surround_beta * e.mean())
        top = torch.topk(e, self.k_e).indices
        e_sparse = torch.zeros_like(e)
        e_sparse[top] = e[top]
        r_new = (1 - self.alpha) * self.r + self.alpha * torch.tanh(
            self.W_in @ e_sparse + self.W_res @ self.r - self.inhib)
        self.r.copy_(r_new)
        self.inhib.zero_()
        p = self.W_prox @ self.r
        return p, self.r, e_sparse

# ---------- two-module RPC network v2 ----------
class RPCNetwork(torch.nn.Module):
    def __init__(self, K=16, n_exp=256, n_res=200, n_out=64,
                 theta_p=1.0, theta_d=1.0, theta_f=1.28,
                 eta_b=0.05, eta_c=0.05, w_max=2.0, s_inh=0.5, g2=1.0,
                 w_dist_std=0.5, norm_rows=True, y2_topk=True,
                 rho=0.9, alpha=0.3, d_in=None, center_proto=False,
                 learn_rule="burst", surround_beta=0.0,
                 d1_sharpen=False, post_topk=False, use_delay=False,
                 mod2_direct=False, rho2=None, y2_frac=0.1,
                 learn_prox=False, smooth_sigma=0.0,
                 learn_prox1=False, split_proto=False, graded_conf=False,
                 dot_proto=False):
        # center_proto: subtract the mean prototype (over classes that
        # appear in train) from every prototype and from d1 before the
        # cosine prediction. Train-data means only, no new learned
        # weights. Added Nov 25 after the diagnostic showed raw cycle
        # prototypes are near-parallel (off-diag cosine 0.890), so no
        # distal vector can discriminate them by raw cosine.
        super().__init__()
        self.K = K
        # d_in: module-1 input dimension. Default K (one-hot token input).
        # Set d_in=200 with K=32 to feed continuous vectors (e.g. the
        # 200-number leaky-integrator input [F(t), F(t-1)]) while the
        # prediction classes stay K bins.
        self.d_in = d_in if d_in is not None else K
        self.mod1 = ReservoirModule(self.d_in, n_exp, n_res, n_out, rho=rho,
                                    alpha=alpha, surround_beta=surround_beta)
        # rho2 / y2_frac: fix variants for the code-degradation probe
        # (section 20). rho2=None keeps the shared rho; rho2=0.0 makes
        # module 2 feedforward (no recurrence). y2_frac sets the y2
        # winner fraction (default 0.1, 6 of 64 units).
        self.mod2 = ReservoirModule(n_res, n_exp, n_res, n_out,
                                    rho=(rho if rho2 is None else rho2),
                                    alpha=alpha, surround_beta=surround_beta)
        # A1: thresholds are z-scores on standardised drives, not raw 0.5.
        self.theta_p, self.theta_d, self.theta_f = theta_p, theta_d, theta_f
        self.eta_b, self.eta_c, self.w_max = eta_b, eta_c, w_max
        self.learn_rule = learn_rule
        # A12: fit W_prox by ridge regression (fit_prox) instead of
        # leaving it random. Set by run() before calibrate().
        self.learn_prox = learn_prox
        # A13: fit mod1.W_prox the same way (fit_prox1). A14: smooth
        # Gaussian-bump input instead of one-hot (sigma in bins, 0 = off).
        self.learn_prox1 = learn_prox1
        self.smooth_sigma = smooth_sigma
        # A15: direction-split prototypes (see module docstring).
        self.split_proto = split_proto
        # A16: graded confirmation (see module docstring).
        self.graded_conf = graded_conf
        # A17: dot-product readout (see module docstring).
        self.dot_proto = dot_proto
        # A9b/c: divisive sharpening of d1 at the readout only, and
        # top-k post term for the coactivity rule.
        self.d1_sharpen = d1_sharpen
        self.post_topk = post_topk
        # A11: module 2 reads the clean token instead of r1.
        self.mod2_direct = mod2_direct
        self.register_buffer("W_12d",
                             torch.randn(n_res, self.d_in) / math.sqrt(self.d_in))
        # A10: second distal matrix reading the code from two steps ago.
        self.use_delay = use_delay
        self.register_buffer("W_dist1b",
                             torch.randn(n_out, n_out) * w_dist_std / math.sqrt(n_out))
        self.register_buffer("W_dist1b_row0",
                             torch.linalg.norm(self.W_dist1b, dim=1))
        self.s_inh, self.g2 = s_inh, g2
        self.register_buffer("W_12", torch.randn(n_res, n_res) / math.sqrt(n_res))
        # A2: fan-in scaling for W_dist1 (v1 used raw std 0.05).
        self.register_buffer("W_dist1",
                             torch.randn(n_out, n_out) * w_dist_std / math.sqrt(n_out))
        self.register_buffer("W_conf", torch.randn(n_out, n_out) / math.sqrt(n_out))
        # A5: initial row norms, preserved after each Hebbian update.
        self.center_proto = center_proto
        self.norm_rows = norm_rows
        # A6: k-winners-take-all y2 (k = 10% of N_o) instead of a fixed
        # threshold; on by default, see module docstring.
        self.y2_topk = y2_topk
        self.y2_k = max(1, int(round(y2_frac * n_out)))
        self.register_buffer("W_dist1_row0",
                             torch.linalg.norm(self.W_dist1, dim=1))
        self.register_buffer("W_prox2_row0",
                             torch.linalg.norm(self.mod2.W_prox, dim=1))
        self.register_buffer("y2_prev", torch.zeros(n_out))
        # A10: module-2 code from two steps ago.
        self.register_buffer("y2_prev2", torch.zeros(n_out))
        self.register_buffer("r2_prev", torch.zeros(n_res))
        self.register_buffer("p2_prev", torch.zeros(n_out))
        self.prototypes = None
        # Calibration scales (A1), set by calibrate() on TRAIN data only.
        self.sd_p1 = 1.0
        self.sd_p2 = 1.0
        self.sd_d1 = 1.0
        self.sd_d2 = 1.0

    def reset(self):
        self.mod1.reset()
        self.mod2.reset()
        self.y2_prev.zero_()
        self.y2_prev2.zero_()
        self.r2_prev.zero_()
        self.p2_prev.zero_()

    def _distal_drive(self):
        """A10: distal drive from the last two module-2 codes. With
        use_delay off this is exactly W_dist1 @ y2_prev."""
        d1 = self.W_dist1 @ self.y2_prev
        if self.use_delay:
            d1 = d1 + self.W_dist1b @ self.y2_prev2
        return d1

    def _mod2_input(self, x, r1):
        """A11: module-2 input. Default is the reservoir state through
        W_12; with mod2_direct the clean token goes straight in
        through a fixed random projection (see A11)."""
        if self.mod2_direct:
            xv = torch.as_tensor(x, dtype=torch.float32)
            assert xv.shape[0] == self.d_in, \
                "mod2_direct needs token input of dim d_in"
            return self.W_12d @ xv
        return torch.tanh(self.W_12 @ r1)

    def _y2_from_p2n(self, p2n):
        """Module-2 output code. Default (A6): top-k winners (fixed
        sparsity, rank-based, scale-invariant). With y2_topk=False:
        fixed z-threshold (A4)."""
        if self.y2_topk:
            y2 = torch.zeros_like(p2n)
            y2[torch.topk(p2n, self.y2_k).indices] = 1.0
            return y2
        return (self.g2 * p2n > self.theta_f).float()

    def calibrate(self, data, n_seq=50):
        """A1: measure drive scales on train data only, plasticity OFF.
        Pass 1 records p1/p2 (y2 held at 0; y2 does not feed the reservoirs,
        so p1/p2 are unaffected). Pass 2 replays with y2 live and records
        d1. Pass 3 replays with y2 live and b1 live and records d2.
        Scales are frozen after; thresholds are z-scores on these scales."""
        with torch.no_grad():
            # Pass 1: p1, p2 scales.
            p1s, p2s = [], []
            for seq in data[:n_seq]:
                self.reset()
                for tok in seq:
                    x = self._encode(tok)
                    p1, r1, _ = self.mod1.forward(x)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    p1s.append(p1.numpy().copy())
                    p2s.append(p2.numpy().copy())
            self.sd_p1 = float(np.std(np.stack(p1s))) or 1.0
            self.sd_p2 = float(np.std(np.stack(p2s))) or 1.0
            # Pass 2: d1 scale, with y2 = (p2n > theta_f).
            d1s = []
            for seq in data[:n_seq]:
                self.reset()
                for tok in seq:
                    x = self._encode(tok)
                    d1 = self._distal_drive()
                    p1, r1, _ = self.mod1.forward(x)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    y2 = self._y2_from_p2n(p2 / self.sd_p2)
                    d1s.append(d1.numpy().copy())
                    self.y2_prev2 = self.y2_prev.clone()
                    self.y2_prev = y2.clone()
                    self.r2_prev = r2.clone()
                    self.p2_prev = (p2 / self.sd_p2).clone()
            self.sd_d1 = float(np.std(np.stack(d1s))) or 1.0
            # Pass 3: d2 scale, with y2 and b1 live.
            d2s = []
            for seq in data[:n_seq]:
                self.reset()
                for tok in seq:
                    x = self._encode(tok)
                    d1 = self._distal_drive()
                    p1, r1, _ = self.mod1.forward(x)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    y2 = self._y2_from_p2n(p2 / self.sd_p2)
                    b1 = (((p1 / self.sd_p1) > self.theta_p) &
                          ((d1 / self.sd_d1) > self.theta_d)).float()
                    d2s.append((self.W_conf @ b1).numpy().copy())
                    self.y2_prev2 = self.y2_prev.clone()
                    self.y2_prev = y2.clone()
                    self.r2_prev = r2.clone()
                    self.p2_prev = (p2 / self.sd_p2).clone()
            self.sd_d2 = float(np.std(np.stack(d2s))) or 1.0
        self.reset()
        return {"sd_p1": self.sd_p1, "sd_p2": self.sd_p2,
                "sd_d1": self.sd_d1, "sd_d2": self.sd_d2}

    def build_prototypes(self, data, proto_n=20):
        sums = torch.zeros(self.K, self.mod1.n_out)
        counts = torch.zeros(self.K)
        if self.split_proto:
            ssums = torch.zeros(3 * self.K, self.mod1.n_out)
            scounts = torch.zeros(3 * self.K)
        self.reset()
        with torch.no_grad():
            for seq in data[:proto_n]:
                self.reset()
                prev = None
                for tok in seq:
                    x = self._encode(tok)
                    p1, _, _ = self.mod1.forward(x)
                    sums[tok] += p1
                    counts[tok] += 1
                    if self.split_proto and prev is not None:
                        d = 1 if tok > prev else (0 if tok < prev else 2)
                        ssums[d * self.K + tok] += p1
                        scounts[d * self.K + tok] += 1
                    prev = tok
        plain = sums / counts[:, None].clamp(min=1)
        if self.split_proto:
            split = ssums / scounts[:, None].clamp(min=1)
            # Fallback: combos with no occurrences take the plain prototype.
            empty = scounts == 0
            idx = torch.arange(3 * self.K) % self.K
            split[empty] = plain[idx[empty]]
            self.prototypes = split
            self._finalize_prototypes(scounts)
        else:
            self.prototypes = plain
            self._finalize_prototypes(counts)
        self.reset()

    def _finalize_prototypes(self, counts):
        """Set _proto_norm (and _proto_mean when center_proto is on).
        Classes with zero train counts are excluded from the mean, so an
        unused token cannot drag the centre."""
        if self.center_proto:
            seen = counts > 0
            self._proto_mean = self.prototypes[seen].mean(dim=0) if seen.any() \
                else torch.zeros_like(self.prototypes[0])
            centered = self.prototypes - self._proto_mean
            self._proto_norm = torch.nn.functional.normalize(centered, dim=1)
        else:
            self._proto_mean = torch.zeros_like(self.prototypes[0])
            self._proto_norm = torch.nn.functional.normalize(self.prototypes, dim=1)

    def fit_prox(self, data, lam=1.0):
        """A12: fit mod2.W_prox by ridge regression from the module-2
        reservoir state r2 onto a fixed random per-token code E
        (K x n_out, unit-norm rows), using train data only. Runs the
        frozen network (no plasticity); r2 does not depend on W_prox
        or on calibration, so this is safe before calibrate(). Sets
        mod2.W_prox in place. Returns the mean squared fit residual
        on train data and the sample count."""
        K, n_out = self.K, self.mod2.n_out
        E = torch.randn(K, n_out) / math.sqrt(n_out)
        E = torch.nn.functional.normalize(E, dim=1)
        self._prox_target = E
        R, Y = [], []
        with torch.no_grad():
            for seq in data:
                self.reset()
                for i, tok in enumerate(seq):
                    out = self.step(tok, train=False)
                    if i > 0:
                        R.append(out["r2"].numpy().copy())
                        Y.append(E[tok].numpy().copy())
        Rm = np.stack(R)
        Ym = np.stack(Y)
        W = np.linalg.solve(Rm.T @ Rm + lam * np.eye(Rm.shape[1]),
                            Rm.T @ Ym).T
        self.mod2.W_prox.copy_(torch.as_tensor(W, dtype=torch.float32))
        self.reset()
        resid = float(np.mean((Rm @ W.T - Ym) ** 2))
        return {"fit_resid_mse": resid, "n_samples": len(R)}

    def _encode(self, tok):
        """Token -> input vector. Default is the one-hot code (the
        thesis token). With smooth_sigma > 0 (A14), a Gaussian bump
        centred on the token, unit L2 norm: neighbouring bins share
        input, so the code carries metric structure instead of making
        every pair of bins orthogonal."""
        if self.smooth_sigma > 0:
            j = torch.arange(self.K, dtype=torch.float32)
            x = torch.exp(-0.5 * ((j - float(tok)) / self.smooth_sigma) ** 2)
            return x / x.norm()
        return torch.nn.functional.one_hot(torch.tensor(tok), self.K).float()

    def fit_prox1(self, data, lam=1.0):
        """A13: fit mod1.W_prox by ridge regression from the module-1
        reservoir state r1 onto the SAME per-token code E used by
        fit_prox, so the prediction (d1, in p1 space) and the post
        term (p1) live in one shared code. Runs the frozen network;
        r1 does not depend on W_prox, so this is safe before
        calibrate(). If fit_prox has not run, creates E first. Sets
        mod1.W_prox in place. Returns the fit residual."""
        K, n_out = self.K, self.mod1.n_out
        if not hasattr(self, "_prox_target"):
            E = torch.randn(K, n_out) / math.sqrt(n_out)
            E = torch.nn.functional.normalize(E, dim=1)
            self._prox_target = E
        E = self._prox_target
        R, Y = [], []
        with torch.no_grad():
            for seq in data:
                self.reset()
                for i, tok in enumerate(seq):
                    out = self.step(tok, train=False)
                    if i > 0:
                        R.append(out["r1"].numpy().copy())
                        Y.append(E[tok].numpy().copy())
        Rm = np.stack(R)
        Ym = np.stack(Y)
        W = np.linalg.solve(Rm.T @ Rm + lam * np.eye(Rm.shape[1]),
                            Rm.T @ Ym).T
        self.mod1.W_prox.copy_(torch.as_tensor(W, dtype=torch.float32))
        self.reset()
        resid = float(np.mean((Rm @ W.T - Ym) ** 2))
        return {"fit_resid_mse": resid, "n_samples": len(R)}

    def calibrate_vec(self, data, n_seq=50):
        """A1 calibration for continuous vector inputs: same three passes
        as calibrate(), but each step feeds the vector x directly instead
        of a one-hot token. data: list of sequences of (x, y) pairs."""
        with torch.no_grad():
            p1s, p2s = [], []
            for seq in data[:n_seq]:
                self.reset()
                for x, y in seq:
                    xv = torch.as_tensor(x, dtype=torch.float32)
                    p1, r1, _ = self.mod1.forward(xv)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    p1s.append(p1.numpy().copy())
                    p2s.append(p2.numpy().copy())
            self.sd_p1 = float(np.std(np.stack(p1s))) or 1.0
            self.sd_p2 = float(np.std(np.stack(p2s))) or 1.0
            d1s = []
            for seq in data[:n_seq]:
                self.reset()
                for x, y in seq:
                    xv = torch.as_tensor(x, dtype=torch.float32)
                    d1 = self._distal_drive()
                    p1, r1, _ = self.mod1.forward(xv)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    y2 = self._y2_from_p2n(p2 / self.sd_p2)
                    d1s.append(d1.numpy().copy())
                    self.y2_prev2 = self.y2_prev.clone()
                    self.y2_prev = y2.clone()
                    self.r2_prev = r2.clone()
                    self.p2_prev = (p2 / self.sd_p2).clone()
            self.sd_d1 = float(np.std(np.stack(d1s))) or 1.0
            d2s = []
            for seq in data[:n_seq]:
                self.reset()
                for x, y in seq:
                    xv = torch.as_tensor(x, dtype=torch.float32)
                    d1 = self._distal_drive()
                    p1, r1, _ = self.mod1.forward(xv)
                    u2 = self._mod2_input(x, r1)
                    p2, r2, _ = self.mod2.forward(u2)
                    y2 = self._y2_from_p2n(p2 / self.sd_p2)
                    b1 = (((p1 / self.sd_p1) > self.theta_p) &
                          ((d1 / self.sd_d1) > self.theta_d)).float()
                    d2s.append((self.W_conf @ b1).numpy().copy())
                    self.y2_prev2 = self.y2_prev.clone()
                    self.y2_prev = y2.clone()
                    self.r2_prev = r2.clone()
                    self.p2_prev = (p2 / self.sd_p2).clone()
            self.sd_d2 = float(np.std(np.stack(d2s))) or 1.0
        self.reset()
        return {"sd_p1": self.sd_p1, "sd_p2": self.sd_p2,
                "sd_d1": self.sd_d1, "sd_d2": self.sd_d2}

    def build_prototypes_vec(self, data, proto_n=20):
        """Mean-p1 prototypes per target class y, for vector inputs.
        Mirrors build_prototypes(): prototype for class c is the mean p1
        over steps whose target is c."""
        sums = torch.zeros(self.K, self.mod1.n_out)
        counts = torch.zeros(self.K)
        self.reset()
        with torch.no_grad():
            for seq in data[:proto_n]:
                self.reset()
                for x, y in seq:
                    xv = torch.as_tensor(x, dtype=torch.float32)
                    p1, _, _ = self.mod1.forward(xv)
                    sums[y] += p1
                    counts[y] += 1
        self.prototypes = sums / counts[:, None].clamp(min=1)
        self._finalize_prototypes(counts)
        self.reset()

    def predict_from_distal(self, d1):
        if self.prototypes is None:
            return -1
        if self.center_proto:
            d1 = d1 - self._proto_mean
        if self.dot_proto:
            # A17: raw dot product against the unnormalised
            # prototypes. The score is ||d1|| * ||proto_k|| * cos,
            # so each prototype is weighted by its stored
            # strength. (A dot product against unit prototypes
            # would be identical to cosine, since ||d1|| is the
            # same for every k, so that variant is not tested.)
            protos = self.prototypes - self._proto_mean \
                if self.center_proto else self.prototypes
            idx = int(torch.argmax(d1 @ protos.T).item())
        else:
            a = torch.nn.functional.normalize(d1[None, :], dim=1)
            idx = int(torch.argmax(a @ self._proto_norm.T).item())
        # A15: split prototypes are laid out dir*K + tok; the token is idx % K.
        return idx % self.K if self.split_proto else idx

    def step(self, tok, train=True, use_confirmation=True, use_burst=True,
             plasticity=True):
        x = self._encode(tok)
        return self._step_core(x, train=train, use_confirmation=use_confirmation,
                               use_burst=use_burst, plasticity=plasticity)

    def step_vec(self, x, train=True, use_confirmation=True, use_burst=True,
                 plasticity=True):
        """Same step, but x is a continuous input vector of dim d_in
        (e.g. a leaky-integrator vector) instead of a token index.
        Prediction is still a class index via the prototypes."""
        return self._step_core(torch.as_tensor(x, dtype=torch.float32),
                               train=train, use_confirmation=use_confirmation,
                               use_burst=use_burst, plasticity=plasticity)

    def _step_core(self, x, train=True, use_confirmation=True, use_burst=True,
                   plasticity=True):
        with torch.no_grad():
            # Snapshot for the probe return values (section 20).
            y2_prev = self.y2_prev.clone()
            d1 = self._distal_drive()
            p1, r1, _ = self.mod1.forward(x)
            u2 = self._mod2_input(x, r1)
            p2, r2, e2 = self.mod2.forward(u2)
            # A1: standardised drives; thresholds are z-scores.
            p1n, d1n, p2n = p1 / self.sd_p1, d1 / self.sd_d1, p2 / self.sd_p2
            y2 = self._y2_from_p2n(p2n)
            if use_burst:
                b1 = ((p1n > self.theta_p) & (d1n > self.theta_d)).float()
            else:
                # A3: graded ungated coincidence, standardised (ablation only).
                b1 = torch.sigmoid(p1n * d1n)
            # A9b: sign-preserving square sharpens contrast inside
            # the distal pattern before the cosine match; the gate
            # above still uses the unsharpened d1n.
            pred_d1 = d1 * d1.abs() if self.d1_sharpen else d1
            pred = self.predict_from_distal(pred_d1)
            # A16: graded confirmation. g is the match quality
            # between the prediction (d1) and the actual input's
            # code (p1), cosine clipped to [0, 1]. When graded_conf
            # is on, the distal Hebbian update is scaled by g, so
            # well-matched steps teach more than poorly matched
            # ones; when off, g is 1 and nothing changes.
            if self.graded_conf:
                g = float(torch.nn.functional.cosine_similarity(
                    d1, p1, dim=0).clamp(0, 1))
            else:
                g = 1.0
            if train and plasticity:
                if self.learn_rule == "coactivity":
                    # A8: ungated co-activity ("hit and solidify").
                    if self.post_topk:  # A9c: winners only.
                        post = torch.zeros_like(p1n)
                        post[torch.topk(p1n, self.y2_k).indices] = 1.0
                    else:
                        post = (p1n > self.theta_p).float()
                else:  # "burst": thesis-faithful burst-gated Hebbian.
                    post = b1
                post = post * g  # A16: grade the distal update by match.
                self.W_dist1 += self.eta_b * torch.outer(post, self.y2_prev)
                self.W_dist1.clamp_(-self.w_max, self.w_max)
                if self.norm_rows:  # A5: keep each row's initial norm.
                    n = self.W_dist1.norm(dim=1, keepdim=True).clamp(min=1e-9)
                    self.W_dist1.mul_(self.W_dist1_row0[:, None] / n)
                if self.use_delay:  # A10: same rule, two-step code.
                    self.W_dist1b += self.eta_b * torch.outer(post, self.y2_prev2)
                    self.W_dist1b.clamp_(-self.w_max, self.w_max)
                    if self.norm_rows:
                        n = self.W_dist1b.norm(dim=1, keepdim=True).clamp(min=1e-9)
                        self.W_dist1b.mul_(self.W_dist1b_row0[:, None] / n)
                if use_confirmation:
                    d2 = self.W_conf @ b1
                    d2n = d2 / self.sd_d2
                    b2 = ((self.p2_prev > self.theta_p) &
                          (d2n > self.theta_d)).float()
                    self.mod2.W_prox += self.eta_c * torch.outer(b2, self.r2_prev)
                    self.mod2.W_prox.clamp_(-self.w_max, self.w_max)
                    if self.norm_rows:  # A5, same for the confirmation target.
                        n = self.mod2.W_prox.norm(dim=1, keepdim=True).clamp(min=1e-9)
                        self.mod2.W_prox.mul_(self.W_prox2_row0[:, None] / n)
                raw = self.mod1.W_prox.T @ b1
                self.mod1.inhib = self.s_inh * raw.clamp(0, 0.8)
            self.y2_prev2.copy_(self.y2_prev)
            self.y2_prev.copy_(y2)
            self.r2_prev.copy_(r2)
            self.p2_prev.copy_(p2n)
        return {"pred": pred, "burst_rate": float(b1.mean()),
                "y2_rate": float(y2.mean()), "g": g,
                # d1 is returned (no behaviour change) so a caller can
                # fit a learned readout on the distal drive (section 18).
                # d1 at step t is computed from y2_prev only, before the
                # current token is fed in, so it carries no leakage of
                # the current token.
                "d1": d1,
                # Returned for the code probe (section 20): y2_prev is
                # the module-2 code the distal drive reads, r1 is the
                # current module-1 reservoir state. The stage values
                # (u2, e2, r2, p2, y2) are for the stage-by-stage probe:
                # where in the mod1-to-mod2 pathway does the token code
                # degrade from r1's perfect 1.000 to y2's 0.361?
                "y2_prev": y2_prev, "r1": r1, "p1": p1,
                "u2": u2, "e2": e2, "r2": r2, "p2": p2, "y2": y2}

def run(train_data, test_data, epochs=5, **kw):
    step_kw = {k: kw.pop(k) for k in
               ("use_confirmation", "use_burst", "plasticity") if k in kw}
    proto_n = kw.pop("proto_n", 20)
    net = RPCNetwork(**kw)
    if getattr(net, "learn_prox", False):
        # A12: fit W_prox before calibrate(), so the calibrated
        # scales are measured with the learned W_prox in place.
        net.fit_prox(train_data)
    if getattr(net, "learn_prox1", False):
        # A13: same, for module 1. Runs after fit_prox so both share
        # the one per-token code E.
        net.fit_prox1(train_data)
    cal = net.calibrate(train_data)
    net.build_prototypes(train_data, proto_n=proto_n)
    gs = []
    for ep in range(epochs):
        for seq in train_data:
            net.reset()
            for tok in seq:
                gs.append(net.step(tok, train=True, **step_kw)["g"])
    correct, total, bursts, y2s = 0, 0, [], []
    for seq in test_data:
        net.reset()
        for i, tok in enumerate(seq):
            out = net.step(tok, train=False)
            if i > 0:
                total += 1
                correct += int(out["pred"] == tok)
            bursts.append(out["burst_rate"])
            y2s.append(out["y2_rate"])
    return {"acc": correct / max(total, 1), "chance": 1 / net.K,
            "mean_burst": float(np.mean(bursts)),
            "mean_y2": float(np.mean(y2s)), "cal": cal,
            "mean_g": float(np.mean(gs)), "std_g": float(np.std(gs))}

def run_vec(train_data, test_data, epochs=5, **kw):
    """run() for continuous vector inputs. Sequences are lists of
    (x, y) pairs: x a float vector of dim d_in, y the target class.
    Identical training, calibration, prototypes, scoring, and controls
    to run(); only the input encoding differs."""
    step_kw = {k: kw.pop(k) for k in
               ("use_confirmation", "use_burst", "plasticity") if k in kw}
    proto_n = kw.pop("proto_n", 20)
    net = RPCNetwork(**kw)
    cal = net.calibrate_vec(train_data)
    net.build_prototypes_vec(train_data, proto_n=proto_n)
    for ep in range(epochs):
        for seq in train_data:
            net.reset()
            for x, y in seq:
                net.step_vec(x, train=True, **step_kw)
    correct, total, bursts, y2s = 0, 0, [], []
    for seq in test_data:
        net.reset()
        for i, (x, y) in enumerate(seq):
            out = net.step_vec(x, train=False)
            if i > 0:
                total += 1
                correct += int(out["pred"] == y)
            bursts.append(out["burst_rate"])
            y2s.append(out["y2_rate"])
    return {"acc": correct / max(total, 1), "chance": 1 / net.K,
            "mean_burst": float(np.mean(bursts)),
            "mean_y2": float(np.mean(y2s)), "cal": cal}

def run_experiment(seed, epochs=5, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = make_data(n_seq=200)
    test = make_data(n_seq=50)
    return run(train, test, epochs=epochs, **kw)

if __name__ == "__main__":
    import json
    # Defaults ARE the sweep winner (sweep_v2.py, 5 seeds 0-4, metric
    # mean(acc - no-plasticity control)): theta_p=1.0, theta_d=1.0,
    # eta_b=eta_c=0.05, N_r=200, N_o=64, gain +0.013, acc mean 0.075.
    # See diag_A.md section 7.2 for the full sweep table.
    defaults = dict(theta_p=1.0, theta_d=1.0, eta_b=0.05, eta_c=0.05,
                    n_res=200, n_out=64)
    conditions = [
        ("full", {}),
        ("no-confirmation", {"use_confirmation": False}),
        ("no-burst-gate", {"use_burst": False}),
        ("no-plasticity", {"plasticity": False}),
    ]
    for name, extra in conditions:
        accs, bursts, y2s = [], [], []
        for seed in range(5):
            r = run_experiment(seed, **{**defaults, **extra})
            accs.append(r["acc"]); bursts.append(r["mean_burst"]); y2s.append(r["mean_y2"])
        print(f"{name}: acc mean={np.mean(accs):.3f} range=[{min(accs):.3f},{max(accs):.3f}] "
              f"chance=0.062 mean_burst={np.mean(bursts):.3f} mean_y2={np.mean(y2s):.3f} "
              f"accs={[round(a,3) for a in accs]}")
