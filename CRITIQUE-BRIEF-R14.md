# Critique brief: round 14 (delta rule)

Read the full record first: `simulation-results.md` (sections 1-43, especially
41-43), `rpc_v2.py` (assumptions A1-A24), `RPC-HUNT-REPORT.md`, the two earlier
critiques (`kimi-critique.md`, `agy-critique.md`), and the third external
critique (`external-critique-3.md`). The review pack is on Beast at
`/home/reza/zfs-1tb/rpc-sim/review/`. The public repo is
https://github.com/Abolfazl-Alipour/random-predictive-coding-sim (head
ac8b34c4 at brief time; round 14 push follows).

## What round 14 claims

1. A23 (drive prototypes) fails its diagnostic on all three versions
   (section 42.3-42.4, 43.7): sine top-6 trained 0.281 vs frozen 0.444,
   sine full code 0.185 vs frozen 0.447, planted top-6 0.132 vs frozen
   0.129. Thesis-claim version (a), all wires fixed random with a fixed
   readout, is dead.
2. A24, the local signed prediction-error (delta) rule, works on both
   tasks at n_out 256 with the unmodified p1-prototype cosine readout
   and all other wires fixed random (section 43.2, 43.5): sine-fixed
   0.484 vs frozen 0.031, gain +0.453, both AGY bars pass (0.46,
   +0.40), first theory-faithful config to beat Markov-1 (0.459);
   planted motifs 0.221 vs frozen 0.062, gain +0.159, motif-only
   0.966, all three AGY bars pass (0.22, +0.15, 0.75), marginal on
   accuracy (+0.001) and gain (+0.009); strict bar 0.2625 NOT met.
3. Mechanism (section 43.3, ridge probe on the delta-256 drive):
   trained drive decodes the next token at 0.913 vs frozen 0.917.
   Delta adds no information; the win is readout alignment.
4. Controls (sections 43.4, 43.6): coactivity and burst collapse to
   chance at n_out 256 (sine 0.024/0.028, planted 0.059/0.059,
   identical per-seed scores on planted); centring alone recovers
   partway (sine 0.151, planted 0.154). Only the error term reaches
   the bars.

## What to check

1. Is the delta win real, or is it test-set selection? Alpha 1.0 and
   smooth sigma 3.0 were selected on the test set (third critique).
   Round 15 (validation split, 20 seeds) was cancelled by Reza, so
   this critique is the check: with 5 seeds and no validation split,
   how much weight can the sine 0.484 and planted 0.221 carry? Name
   the exact failure mode if the win would not survive a clean split.
2. The planted passes are marginal (accuracy +0.001, gain +0.009
   over AGY's bars). Is 0.221 meaningfully above 0.22, or is it
   noise? What would you need to see to call planted solved?
3. The planted coactivity/burst controls give byte-identical
   per-seed scores (section 43.6). The record says both rules
   collapse to the same constant prediction and the driver was
   verified correct. Check that claim against the code: is there
   any path (shared state, a bug in `run_planted_motif`, the
   `merged` dict) that would make two different rules produce
   identical networks?
4. The probe claim (section 43.3): frozen drive decodes at 0.917,
   trained at 0.913, so "the win is readout alignment". Check the
   probe code (`rpc_sine_fix14c.py`, `probe_delta`): same test data
   as `run_task` (seed+1000, proto_n=50)? Does the probe train with
   the same config as the scored network (use_confirmation=False,
   5 epochs)? Is "alignment" the right word, or is there a
   confound (e.g. the probe's ridge regularisation, or the frozen
   drive being probed after calibration on train data)?
5. Two driver bugs were found and fixed mid-round (section 43.2,
   43.6): proto_n/use_confirmation passed to the network
   constructor, and alpha passed twice to `run_planted_motif`.
   Audit the fix14 drivers for any remaining config mismatch
   between the scored conditions and the stated config, especially
   anything that would make the delta cell differ from its
   controls in more than the learning rule.
6. The sine 0.484 beats Markov-1 (0.459) but is far below Markov-2
   (0.802). What does that say about what delta actually learned:
   one step of history, or more? What test would settle it?
7. Variable sine, Lorenz, and the 1D track were not run under
   delta. Is the sine/planted win enough to claim the theory
   "learns in noise", or does the claim need those tasks? Name
   the cheapest next experiment that would change the verdict.
8. The thesis question: does the delta rule still count as the
   thesis theory (Chapter 3), or is it a different mechanism
   (local error-driven plasticity, Urbanczik and Senn 2014) wearing
   the thesis label? Say plainly what survives of the original
   claim and what does not.

## Output

Write `r14-kimi-critique.md` (Kimi) or `r14-agy-critique.md` (AGY) in the
review directory. Lead with a verdict: does the delta win stand, and
what is the single strongest reason to doubt it? Then the checks above,
each with a file/line reference and a verdict (verified / refuted /
unresolved). End with the next experiment, if any, that would settle
the open question.
