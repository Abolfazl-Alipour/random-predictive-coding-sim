# Critique brief: rounds 17 and 18 (top-down clamp, continuous wall)

Read the full record first: `simulation-results.md` (sections 44-53,
especially 46, 48, 52, 53), `PROTOCOL-R17-CLAMP.md`,
`PROTOCOL-R18-CONTINUOUS.md`, `rpc_v2.py` (assumptions A1-A27, especially
A26 feedback wire and A27 continuous bump), `rpc_round17_clamp.py`,
`rpc_round18_continuous.py`, `r17/r17_analysis.md`, `r18/r18_analysis.md`,
and the round-14 critiques (`r14-kimi-critique.md`, `r14-agy-critique.md`)
for how hard the last review was. The review pack is on Beast at
`/home/reza/zfs-1tb/rpc-sim/review/`. The public repo is
https://github.com/Abolfazl-Alipour/random-predictive-coding-sim (head
b2543c73, sections 1-48; sections 49-53 are NOT yet pushed, so read the
Beast pack, not the repo, for rounds 15-19).

## What rounds 17 and 18 claim

Round 17 (section 53, PROTOCOL-R17-CLAMP.md): experiment B FAILS.
The fixed random top-down wire (A26, fb_gain 0.1, picked on validation
acc 0.226 against gain-0.0 0.227) carries no causal role. Lag-1:
C1 FAIL (cut degrades omit_m2 by +0.005, CI +0.000 to +0.010, bar
needs >= 0.10, fails by a factor of 20); C2 FAIL (shuffle -0.001);
C3 FAIL (p1_cos wire-on vs wire-cut 0.998, bar needs < 0.90, the wire
moves module 1's code by 0.2 percent); C4 FAIL (wire-off +0.004, CI
includes zero); C5 guard PASS (0.228 vs 0.229). Lag-2: only C1
passes (+0.004, a 0.4-point effect). Total 2/9 bars. Omit_m2 lag-1:
wire-on 0.953, cut 0.948, shuffle 0.953, off 0.948, frozen 0.083.
Feedback RMS is 6.5 percent of the reservoir drive, yet module 1's
code is 99.8 percent identical with the wire on or cut. Error
magnitude identical (last epoch 1.619 on vs 1.617 off). Strong fixed
feedback is worse (gain 1.0 drops validation to 0.182), so a learned
feedback wire is the untested option, now a hierarchy question
(round 19).

Round 18 (section 52, PROTOCOL-R18-CONTINUOUS.md): the continuous
wall survives correct scoring. Round 15 scored the model two steps
ahead while baselines scored one step ahead; correct scoring roughly
doubles accuracy (Lorenz C1 0.333 vs mis-scored 0.164, var-sine C1
0.295 vs 0.150), so the pilot wall was partly a scoring artifact.
But the wall survives on MSE, the metric continuous prediction is
judged by. Bars D1-D8: 5 pass, 3 fail. D1 PASS (Lorenz C1 acc beats
persistence and Markov-1). D2 PASS, barely (var-sine C1 acc beats
Markov-1 by 0.011, CI 0.000-0.021, lower bound exactly zero).
D3 FAIL (var-sine C1 MSE 0.064 vs persistence 0.056, paired diff
-0.008, CI includes zero). D4 FAIL, badly (Lorenz C1 MSE 22.390 vs
persistence 12.605, paired diff -9.785). D5 PASS (ridge halves
Lorenz MSE, C2 11.403 vs C1 22.390). D6 PASS (bump cuts it by two
thirds, C3 7.292). D7 PASS (combined C4 3.795 beats C2, C3, and the
ESN's 4.308). D8 FAIL: frozen ridge C2f 10.818 beats trained ridge
C2 11.403, and C4f 3.615 beats C4 3.795, so delta adds no continuous
information beyond what the frozen reservoir already holds. H0
(D1, D2, D3) FAILS because D3 fails. Markov-2 (not a bar) scores
Lorenz acc 0.556 / MSE 1.416 and var-sine MSE 0.005, beating every
model configuration on continuous MSE. Settled: delta does not learn
continuous information; the best continuous configuration is bump
input plus ridge readout of the frozen reservoir.

## What to check

1. Round 17's wire: is it really live in the code? Feedback RMS is
   6.5 percent of the reservoir drive, yet p1_cos is 0.998 and the
   error magnitude does not move. Read the A26 implementation in
   `rpc_v2.py`: where exactly does the feedback enter module 1's
   update (before or after the leak, before or after the
   normalisation)? Is there any step (a normalisation, a
   threshold, the leak rate) that would wash out a 6.5 percent
   drive term, so that the "wire is inert" verdict is a property
   of the implementation rather than a finding about feedback?
   If the wire cannot matter at gain 0.1 by construction, say so
   and name the gain or the code change that would make the test
   real.
2. Round 17's lag-2 C1 pass: a 0.4-point effect passes because the
   lag-2 C1 bar has no 0.10 floor (any degradation with CI above
   zero). Is that pass meaningful, or is it a bar artifact? Would
   the lag-2 bar have passed with the same 0.10 floor as lag-1?
   Does it change the experiment-B verdict either way?
3. Round 17's winner pick: validation picked fb_gain 0.1 at
   0.226 against 0.227 for gain 0.0, a tie with no wire. The test
   then compares a wire that validation already said does not
   help. Is the clamp test biased toward a null by its own winner
   pick? Would picking on omission accuracy, or on the cut gap
   itself, have changed which gain was tested?
4. Round 18's D2 and D3 are both borderline: D2 passes with its
   lower CI bound exactly at zero, D3 fails with its CI including
   zero. Is "H0 fails" the right verdict when the deciding bars
   sit on the boundary? What would a 40-seed rerun do to D2 and
   D3, and is the var-sine MSE wall established at all?
5. Round 18's D8: the frozen ridge BEATS the trained ridge
   (C2f 10.818 vs C2 11.403, C4f 3.615 vs C4 3.795). Check the
   ridge implementation in `rpc_round18_continuous.py`: same
   data, same regularisation, same probe for trained and frozen?
   The round-14 probe confound was that delta shrinks the drive
   norm 4.4x, which changes ridge conditioning. Is D8 "delta adds
   nothing", or "delta actively harms the ridge"? The two have
   different implications for the next design.
6. Round 18's scoring correction: verify it in the code. Does
   `rpc_round18_continuous.py` convention A (pred(t) scored
   against token t) match what `rpc_v2.py` `run()` actually
   scores, and are the baselines (Markov-1/2, persistence, ESN)
   now on the same footing? Audit the fix against the original
   bug in `rpc_round15.py` `run_cont` and `baselines.py`
   `mse_from_bin_preds`. Is there any remaining one-step shift
   anywhere in the round-18 numbers?
7. Markov-2 beats every model configuration on continuous MSE by
   a wide margin (Lorenz 1.416, var-sine 0.005). Both tasks are
   smooth and nearly deterministic given two past bins. Is the
   continuous wall a property of the tasks (too easy for a
   lookup, too hard for a one-step reservoir) rather than of the
   model? What continuous task would actually discriminate
   learned prediction from lookup, and does the current task
   choice make D4/D8 uninformative?
8. The mapping tension: omission (A) passes and the clamp (B)
   fails, which matches the mapping research (the model explains
   omission only, and a fixed random wire cannot deliver a
   stimulus-specific template). Does A-pass plus B-fail strengthen
   or weaken the "predictive coding" claim? What does the model
   now explain that a plain reservoir plus NLMS (Widrow-Hoff)
   does not? Say plainly what survives of the thesis claim.
9. Round 19 interaction: round 19 tests a hierarchy with fixed
   feedback gains 0.1/0.3, val-picked on overall accuracy
   (PROTOCOL-R19-HIERARCHY.md). Given round 17's wire was inert
   at 6.5 percent drive and stronger fixed feedback was worse,
   does round 19 risk repeating round 17's null by construction?
   What feedback magnitude would the wire need to matter, and
   does the round-19 design measure it? Name the cheapest
   change to round 19 that would make its feedback test real.
10. Next experiment: the generative test (D) is next per the
    round-18 scope rule (D8 failed), and precision (C) is still
    unrun. Given B failed and D8 failed, which one settles more,
    and what exactly would each one need to show to change the
    verdict that this is local error learning, not predictive
    coding?

## Pack contents for the launch (Beast review pack)

Sync these into `/home/reza/zfs-1tb/rpc-sim/review/` before launching:
`simulation-results.md` (sections 44-54), `PROTOCOL-R17-CLAMP.md`,
`PROTOCOL-R18-CONTINUOUS.md`, `PROTOCOL-R19-HIERARCHY.md`,
`rpc_v2.py`, `rpc_round17_clamp.py`, `rpc_round18_continuous.py`,
`r17/` (monitor, analyzer, winner.json, test JSONLs, analysis),
`r18/` (monitor, analyzer, winner.json, test JSONLs, analysis),
`CRITIQUE-BRIEF-R1718.md` (this file). Launch pattern: same as
round 14 (Beast-side launch script, Kimi PID + AGY Gemini PID,
separate logs). AGY Opus is quota-blocked until about Dec 1, 2026;
do NOT attempt Opus (its round-14 retry is separately scheduled
Dec 2). Paste the whole brief to each reviewer in ONE message.

## Output

Write `r1718-kimi-critique.md` (Kimi) or `r1718-agy-critique.md`
(AGY) in the review directory. Lead with a verdict: does
experiment B's failure stand, does the continuous wall stand, and
what is the single strongest reason to doubt each? Then the checks
above, each with a file/line reference and a verdict (verified /
refuted / unresolved). End with the next experiment, if any, that
would settle the open question.
