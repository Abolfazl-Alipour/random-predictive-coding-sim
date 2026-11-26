You are critiquing a simulation study, not building anything.
You are the Opus model running under AGY. A Gemini run under
AGY already critiqued this work (agy-critique.md) and Kimi
critiqued it too (kimi-critique.md). Your job is different:
check their work, and critique what is new since they read it.

Working directory: /home/reza/zfs-1tb/rpc-sim/review
Read first: RPC-HUNT-REPORT.md (the whole report, especially
section 7, which was the first brief). Then read the full
record simulation-results.md (sections 1-41; section 41 is
the synthesis of the Kimi and Gemini critiques, with every
checkable claim verified against the code). Then read the code
rpc_v2.py (now with ASSUMPTION A24, the delta rule) plus the
new round-14 drivers rpc_sine_fix14a.py, rpc_sine_fix14b.py,
rpc_planted_fix14a.py, rpc_sine_fix14c.py, rpc_sine_fix14d.py,
rpc_planted_fix14c.py, rpc_planted_fix14d.py. Then read the
two existing critiques (kimi-critique.md, agy-critique.md).

Do exactly this:
1. Check the section-41 corrections. Are the verified
   corrections right? Did Muse verify them correctly, or is
   any verification wrong? Name anything Kimi or Gemini got
   wrong, and anything they missed.
2. Critique the A24 delta-rule design (rpc_v2.py, ASSUMPTION
   A24, and the round-14 drivers) BEFORE its results land.
   Is the rule implemented as AGY specified it? Is eta_d
   0.2 (sine) / 0.05 (planted) the right scale, or will it
   be too small or too large? Is the centred-covariance
   control a fair control? Is there a design flaw in the
   2x2 factorial (delta/coactivity/burst/cov x n_out 64/256)
   that will make the result hard to read?
3. Critique the A23 diagnostic setup (drivers fix14a/fix14b
   and planted fix14a). Is the frozen control the right
   leakage control? Will the 5-seed 5-epoch run settle the
   Kimi-vs-AGY disagreement, or is something still missing?
4. Name the single most promising next experiment AFTER
   round 14, and say what would change your mind about it.

Write your critique as a report file in this directory:
agy-critique-opus.md. Plain English. Quote the exact file
and line or the exact results file when you say something is
wrong. Do not invent numbers. If you cannot check something,
say so.

Rules: do not modify the report, the record, or the code.
You may run small checks of the code (1 seed, 1 epoch only)
to verify a claim. If you need a Python environment, create
it under /home/reza/zfs-1tb/rpc-sim/review-venv (requirements
are in requirements.txt). Do NOT write anything to Beast's
old disk (it is 99 percent full). Do NOT touch
/home/reza/projects/agescape-revision or any Agescape files.
Keep CPU load low: a SAMap job is running on Beast and must
not be disturbed. Do not launch any run longer than 15
minutes.
