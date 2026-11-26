You are critiquing a simulation study, not building anything.

Working directory: /home/reza/zfs-1tb/rpc-sim/review
Read first: RPC-HUNT-REPORT.md (the whole report, especially
section 7, which is your brief). Then read the full record
simulation-results.md (sections 1-40) and the code rpc_v2.py
plus the drivers rpc_*.py and the *_results.txt files.

Do exactly what section 7 asks: (1) critique the work,
(2) check the root-cause chain, (3) tell Muse how to do it
right and name the single most promising next experiment,
(4) say whether A23 (drive prototypes) is the right next test
or another no-op like A22, working through the linear algebra
first, (5) say whether the theory can keep fixed random wires
or needs learned wires somewhere, and where exactly.

Write your critique as a report file in this directory:
kimi-critique.md (if you are Kimi) or agy-critique.md (if you
are agy). Plain English. Quote the exact file and line or the
exact results file when you say something is wrong. Do not
invent numbers. If you cannot check something, say so.

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
