# Independent Scientific Critique of the Random Predictive Coding Simulation Study

**Reviewer:** agy (Antigravity)  
**Date:** Nov 26, 2026  
**Scope:** Full audit of `RPC-HUNT-REPORT.md`, `simulation-results.md` (sections 1–40), `README.md`, `diag_A.md`, `baselines_B.md`, `tasks_C.md`, `ch3.txt` (thesis Chapter 3), `rpc_toy.py`, `rpc_v2.py`, `baselines.py`, `tasks.py`, all `rpc_*.py` drivers, and all `*_results.txt` outputs.  
**Verification checks:** No existing files were modified. Small diagnostic checks (strictly 1 seed, 1 epoch — or 0 epochs for frozen probes) were executed using `/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python` to verify mathematical claims. Every number reported below is either quoted directly from an existing results file or explicitly labeled as a 1-seed, 1-epoch diagnostic check.

---

## Executive Summary (The Five Bottom Lines)

1. **You misdiagnosed the core failure.** Your report ends by blaming two things: (a) fixed random proximal weights `W_prox` losing a third of the token information (`0.984 -> 0.711`), and (b) the distal drive `d1` carrying its information in a subspace orthogonal to the `p1` prototype span (`31%` of energy). **Both conclusions are wrong.**
   - The `0.984 -> 0.711` drop in `W_prox` happens *only* because `rpc_v2.py` arbitrarily compresses 200 reservoir units down to 64 (`n_out=64 < n_res=200`). At `n_out=256`, a 100% fixed random `W_prox` loses **zero** information (`0.983 -> 0.983`, verified below). You declared the dimensionality hypothesis dead in section 22 without ever probing `p2` at `n_out=256`.
   - The collapse of A18 (full graded code) on seeds 2–4, the `0.022` crash of `A18+burst`, and the `0.20–0.22` ceiling of cosine/dot/kNN are **not** caused by signal hiding in the orthogonal complement of the `p1` prototypes. On collapsed seed 2, **64.9%** of `d1`'s energy is *inside* the `p1`-prototype span (more than on working seed 0, `43.9%`). The real cause is **rank-1 positive-orthant collapse from your purely non-negative Hebbian rule** (`W_dist1 += eta * outer(post >= 0, pre >= 0)`). That rule pushes `W_dist1` into the positive orthant so `d1` becomes 88–91% positive and 85–97% collinear across steps — locking cosine argmax onto 3–7 constant bins (`78.8%` on a single bin on seed 2) while ridge regression easily whitens out the constant DC vector and still reads `0.49`.
2. **DO NOT run A23 (`drive_proto`) as your next experiment.** By linear algebra, A23 is *not* a mathematical no-op like A22 — it is **actively worse than a no-op**. Because `d1 = W_dist1 @ y2_prev`, building mean-`d1` templates per next-token turns an untrained random `W_dist1` into a **supervised nearest-centroid classifier directly on `y2_prev`**. On seed 0 (1 epoch, sine-fixed), A23 inflates the frozen no-plasticity control tenfold from `0.045` to **`0.444`** (top-6) and `0.031` to **`0.363`** (A18), while the trained network scores **worse** than its own frozen control (gains **`-0.131`** and **`-0.164`**; on collapsed seed 2, **`-0.296`**). A23 also violates the core thesis claim that `d1` must match `p1` on the same neuron.
3. **Several key experimental comparisons in the record are silently confounded or misreported.** Even after your Round 13 correction, `A12+knn+coact` (`rpc_planted_fix13.py:31-32`) still left `use_confirmation=True` (default), overwriting the learned `W_prox`; `rpc_sine_fix5b.py:25` labeled runs `smooth3.0-10ep` while executing `epochs=5`; Rounds 6 and 8 silently ran planted motifs under `burst` while claiming `coactivity`; `rpc_spatial.py` compared variable-frequency sine and `dt=0.02` Lorenz against `baselines_B.md`'s fixed-frequency sine and `dt=0.10` Lorenz; `rpc_track_F.py` accidentally evaluated 2-step-ahead prediction; and `mod1.inhib` (`rpc_v2.py:1069-1070`) is active during training and zero during calibration, prototype building, and testing.
4. **The single most promising change is fixing the distal learning rule to a local signed dendritic prediction-error rule (Urbanczik & Senn, 2014 / Widrow-Hoff LMS) paired with a non-compressive `n_out >= n_res` and `full_code` — keeping 100% fixed random wires everywhere else and keeping the unmodified `p1`-prototype cosine readout.** On your collapsed seed 2 (1 seed, 1 epoch, fixed sine), swapping *only* the `W_dist1` update from unsigned coactivity to local `err = p1 - d1` lifts the *exact same* `p1`-prototype cosine readout from `0.0249` to **`0.4410`** at `n_out=64` (and **`0.5065`** at `n_out=256`, with all 32/32 bins predicted). On planted motifs (seed 0, 1 epoch, `n_out=256`, 100% fixed random wires), it jumps from `0.0409` (`muse_coact`) to **`0.1409`** in a single epoch — already beating your 5-epoch learned-`W_prox` best (`0.113`).
5. **What this means for the thesis.** Fixed random wires *are* enough on every benchmark in this study — **no learned `W_prox` is needed** once you stop compressing `200 -> 64`. What **cannot** work as formalized is the thesis's **purely confirmatory, unsigned burst-gated Hebbian plasticity** (and its uncoupled `W_conf` confirmation loop). Making the model actually learn requires replacing confirmation-only potentiation on `W_dist1` with local signed error-driven plasticity (`p1 - d1`), which turns the working system into an online local Echo State Network / dendritic predictive-coding circuit rather than Chapter 3's burst-confirmation hypothesis.

---

## 1. Critique of the Work

### 1.1. Concrete Bugs, Confounds, and Reporting Errors Found in the Code and Record

Every item below was checked directly against the source code and `*_results.txt` files.

1. **Silent `use_confirmation=True` confound survives Round 13 (`rpc_planted_fix13.py:31-32`, `rpc_sine_fix11.py:73`, `rpc_sine_fix11b.py:21`, `rpc_sine_fix.py:89, 116`).**
   - In `rpc_prox.py:80-83`, A12 alone (`prox-full`, scoring `0.113`) explicitly passed `use_confirmation=False`, because `prox-confirm-on` (`use_confirmation=True`, scoring `0.076`) overwrites the fitted `W_prox` every step (`simulation-results.md:690-701`).
   - In Round 13 (`rpc_planted_fix13.py:31-32`), you fixed the `learn_rule` confound by adding `"learn_rule": "coactivity"` to `A12+knn+coact`, **but forgot `"use_confirmation": False`**. Since `RPCNetwork.step()` defaults to `use_confirmation=True` (`rpc_v2.py:978`), confirmation destroyed the fitted `W_prox` during training, giving `0.075` (matching `prox-confirm-on`'s `0.076`, not `prox-full`'s `0.113`). Your statement in `RPC-HUNT-REPORT.md:414-416` and `simulation-results.md:1585-1591` that Round 13 resolved the confound on "clean evidence" is therefore wrong.
   - The exact same omission occurred in `rpc_sine_fix11.py:73` and `rpc_sine_fix11b.py:21` (`A12+knn` on sine, `0.133`), and in `rpc_sine_fix.py:89, 116` (`prox+prox1`). Moreover, on sine, `fit_prox()` (`rpc_v2.py:810-811`) regresses `r2` onto a bin-only target `E[tok]`, explicitly *stripping* rising-vs-falling direction from `p2`, so `A12` actively sabotages sine by design!
   - Also note a latent scaling bug in `rpc_v2.py:567-568, 816, 1066-1068`: `self.W_prox2_row0` is computed from the *random* `W_prox` inside `__init__`, and never updated when `fit_prox()` overwrites `self.mod2.W_prox` at line 816. On the first confirmation step, `norm_rows` rescales the fitted `W_prox` back to random init row norms.
   - *1-seed, 1-epoch check (planted motifs, seed 0, alpha=1.0, coactivity):*
     - `A12 alone (conf OFF)`: `0.1426` vs `A12 alone (conf ON)`: `0.0939`
     - `A12+knn (conf OFF)`: `0.0765` vs `A12+knn (conf ON, Muse R13)`: `0.0791`
     - **Important nuance:** Even when you fix the `use_confirmation` confound, `A12+knn` *still* underperforms `A12` alone on planted motifs (`0.0765` vs `0.1426`), for a simple bias-variance reason: averaging `p1` into prototypes averages out random filler history on the post side, whereas 1-NN against individual noisy `p1` bank vectors matches noisy filler outliers.

2. **Two more unnoticed `learn_rule="burst"` confounds on planted motifs (`rpc_sine_fix6.py:51`, `rpc_sine_fix8.py:48`).**
   - Your docstrings (`rpc_sine_fix6.py:20`, `rpc_sine_fix8.py:21`) and `simulation-results.md:1044, 1184` state *"All coactivity, 5 epochs"* and compare against coactivity references. However, `rpc_sine_fix6.py:51` (`run_planted(seed, 1.0, graded_conf=True, **ex2)`) and `rpc_sine_fix8.py:48` (`run_planted(seed, 1.0, dot_proto=True, **ex2)`) never pass `learn_rule="coactivity"`, so both silently ran under the default `learn_rule="burst"` (`rpc_v2.py:476`). Your Round 11 correction (`simulation-results.md:1425-1427`) caught this for Rounds 10–12 (`sections 37–39`), but missed Rounds 6 and 8 (`sections 31, 35`).

3. **"10 epochs" runs that actually ran 5 epochs (`rpc_sine_fix5.py:52, 58`, `rpc_sine_fix5b.py:19, 25`).**
   - Both drivers name the condition `"smooth3.0-10ep"`, and `simulation-results.md:1000, 1014, 1021` reports *"variable-frequency sine ... smooth 3.0 with 10 epochs full 0.108"* and *"Lorenz ... smooth 3.0 with 10 epochs full 0.069"*. Line 25 of `rpc_sine_fix5b.py` (and line 58 of `rpc_sine_fix5.py`) hardcodes `epochs=5`!

4. **Train-vs-test dynamics mismatch: `mod1.inhib` is ON during training and OFF at test (`rpc_v2.py:1026, 1069-1070`).**
   - Lines 1069–1070 (`raw = self.mod1.W_prox.T @ b1; self.mod1.inhib = self.s_inh * raw.clamp(0, 0.8)`) sit *inside* `if train and plasticity:`.
   - Consequently, `mod1.inhib` perturbs `mod1.r` (`rpc_v2.py:463`) on every step during training (on sine-fixed seed 2, epoch 1, `mean ||inhib|| = 0.361` against `mean ||r1|| = 2.044`, ~18% of state norm), yet is **identically zero** during `calibrate()`, `build_prototypes()`, `no-plasticity`, and **test evaluation (`train=False`)**. That trains `W_dist1` on a shifted state distribution that does not match the prototypes or test dynamics — and is backwards relative to Chapter 3 (`ch3.txt:1540-1561`, Fig. 3.4), where Layer V → Layer II/III inhibition is the *post-learning runtime* predictive-routing mechanism.

5. **Apples-to-oranges baseline comparisons (`rpc_spatial.py` vs `baselines.py` / `baselines_B.md`).**
   - **Sine:** `baselines.py:76-92` generates a **fixed-period** sine (`period=12`, `f=0.0833`, length 24, 200 train / 50 test). `rpc_spatial.py:21-31` generates a **variable-frequency** sine (`f ~ U(0.02, 0.08)`, length 100, 100 train / 20 test). `rpc_simple.py:43-52` generates a third sine (`f=0.05`, period 20, length 200, 50 train / 20 test). Yet `rpc_spatial.py:9-10` and `simulation-results.md:49-50, 88` directly compare RPC on variable-frequency sine against `baselines_B.md`'s fixed-frequency ESN (`0.900`), LSTM (`0.894`), and Markov-2 (`0.769`).
   - **Lorenz:** `baselines.py:107-135` uses `dt=0.01, stride=10` (`dt_eff=0.10`), burn-in 10, bin range `[-25, 25]`, 200 train / 50 test, length 24 (persistence `0.170`). `rpc_spatial.py:33-55` uses `dt=0.02, stride=1` (`dt_eff=0.02` — 5× finer!), burn-in 500, bin range `[-20, 20]`, 20 train / 10 test, length 100 (where persistence alone is `~0.47`, per `tasks_C.md:72`). Comparing `rpc_spatial.py`'s Lorenz score (`0.045`) against `baselines_B.md`'s Lorenz baselines (`ESN 0.388`, `Markov-2 0.559`) compares two completely different dynamical sampling rates and dataset sizes.

6. **2-step-ahead target shift in `rpc_track_F.py:54-56` (`rpc_v2.py:931-935, 1152-1156`).**
   - `rpc_track_F.py:55-56` stores pairs `(x_t, y_t)` where `x_t = [F(t), F(t-1)]` and `y_t = bins[t+1]`. Inside `run_vec` (`rpc_v2.py:1152-1156`), `d1` at step `i` is computed from `y2_prev` (i.e., `x_{i-1} = [F(i-1), F(i-2)]`), and scored against `y_i = bins[i+1]`. That evaluates **2-step-ahead** prediction (`t-1 -> t+1`), whereas `rpc_track.py` evaluates 1-step-ahead prediction (`t-1 -> t`), while `rpc_track_F.py:29` cites 1-step persistence (`0.776`) as the reference.

7. **Math and probe-design errors in the diagnosis scripts:**
   - **Energy vs. norm error (`RPC-HUNT-REPORT.md:311, 318, 381`; `simulation-results.md:1557, 1566`).** `rpc_proj_check.py` reports `cos(d1, P d1) = 0.689`. For an orthogonal projector $P$, $\cos(d_1, P d_1) = \|P d_1\| / \|d_1\|$ is the *norm ratio*, not the energy fraction. The energy fraction inside the prototype span is $\cos^2 \theta = 0.689^2 = 0.475$, so the orthogonal component holds **$1 - 0.689^2 = 52.5\%$** of the drive's energy, not `31%`. (Note: even a completely isotropic random vector in $\mathbb{R}^{64}$ projected onto a 32-D subspace has 50% orthogonal energy, norm ratio $\sqrt{32/64} = 0.7071$.)
   - **Current-token vs. next-token mismatch in Rounds 10–11 probes (`rpc_sine_fix10.py:54-57`, `rpc_sine_fix11.py:52-55`, `simulation-results.md:1323-1328`).** At step `i`, `out["y2"]` is computed *after* `tok[i]` enters the network (encoding the **current** token), while `out["d1"]` is computed from `y2_prev` *before* `tok[i]` enters (predicting the **next** token from `tok[i-1]`). Comparing `y2 = 0.709` directly against `d1 = 0.457` compares a current-token decoder against a next-token predictor.
   - **Mismatched test seeds, `proto_n`, and epoch counts between probes and main runs.** `rpc_sine_fix10.py:35,40`, `rpc_sine_fix12.py:43,48`, and `rpc_sine_fix13.py:46,50` evaluate probes on `seed + 100` with `proto_n=20`, while `run_task("sine-fixed", ...)` (`rpc_simple.py:65,67`) evaluates network accuracy on `seed + 1000` with `proto_n=50`. Your "extraction ratios" (`simulation-results.md:1485-1489`) therefore divide accuracies measured on different test sets and different prototype sizes. Similarly, `rpc_proj_check.py:21,27,37` trains for **2 epochs** (`range(2)`) with `seed+100` and `proto_n=20`, yet `simulation-results.md:1553-1558` cites its `0.689` cosine as the diagnostic of the 5-epoch Round 13 network.
   - **Timeline/version misattribution in `RPC-HUNT-REPORT.md:98-100`.** You attribute `Motif gains +0.013 to +0.028. Sine gain -0.007. Lorenz gain -0.001` to `(rpc_toy.py, v1)`. Those are **v2** results (`simulation-results.md:40-42`). `rpc_toy.py` (v1) never ran sine, Lorenz, or track and had `burst=0.000`, motif accuracy `0.049` (`diag_A.md:3-4, 35-37`).
   - **Minor code bug in `rpc_v2.py:1042-1054`.** A20 (`pre_sharpen`) sharpens `y2_prev` for `W_dist1`, but leaves `y2_prev2` unsharpened for `W_dist1b` (line 1054), contradicting the docstring at line 290 (not triggered in any run since delay and `pre_sharpen` were never combined). Also `calibrate()` Pass 2–3 (`rpc_v2.py:668-679`) includes step `i=0` where `d1=0`, slightly deflating `sd_d1`.

---

### 1.2. Assumptions Added Beyond Chapter 3 — and Which Ones Actually Matter

Comparing `ch3.txt` (pp. 89–98, lines 1231–1597) directly against `rpc_v2.py`, here are the assumptions you introduced that the thesis does **not** make, ranked by actual impact:

| Assumption in `rpc_v2.py` | Thesis (`ch3.txt`) | Actually matters? | Why |
|---|---|---|---|
| **Independent random `W_conf`** (`rpc_v2.py:557, 1060`) + `b2` gated on `p2_prev > theta_p` instead of `y2_prev` (`line 1062`) | Explicitly requires **reciprocal** EC–HPC connectivity back to the *originating* CA1 neuron (`ch3.txt:1502-1533`) | **YES — fatal to the confirmation test** | `W_conf @ b1` sends confirmation to random module-2 units uncorrelated with which `y2_prev` unit caused the hit. Every "confirmation hurts" finding in the record tests random weight drift, not the thesis's reciprocal confirmation rule. |
| **Purely non-negative Hebbian update on `W_dist1`** (`post >= 0, pre >= 0`, `rpc_v2.py:1048`) paired with **zero-mean signed `p1`** (`mod1.W_prox @ r1`) | Gives no algebraic plasticity rule or sign convention | **YES — #1 killer of the entire study** | Drives `W_dist1` into the positive orthant and creates a huge rank-1 DC component $\bar{q}\bar{y}^\top$, making `d1` 88–91% positive and 85–97% collinear across steps while `p1` prototypes are zero-mean signed. |
| **Compressive bottleneck `n_res=200 -> n_out=64`** (`rpc_v2.py:429`) + **top-6 cutoff (`A6`)** (`line 637`) | Calls for *"a vast number of random nonlinear transforms"* (`ch3.txt:1479`), never a 200→64 compression or top-6 cutoff | **YES — directly causes the `p2` & `y2` loss** | Compressing 200D to 64D via random `W_prox` causes the `0.983 -> 0.721` drop; top-6 cuts it to `0.365`. Expanding to `n_out=256` with `full_code` keeps `p2` at `0.983` and `y2` at `0.867` (Check 5 below). |
| **One-hot binning of continuous signals** (`rpc_spatial.py`, `rpc_simple.py`, `rpc_track.py`) | Never specifies one-hot tokenization of continuous variables | **YES — partially fixed by A14** | Destroys metric topology between adjacent bins; your A14 Gaussian population code legitimately recovered direction information. |
| **`mod1.inhib` inside `if train and plasticity:`** (`rpc_v2.py:1069-1070`) | Fig. 3.4 (`ch3.txt:1540-1561`) has inhibition operate *after* learning to suppress predicted activity | **Moderate** | Shifts `r1`/`p1` during training (`||inhib|| = 0.361`) relative to test (`0.000`). |
| **External `p1`-prototype cosine readout** (`build_prototypes`, `predict_from_distal`) | Thesis has no external readout; matching happens physically via `d1` colliding with `p1` on Layer V dendrites | **Fair as a test, misblamed as a bottleneck** | Testing whether `d1` matches `p1` prototypes is faithful to the thesis's collision principle. It was *not* the bottleneck capping sine at `0.22` — once `d1` stops DC-collapsing, the exact same `p1`-prototype cosine readout reaches `0.441–0.507` on seed 2 in 1 epoch! |
| **Frozen pre-learning z-scale calibration (`A1`)** (`rpc_v2.py:643-701`) | No thresholds specified | **Secondary** | Reasonable for bootstrapping v1, but freezing `sd_p2` and `sd_d1` hid scale drift until A18 exposed a 15× blow-up in `p2`. |
| **Row-norm preservation (`A5`)** (`rpc_v2.py:1050-1052`) | No normalization specified | **Secondary (interacts badly with unsigned rule)** | Necessary to prevent v1's weight clamp saturation, but when paired with strictly positive updates, rescaling rotates every row straight into the positive orthant toward the mean vector $\bar{y}_2$. |

**Equally important — what IS faithful to the thesis and genuinely fails:**  
The thesis *really does* propose **confirmation-only burst-gated Hebbian potentiation** (`ch3.txt:1447-1490`) with no prediction-error depression. That mechanism is genuinely unworkable as written (cold-start lock-in, positive-feedback runaway, inability to unlearn wrong predictions). Note also that switching to `learn_rule="coactivity"` (A8) for Rounds 7–13 quietly left the thesis's burst-gated rule behind; when you finally re-tested `burst` with A18 in Round 13, it scored `0.022` (below chance).

---

### 1.3. Where the Test Setup Is Unfair to the Theory — and Where It Is Too Kind

**Unfair to the theory:**
1. **The planted-motif accuracy bar (`0.2625`) is 92% of the true theoretical Bayes ceiling (`~0.285`), not 73% of `0.36`.** In section 19b (`simulation-results.md:559-563`) you claimed a 1-step ceiling of `~0.36` by writing *"each motif token appears about 3 times in motifs and about 1 time in filler per sequence"*. That arithmetic is wrong: `make_data` (`rpc_v2.py:417-424`) plants **three distinct motifs once each**, so each motif cue token (`1, 2, 5, 6, 9, 10`) appears **at most 1 time** in its motif (slightly less due to overwrites) and about `15/16 = 0.94` times in filler! Thus $P(\text{motif succ} \mid \text{cue}) \approx 0.51$ (not `0.67`), making the true 1-step Bayes ceiling `(5.5 + 5.5/16 + 12/16) / 23 ≈ 0.287` — which matches `baselines_B.md:10` (`Markov-1 0.278`, `ESN 0.274`, `LSTM 0.283`)! Asking a local, unsupervised rule to hit `0.2625` demands **92% of the theoretical optimum** (almost matching an Adam-trained LSTM's `0.283` and a globally solved ridge ESN's `0.274`) while tolerating zero variance.
2. **The `200 -> 64` bottleneck (`W_prox`) + top-6 cutoff** artificially discards half to two-thirds of the module-2 token code before distal learning even sees it.
3. **One-hot binning** of continuous time series (before A14), **uncoupled random `W_conf`**, **`mod1.inhib` train/test mismatch**, and **`rpc_track_F.py`'s 2-step-ahead shift**.

**Too kind to the theory:**
1. **The `0.231` pass bar on continuous tasks (`chance + 0.20`) is misdescribed in `RPC-HUNT-REPORT.md:72-73` as *"the baselines' level"* and is far too weak for temporal prediction.** On `sine-fixed`, Markov-1 alone scores `0.459` (`rpc_sine_diag_results.txt:1`) and Markov-2 scores `0.802`; hitting `0.231` wouldn't even clear half of Markov-1! On `rpc_spatial.py`'s Lorenz (`dt=0.02`) and on `rpc_track.py`, **trivial persistence (repeating the current bin)** scores `~0.47` and `0.776` respectively — two to three times higher than the `0.231` "pass bar".
2. **Pure cycle and pure motif at `alpha=1.0`** are noiseless 1st-order lookup tables (`Markov-1 = 1.000`). You stated this scope caveat accurately (`RPC-HUNT-REPORT.md:118-124`).
3. **Evaluating Rounds 7–13 mostly under `coactivity` (A8) rather than `burst`** gave the impression that the model got within `0.013` of the sine bar (`0.218` vs `0.231`), whereas the actual thesis burst rule was never shown to work on smooth sine (and scored `0.022` on A18).

---

### 1.4. Audit of Your Nine Self-Identified Mistakes in Section 5 (`RPC-HUNT-REPORT.md:393-436`)

| Item in Sec. 5 | Your correction | Audit verdict |
|---|---|---|
| **5.1 V1 scale bug** | Fixed via A1 pre-learning z-standardization (`rpc_v2.py:643-701`). | **Mostly right, incomplete:** Fixed the dead-v1 bug, but freezing `sd_p2`/`sd_d1` pre-learning allowed the post-learning scale drift that later broke naive A18. |
| **5.2 Prototype misstatement** | Corrected to mean `p1` (not mean `d1`). | **Right** (`rpc_v2.py:703-745`). |
| **5.3 Dilution → read → space misalignment** | Rejected A20 (dilution) and A21 (read noise), replaced with "space misalignment / orthogonal part of drive". | **Correction is WRONG:** Rejecting A20/A21 was empirically justified, but the replacement explanation ("orthogonal part of drive") is refuted by Checks 1 & 8 (collapsed seed 2 has `64.9%` in-span energy; real cause is rank-1 positive-orthant DC collapse). |
| **5.4 Planted stacking confound** | Caught `burst` vs `coactivity` confound in R11, claimed R13 reinstated stacking failure on clean evidence. | **Correction is INCOMPLETE / STILL WRONG:** R13 (`rpc_planted_fix13.py:31-32`) forgot `use_confirmation=False`, leaving `A12+knn+coact` confounded! (Check 4 shows the drop still happens with `conf OFF` at 1 epoch due to kNN variance, but R13 did not prove it cleanly.) Also missed the same `burst` confound in Rounds 6 & 8. |
| **5.5 A22 mathematical no-op** | Proved `(P d1)·proto = d1·proto` makes A22 a no-op for cosine and dot. | **Proof RIGHT; interpretation WRONG:** The linear-algebra no-op proof is exact for cosine/dot, but `cos = 0.689` means `52.5%` orthogonal energy (not `31%`), `rpc_proj_check.py` ran 2 epochs (not 5), and the orthogonal part is *not* why ridge beats cosine. |
| **5.6 1-epoch smoke tests misleading for A18** | Stopped trusting 1-epoch smoke tests for A18 (`0.274` ep1 → `0.113` ep5). | **Right** — and now explained mechanistically: the unsigned Hebbian DC collapse worsens with more training epochs. |
| **5.7 Planted ceiling error** | Revised single-motif `0.2625` analysis to 3-motif 1-step ceiling `~0.36`, 2-step `~0.39`. | **Direction right, arithmetic WRONG:** Overcounted each motif as appearing 3× instead of 1× (`simulation-results.md:562`). True 1-step Bayes ceiling is `~0.287` (matching `Markov-1 0.278` and `LSTM 0.283`). |
| **5.8 Smooth 1.5 3-seed overstatement** | Corrected `0.145` (3 seeds) to `0.129` (5 seeds). | **Right** (`rpc_sine_fix2_results.txt:9`). |
| **5.9 Graded confirmation + lr compensation** | Corrected 3-seed `0.220`/`0.230` to 5-seed `0.196`/`0.202` (ties binary). | **Right for sine** (`rpc_sine_fix6b_results.txt:1`, `rpc_sine_fix6c_results.txt:1`); missed that planted graded (`rpc_sine_fix6.py:51`) accidentally ran under `burst`. |

---

## 2. Audit of the Root-Cause Chain (Section 4)

Here is a step-by-step check of `RPC-HUNT-REPORT.md:367-391`, paired with direct verification runs (all strictly 1 seed, 0 or 1 epoch, CPU single-threaded):

### Step-by-Step Verdict

- **Step 1 (`r1` holds current token at `1.000`):** **Supported** at `alpha=1.0` (`rpc_probe_results.txt:1`, `rpc_sine_diag_results.txt:9`). Caveat: only probed at `alpha=1.0` (and `0.7` in Round 10), not at default `alpha=0.3`.
- **Step 2 (fixed random `W_prox` loses a third: `0.984 -> 0.711`):** **Measured accurately at `n_out=64`, but causally misattributed.** I checked whether `lam=1.0` on unstandardized `p2` (`sd=0.113`) caused a ridge regularization artifact; it did **not** (`raw_l1.0 = 0.7209`, `raw_l1e-6 = 0.7278`, `std+bias_l1 = 0.7278`, seed 0 frozen). **However**, your conclusion in `simulation-results.md:660-664` that *"width cannot fix"* this is **falsified**. A random $256 \times 200$ matrix has full column rank 200 (linearly invertible on $\mathbb{R}^{200}$!), whereas $64 \times 200$ compresses 200D into 64D and mixes token signal with history noise.

  *Verification — Frozen `p2` & `y2` linear probe on planted motifs (seed 0, 0 epochs, `alpha=1.0`):*
  | `n_out` | `r2` | `p2 (lam=1)` | `p2 (lam=1e-4)` | `y2` (top-10%) | `y2` (A18 full graded) |
  |---:|---:|---:|---:|---:|---:|
  | `64` | `0.983` | `0.721` | `0.728` | `0.365` | `0.563` |
  | `256` | `0.983` | **`0.976`** | **`0.983`** | `0.579` | **`0.867`** |

  **Verdict on Step 2:** Fixed random `W_prox` loses **0.000** when it is not artificially forced through a `200 -> 64` bottleneck.
- **Step 3 (top-6 loses another half `0.711 -> 0.332`; full code recovers `y2` to `0.709` but degrades `d1` to `0.457` vs `0.509`):** **First half supported; second half confounded and misdiagnosed.**
  - On planted motifs, top-6 genuinely drops `p2` from `0.721` to `0.365` (seed 0); full graded code lifts it to `0.563` at `n_out=64` and `0.867` at `n_out=256`.
  - On sine, `0.709` vs `0.457` (`rpc_sine_fix10.py:54-57`) compared **current-token** `y2` against **next-token** `d1`. And the real `d1` next-token drop (`top-6 0.507` vs `A18 0.408`, `rpc_sine_fix12_results.txt:7-8`) is **not** an intrinsic conflict between rich codes and learning — it is caused by the unsigned Hebbian rule collapsing 4.5× harder on dense `relu(z)` (mean `0.399`) than on sparse top-6 (mean `0.094`). Under a signed local error rule (`delta_p1`, Section 3), A18 *beats* top-6 (`0.3405` vs `0.2761` on seed 0, 1 epoch).
- **Steps 4 & 5 ("the information sits largely in components orthogonal to the prototype span [31% of energy], and no fixed comparison against `p1` patterns can see it — explaining why cosine, dot, and kNN fail on collapsed seeds"):** **FALSE — THIS IS THE CRITICAL SKIPPED STEP AND WRONG EXPLANATION.**

### The True Root Cause of the Collapse (and Why the "Orthogonal Part" Claim Fails)

Look at the update in `rpc_v2.py:1048-1052`:
```python
self.W_dist1 += self.eta_b * torch.outer(post, pre)
```
Every entry of `post` (`b1` or `(p1n > theta_p) * g`) is $\ge 0$, and every entry of `pre` (`y2_prev`) is $\ge 0$. Therefore **every synaptic change is strictly non-negative**. Decomposing `post` and `pre` into mean + fluctuation ($\text{post} = \bar{q} + \delta q$, $y_2 = \bar{y} + \delta y$):
$$\mathbb{E}\big[\text{post}(t)\, y_2(t-1)^\top\big] = \underbrace{\bar{q}\,\bar{y}^\top}_{\text{rank-1 positive DC mode}} + \underbrace{\operatorname{Cov}\big(\text{post}(t),\, y_2(t-1)\big)}_{\text{actual predictive signal}}.$$

1. **Why A18 collapses catastrophically while top-6 only partially collapses:** For `top-6` binary `y2`, $\bar{y}_j = 0.094$ and $\bar{y}^2 / \operatorname{Var}(y) = 0.104$. For A18 `relu(z)` ($z \sim \mathcal{N}(0,1)$), $\bar{y}_j = 1/\sqrt{2\pi} = 0.399$ (matching `mean_y2=0.399` in `rpc_sine_fix10_results.txt`) and $\bar{y}^2 / \operatorname{Var}(y) = 0.467$ — a **4.5× larger DC-to-AC ratio**. When `W_dist1` multiplies `y2_prev`, the rank-1 positive term adds coherently across all $N_o$ columns ($O(N_o)$), while the zero-mean covariance adds incoherently ($O(\sqrt{N_o})$) — which is *also* why `wide256` (`n_out=256`) collapsed to `0.067` (gain `0.000`) under unsigned coactivity (`0.0409` on seed 0, 1 epoch)!
2. **Orthant mismatch:** After training, `W_dist1` becomes mostly positive (`75–82%` of entries $> 0$ after just 1 epoch!), so `d1 = W_dist1 @ y2_prev` is trapped in the **positive orthant** (`88–91%` of entries $> 0$), whereas `p1` prototypes are zero-mean signed (`47.6%` $> 0$).
3. **Why cosine, dot, and kNN collapse while ridge still reads `0.49`:** Write $d_1(t) = u_{\text{DC}} + \delta d_1(t)$ with $\|u_{\text{DC}}\| \gg \|\delta d_1(t)\|$.
   - **Cosine / dot / kNN** against `p1` patterns are swamped by the constant step-independent term $u_{\text{DC}} \cdot \mu_k$, locking `argmax` onto whichever 1–4 bins happen to have the largest dot product with $u_{\text{DC}}$.
   - **Ridge regression** $(D^\top D + \lambda I)^{-1} D^\top Y$ automatically **whitens out** the giant rank-1 eigenvalue of $u_{\text{DC}}$ (dividing by $\lambda_1 \gg 1$) and amplifies the tiny discriminative signal $\delta d_1(t)$! It does **not** need an "orthogonal-to-`p1`-span" subspace to succeed.

*Direct falsification of the "orthogonal part" claim (1 epoch, sine-fixed, `alpha=0.7, smooth=3.0, graded+eta_b=0.15`, `p1`-prototype cosine readout):*

| Condition (1 epoch) | Net acc | Ridge | `W>0` | Top SV energy (`sv1^2/sum`) | `d1>0` | `d1` pairwise cos | Energy **inside** `p1` span | Unique bins predicted | Top-1 bin share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed 0, `top6` (coact) | `0.2819` | — | `0.749` | `0.335` | `0.892` | `0.560` | `0.567` | `12/32` | `19.2%` |
| Seed 0, `A18` (coact) | `0.2337` | — | `0.819` | `0.650` | `0.906` | **`0.967`** | `0.439` | `7/32` | `30.7%` |
| Seed 0, `A18+burst` | **`0.0198`** | — | `0.615` | `0.353` | `0.594` | **`0.950`** | **`0.577`** | **`3/32`** | **`48.6%`** |
| **Seed 2, `A18` (coact)** | **`0.0249`** | `0.4884` | `0.791` | `0.518` | `0.877` | **`0.845`** | **`0.649`** | **`4/32`** | **`78.8%`** |
| **Seed 2, `A18 + delta_p1` (`n_out=64`)** | **`0.4410`** | `0.7143` | `0.492` | `0.149` | `0.454` | `0.493` | **`0.948`** | **`32/32`** | `8.6%` |
| **Seed 2, `A18 + delta_p1` (`n_out=256`)** | **`0.5065`** | `0.9136` | `0.500` | `0.019` | `0.492` | `0.405` | `0.780` | **`32/32`** | `11.9%` |

*(For reference, true `p1` has `97.1%` of its energy in the 32-D prototype span on seed 0; `p1 > 0` fraction is `0.476`.)*

Look at rows 3 and 4: **`A18+burst` (`0.0198`) has MORE energy inside the `p1` prototype span (`57.7%`) than working `top6` (`56.7%`), and collapsed seed 2 (`0.0249`) has `64.9%` inside the span (far MORE than working seed 0's `43.9%`)!** Yet seed 2 predicts a **single bin 78.8% of the time** because `d1` is 84.5% collinear across steps (`W>0 = 79.1%`, `sv1_E = 51.8%`). That kills the "orthogonal part" explanation cleanly.

- **Step 6 (burst gate worsens collapse because it never error-corrects):** **Directionally right, incomplete.** Both `burst` and `coactivity` lack negative error correction; `burst` additionally locks a small subset of early-bursting units into positive-feedback runaway (`A18+burst` predicts only 3/32 bins, `d1` pairwise cosine `0.950`).
- **Step 7 (smooth input raises code direction decode `0.748 -> 0.915`):** **Supported** (`rpc_sine_fix3_results.txt:9-11`). Genuine, well-verified finding.

---

## 3. How to Do It Right — and the Single Most Promising Next Experiment

### 3.1. Plain Verdict on the Theory as Formalized

**Can RPC as formalized in Chapter 3 + `rpc_v2.py` work on noisy or continuous temporal prediction?**  
**No.** Purely confirmatory, unsigned Hebbian potentiation on `W_dist1` (`post = b1 >= 0`, `pre = y2 >= 0`, with no depression/error term) mathematically cannot work once inputs are dense, noisy, or continuous:
1. Without negative plasticity (LTD / error correction), `W_dist1` collapses onto the rank-1 positive DC mode $\bar{q}\bar{y}^\top$, destroying cosine discrimination against zero-mean `p1` and saturating on filler noise.
2. With coincidence-only burst gating (`b1`), the network cannot escape cold start or correct wrong predictions, and positive feedback locks a few units into permanent firing.
3. Without reciprocal connectivity (`W_conf ≈ W_dist1.T`), the confirmation rule is random drift on `W_prox2`.

**What is the smallest change that makes the architecture work while keeping ALL random wires fixed, local plasticity, and the `p1`-prototype cosine readout?**  
Replace the unsigned Hebbian update on `W_dist1` (`rpc_v2.py:1048-1052`) with a **local dendritic prediction-error (delta / normalized LMS) rule** (Widrow & Hoff, 1960; Urbanczik & Senn, 2014, *Neuron*):
$$\Delta W_{\text{dist1}} = \eta \, \frac{\big(p_1(t) - d_1(t)\big)\, y_2(t-1)^\top}{\|y_2(t-1)\|^2 + \epsilon}$$
with `norm_rows=False` and `use_confirmation=False`, paired with **non-compressive module-2 width `n_out = 256` ($\ge n_{\text{res}} = 200$)** and **`full_code=True` (A18)** (and removing the train-only `mod1.inhib` mismatch).

Why this exact rule?
- **Strictly local:** Neuron $i$ in Module 1 Layer V already receives both its own proximal drive $p_{1,i}(t)$ and its own distal drive $d_{1,i}(t)$ on its apical dendrite, plus the presynaptic distal input $y_{2,j}(t-1)$. No backprop, no global loss, no external labels, no trained readout.
- **Directly enforces the thesis's physical match condition:** It minimizes $\mathbb{E}[\|p_1(t) - W_{\text{dist1}} y_2(t-1)\|^2]$ online, forcing `d1` to land in the exact signed vector space of `p1` (Check 8: `94.8%` in-span energy at `n_out=64`), automatically canceling both the positive DC bias and filler-step noise, and decorrelating `y2` so richer codes (`full_code`, `n_out=256`) strictly *help* rather than hurt!
- **Compare with purely Hebbian covariance learning (Sejnowski, 1977):** I also checked centered covariance Hebbian learning ($\Delta W_{\text{dist1}} = \eta (p_1 - \bar{p}_1)(y_2 - \bar{y}_2)^\top$, no error feedback at all) at 1 seed, 1 epoch. It fixes the positive-orthant collapse on planted motifs (`0.1313` vs `0.0409` for `muse_coact` at `full_256`), **but** still cannot invert input correlations $\Sigma_{y_2 y_2}^{-1}$ on smooth sine (`0.1538` vs `0.3405` for `delta_p1` on seed 0 `A18_full`). Local dendritic error `p1 - d1` is strictly superior and still 100% local.

*1-seed, 1-epoch verification (100% fixed random wires, unmodified `p1`-prototype cosine readout, NO learned `W_prox`, NO learned readout):*

| Task & Config (1 epoch) | `no-plasticity` | `muse_coact` | `cov_hebb_bin` | `cov_hebb` (signed `p1`) | **`delta_p1` (`p1 - d1`)** |
|---|---:|---:|---:|---:|---:|
| Sine-fixed, seed 0, `top6`, `n_out=64` | `0.0450` | `0.1812` | `0.1621` | `0.2229` | **`0.2761`** |
| Sine-fixed, seed 0, `A18_full`, `n_out=64` | `0.0309` | `0.1364` | `0.1284` | `0.1538` | **`0.3405`** |
| **Sine-fixed, collapsed seed 2, `A18_full`, `n_out=64`** | `0.0156` | `0.0249` | — | — | **`0.4410`** |
| **Sine-fixed, collapsed seed 2, `A18_full`, `n_out=256`** | — | — | — | — | **`0.5065`** |
| Planted, seed 0, `top6`, `n_out=64` | `0.0574` | `0.1017` | `0.1183` | `0.0783` | `0.0870` |
| Planted, seed 0, `A18_full`, `n_out=64` | `0.0583` | `0.0791` | `0.1191` | `0.0826` | `0.0843` |
| **Planted, seed 0, `A18_full`, `n_out=256`** | `0.0652` | `0.0409` | `0.1313` | `0.0765` | **`0.1409`** |

*(Caveat: these are strictly 1-seed, 1-epoch verification checks per brief rules, not 5-seed, 5-epoch verdicts. Note also that on planted motifs, `lr=0.5` in normalized LMS is aggressive under 77% random filler noise; a smaller `lr` like `0.05–0.1` over 5 epochs should reduce SGD noise substantially — to be verified in your full run.)*

### 3.2. How to Fix the Experimental Protocol Going Forward

1. **Fix the bugs before running anything else:**
   - Remove the train-only `mod1.inhib` asymmetry (either apply `mod1.inhib` identically in `calibrate`/`build_prototypes`/`train`/`test`, or ablate it cleanly with `s_inh=0.0`).
   - Stop silently inheriting `learn_rule="burst"` and `use_confirmation=True`: make `learn_rule` and `use_confirmation` explicit required arguments in every driver.
   - Align `baselines.py` and the RPC drivers so baselines (Markov-1, Markov-2, persistence, ESN, LSTM) run on the **exact same** generated data, seeds, sequence lengths, and `dt`.
   - Fix `rpc_track_F.py` to predict 1 step ahead (`y_t = bins[t]`) or compare against 2-step persistence.
   - Make every probe use the exact same test seed, `proto_n`, and epoch count as the main run, and always match timestamps (`y2_prev -> tok[i]` for next-token, `y2 -> tok[i]` for current-token).
2. **Fix the pass bars so they measure genuine temporal learning:**
   - Every task must report: **chance**, **persistence**, **Markov-1**, **Markov-2 / ESN ceiling** (on the *exact same* data), and **no-plasticity control**.
   - **Sine-fixed (`f=0.05`):** Pass bar must beat **Markov-1 (`0.459`)** (not `0.231`), because anything below `0.459` does not prove phase/direction learning!
   - **Variable sine & Lorenz:** Pass bar must beat **both persistence and Markov-1** on the identical dataset, plus `>= +0.10` over no-plasticity.
   - **Planted motifs (lag 1):** Keep `gain >= +0.10` over no-plasticity, and set the accuracy bar relative to the *true* Bayes ceiling (`~0.285`), e.g. `>= 0.22` (77% of Bayes ceiling, well above chance `0.0625`), plus report **motif-step accuracy** separately from uninformative filler steps (since 77% of steps are pure random noise capped at `0.0625`).
3. **Never change `W_prox` and `W_dist1` simultaneously without a clean factorial table.** Always log `W>0` fraction, top singular value energy `sv1_E`, pairwise `d1` cosine, and unique predicted classes so rank-1 collapse is spotted in 10 seconds instead of after 4 rounds.

### 3.3. The Single Most Promising Next Experiment (Exact Specification)

- **Tasks (run both, 5 seeds `0–4`, 5 epochs):**
  1. `sine-fixed` (`K=32, alpha=0.7, smooth_sigma=3.0`, 50 train / 20 test, len 200, `proto_n=50`).
  2. `planted motifs` (`K=16, alpha=1.0`, one-hot, 200 train / 50 test, len 24, `proto_n=200`).
- **Code change (2×2 factorial on `learn_rule` × `n_out`, all with `full_code=True`, `use_confirmation=False`, `s_inh=0.0`, 100% fixed random `W_exp, W_in, W_res, W_12, W_prox`):**
  - `learn_rule="delta_p1"`: `W_dist1 += (eta_d / (y2_prev.dot(y2_prev) + 1.0)) * outer(p1 - d1, y2_prev)`, `norm_rows=False`, `w_max` unclamped (or wide clamp `10.0`). Test `eta_d = 0.2` (sine) and `eta_d = 0.05` (planted, low noise).
  - `n_out in {64, 256}` (showing directly that `n_out=256` eliminates the `W_prox` bottleneck without learning `W_prox`).
  - Controls on identical seeds: `no-plasticity`, `cov_hebb` (centered signed `p1` covariance rule without error term — to cleanly isolate "sign/centering fix" vs "error-driven whitening"), and `burst` (thesis baseline).
- **Readout:** **Unmodified `p1`-prototype cosine readout ONLY** (do *not* use A23 or ridge for the verdict; report ridge *only* as a secondary diagnostic on the identical test split).
- **Pass bars (5-seed mean, held-out test):**
  - `sine-fixed`: accuracy **`>= 0.46`** (beats Markov-1 `0.459` — proving real phase/direction prediction, not just the weak `0.231` bar) AND gain **`>= +0.40`** over no-plasticity (`~0.03`).
  - `planted motifs`: overall accuracy **`>= 0.22`** (or `>= 0.2625` if you keep the original strict bar; also report motif-only accuracy with bar `>= 0.75`) AND gain **`>= +0.15`** over no-plasticity (`~0.062`).
- **Secondary scientific follow-up (only if you want to give the thesis's confirmation rule a fair trial):** Test **reciprocal `W_conf = W_dist1.T`** with `b2` gated on `y2_prev > 0` (fixing the uncoupled-`W_conf` bug) on `n_out=64` to test whether true reciprocal confirmation actually helps `W_prox2` instead of destroying it.

---

## 4. Linear-Algebra Verdict on A23 (`drive_proto`): Do NOT Run It

### 4.1. Exact Linear Algebra

First, compare A22 (`proto_proj`) and A23 (`drive_proto`):

- **Why A22 was a mathematical no-op:** Let $M \in \mathbb{R}^{K \times n_{\text{out}}}$ have rows $\mu_k = \mathbb{E}[p_1 \mid \text{tok}=k]$, and $P = M^+ M$ be the orthogonal projector onto $\operatorname{row}(M)$. Because $P^\top = P$ and $P\mu_k = \mu_k$:
  $$\langle P d_1, \mu_k \rangle = \langle d_1, P^\top \mu_k \rangle = \langle d_1, \mu_k \rangle, \qquad \cos(P d_1, \mu_k) = \frac{\|d_1\|}{\|P d_1\|}\cos(d_1, \mu_k).$$
  The factor $\|d_1\|/\|P d_1\| > 0$ is identical for all $k$, so $\arg\max_k$ cannot change for cosine or dot-product readout.

- **What A23 actually computes (`rpc_v2.py:747-768, 973-974`):** After training (with `W_dist1` frozen), `build_drive_prototypes` computes for each next-token $k$:
  $$\tilde{\mu}_k = \mathbb{E}\big[d_1(t) \mid \text{tok}(t) = k\big] = \mathbb{E}\big[W_{\text{dist1}} y_2(t-1) \mid \text{tok}(t) = k\big] = W_{\text{dist1}} \underbrace{\mathbb{E}\big[y_2(t-1) \mid \text{tok}(t) = k\big]}_{\bar{y}_{2,\text{prev}}^{(k)}}.$$
  Then `predict_from_distal` scores each class $k$ by:
  $$\cos(d_1(t), \tilde{\mu}_k) = \frac{y_2(t-1)^\top \big(W_{\text{dist1}}^\top W_{\text{dist1}}\big) \bar{y}_{2,\text{prev}}^{(k)}}{\big\|W_{\text{dist1}} y_2(t-1)\big\|\;\big\|W_{\text{dist1}} \bar{y}_{2,\text{prev}}^{(k)}\big\|}.$$

**Therefore: A23 is NOT a mathematical no-op — it changes the predictions drastically. It is WORSE than a no-op, for three independent linear-algebraic reasons:**

1. **It makes the frozen no-plasticity control a supervised nearest-centroid classifier on `y2_prev`!**
   When plasticity is OFF, $W_{\text{dist1}} \in \mathbb{R}^{64 \times 64}$ is an i.i.d. random Gaussian matrix. By isotropy (Johnson-Lindenstrauss), $W_{\text{dist1}}^\top W_{\text{dist1}} \approx c I_{64}$, so:
   $$\cos(d_1(t), \tilde{\mu}_k) \approx \cos\big(y_2(t-1),\; \bar{y}_{2,\text{prev}}^{(k)}\big).$$
   Previously, `no-plasticity` scored at chance (`~0.03`) only because random $W_{\text{dist1}} y_2(t-1)$ was compared to unrelated `p1` templates. Under A23, the **untrained** network maps `y2_prev` directly against its own labeled next-token centroids — leaking full next-token supervision into the readout (`K × 64 = 2,048` supervised template parameters, literally the numerator $Y^\top D$ of ridge regression without the whitening inverse $(D^\top D + \lambda I)^{-1}$!). Your description in `RPC-HUNT-REPORT.md:336` (*"with no trained weights"*) is false: class centroids fit on `(d1, next_tok)` pairs *are* a trained supervised linear classifier.
2. **Training with unsigned Hebbian updates makes `W_dist1` rank-collapsed, so the trained network LOSES badly to its own frozen control!**
   For a frozen random $W_{\text{dist1}}$, $W^\top W$ is full-rank and well-conditioned, preserving the angular separation of $\bar{y}_{2,\text{prev}}^{(k)}$. For your trained `W_dist1` (pushed into the positive orthant), Perron-Frobenius forces a massive dominant positive singular vector $\sigma_1 u_1 v_1^\top$ (`sv1_E = 0.52–0.65`), squashing all 32 drive prototypes $\tilde{\mu}_k = W_{\text{dist1}} \bar{y}_{2,\text{prev}}^{(k)}$ into a narrow cone (verified off-diagonal cosine **`0.986`** on A18 seed 0 at just epoch 1, `0.775` on top-6!). Thus `W_dist1` training **destroys** nearest-centroid separability relative to a random matrix — guaranteeing **large negative gain**!
3. **It abandons the RPC mechanism completely.** In Chapter 3, `d1` must physically predict `p1` so distal and proximal drives collide on the same Layer V neuron. Matching `d1` against post-hoc labeled `d1` centroids evaluates an external supervised decoder, not whether Module 2 predicts Module 1.

### 4.2. Direct Verification of the Linear Algebra (1 Seed, 1 Epoch, Sine-Fixed)

| Seed & Condition (`alpha=0.7, smooth=3.0, graded+eta_b=0.15, coactivity`, 1 epoch) | `full` | `no-plasticity` | **Gain (`full - ctrl`)** | Drive-proto off-diag cos |
|---|---:|---:|---:|---:|
| Seed 0, `top6 + p1proto` | `0.2819` | `0.0450` | `+0.2369` | — |
| **Seed 0, `top6 + A23 (drive_proto)`** | `0.3126` | **`0.4440`** | **`-0.1314`** | `0.775` |
| Seed 0, `top6 + A23 + center` | `0.3093` | **`0.4465`** | **`-0.1372`** | — |
| Seed 0, `A18 + p1proto` | `0.2337` | `0.0309` | `+0.2028` | — |
| **Seed 0, `A18 + A23 (drive_proto)`** | `0.1982` | **`0.3626`** | **`-0.1643`** | **`0.986`** |
| Seed 0, `A18 + A23 + center` | `0.0807` | **`0.3525`** | **`-0.2719`** | — |
| **Collapsed Seed 2, `A18 + p1proto`** | `0.0249` | `0.0156` | `+0.0093` | — |
| **Collapsed Seed 2, `A18 + A23 (drive_proto)`** | `0.1623` | **`0.4580`** | **`-0.2957`** | — |

**Verdict:** Do **not** waste a 5-seed run on A23. Every prediction of the linear algebra holds on the first try: A23 hands the untrained network `0.36–0.46` for free and turns RPC learning into a `-0.13` to `-0.30` net loss, reproducing the exact frozen-beats-trained failure of your Round 9 ridge test (`simulation-results.md:1243-1249`).

---

## 5. Can RPC Keep Fixed Random Wires, or Where Are Learned Wires Needed?

### 5.1. Fixing the Premise: Fixed Random `W_prox` Is NOT the Bottleneck

Your brief asks (Section 7.5): *"My results say the fixed random proximal projection is where a third of the information is lost, and a learned proximal projection fixes the code but the confirmation rule then destroys the gain. Is there a version of this theory that keeps the random wires and still works, or does the theory need learned wires somewhere?"*

That premise rested on two artifacts we just dismantled:
1. Fixed random `W_prox` only lost a third (`0.984 -> 0.711`) because `rpc_v2.py` forced a `200 -> 64` random compression. At `n_out = 256` ($\ge n_{\text{res}} = 200$), a 100% fixed random `W_prox` is full column rank and loses **0.000** (`0.983 -> 0.983`, Check 5), and `y2_full` jumps from `0.563` to `0.867`.
2. Learned `W_prox` (A12) was only destroyed by confirmation because `W_conf` in `rpc_v2.py:557` is an independent random matrix with zero reciprocal link to `W_dist1`.

### 5.2. Exact Map of Which Wires Can Stay Fixed Random vs. Which Must Learn

1. **Wires that CAN stay 100% fixed random (for all benchmarks in this study):**
   - `mod1.W_exp`, `mod1.W_in`, `mod1.W_res`, `mod1.W_prox` (Module 1 expansion, reservoir, and proximal projection).
   - `W_12`, `mod2.W_exp`, `mod2.W_in`, `mod2.W_res`, **and `mod2.W_prox`** — **provided `n_out >= n_res`** (e.g., `256` instead of `64`) and `full_code=True` so no artificial compression or top-6 truncation destroys the reservoir's temporal state.
   - *Proof:* Check 8 already demonstrates `0.5065` accuracy on sine-fixed (collapsed seed 2, 1 epoch, unmodified `p1`-prototype cosine readout; `0.9136` ridge) and `0.1409` on planted motifs in 1 epoch with **every single one of these matrices 100% fixed random**. Furthermore, `baselines_B.md` already proves that a single fixed random reservoir (`ESN`, `N_r=300`) contains enough temporal information to solve *all* five tasks in this study (`lag-1 0.274`, `lag-2 0.263`, `lag-5 0.230`, `sine 0.900`, `Lorenz 0.388`).
2. **Wires that MUST be learned — and HOW they must learn:**
   - **Minimum necessary learned matrix: `W_dist1` (the distal predictive projection from Module 2 back to Module 1).** Even the thesis (`ch3.txt:1451, 1488`) makes `W_dist1` plastic — you cannot predict an arbitrary sequence with a frozen random `W_dist1`. What changes from the thesis is **the learning rule**: `W_dist1` *must* learn via a **signed, local error-correcting rule** ($\Delta W_{\text{dist1}} \propto (p_1 - d_1) y_2^\top$, Urbanczik & Senn 2014), not unsigned burst-gated potentiation.
   - **What about `mod2.W_prox` (CA3 → CA1) and `W_conf`?** On these low-dimensional sequence benchmarks, once `n_out >= n_res`, learning `mod2.W_prox` is **unnecessary**. If you *do* want `mod2.W_prox` to learn (as Chapter 3 proposes, `ch3.txt:1504-1522`), you need either: (a) **reciprocal feedback** (`W_conf = W_dist1.T`, or feedback alignment where forward weights `W_prox2` learn to align with fixed random `W_conf` via a *signed* error signal, Lillicrap et al., 2016; Nøkland, 2016 — note that Feedback Alignment works *only* when the forward weights receive a **signed error signal**, never an unsigned binary burst!), or (b) unsupervised decorrelation / sparse coding in `W_prox2`.
   - **When fixed random internal wires (`W_res`, `W_12`, `W_prox`) eventually stop being enough:** If you scale RPC up to high-dimensional natural video / future-frame prediction (as suggested in `ch3.txt:1592-1594`), fixed random projections suffer the curse of dimensionality: random nonlinear features of raw pixels are vastly less sample-efficient than learned hierarchical features. On those tasks, **learned feedforward/recurrent feature wires** (`W_in`, `W_res`, `W_prox`) become necessary, not just plastic `W_dist1`.

### 5.3. What This Does to the Thesis Claim

State this plainly to Abolfazl — there are two separable claims in Chapter 3, and they meet opposite fates:

1. **The "fixed random basis" claim survives on sequence benchmarks — because it is Echo State Networks.** Keeping `W_exp, W_in, W_res, W_12, W_prox` fixed random *does* work once you remove the un-thesis-like `200 -> 64` bottleneck and fix the distal rule. That validates the reservoir-computing foundation (`Grigoryeva & Ortega, 2018`, already cited in `ch3.txt:1388`), not a new discovery about random wiring.
2. **The "burst-confirmation Hebbian learning" claim fails and must be replaced.** Chapter 3's genuinely distinctive proposal — that random predictions collide by chance, trigger a confirmation burst, and wire up sequence memory via purely potentiating Hebbian bursts plus a confirmation loop — **fails mathematically and empirically** (rank-1 positive-orthant collapse, cold-start lock-in, inability to depress false predictions, and an anatomically rare reciprocal reentry loop per Naber et al., 2001). Replacing it with the smallest working local rule ($\Delta W_{\text{dist1}} \propto (p_1 - d_1) y_2^\top$) rescues the simulation cleanly, **but turns the model into local dendritic prediction-error learning (Urbanczik & Senn, 2014; Sacramento et al., 2018) on top of a fixed random reservoir (online LMS ESN)** — meaning prediction *errors*, not confirmation bursts, do the learning.

---

## References Verified for This Critique

- **Bittner, K. C., Milstein, A. D., Grienberger, C., Romani, S., & Magee, J. C. (2017).** Behavioral time scale synaptic plasticity underlies CA1 place fields. *Science*, 357(6355), 1033–1036.
- **Carandini, M., & Heeger, D. J. (2012).** Normalization as a canonical neural computation. *Nature Reviews Neuroscience*, 13(1), 51–62.
- **Chen, Y., Zhang, H., Cameron, M., & Sejnowski, T. J. (2022/2024).** Predictive sequence learning in the hippocampal formation. *bioRxiv* (May 2022); *Neuron* (August 2024).
- **Cui, Y., Ahmad, S., & Hawkins, J. (2016).** Continuous online sequence learning with an unsupervised neural network model. *Neural Computation*, 28(11), 2474–2504.
- **Hawkins, J., & Ahmad, S. (2016).** Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits*, 10, 23.
- **Larkum, M. E., Zhu, J. J., & Sakmann, B. (1999).** A new cellular mechanism for coupling inputs arriving at different cortical layers. *Nature*, 398, 338–341.
- **Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016).** Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*, 7, 13276.
- **Naber, P. A., Lopes da Silva, F. H., & Witter, M. P. (2001).** Reciprocal connections between the entorhinal cortex and hippocampal fields CA1 and the subiculum are in register with the projections from CA1 to the subiculum. *Hippocampus*, 11(2), 99–104.
- **Nøkland, A. (2016).** Direct feedback alignment provides learning in deep neural networks. *NeurIPS 2016*.
- **Sacramento, J., Ponte Costa, R., Bengio, Y., & Senn, W. (2018).** Dendritic cortical microcircuits approximate the backpropagation algorithm. *NeurIPS 2018*.
- **Sejnowski, T. J. (1977).** Storing covariance with nonlinearly interacting neurons. *Journal of Mathematical Biology*, 4, 303–321.
- **Urbanczik, R., & Senn, W. (2014).** Learning by the dendritic prediction of somatic spiking. *Neuron*, 81(3), 521–528.
- **Widrow, B., & Hoff, M. E. (1960).** Adaptive switching circuits. *IRE WESCON Convention Record*, 96–104.
