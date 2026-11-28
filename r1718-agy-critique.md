# Independent Scientific Critique: Rounds 17 & 18 (Top-Down Clamp & Continuous Wall)

**Reviewer:** AGY (Antigravity)  
**Date:** Nov 27, 2026  
**Scope:** Full audit of `simulation-results.md` (sections 44–54), `PROTOCOL-R17-CLAMP.md`, `PROTOCOL-R18-CONTINUOUS.md`, `PROTOCOL-R19-HIERARCHY.md`, `rpc_v2.py` (A1–A27, focusing on A26–A27), `rpc_round17_clamp.py`, `rpc_round18_continuous.py`, `baselines.py`, `rpc_spatial.py`, `ch3.txt`, `r17/` (`monitor.py`, `r17_analyze.py`, `winner.json`, `r17_analysis.md`, all 200 test JSONL lines), `r18/` (`monitor.py`, `r18_analyze.py`, `winner.json`, `r18_analysis.md`, all 360 test JSONL lines), and prior critiques (`r14-kimi-critique.md`, `r14-agy-critique.md`).  
**Missing pack files noted upfront:** Five items imported by the drivers or cited in the brief were **not synced** into `/home/reza/zfs-1tb/rpc-sim/review/`: (1) `rpc_round15.py` (imported at `rpc_round18_continuous.py:58` and required for Check 6) and (2) `rpc_round16_omission.py` (imported at `rpc_round17_clamp.py:47`) — both were retrieved directly from the public repository at the prompt-specified commit `b2543c73` into an isolated scratch directory so the review pack stayed unmodified; (3) `r17/r17_planted1_val_*.jsonl` (20 validation lines read by `r17/r17_analyze.py:70`) and (4) `r18/r18_*_val_*.jsonl` (30 validation lines read by `r18/monitor.py:94`) — absent from disk, though `winner.json` and our direct 5-seed re-runs reproduce the R17 validation means to 3 decimals; (5) `rpc_hier.py` / `rpc_round19_hierarchy.py` (Check 9 audits `PROTOCOL-R19-HIERARCHY.md` and `simulation-results.md` section 54).  
**Verification method:** Zero existing files in `/home/reza/zfs-1tb/rpc-sim/review/` were modified. All 560 JSONL records were verified for integrity (0 duplicate keys, 0 NaNs, `clamp_frac == 0.0` everywhere) and re-analyzed with both normal `z = 1.96` and exact Student-$t_{19}$ (`2.0930`) / Wilcoxon signed-rank tests. Targeted read-only mechanistic diagnostics were executed with `/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python` (single-threaded CPU, `torch 2.15.0+cpu`, `numpy 2.5.4`). Every number below is explicitly labeled as either quoted from the record/JSONL or measured in a named diagnostic.

---

## Lead Verdict

### 1. Does Experiment B's failure (Round 17) stand?

**YES — Experiment B's failure stands decisively for the tested mechanism (fixed random additive `W_fb`), and the raw data are even harsher than section 53 reports.**  
All four primary lag-1 bars fail (`C1` `+0.005` vs `>= 0.10`, exact Student-$t_{19}$ $p = 0.059$; `C2` `-0.001`, $p = 0.85$; `C3` `p1_cos = 0.9978` vs `< 0.90`; `C4` `+0.004`, $p = 0.084$), and the lone lag-2 pass (`C1-lag2`, `+0.004`) is a bar-definition artifact (`r17_analyze.py:185` silently omitted the `0.10` floor). Worse, three unhighlighted metrics move **against** predictive coding: enabling the wire significantly *increases* final training error `err_mag_last` on `19/20` seeds on both lag-1 (`+0.00203`, $p < 10^{-7}$) and lag-2 (`+0.00188`, $p < 10^{-8}$), significantly *degrades* lag-1 `omit_m1` (`-0.0157` vs `wire-off`, $p = 0.0069$), and significantly *reduces* `cos_p1_omit_m2` during omissions ($p = 0.035$ lag-1, $p = 0.047$ lag-2). Crucially, our 5-seed gain sweep (`fb_gain ∈ {0.1, 0.3, 1.0, 2.0}`) proves validation's `0.1` pick did **not** hide a working gain: at `0.3`, `1.0`, and `2.0`, cutting the wire **improves** `omit_m2` by `+0.018`, `+0.111`, and `+0.180`, respectively.

**Single strongest reason to doubt the broader conclusion ("top-down feedback into module 1 carries no causal role without learned feedback weights"):** **Round 17 implemented a *fixed random additive* wire (`+W_fb @ d1`) at a Pythagorean-negligible drive ratio (`fb/drv = 3.9%`), instead of the *aligned subtractive* feedback (`-W_prox.T @ d1`) actually specified by Rao & Ballard (1999) and Chapter 3 (`ch3.txt:1540-1560`).** At `g = 0.1`, `p1_cos = 0.998` is guaranteed by elementary geometry ($1/\sqrt{1 + 0.066^2} = 0.9978$), making C3 (`< 0.90`) impossible. At `g = 1.0`, random `+W_fb` injects pure orthogonal interference (`omit_m2` drops to `0.769`, cut rescues it to `0.879`), and positive aligned `+W_prox.T` triggers self-reinforcing runaway collapse (`omit_m2 = 0.000`). Yet on seed `241`, merely replacing random `W_fb` with **fixed norm-matched subtractive `-W_prox.T`** (zero extra learned weights!) at `g = 1.0` stays completely stable, **improves `normal_acc` (`0.2330` vs `0.2217` wire-off)**, reaches **`omit_m2 = 1.0000`** (`+0.0342` over `cut`), **clears C3 (`p1_cos = 0.8565 < 0.90`)**, and at `g = 0.3` lowers `err_mag_last` (`1.604` vs `1.617`). Round 17 falsified random additive feedback — not aligned inhibitory feedback.

---

### 2. Does the continuous wall (Round 18) stand?

**NO — Round 18's headline verdict ("the continuous wall survives correct scoring; delta adds no continuous information; the best continuous configuration is bump input plus ridge readout of the frozen reservoir") is REFUTED on three independent grounds.**

**Single strongest reason to doubt the continuous wall:** **Round 18 froze hyperparameters (`ROUND15_WINNER`, `rpc_round18_continuous.py:70-73`) selected under Round 15's *buggy two-step-ahead scoring* AND repeated Round 14's exact unstandardized `lam = 1.0` ridge-scale confound in D8:**

1. **Both D3 FAIL (`varsine C1 MSE 0.064 > 0.056`) and D4 FAIL (`Lorenz C1 MSE 22.39 > 12.61`) dissolve immediately under valid Convention-A hyperparameters already inside Round 15's own 18-config discrete-bin grid — with ZERO A27 continuous bump and ZERO ridge readout:**
   - **Lorenz (`5 seeds 262..266`, unmodified `p1`-prototype readout, discrete bin input only):** Switching from Round 15's Convention-B carryover (`a=0.7, smooth_sigma=0.0, eta_d=0.2`: `acc = 0.317, MSE = 26.61`) to **`a=1.0, smooth_sigma=2.0, eta_d=0.2`** hits **`acc = 0.4285`** (beating ESN `0.3970`), **`within1 = 0.8393`** (beating ESN `0.7946`), and **`MSE = 5.07`** — crushing persistence (`13.05`) by `2.6×` on **5/5 seeds** (`[5.68, 4.62, 3.75, 5.83, 5.48]`) and beating Round 18's own A27 `C3` (`7.73`)! Furthermore, at `a=0.7`, discrete bin-smoothing (`A14`, `MSE = 8.43`) accounts for **96.3%** of Round 18's D6 drop (`26.61 -> 7.73`); true continuous within-bin input (`A27`) adds only `3.7%` (`-0.70`).
   - **Var-sine (`5 seeds 262..266`, unmodified `p1`-prototype readout, discrete bin input only):** Switching from Round 15's Convention-B carryover (`a=1.0, s=2.0, eta_d=0.02`: `acc = 0.294, MSE = 0.0678`) to **`a=0.7, s=2.0, eta_d=0.2`** hits **`acc = 0.3230`** (`+0.040` above Markov-1 `0.2826`) and **`MSE = 0.0370`** — beating persistence (`0.0548`) on **5/5 seeds** (`-32%` MSE) and ESN (`0.0483`) on 4/5 seeds (`a=0.7, s=3.0, eta_d=0.2` hits **`MSE = 0.0317`**, beating both persistence and ESN on **5/5 seeds**).
2. **D8 (`C2f < C2`, `C4f < C4`) is 100% the unstandardized `lam = 1.0` norm-shrinkage confound + a linear Data Processing Inequality (DPI) tautology:** Delta shrinks centered drive norm `||dc||` by **`4.96×`** (Lorenz one-hot) and **`3.91×`** (Lorenz bump), making `rpc_round18_continuous.py:244`'s unstandardized `lam = 1.0` penalize trained `d1` **`24.6×` and `15.3×` harder** than frozen `d1`. Once scale-matched (`[rms]` or `[std]` at `lam = 1.0`), **trained ridge BEATS frozen ridge** (`12.52` vs `13.33` one-hot; `3.97` vs `4.36` bump), and at `lam = 1e-6`, trained `d1`, frozen `d1`, and `y2_prev` are algebraically identical to 5 significant digits (`11.9528`, `3.4855`) because both `W_dist1` matrices are full-rank invertible maps of the identical frozen `y2_prev`.
3. **Even within Round 18's own data, "frozen ridge is the best continuous configuration" is false on var-sine, and "C4 beats ESN" is an apples-to-oranges baseline:** On var-sine, learned prototype readout **`C3_bump_delta_proto` (`0.039`)** beats persistence (`0.056`, $p = 0.00029$), beats ESN (`0.046`, `15/20`), and beats **both** `C4` (`0.044`, `15/20`) and frozen ridge `C4f` (`0.045`, `15/20`, Wilcoxon $p = 0.050$) — obscured by a bug in `r18_analyze.py:288-291` that mislabeled `D7-var` while omitting `C3 - C4`. Conversely, giving the exact same `baselines.py` 300-unit ESN a continuous scalar ridge readout (matching `C2/C4`) drops ESN MSE to **`3.287` (one-hot) / `2.097` (bump)** on Lorenz and **`0.0191` / `0.0072`** on var-sine — crushing `C4` (`3.795` / `0.044`) and `C4f` (`3.615` / `0.045`).

What **does** survive of the continuous wall is much narrower: on discrete-bin or bump inputs, the unmodified 32-prototype delta model still trails Markov-2 (`1.416` / `0.0053`) and a matched continuous-ridge ESN (`2.097` / `0.0072`), and because `mod1`/`mod2` are frozen, `W_dist1` can never create linear information absent from `y2_prev`.

---

### Summary Table of Checks (1–10)

| # | Check | Verdict | Key Finding |
|---|---|---|---|
| 1 | R17 wire live & `p1_cos = 0.998` | **Verified (live) / Refuted (mysterious or testable at `g=0.1`)** | No code bug washes out `fb`. `fb/drv = 3.9%` (`rec = 2.87 > bu = 2.11`), `dp1_rel = 0.066` $\implies$ Pythagorean $\cos = 0.9978$ exactly; C3 (`<0.90`) requires `dp1_rel > 0.484` (`g >= 1.0`). Also `build_prototypes` runs with `fb=None`, `d1` shrinks `3.5×`, and `d1(t)` is read out *before* `fb(t)` enters `mod1`. Fixed `-W_prox.T` at `g=1.0` clears C3 (`0.8565`), raises `normal_acc`, and hits `omit_m2 = 1.000`. |
| 2 | R17 lag-2 C1 pass (`+0.004`) | **Refuted (as meaningful pass) / Verified (bar artifact)** | `r17_analyze.py:185` silently dropped the `>= 0.10` floor from `PROTOCOL-R17-CLAMP.md:156`. Fails `0.10` by `24×`; `8/20` seeds are exact ties (`W/L/T = 9/3/8`); held-out 15 seeds $p = 0.053$; `on - shuf` and `on - off` are null. Does not change Experiment B's fail verdict. |
| 3 | R17 winner-pick bias (`g = 0.1`) | **Verified (selection bias toward `0.1`) / Refuted (hid a passing gain)** | `val_acc` decreases monotonically with random `fb_gain` (`0.227 -> 0.226 -> 0.221 -> 0.182`), guaranteeing `0.1` wins. However, 5-seed sweeps at `g = 0.3, 1.0, 2.0` show cutting the wire *improves* `omit_m2` (`-0.018`, `-0.111`, `-0.180`), so picking on `omit_m2` or `on - cut` still picks `0.1`. |
| 4 | R18 D2/D3 borderline & 40-seed rerun | **Refuted (D2 pass & var-sine MSE wall) / Verified (H0 fails for frozen C1)** | `r18_analyze.py:67` uses `z=1.96` instead of `t_19=2.093`; under exact $t_{19}$, **D2b FAILS** (`[-0.0004, +0.0220]`, $p = 0.0579$). On D3, frozen C1 *loses* to persistence on `13/20` seeds (`-0.0083`); a 40-seed rerun would confirm C1 loses (`[-0.0156, -0.0009]`). Yet `C3` (`0.039`, $p = 0.00029$) and Convention-A `a=0.7, s=2.0, eta=0.2` (`0.0370`, 5/5) crush persistence. |
| 5 | R18 D8 (`C2f < C2`, `C4f < C4`) | **Verified (code parity) / Refuted ("delta harms ridge" & "adds nothing")** | Same data & `lam=1.0`, repeating R14's exact scale confound: delta shrinks `||dc||` `4.96×` (one-hot) and `3.91×` (bump), penalizing trained `24.6×` / `15.3×` harder. Scale-matched `[rms]`/`[std]` flips **trained > frozen**; `lam=1e-6` gives exact algebraic equality (`11.9528`, `3.4855`) by invertible-map DPI. |
| 6 | R18 scoring correction audit | **Verified (no remaining 1-step shift) / Refuted (blame on `baselines.py` & ESN parity)** | `collect()` matches `run()` and all baselines on steps `i = 1..L-1`. Bug was solely `rpc_round15.py:211-223` (`run_cont`, which also scored zero-state `pred(0)` against `seq[1]`), NOT `baselines.py:322-330`'s docstring. Remaining asymmetries: `C2/C4` scalar ridge vs `ESN` class-argmax MSE, continuous vs quantized `persistence_mse`, `D7-var` analyzer bug, and 5/20 val-test seed overlap. |
| 7 | Markov-2 dominance & task choice | **Verified (tasks trivial for 2-gram lookup) / Partially Refuted (D4 uninformative)** | `99.50%` (Lorenz) and `99.82%` (var-sine) of test 2-grams appear in train; $(b_{t-2}, b_{t-1})$ explains $R^2 = 0.9892$ (Lorenz) and $0.9954$ (var-sine) of $x_t$ variance. D4 is still informative against persistence (and passes once properly validated); D8 is uninformative due to DPI/scale confound. Use NARMA-10 or Mackey-Glass ($\tau=17, h=5$–$10$). |
| 8 | Mapping tension (A-pass + B-fail) | **Refuted (strengthens PC) / Verified (reduces to ESN + NLMS)** | A-pass is a trivial 1-step readout property shared by any ESN + NLMS; B-fail removes top-down causality (`cos_p1_omit` drops, `err_mag` rises). Only fixed random reservoir + local NLMS alignment to internal `p1`-prototypes survives. |
| 9 | R19 hierarchy interaction | **Verified (repeats R17 null by construction)** | R19 reuses random additive `W_fb_l` at `g ∈ {0.1, 0.3}` validated on overall acc (`83%` filler). Worse, `mod3` reaches `d1` *only* through `fb_2`, bottlenecking both `H8` and `H4`, while L1 already decodes lag-11 context at `0.70`, compressing `H2`. |
| 10 | Next experiment (C vs D) | **Unresolved (D + fixed B settles more than C)** | Precision (C) collapses to adaptive learning-rate scheduling without a causal top-down wire. Fix feedback geometry first (`-W_prox.T` + omission substitution), then run closed-loop generative rollout (D) + revised B, alongside a clean Convention-A re-validation of Lorenz/var-sine + NARMA-10/Mackey-Glass. |

---

## Detailed Audit of Checks 1–10

### Check 1: Is Round 17's wire really live, and why is `p1_cos = 0.998`?

**Verdict:** **Verified** that the wire is live with no normalization/threshold bug washing it out; **Refuted** that `p1_cos = 0.998` is surprising or that Experiment B's C3 bar (`< 0.90`) was reachable at `fb_gain = 0.1`.  
**References:** `rpc_v2.py:554-574, 697-700, 770-872, 1122-1156`; `rpc_round17_clamp.py:89-114, 130-145, 211-244`; `r17/r17_analysis.md:20, 40-44`; `simulation-results.md:2988-2993`.

1. **Exact data flow in `rpc_v2.py` (wire IS live):**
   - `W_fb` is initialized as `randn(200, 256) / sqrt(256)` (`rpc_v2.py:698-699`), drawn after all other buffers so `fb_gain = 0.0` reproduces Round 16 identically.
   - In `_step_core` (`rpc_v2.py:1127-1141`), `d1 = self._distal_drive()`, then `fb = self.fb_gain * (self.W_fb @ d1)` (`"on"`), `fb = self.fb_gain * (self.W_fb @ d1[self.fb_perm])` (`"shuffle"`), or `fb = None` (`"cut"`).
   - In `ReservoirModule.forward` (`rpc_v2.py:564-573`):
     ```python
     bu = self.W_in @ e_sparse
     drive = bu + self.W_res @ self.r - self.inhib
     if fb is not None:
         drive = drive + fb
     r_new = (1 - self.alpha) * self.r + self.alpha * torch.tanh(drive)
     p = self.W_prox @ self.r
     ```
   - `fb` enters **before** `tanh` and **before** the leak (here `alpha = 1.0`, so `r_new = tanh(drive)`), with **no** normalization, clipping, or threshold applied to `r` or `p1`. There is zero implementation bug suppressing `fb`.

2. **Why `fb_rms_ratio = 0.065` produces `p1_cos = 0.998` (exact algebra + 5 structural factors):**
   - **Factor A — `fb_rms_ratio` omits the dominant recurrent drive `W_res @ r`:** `rpc_v2.py:565-566, 1286-1288` computes `||fb|| / ||bu||`, ignoring `W_res @ r`! Measuring exact internal norms on `planted1` seed `241` (`diag_r17_wire.py`, epoch 5, `g = 0.1`): `||bu|| = 2.111`, **`||W_res @ r|| = 2.871`**, total pre-activation drive `||bu + W_res @ r|| = 3.557`, and `||fb|| = 0.139`. Thus `||fb|| / ||drive||` is only **`0.039` (`3.9%`)**, not `6.5%` (`tanh'` is `0.943`, so `tanh` saturation is *not* washing it out).
   - **Factor B — Pythagorean cosine geometry ($\cos \theta \approx 1 - \frac{1}{2}\varepsilon^2$):** Because `W_fb` is random and orthogonal to `drive` (`cos ≈ 0`), a perturbation $\Delta p_1$ of relative L2 norm $\varepsilon = \|\Delta p_1\| / \|p_1\|$ yields $\cos(p_1, p_1 + \Delta p_1) \approx (1 + \varepsilon^2)^{-1/2}$. On seed `241` (`g = 0.1`), measured $\varepsilon = 0.063$ (5-seed mean `0.066`), predicting $(1 + 0.063^2)^{-1/2} = \mathbf{0.9980}$ against the exact measured **`p1_cos = 0.9980`**! To clear **Bar C3 (`p1_cos < 0.90`)**, an orthogonal perturbation MUST satisfy:
     $$\varepsilon = \frac{\|\Delta p_1\|}{\|p_1\|} > \sqrt{0.90^{-2} - 1} = \mathbf{0.4843}$$
     Passing C3 at `g = 0.1` (`ε = 0.066`) is **mathematically impossible by a factor of `7.3×` in norm (`54×` in energy)**. Even on a **blank omission step** (`bu = 0`), `W_res @ r` remains `~2.87`, keeping `p1_cos_blank = 0.9965`!
   - **Factor C — `3.5×` norm shrinkage of `d1` during training:** In `r17/r17_analysis.md:20, 40`, `frozen fb_rms_ratio = 0.229` (`||d1|| = 5.621`) while trained `wire-on fb_rms_ratio = 0.065` (`||d1|| = 1.632`). Because `W_dist1` starts `3.5×` larger than `p1`, high gains like `g = 1.0` inject **`228%` of bottom-up drive at epoch 0 (`fb/drv = 0.897`)**, wrecking early learning before delta shrinks `d1` down to `fb/drv = 0.395` at epoch 5!
   - **Factor D — `build_prototypes` runs with `fb = None`:** `rpc_v2.py:846` calls `self.mod1.forward(x)` without `fb`, so prototypes are frozen in the wire-off coordinate system while large `g` drags `p1` away from them.
   - **Factor E — Causal timing mismatch on `omit_m2`:** In `_step_core` (`rpc_v2.py:1127-1156`), `d1(t)` and `pred(t)` are computed **before** `fb(t)` enters `mod1`! So `omit_m2` at step `t` never tests whether `fb(t)` injects a top-down template during the omission; it tests only whether past `fb(t-1)` improved `r1(t-1) -> y2(t-1)`, on a task where bottom-up input alone already gives `omit_m2 = 0.948`. Even on `cont_m1` (where `fb(t)` *does* enter during the blank step `t` before `t+1` is read out), random `W_fb` cannot map `d1(t)` into `r1(t)`'s `m1` subspace (`wire-on 0.581` vs `wire-cut 0.587`, $p = 0.29$).

3. **Empirical gain sweep + aligned-feedback control (`diag_r17_wire.py`, `planted1`, 5 epochs):**

| Condition (`planted1`, `n=5` seeds `241..245` unless noted) | `val_acc` | `fb/bu` (ep0$\to$ep5) | `fb/drv` (ep5) | `dp1_rel` | `p1_cos` | `omit_m2` (`on / cut / shuf`) | `on - cut` | `err_last` | `cosp1_omit` (`on/cut`) |
|---|---:|---|---:|---:|---:|---|---:|---:|---|
| Random `W_fb`, `g = 0.1` (R17 winner) | `0.2261` | `0.227 -> 0.067` | `0.039` | `0.066` | `0.9978` | `0.950 / 0.946 / 0.948` | `+0.0033` | `1.6203` | `0.397 / 0.398` |
| Random `W_fb`, `g = 0.3` | `0.2209` | `0.682 -> 0.202` | `0.117` | `0.199` | `0.9805` | `0.936 / 0.954 / 0.945` | **`-0.0182`** | `1.6325` | `0.386 / 0.398` |
| Random `W_fb`, `g = 1.0` | `0.1816` | `2.281 -> 0.756` | `0.397` | `0.699` | **`0.8042`** | `0.769 / 0.879 / 0.735` | **`-0.1109`** | `1.7544` | `0.302 / 0.398` |
| Random `W_fb`, `g = 2.0` | `0.0850` | `4.563 -> 3.085` | `1.011` | `2.047` | `0.2986` | `0.140 / 0.321 / 0.090` | **`-0.1804`** | `1.7913` | `0.097 / 0.398` |
| *Seed 241:* Aligned **`+W_prox.T`**, `g = 0.3` | *(norm `0.158`)* | — | — | — | `0.8493` | `0.752 / 0.940 / 0.838` | `-0.1880` | `1.6870` | `0.407 / 0.401` |
| *Seed 241:* Aligned **`+W_prox.T`**, `g = 1.0` | *(norm `0.045`)* | — | — | — | `0.0688` | `0.000 / 0.000 / 0.000` | `0.0000` | `0.8161` | `0.094 / 0.401` |
| *Seed 241:* Aligned **`-W_prox.T`**, `g = 0.3` | *(norm **`0.232`**)* | — | — | — | `0.9667` | `0.957 / 0.957 / 0.974` | `0.0000` | **`1.6040`** | `0.271 / 0.401` |
| *Seed 241:* Aligned **`-W_prox.T`**, `g = 1.0` | *(norm **`0.233`**)* | — | — | — | **`0.8565`** | **`1.000 / 0.966 / 0.974`** | **`+0.0342`** | `1.6162` | `0.036 / 0.401` |

**Gain and code change required to make the test real:**
- **Gain:** Clearing `p1_cos < 0.90` requires `fb_gain ≈ 0.8–1.0` (after scaling `W_dist1` init down `3.5×` to match `p1`'s norm so epoch 0 does not blow up).
- **Code change:** Replace fixed random `W_fb` with **aligned inhibitory feedback `W_fb = -mod1.W_prox.T`** (norm-matched, i.e., Rao & Ballard's / Chapter 3's Layer V $\to$ Layer II/III subtraction!) — or locally learned `W_fb` minimizing `||r1_actual - W_fb @ d1||` — and evaluate on a task where bottom-up input is noisy/occluded *before* readout (so bottom-up alone cannot already saturate at `95%`). Notice the stark sign asymmetry above: `+W_prox.T` collapses via positive feedback (`0.000`), whereas `-W_prox.T` is stable, boosts accuracy above `wire-off` (`0.233` vs `0.222`), hits `omit_m2 = 1.000`, clears `p1_cos < 0.90` (`0.8565`), boosts `cont_m1` (`0.615` vs `0.581` cut), and produces genuine subtractive negative error on omission (`cosp1_omit` drops `0.401 -> 0.036`)!

---

### Check 2: Is Round 17's lag-2 C1 pass meaningful, or a bar artifact?

**Verdict:** **Refuted** as a meaningful causal pass; **Verified** as an analyzer bar-definition artifact that does not alter Experiment B's fail verdict.  
**References:** `PROTOCOL-R17-CLAMP.md:154-177`; `r17/r17_analyze.py:25-31, 169-191`; `r17/r17_analysis.md:45-53, 86-97`; `r17/r17_planted{1,2}_test_{a,b}.jsonl`.

1. **Silent protocol divergence:** `PROTOCOL-R17-CLAMP.md:156-157` defines C1 with a **`>= 0.10` degradation floor**, and `r17/r17_analysis.md:97` states *"Lag-2 bars are reported the same way."* Yet `r17/r17_analyze.py:184-186` **silently dropped `c1 >= 0.10` and `c2 >= 0.10`** for `planted2`, checking only `c1lo > 0`. With the same `0.10` floor as lag-1, lag-2 C1 (`+0.00419`) **fails by a factor of `23.9×`**.
2. **Statistical fragility (exact JSONL recomputation across all 20 seeds `241..260`):**

| Metric (`n = 20`, exact JSONL audit) | Mean diff | SD | `z=1.96` 95% CI (analyzer) | Exact $t_{19}$ 95% CI | $t_{19}$ | Two-sided $p$ | Wilcoxon $p$ | W / L / T |
|---|---:|---:|---|---|---:|---:|---:|---|
| Lag-1 `omit_m2` (`on - cut`) | `+0.00494` | `0.01099` | `[+0.00012, +0.00975]` | **`[-0.00021, +0.01008]`** | `+2.008` | `0.0591` | `0.1165` | `13 / 5 / 2` |
| Lag-2 `omit_m2` (`on - cut`) | `+0.00419` | `0.00866` | `[+0.00040, +0.00799]` | `[+0.00014, +0.00825]` | `+2.166` | `0.0432` | `0.0253` | **`9 / 3 / 8`** |
| Lag-2 `omit_m2` (`on - cut`, **15 held-out**) | `+0.00503` | `0.00921` | `[+0.00037, +0.00969]` | **`[-0.00007, +0.01013]`** | `+2.114` | `0.0529` | `0.0330` | `7 / 2 / 6` |
| Lag-2 `omit_m2` (`on - shuf`) | `+0.00163` | `0.01423` | `[-0.00461, +0.00786]` | `[-0.00503, +0.00829]` | `+0.511` | `0.6150` | `0.4561` | `6 / 6 / 8` |
| Lag-2 `omit_m2` (`on - off`) | `+0.00235` | `0.01741` | `[-0.00529, +0.00998]` | `[-0.00580, +0.01050]` | `+0.603` | `0.5538` | `0.8971` | `9 / 7 / 4` |
| Lag-1 `omit_m1` (`on - off`) | **`-0.01571`** | `0.02320` | `[-0.02587, -0.00554]` | **`[-0.02656, -0.00485]`** | **`-3.028`** | **`0.0069`** | **`0.0130`** | `4 / 14 / 2` |
| Lag-1 `err_mag_last` (`on - off`) | **`+0.00203`** | `0.00104` | `[+0.00157, +0.00249]` | **`[+0.00154, +0.00252]`** | **`+8.706`** | **`<1e-7`** | `0.0001` | **`19 / 1 / 0`** |
| Lag-2 `err_mag_last` (`on - off`) | **`+0.00188`** | `0.00082` | `[+0.00152, +0.00224]` | **`[+0.00149, +0.00227]`** | **`+10.194`** | **`<1e-8`** | `0.0001` | **`19 / 1 / 0`** |

- On **`8/20` lag-2 seeds**, cutting the wire changes **zero** `m2` predictions out of `~115` events (`W/L/T = 9/3/8`), and on the 15 strictly held-out seeds (`246..260`, excluding val seeds `241..245`), exact Student-$t_{14}$ crosses zero ($p = 0.0529$).
- Even taking the `+0.004` lag-2 and `+0.005` lag-1 shifts at face value, they reflect **train-test distribution shift** (`W_dist1` was trained with `fb = on` and tested with `fb = cut`), NOT predictive content: `on - shuf` is `-0.0005` (lag-1) and `+0.0016` (lag-2), `on - off` is null on both, `margin_m2` (`on - cut`) is actually negative on lag-1 (`-0.00095`, `4/16` wins), and `on` slightly degrades `omit_m1` and `cont_m1`.
- **Does it change Experiment B's verdict?** No. Experiment B requires C1–C4 jointly on lag-1 (`PROTOCOL-R17-CLAMP.md:170`), and lag-2 itself fails C2, C3, and C4 (and C1 under the protocol's `0.10` floor).

---

### Check 3: Did Round 17's winner pick (`fb_gain = 0.1`) bias the clamp test toward a null?

**Verdict:** **Verified** that selecting `fb_gain` on `val_acc` structurally favors the smallest gain (`0.1`); **Refuted** that any alternative selection rule (`omit_m2` or `on - cut`) would have changed the winner or rescued Experiment B.  
**References:** `PROTOCOL-R17-CLAMP.md:99-108, 236-239`; `r17/monitor.py:71-100`; `r17/winner.json`; `r17/r17_analysis.md:8-12`.

1. **Structural selection bias (Verified):** Validation chose `max` over `{0.1, 0.3, 1.0}` on `val_acc` (`r17/monitor.py:83-85`), where `fb_gain = 0.0` scored `0.2266`, `0.1` scored `0.2261`, `0.3` scored `0.2209`, and `1.0` scored `0.1816` (`diag_r17_wire.py` reproduces `r17/r17_analysis.md:8-12` to 3 decimals). Because fixed random `W_fb` adds monotonic interference into `mod1`, maximizing `val_acc` **automatically picks the smallest nonzero gain on the grid (`0.1`)** — the setting closest to no wire at all.
2. **Counterfactual alternative picks (Refuted — none change the winner!):** Using our 5-seed (`241..245`) full-grid evaluation from Check 1:
   - **Pick on `omit_m2`:** `g = 0.1` (`0.9497`) > `g = 0.3` (`0.9361`) > `g = 1.0` (`0.7685`) > `g = 2.0` (`0.1403`) $\implies$ **still picks `0.1`**.
   - **Pick on positive cut gap (`omit_m2 on - cut`):** `g = 0.1` (`+0.0033`) > `g = 0.3` (`-0.0182`) > `g = 1.0` (`-0.1109`) > `g = 2.0` (`-0.1804`) $\implies$ **still picks `0.1`!** At every stronger gain (`0.3, 1.0, 2.0`), cutting the wire **improves** `omit_m2`, `omit_m1`, `cont_m1`, and `normal_acc`.
   - **Pick on `|on - cut|` or `on - shuf` (`g = 1.0` or `2.0`):** `g = 1.0` barely clears the `0.80×` guard (`0.1816 / 0.2266 = 0.8014`) and passes C3 (`p1_cos = 0.8042`), but **catastrophically fails C1 in the wrong direction (`-0.1109`)** and fails C2 (`+0.0334 < 0.10`, because `cut` `0.879` beats both `on` `0.769` and `shuf` `0.735`).
3. **Conclusion:** Validation bias toward `0.1` is a real design flaw (especially dangerous for Round 19 — see Check 9), but on Round 17's fixed random `W_fb`, **no gain on `[0.1, 2.0]` could have passed Experiment B**.

---

### Check 4: Round 18's borderline D2 and D3 — is "H0 fails" right, what would 40 seeds do, and is the var-sine MSE wall established?

**Verdict:** **Refuted** that D2 passes under exact Student-$t$ inference; **Verified** that `H0` fails for the *frozen Convention-B C1 configuration*; **Refuted** that a genuine var-sine MSE wall is established.  
**References:** `PROTOCOL-R18-CONTINUOUS.md:137-147, 211-225, 336-340`; `rpc_round18_continuous.py:70-73`; `r18/r18_analyze.py:58-68, 196-214`; `r18/r18_analysis.md:31-60`; `r18/r18_varsine_test_262-281.jsonl`.

1. **Exact Student-$t_{19}$ audit flips D2 to FAIL (`r18_analyze.py:67` uses `z = 1.96` instead of `t_{0.975, 19} = 2.0930`):**

| Var-sine Contrast (`r18_varsine_test_262-281.jsonl`) | $n$ | Mean | SD | `z=1.96` CI (R18) | Exact $t$ 95% CI | $t$ | $p$ | Wilcoxon $p$ | W / L |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| **D2b:** `C1 acc - Markov-1 acc` (all 20) | 20 | `+0.01081` | `0.02395` | `[+0.00031, +0.02130]` *(PASS)* | **`[-0.00040, +0.02202]` *(FAIL)*** | `+2.018` | `0.0579` | `0.0645` | `13 / 7` |
| **D2b:** held-out 15 seeds (`267..281`) | 15 | `+0.01054` | `0.02647` | `[-0.00286, +0.02394]` | `[-0.00412, +0.02520]` | `+1.542` | `0.1454` | `0.1818` | `9 / 6` |
| **D3:** `persist MSE - C1 MSE` (all 20) | 20 | **`-0.00828`** | `0.02293` | `[-0.01833, +0.00177]` *(FAIL)* | `[-0.01901, +0.00246]` *(FAIL)* | `-1.614` | `0.1230` | `0.1506` | `7 / 13` |
| **D3:** held-out 15 seeds (`267..281`) | 15 | `-0.00670` | `0.02625` | `[-0.01999, +0.00658]` | `[-0.02124, +0.00784]` | `-0.989` | `0.3396` | `0.6701` | `7 / 8` |
| **D3:** val-5 seeds (`262..266`) | 5 | `-0.01300` | `0.00699` | `[-0.01912, -0.00687]` | `[-0.02167, -0.00432]` | `-4.158` | `0.0142` | `0.0591` | `0 / 5` |
| **C3 vs persist:** `persist MSE - C3 MSE` | 20 | **`+0.01667`** | `0.01684` | `[+0.00929, +0.02405]` | **`[+0.00879, +0.02455]`** | **`+4.428`** | **`0.00029`** | **`0.0014`** | **`17 / 3`** |

   - Using the protocol's own `z = 1.96` rule, D2 barely passed (`ci_lo = +0.00031`, printed as `0.000`) while D3 failed (`-0.00828`). Under statistically valid finite-sample $t_{19}$, **BOTH D2 AND D3 FAIL**, so `"H0 fails"` for the frozen C1 config is **even more firmly supported** than section 52 states.
   - Note also a naming defect: on var-sine, `C1_onehot_delta_proto` is **NOT one-hot** (`rpc_round18_continuous.py:72`: `smooth_sigma = 2.0`), unlike Lorenz C1 (`smooth_sigma = 0.0`).

2. **What a 40-seed rerun would do (exact power projection):**
   - Assuming the observed means and SDs hold at $n = 40$ ($t_{0.975, 39} = 2.0227$):
     - **D2b** (`m = +0.01081, sd = 0.02395`): 40-seed $t$-CI tightens to **`[+0.00315, +0.01847]`** (passes; 80% power requires $n \approx 39$).
     - **D3** (`m = -0.00828, sd = 0.02293`): 40-seed $t$-CI tightens to **`[-0.01561, -0.00094]`** — **confirming that frozen C1 *loses* to persistence**, NOT turning D3 into a pass! (Also notice C1's var-sine MSE is right-skewed by catastrophic bin-jump outlier seeds: seed `274` `0.1132`, seed `268` `0.1087`, seed `267` `0.0903`, vs median `0.059` against persistence `0.056`.)
   - Therefore, spending 40 seeds re-running Round 15's stale C1 config is **wasted compute**.

3. **Is a var-sine MSE wall established at all? NO — refuted twice over!**
   - **First**, even inside Round 18's own 20-seed data, **`C3_bump_delta_proto`** (pure delta + unmodified `p1`-prototype readout, no ridge) gets **`MSE = 0.039`** vs persistence **`0.056`** (`+0.0167`, $t_{19} = +4.43, p = 0.00029$, `17/20` wins) and beats ESN (`0.046`, `15/20` wins).
   - **Second**, why did `C1` fail D3 (`0.064`)? Because `C1` froze Round 15's winner (`alpha = 1.0, smooth_sigma = 2.0, eta_d = 0.02`), selected under **buggy two-step-ahead Convention B**, which picked a `10×` too-small learning rate (`eta_d = 0.02`) and memoryless `alpha = 1.0`! Our 5-seed check (`262..266`, `diag_r18_a14_vs_a27.py`) using **pure discrete bin input (NO A27 bump, NO ridge)** from Round 15's own 18-config grid proves it:

| Var-sine (`n = 5` seeds `262..266`, discrete bin input, unmodified prototype readout) | `acc` | `within1` | `MSE` | Per-seed MSE (`262..266`) |
|---|---:|---:|---:|---|
| `C1_R18` (`a=1.0, s=2.0, eta=0.02`, Conv-B frozen) | `0.2942` | `0.5808` | `0.0678` | `[0.0613, 0.0676, 0.0806, 0.0588, 0.0709]` |
| `A14_bin` (`a=1.0, s=2.0, eta=0.2`) | **`0.3339`** | **`0.6518`** | `0.0530` | `[0.0370, 0.0493, 0.0719, 0.0613, 0.0457]` |
| **`A14_bin` (`a=0.7, s=2.0, eta=0.2`)** | **`0.3230`** | **`0.6363`** | **`0.0370`** | **`[0.0423, 0.0368, 0.0371, 0.0207, 0.0479]`** |
| **`A14_bin` (`a=0.7, s=3.0, eta=0.2`)** | `0.2989` | `0.6053` | **`0.0317`** | **`[0.0391, 0.0332, 0.0285, 0.0173, 0.0404]`** |
| Persistence / Markov-1 / ESN (matched 5 seeds) | `0.124 / 0.283 / 0.381` | `— / — / 0.692` | `0.0548 / 0.1148 / 0.0483` | `persist: [0.0493..0.0589]`, `ESN: [0.0371..0.0573]` |

   **And the exact same Convention-B carryover manufactured Lorenz D4 FAIL (`22.39` vs `12.61`) and `96.3%` of Lorenz D6!** Across the same 5 seeds (`262..266`) on Lorenz:

| Lorenz (`n = 5` seeds `262..266`, unmodified prototype readout, NO ridge) | `acc` | `within1` | `MSE` | Per-seed MSE (`262..266`) |
|---|---:|---:|---:|---|
| `C1_R18` (`a=0.7, s=0.0, eta=0.2`, Conv-B frozen) | `0.3169` | `0.6301` | `26.61` | `[24.88, 20.06, 25.52, 31.16, 31.44]` |
| `A14_bin` (`a=0.7, s=2.0, eta=0.2`, discrete bin only) | `0.3376` | `0.7656` | **`8.43`** | `[14.00, 7.21, 7.75, 5.26, 7.92]` |
| `A27_bump` (`a=0.7, s=2.0, eta=0.2`, R18 `C3`) | `0.3343` | `0.7682` | `7.73` | `[9.68, 8.94, 7.73, 5.55, 6.73]` |
| **`A14_bin` (`a=1.0, s=2.0, eta=0.2`, discrete bin only!)** | **`0.4285`** | **`0.8393`** | **`5.07`** | **`[5.68, 4.62, 3.75, 5.83, 5.48]`** |
| **`A27_bump` (`a=1.0, s=2.0, eta=0.2`)** | **`0.4301`** | **`0.8463`** | **`4.36`** | **`[5.57, 3.94, 3.35, 4.61, 4.34]`** |
| Persistence / ESN (matched 5 seeds) | `0.170 / 0.397` | `— / 0.795` | `13.05 / 4.29` | `persist: [12.80..13.61]`, `ESN: [3.77..4.72]` |

   Neither a var-sine MSE wall nor a Lorenz persistence-MSE wall survives proper Convention-A validation of Round 15's own existing discrete-bin grid.

---

### Check 5: Round 18's D8 — does delta "add nothing" or "actively harm the ridge"?

**Verdict:** **Verified** (code parity on train/test splits and `lam = 1.0`); **Refuted** that delta "actively harms the ridge" or that D8 validly tests whether delta learns continuous information. **D8 repeated Round 14's exact unstandardized `lam = 1.0` norm-shrinkage confound (`r14-agy-critique.md:156-168`), layered on top of a linear Data Processing Inequality (DPI) tautology.**  
**References:** `rpc_round18_continuous.py:66, 231-262, 324-368`; `r18/r18_analysis.md:15, 17, 20-21, 74-75`; `simulation-results.md:2902-2906, 2931-2936`; `r14-agy-critique.md:156-168`.

1. **Code parity verified — with the fatal unstandardized penalty intact:** `run_test_nets` (`rpc_round18_continuous.py:324-368`) feeds identical train/test sequences, identical `torch`/`numpy` seeds, identical `collect_train` (`t >= 1`), and identical `fit_ridge(..., lam=1.0)` (`lines 231-246`) to `C2`/`C2f` and `C4`/`C4f`. Unlike Round 14, `fit_ridge` now centers `D` and `V` (`Dc = D - d_mean`), fixing the missing intercept — **but it still omits feature/norm standardization while fixing `lam = 1.0`** (`line 244`: `Dc.T @ Dc + 1.0 * I`).
2. **Direct forensic sweep (`diag_r18_ridge.py`, 5 Lorenz seeds `262..266`, 3 var-sine seeds `262..264`):**

| Task / Representation (`n=5` Lorenz, `n=3` var-sine) | `||d||` | `||dc||` | Eff. rank (PR) | `s_min` | Raw `lam=1.0` (R18) | **RMS-matched `lam=1.0`** | **Col-std `lam=1.0`** | **Unreg `lam=1e-6`** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Lorenz one-hot** `tr` (`C2`) | `1.042` | **`0.784`** | `18.9` | `0.0038` | `12.7244` *(worse)* | **`12.5195` *(WINS)*** | **`11.9944` *(WINS)*** | **`11.9543`** |
| **Lorenz one-hot** `fr` (`C2f`) | `5.763` | **`3.890`** | `26.5` | `0.0232` | `12.2241` | `13.3313` | `12.0004` | **`11.9528`** |
| **Lorenz one-hot** `y2_prev` | `11.324` | `7.766` | `30.5` | `3.0532` | `11.9445` | `12.1093` | `11.9516` | **`11.9528`** |
| **Lorenz bump** `tr` (`C4`) | `1.269` | **`1.001`** | `5.6` | `0.0019` | `3.9766` *(worse)* | **`3.9727` *(WINS)*** | **`3.5100` *(WINS)*** | **`3.4859`** |
| **Lorenz bump** `fr` (`C4f`) | `5.744` | **`3.914`** | `10.5` | `0.0122` | `3.7275` | `4.3637` | `3.5448` | **`3.4854`** |
| **Lorenz bump** `y2_prev` | `11.293` | `7.786` | `11.2` | `1.5971` | `3.4785` | `3.6567` | `3.4844` | **`3.4855`** |
| **Var-sine `R15_C1`** `tr` / `fr` / `y2` | `1.599 / 5.670 / —` | — | — | — | `0.06621 / 0.06547` | — | — | **`0.06381 / 0.06384 / 0.06377`** |

3. **Exact diagnosis (neither "delta actively harms ridge" nor "delta adds nothing continuous" is right):**
   - **Why `C2f < C2` (`-0.584`, `19/20` seeds) and `C4f < C4` (`-0.180`, `19/20` seeds) happened in Round 18:** Delta shrinks `||dc||` by **`4.96×`** (one-hot) and **`3.91×`** (bump), shrinking `Dc.T @ Dc` eigenvalues by **`24.6×`** and **`15.3×`**! Fixed `lam = 1.0` therefore over-regularizes trained `d1` by `15–25×` relative to frozen `d1`. Remove that global scale confound (`[rms]` at `lam = 1.0`), and **trained ridge BEATS frozen ridge by `+0.812` (one-hot) and `+0.391` (bump)** (and by `+2.86` and `+1.28` at `lam = 10.0`), because delta concentrates predictive variance into fewer principal directions (participation ratio drops `26.5 -> 18.9` and `10.5 -> 5.6`)!
   - **Why D8 is a mathematical tautology at `lam -> 0`:** Because `mod1` and `mod2` are frozen (`use_confirmation=False, fb_gain=0.0`), `y2_prev` is byte-identical between trained and frozen nets. Both `W_dist1_frozen` and `W_dist1_trained` are invertible full-rank $256 \times 256$ matrices (`s_min > 0`). By linear algebra, OLS on $W y_2$ equals OLS on $y_2$ for **every** invertible $W$ (`11.9528` and `3.4855` across all three!). By the linear Data Processing Inequality, **NO linear plastic matrix `W_dist1` on a frozen reservoir can EVER exceed the linear decodability of `y2_prev`**!
   - **Design implication:** Delta does **not** harm continuous representations, nor did it fail to learn continuous trajectories (it slashes Lorenz bump prototype MSE `35×`, from `C6 = 257.4` to `C3 = 7.29`, and var-sine `22×`, from `0.879` to `0.039`, beating frozen ridge `C4f = 0.045`!). Rather: (a) **if the goal is better features than a frozen reservoir**, plasticity MUST enter the reservoir (`mod1`/`mod2`, e.g., confirmation or recurrent plasticity), because `W_dist1` alone is linear post-processing of frozen `y2`; (b) **if the goal is lower continuous MSE without an offline ridge**, replace hard `argmax` over prototypes with a **local continuous population-vector / softmax-weighted centroid readout** over the `K` prototypes (zero trained weights, continuous interpolation).

---

### Check 6: Audit of Round 18's scoring correction and remaining baseline shifts/asymmetries

**Verdict:** **Verified** that Round 18's convention A matches `rpc_v2.py:run()` and puts all models and baselines on the exact same 1-step-ahead test transitions (`i = 1..L-1`, zero remaining 1-step shift); **Refuted** on two secondary claims (`baselines.py` docstring blame and `C4` vs `ESN` parity), plus two minor analyzer/design defects.  
**References:** `rpc_v2.py:1126-1156, 1330-1340`; `rpc_round18_continuous.py:188-228, 249-262, 471-474`; `rpc_round15.py:205-249` (from commit `b2543c73`); `baselines.py:167-256, 307-341`; `r18/r18_analyze.py:288-295`; `simulation-results.md:2362-2364, 2723-2725, 2901`.

1. **Exact alignment audit (Zero remaining 1-step shift — VERIFIED):**
   - In `rpc_v2.py:_step_core` (`lines 1126-1156`), at step `i`, `d1(i) = W_dist1 @ y2_prev` is computed from `seq[0..i-1]` *before* `seq[i]` enters `mod1`, and `pred(i)` predicts `seq[i]`. `rpc_v2.py:run()` (`lines 1333-1337`) scores `out["pred"] == tok` for `i = 1..L-1`.
   - `rpc_round18_continuous.py:collect` (`lines 188-213`) steps through `i = 0..L-1` and appends `(out["pred"], out["d1"], cont[i], y)` for `i >= 1`. Both `proto_metrics` (`lines 216-228`) and `ridge_metrics` (`lines 249-262`) score these `L - 1` predictions against `bins[i]` and `cont[i]` (`i = 1..L-1`). Zero target leakage; exact match to `run()`.
   - In `baselines.py`, `markov_predict` (`lines 167-193`), `persistence_preds` (`lines 317-319`), and `ESN.predict` (`lines 242-256`) all output length `L - 1` lists where `preds[t]` (`t = 0..L-2`) uses `seq[0..t]` to predict `seq[t+1]`, scored by `accuracy` (`line 313`) and `mse_from_bin_preds` (`line 328`) against `seq[t+1]` and `cont[t+1]` (`i = 1..L-1`). `persistence_mse` (`line 338`) scores `(cont[t] - cont[t+1])^2` over `t = 0..L-2`. Every baseline and every Round-18 condition now evaluates the **identical `L - 1` 1-step-ahead transitions** (`23` per Lorenz seq, `99` per var-sine seq).

2. **Audit against the original Round-15 bug (`rpc_round15.py:205-226`) — record misattributes docstring blame:**
   - `rpc_round15.py:212-218` (`run_cont`) collected `p = [pred(0), pred(1), ..., pred(L-2)]` and passed it to `baselines.accuracy(preds, ev)` and `baselines.mse_from_bin_preds(preds, ev_cont, decode)`. That had **two** consequences: (a) `p[0] = pred(0)` (computed from `y2_prev = 0`, constant garbage!) was scored against `seq[1]` (`4.3%` of Lorenz steps, `1.0%` of var-sine steps); (b) `p[t]` (`t >= 1`) was scored against `seq[t+1]` (2-step-ahead), while `pred(L-1)` was discarded.
   - **Record error (`simulation-results.md:2362-2364, 2723-2725`):** Blaming `baselines.py:322-330` (*"whose docstring wrongly claims `preds[t] predicts token t+1`"*) is **factually backwards**. `baselines.py` is 100% internally consistent for all baselines; the bug lived solely in `rpc_round15.py:215-216` (`if i < len(seq) - 1:` instead of `if i >= 1:`).

3. **Four remaining non-shift asymmetries / bugs in Round 18:**
   - **Apples-to-oranges `C4` vs `ESN` MSE comparison (`simulation-results.md:2901`):** `baselines.py:232, 254` trains `ESN` with a **32-class one-hot classification** ridge readout on **one-hot bin inputs** and computes `esn_mse` from the argmax bin centre (`rpc_round15.py:244`). By contrast, `C4`/`C4f` receive **continuous bump inputs** AND fit a **continuous scalar MSE ridge readout** directly to `cont[t]`! Running the **exact same 300-unit `ESN` (`baselines.py`, `leak=0.3, lam=1.0`)** with a continuous scalar ridge readout (`diag_fast.py`, 5 seeds `262..266`):
     - **Lorenz:** `ESN` class-argmax MSE = `4.308` $\to$ **ESN scalar ridge (one-hot) = `3.287`** (`acc = 0.388`) $\to$ **ESN scalar ridge (bump) = `2.097` (`acc = 0.562`, beating Markov-2 acc!)** — nearly **half** `C4`'s `3.795` and `C4f`'s `3.615`!
     - **Var-sine:** `ESN` class-argmax MSE = `0.0458` $\to$ **ESN scalar ridge (one-hot) = `0.0191`** $\to$ **ESN scalar ridge (bump) = `0.0072`** — **6× better** than `C4` (`0.044`) and `C4f` (`0.045`)!
   - **Continuous vs quantized `persistence_mse` (`baselines.py:338`):** `persistence_mse` uses the exact continuous value `cont[t]` rather than the quantized bin centre `decode(seq[t])` given to `C1`. Across all 20 seeds (`diag_fast.py`), quantized persistence MSE is `12.827` vs continuous `12.605` on Lorenz (`+0.222`, `1.7%` edge to persistence) and `0.05590` vs `0.05593` on var-sine (`-0.00003`, negligible).
   - **Analyzer bug in `D7-var` (`r18/r18_analyze.py:288-295`):** Prints label `"D7-var: C4 mse < C2 mse and C4 mse < C3 mse"` while **only computing `C2 - C4` (`+0.022`)** and silently dropping `C3 - C4` (`-0.00523`, $t_{19} = -1.66$, `5/15` wins — `C3` actually beats `C4`!).
   - **Val-test seed overlap (`5/20` seeds `262..266`):** `bump_sigma` was chosen on `262..266` and evaluated on `262..281`. Our held-out 15-seed (`267..281`) audit confirms Lorenz D6, D7b, and D8 are unchanged, while var-sine D2b weakens further ($p = 0.145$) and D3 stays null (`7/8` wins, $p = 0.340$).

---

### Check 7: Markov-2 dominance — is the continuous wall a task property, and what task discriminates learned prediction from lookup?

**Verdict:** **Verified** that both Lorenz (`dt_eff = 0.10`, 1D $x$) and var-sine are structurally trivial for a 2-gram lookup table; **Partially Refuted** that this makes D4 uninformative (D4 tests against persistence and passes under Convention-A validation; D8 is uninformative because of the DPI/scale confound, not Markov-2).  
**References:** `baselines.py:107-135, 167-193`; `rpc_round15.py:59-75`; `r18/r18_lorenz_test_262-281.jsonl`; `r18/r18_varsine_test_262-281.jsonl`; `simulation-results.md:2922-2929`.

1. **Quantifying why Markov-2 crushes every model (`diag_fast.py`, 20 seeds `262..281`):**
   - **Lorenz (`K = 32`, `4,600` train transitions/seed):** **`99.50%`** of all test 2-grams (`min = 98.55%`) appear in training! Because 1D Lorenz $x(t)$ at `dt_eff = 0.10` is locally smooth with nearly deterministic local velocity $(x_t - x_{t-1})$, conditioning on one bin leaves $\mathbb{E}[\operatorname{Var}(x_t \mid b_{t-1})] = 9.520$ ($R^2 = 0.8551$), while conditioning on **two bins** $(b_{t-2}, b_{t-1})$ collapses conditional variance to **`0.7120` ($R^2 = 0.9892$)**! Thus a 2-gram table achieves `acc = 0.556, MSE = 1.416` with zero dynamics learning.
   - **Var-sine (`K = 32`, `9,900` train transitions/seed):** **`99.82%`** of all test 2-grams (`min = 99.29%`) appear in training! Conditioning on $(b_{t-2}, b_{t-1})$ collapses conditional variance from `0.0532` ($R^2 = 0.8936$) to **`0.0023` ($R^2 = 0.9954$)** (`Markov-2 MSE = 0.00529`)! Even a variable-frequency sine is almost uniquely determined 1 step ahead by two consecutive bins (because $\Delta x$ pins down $\omega$ and phase up to a tiny curvature residual).
   - **Why the two-module RPC trails Markov-2:** Not merely "a one-step reservoir" — `alpha = 1.0` (`varsine`) or `alpha = 0.7` (`Lorenz`) with fixed random `mod1 -> mod2` (`ReLU(z-score)`) dilutes the exact 2-step nonlinear interaction $x_{t+1} \approx 2\cos(\omega)x_t - x_{t-1}$ across 256 noisy features, whereas `ESN-300` (`leak = 0.3`) with bump + scalar ridge already reaches `2.097` (Lorenz) and `0.0072` (var-sine).

2. **Continuous tasks that genuinely separate learned temporal prediction from $k$-gram lookup:**
   - **NARMA-10** (Atiya & Parlos, 2000; Jaeger, 2002): $y_{t+1} = 0.3 y_t + 0.05 y_t \sum_{i=0}^9 y_{t-i} + 1.5 u_{t-9} u_t + 0.1$ driven by i.i.d. $u_t \sim U[0, 0.5]$. Requires both 10-step fading memory and nonlinear cross-terms; Markov-2 is near chance ($R^2 \approx 0.1$), while a reservoir + learned readout achieves NRMSE $\approx 0.2$–$0.4$.
   - **Mackey-Glass delay-differential equation ($\tau = 17$ or $\tau = 30$, multi-step horizon $h = 5$–$10$)** (Jaeger & Haas, 2004): infinite-dimensional delay state where two recent bins $(b_{t-2}, b_{t-1})$ cannot determine $x_{t+h}$, breaking Markov-2 while staying smooth and deterministic.
   - **Multi-frequency sum of 3 incommensurate sines (or 1D Lorenz $x$ with $\sigma_{\text{obs}} = 1.5$ noise and horizon $h = 3$):** Forces multi-step temporal integration/filtering beyond 2 bins and destroys exact 2-gram lookup.

---

### Check 8: The mapping tension — does A-pass + B-fail strengthen or weaken the predictive-coding claim?

**Verdict:** **Refuted** that A-pass + B-fail strengthens the predictive-coding claim; **Verified** that what survives is functionally a **fixed two-stage Echo State Network with online NLMS (`W_dist1`) and a fixed `p1`-prototype nearest-centroid readout**.  
**References:** `ch3.txt:160-188, 381-470, 1352-1561`; `rpc_v2.py:1126-1180`; `simulation-results.md:2140-2160, 2582-2593, 2810-2846, 3003-3015`.

1. **Why A-pass + B-fail *weakens* the predictive-coding claim:**
   - Look closely at what Round 16 (Experiment A) actually demonstrated (`rpc_v2.py:1127-1156`, `simulation-results.md:2435-2443, 2583-2589`): at the omission step $t$, `d1(t) = W_dist1 @ y2(t-1)` is computed **before** the blank input is even seen (`omit_m2 == normal_m2` identically!). **Any** Echo State Network with an online NLMS readout produces an identical `0.966` "omission decode" at step $t$ (its pre-stimulus next-step readout hasn't seen the blank yet!) and above-chance continuation at `t+1` (`cont_m1 = 0.610`) purely because the recurrent reservoir `W_res @ r` still holds a decaying echo of cue `a` (`1`)!
   - What actually distinguishes cortical **predictive coding** (Rao & Ballard, 1999; Keller & Mrsic-Flogel, *Neuron* 2018; Aitken, Kok et al., *PLoS Biol* 2020) from a feedforward/reservoir predictor is that the **top-down prediction causally enters the lower sensory module**: either suppressing predictable sensory drive (expectation suppression / explaining away, lowering lower-level error) or evoking a stimulus-specific deep-layer predictive template / negative prediction error upon omission.
   - Round 17 (Experiment B) directly tested that discriminating criterion — and **failed**: `p1` barely moved (`0.998`), `err_mag_last` *increased* (`19/20` seeds, $p < 10^{-7}$), and `cos_p1_omit_m2` *decreased* ($p = 0.035$). Saying this "matches the mapping research" because a fixed random wire cannot produce a Kok/Aitken template is accurate as a *post-hoc failure diagnosis*, **not** as corroboration of predictive coding: it confirms the current model **does not implement** the cortical phenomenon!

2. **What does the model currently explain that a plain reservoir + NLMS (Widrow-Hoff) does NOT?**
   - **Honestly: almost nothing at the computational/phenomenological level.** The single genuine architectural distinction from a standard ESN + supervised NLMS readout is that the NLMS target is an **unsupervised internal sensory code `p1(t)`** (from `mod1`), rather than an external class label, so one local dendritic rule (`err = p1 - d1`, `rpc_v2.py:1176-1179`) aligns the distal drive onto sensory prototypes without an externally supervised readout matrix.
   - **Plain statement of what survives of the Chapter 3 thesis claim:**
     - **Survives:** (1) Fixed random expansion + recurrent reservoir as a temporal basis (`ch3.txt:1352-1395`); (2) local, non-backprop online error learning (Widrow-Hoff NLMS / linear rate-coded Urbanczik & Senn 2014) on `W_dist1` alone can map `y2(t-1)` onto `p1(t)` and solve discrete sequences (`sine` `0.868`, `planted2` `0.222`) and continuous smooth sequences (`Lorenz` `5.07`, `var-sine` `0.0317–0.0370` once properly validated under Convention A) using fixed `p1`-prototypes.
     - **Does NOT survive:** (1) Chapter 3's burst-coincidence potentiation (`mean_burst = 0.00037` in R17, `0.0000` in Lorenz R18; unsigned burst/coactivity collapse to chance); (2) reciprocal EC $\to$ CA1 confirmation (`use_confirmation=False`); (3) Layer V $\to$ II/III predictive inhibition (`s_inh=0.0`, and A26 random feedback failed Experiment B); (4) Reza's "random prediction + selection" evolutionary framing (`simulation-results.md:2822-2826`) — NLMS is **gradient descent on squared error** $\frac{1}{2}\|p_1 - W_{\text{dist1}} y_2\|^2$, not selection of random predictions! (To explain canonical microcircuit / hippocampal electrophysiology — expectation suppression, negative prediction-error neurons, Aitken/Kok templates — the model needs **aligned subtractive feedback `-W_prox.T`** turned on, as our Check-1 control just demonstrated.)

---

### Check 9: Round 19 hierarchy interaction — does it risk repeating Round 17's null by construction?

**Verdict:** **Verified — Round 19 (`PROTOCOL-R19-HIERARCHY.md`) is structurally rigged to repeat Round 17's feedback null on `H8` AND bottleneck `H4` (3-level vs 2-level) through the exact same inert random wire.**  
**References:** `PROTOCOL-R19-HIERARCHY.md:67-79, 107-124, 196-210, 255-257, 281-297`; `simulation-results.md:3153-3158, 3190-3195, 3222-3228`.

1. **Three stacked failure modes already baked into Round 19:**
   - **Exact repeat of Round 17's inert/toxic wire + validation bias:** `PROTOCOL-R19-HIERARCHY.md:69-72, 199-210` uses **fixed random additive `W_fb_l ~ N(0, 1/n_out)`** at **`fb_gain ∈ {0.1, 0.3}`**, validated on **overall accuracy** (where `~83%` of steps in `make_hier_data` are uninformative random filler or locally determined `m2`!). We just proved in Check 1 that random `W_fb` at `g = 0.1` has `fb/drv = 3.9%` (`p1_cos = 0.998`) and at `g = 0.3` has `fb/drv = 11.7%` (`p1_cos = 0.981`) while *degrading* accuracy. Validation will therefore favor `g = 0.1` (or at worst `0.3`), where `H8` (`cand_on - cand_cut >= +0.05`) is nearly impossible to pass!
   - **Hidden architectural bottleneck killing `H4` (`3-level > 2-level-ladder` by `+0.10` on late `m1`):** Trace `PROTOCOL-R19-HIERARCHY.md:107-123`! The bottom readout reads `d1 = W_dist_1 @ y2_prev`. Level 3 (`alpha = 0.1`, slow context code `y3`) has **NO direct connection to `d1`** — the **ONLY** way level 3's slow context can ever reach `mod2` (and hence `d1`) is through **`fb_2 = g_2 * (W_fb_2 @ d_2)`**! If `g_2 = 0.1` (`fb/drv ≈ 4%`), level 3's context is attenuated by `~25×` before touching `r2`, starving `mod2` of level 3's signal; if `g_2 = 0.3` (or higher), random `W_fb_2` injects unaligned interference into `mod2`! Even worse, `d2` is trained to predict `p2` (which ALREADY depends on `r1` and `r2`), NOT the future `m1` token!
   - **Dented task premise (already caught in smoke test, `simulation-results.md:3153-3158, 3190-3195`):** Level 1 (`alpha = 1.0, rho = 0.9`) already decodes the block context at lag 11 with **`0.70` accuracy** (and Level 2 at `alpha = 0.3` hits **`0.97`**, *higher* than Level 3's `0.93`!!). So `hier2-ladder` (`alpha_2 = 0.3`, `0.97` lag-11 context!) already has MORE context memory than Level 3 (`0.93`), making `H4` (`L3 - L2_ladder >= +0.10`) doubly uphill!
2. **Feedback magnitude needed & whether Round 19 measures it:**
   - To move `mod_l`'s state meaningfully (`p_l_cos < 0.90`, `dp_l_rel > 0.48`), feedback needs `||fb_l|| / ||drive_l|| ≈ 0.35–0.50` (around `fb_gain ≈ 0.8–1.0` after `d_l` shrinks to `p_l`'s scale — well outside `{0.1, 0.3}`!), AND must be **aligned** (`-W_prox_l.T` or learned) so that `0.40` drive fraction does not destroy accuracy.
   - `PROTOCOL-R19-HIERARCHY.md:255` logs only `fb_rms_ratio = ||fb_l|| / ||bu_l||` (omitting `W_res @ r`, overstating the true drive fraction by `~1.7×`), and **omits `p_l_cos` (`on` vs `cut`) from section 8's test metrics entirely** (measuring `r1_cos` only in the 1-seed smoke test!).
3. **Cheapest single change to make Round 19's feedback test real:**
   - Replace fixed random `W_fb_l` with **norm-matched aligned feedback (`-mod_l.W_prox.T` for `fb_1` explaining away; or for `fb_2`, a local delta-learned `W_fb_2` trained via `err_r2 = r2 - W_fb_2 @ d2` or direct `-mod2.W_prox.T` at `g ∈ {0.3, 1.0}`)**, scale `W_dist_l` init down `3.5×` (`w_dist_std = 0.15`) so epoch-0 feedback doesn't explode, validate on **val `late_m1` accuracy** (not filler-dominated overall accuracy!), and log `||fb_l|| / ||drive_l||` and `p_l_cos` on test. (Even cheaper, if testing whether a slow L3 can *ever* help `d1` before fixing `W_fb_2`, add a 1-line upper-bound diagnostic condition where `d1` reads `[y2_prev, y3_prev]` directly.)

---

### Check 10: Next experiment — Generative (D) vs Precision (C), and what would change the verdict?

**Verdict:** **Unresolved** (neither C nor D has run). Between the two, **Precision (C) is premature without a causal top-down wire, whereas a combined Fixed-B + Generative Rollout (D) experiment settles far more** — *provided* it runs on an architecture with aligned feedback rather than the inert random A26 wire.  
**References:** `PROTOCOL-R17-CLAMP.md`; `PROTOCOL-R18-CONTINUOUS.md`; `simulation-results.md:2153-2160, 2856-2860, 2937-2942, 3003-3015`; `ch3.txt:653-707, 1540-1561`.

1. **Why Precision (C) settles little right now:**
   - In predictive coding (Friston, 2018; Bastos et al., *Neuron* 2012; `ch3.txt:653-707`), precision $\Pi$ dynamically weights the relative influence of **bottom-up sensory evidence vs top-down prior** during **inference** (updating `r1`/`p1` at test time) as well as learning.
   - Because B failed (no causal top-down pathway into `mod1`), testing precision on `err = p1 - d1` in `W_dist1` can only modulate the NLMS step size $\eta_d$ — i.e., **adaptive learning-rate scheduling / Kalman gain in standard adaptive filtering**, which does *not* discriminate predictive coding from reservoir + NLMS!
   - **What C would need to show to change the verdict:** *With a causal top-down wire active and weights frozen*, degrading bottom-up input SNR at test time must smoothly shift `mod1`'s **sensory representation `p1`** (and downstream `y2`) away from the noisy sensory input toward the top-down prediction `d1` (precision-weighted Bayesian fusion), without hand-coding a test-time oracle switch.

2. **Why Generative Rollout (D) settles more — and its exact bar against FORCE / ESN:**
   - Note the crucial literature bar: **FORCE learning** (Sussillo & Abbott, *Neuron* 2009) ALREADY achieves closed-loop autonomous generation from a fixed random reservoir using RLS + feedback. So merely rolling out a sine wave with feedback does *not* prove predictive coding unless it works via **strictly local online plasticity (`delta`) + internal sensory-space matching (`p1`)** (no external target supervision, no global RLS inverse-covariance matrix).
   - **What D would need to show to change the verdict:** Starting from a brief cue (e.g., 3–5 steps of `sine-fixed`, `Lorenz`, or a 5-token motif), blank the bottom-up input for $H = 5$–$20$ consecutive steps with **frozen weights**, letting top-down feedback `W_fb @ d1(t)` drive `mod1` autonomously (`d1(t) -> r1(t) -> r2(t) -> y2(t) -> d1(t+1)`). Passing requires:
     1. Autonomous closed-loop rollout sustains accurate multi-step prediction (`MSE` and phase coherence) **far beyond the open-loop decay baseline (`fb_mode = "cut"`)** and **shuffled-feedback baseline (`fb_mode = "shuffle"`)** across $H \ge 5$–$10$ blank steps;
     2. The exact same top-down pathway produces **stimulus-specific lower-level templates (`cos(p1_omit, proto[omitted])` significantly higher than `cut` and `shuffle`, matching Aitken, Kok et al. 2020)** during omissions AND **expectation suppression / error reduction (`err_mag_on < err_mag_off`, matching Rao & Ballard 1999)** during predictable inputs.

---

## Recommended Next Experiment (Settling Both Open Questions in Minimal Compute)

Before spending another 30+ CPU-hours on Round 19's random-wire hierarchy or a 40-seed rerun of Round 18's stale C1 config, run **two cheap, decisive preregistered experiments** (~6 CPU-hours total):

### Experiment 1 (Mechanistic — fixes Experiment B and runs Experiment D together): **Aligned Inhibitory vs Generative Feedback Clamp + Multi-Step Omission Rollout**
- **Architecture change (2 lines in `rpc_v2.py`):** Test fixed norm-matched **subtractive `-W_prox.T`** (Rao-Ballard / Chapter-3 L5 $\to$ L2/3 inhibition) and **locally delta-learned `W_fb`** ($\Delta W_{\text{fb}} \propto (r_1 - W_{\text{fb}} d_1) d_1^\top$, local to module 1, matching `d1` back into `r1`'s coordinate frame!), with `w_dist_std = 0.15` so epoch-0 feedback scale matches epoch 5.
- **Tasks (20 fresh seeds `302..321`, `5` epochs):**
  1. **Noisy / 40%-occluded `planted1` + `planted2`** (where bottom-up alone cannot saturate at `95%`, giving top-down feedback actual work to do on `normal_acc` and `omit_m2`), testing **Experiment B bars C1–C5** (`on` vs `cut` vs `shuffle` vs `off`) PLUS `err_mag_on < err_mag_off` (explaining away) and `cos_p1_omit` (template/negative-PE signature).
  2. **Multi-step closed-loop omission rollout (Experiment D)** on `sine-fixed` (5-step cue + 15-step blank) and `planted` (`m1` + `m2` consecutive blanks), requiring closed-loop `on` to beat both `cut` (open-loop reservoir decay) and `shuffle` by paired 95% Student-$t$ CI excluding zero across all $H$ blank steps.
- **Decision rule:** If aligned/locally-learned feedback clears C1–C4 (including `on - shuf`), reduces `err_mag`, and sustains multi-step rollout over `cut`/`shuffle`, **predictive coding causality is genuinely established**. If even locally learned `W_fb` fails C1/C2 or rollout, then this two-module architecture is definitively settled as **open-loop reservoir + local NLMS**.

### Experiment 2 (Continuous-wall settlement — replaces Round 18's confounded verdict): **Convention-A Re-Validation + Scale-Invariant Readout Test + Non-Markov-2 Benchmark**
- **Setup (20 fresh seeds `302..321`, 5 val seeds strictly disjoint from 20 test seeds!):**
  1. Re-validate `(alpha, smooth_sigma, eta_d)` under **Convention A** (not Round 15's buggy Convention B!) on Lorenz and var-sine, testing **discrete `A14` (`smooth_sigma`) vs continuous `A27` (`bump_sigma`)** separately, plus a **local parameter-free continuous readout** (softmax-weighted prototype centroid $\hat{v} = \sum_k \operatorname{softmax}(\beta \cos(d_1, \mu_k)) c_k$) against persistence, Markov-1/2, and **matched continuous-scalar-ridge ESN**.
  2. Add **NARMA-10** and **Mackey-Glass ($\tau = 17$, horizon $h = 5$)** — where Markov-2 fails — to test whether local delta learning genuinely learns multi-step continuous nonlinear dynamics.
  3. Whenever any ridge probe is reported, use **exact Student-$t$ CIs** and **column-standardized / cross-validated $\lambda$** so norm shrinkage can never manufacture a fake D8 failure again.
