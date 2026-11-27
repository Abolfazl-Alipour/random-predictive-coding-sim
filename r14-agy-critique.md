# Independent Scientific Critique: Round 14 (A23 & A24 Delta Rule)

**Reviewer:** AGY (Antigravity)  
**Date:** Nov 26, 2026  
**Scope:** Full audit of `simulation-results.md` (sections 1–43, focusing on 41–43), `README.md`, `RPC-HUNT-REPORT.md`, `rpc_v2.py` (assumptions A1–A24), `ch3.txt`, `baselines.py`, `baselines_B.md`, `rpc_simple.py`, `rpc_planted_alpha.py`, `rpc_sine_diag.py`, `rpc_spatial.py`, `rpc_track.py`, earlier critiques (`kimi-critique.md`, `agy-critique.md`), and all Round 14 drivers and results files (`rpc_sine_fix14*.py`, `rpc_planted_fix14*.py`, `*_results.txt`).  
**Missing files noted upfront:** Three files cited in the brief or record—`external-critique-3.md`, `rpc_sine_fix14a_results.txt`, and `rpc_sine_fix14b_results.txt` (`simulation-results.md:1961, 1974`)—are **not present** in `/home/reza/zfs-1tb/rpc-sim/review/`. For the third critique, only Muse's summary in `simulation-results.md:1959-1982` and `README.md:210-218` could be audited; for A23 sine top-6 and A18, the driver code was audited directly but the raw text outputs could not be cross-checked against disk.  
**Verification method:** Zero existing files were modified. Targeted read-only diagnostic runs were executed against the unmodified codebase using `/home/reza/zfs-1tb/rpc-sim/review-venv/bin/python` (single-threaded CPU), including 10 completely untouched fresh seeds (`100..109`). Every number below is explicitly labeled as either quoted from the record or measured in a named diagnostic.

---

## Lead Verdict: Does the Delta Win Stand?

**Short answer:** **Partially YES as a local learning-rule engineering result; NO on the marginal planted pass and the "only the error term works" control claim; NO as a vindication of Chapter 3's thesis theory.**

Separating the three distinct claims bundled inside Round 14:

1. **Does A24 (`delta`) genuinely learn with 100% fixed random internal wires and the unmodified `p1`-prototype cosine readout?** **YES.** That win is **not** a test-set-selection mirage. On 10 completely untouched fresh seeds (`100..109`), `delta-256` replicates large gains over chance (`sine-fixed` `0.467` vs `0.031`, `planted` `0.217` vs `0.062`). Even more strikingly, removing the test-set-selected `smooth_sigma=3.0` carryover and reverting to plain one-hot input (`smooth_sigma=0.0, alpha=1.0`) lifts `delta-256` on 3 fresh sine seeds (`100..102`) to **`0.847`**, crushing Markov-2 (`0.795`).
2. **Do Round 14's specific bar passes and control conclusions survive audit?** **NO.**
   - **Planted fails on fresh seeds:** The marginal 5-seed pass (`0.221` vs `0.22`) drops to **`0.217`** on 10 fresh seeds (`6/10` below `0.22`; 15-seed mean `0.218`, $t_{14} = -0.38$, $p = 0.71$). Worse, `motif_acc = 0.966` (`rpc_planted_fix14c.py:62-68`) silently scores **only** the easy 2nd-step motif transitions (`(1,2)->3`, etc.) and omits the 1st-step motif transitions (`1->2`, etc.), where `delta-256` scores only **`0.628`** (`all-motif = 0.789`), accounting for ~69% of the gap to Markov-1 (`0.281`).
   - **Round 14's smooth-sine config does not reliably beat Markov-1:** On 10 fresh seeds (`alpha=0.7, smooth_sigma=3.0`), `delta-256` (`0.467`) beats matched Markov-1 (`0.461`) by only `+0.006` ($t_9 = 0.37$, $p = 0.72$, wins `5/10`), though our diagnostics show this is a readout artifact of `smooth_sigma=3.0` blurring `p1` prototypes rather than a failure of `delta` to learn phase (`mid_dir = 99.4%`).
   - **The `cov` control was broken by a centering bug and clamp saturation:** `rpc_v2.py:1082-1085` subtracts **spatial** per-step means instead of **temporal** per-neuron means (Sejnowski, 1977) and runs unnormalized `eta_b` on `p1n`, slamming `17.9%` (planted) and `44.0%` (sine) of weights into the `±2.0` clamp. Fixing `cov` to true temporal covariance Hebbian learning (**zero error term**) jumps planted from `0.154` (`m2=0.645`) to **`0.199` overall and `0.939` on `m2`** across seeds `0..4`—90% of delta's overall score and 97% of its `m2` score without any error feedback!
   - **Untested continuous tasks still fail persistence:** 1-seed checks on Lorenz (`0.438` smooth / `0.395` one-hot vs persistence `0.587`) and 1D track (`0.489` smooth / `0.597` one-hot vs persistence `0.776`) fail to beat trivial persistence.
3. **Does A24 rescue Chapter 3's thesis theory?** **NO.** Every passing Round 14 run disables confirmation (`use_confirmation=False`), disables Layer V $\to$ II/III inhibition (`s_inh=0.0`), discards burst-gated Hebbian plasticity in favor of signed dendritic prediction-error LMS (Urbanczik & Senn, 2014; Widrow & Hoff, 1960), and—unmentioned anywhere in section 43—drives **`mean_burst` to `0.000`** (`rpc_sine_fix14c_results.txt:1-2`; `0.0008` on planted) because `d1` shrinks ~4.6× below its frozen calibration scale `sd_d1`.

### The Single Strongest Reason to Doubt the Round 14 Claim

**Round 14's headline claim ("delta solves both tasks past the bars via error-driven learning in noise, rescuing the model") rests on a fragile, mis-scored planted result and broken controls, while silently replacing Chapter 3's theory:** on 10 fresh unseen seeds, planted drops below the `0.22` bar (`0.217`), its `0.966` motif metric omits half the motif transitions (where accuracy is `0.628`), its `cov` control was invalidated by spatial-instead-of-temporal centering and clamp saturation (true temporal-covariance Hebb reaches `0.199` / `0.939` `m2` with zero error term), untested Lorenz/1D-track benchmarks still lose to trivial persistence, and the winning rule is online dendritic LMS (Urbanczik & Senn, 2014) with **zero bursts (`mean_burst = 0.000`)**, not Chapter 3's burst-confirmation theory.

### Summary Table of All Eight Checks

| # | Check | Verdict | One-line finding |
|---|---|---|---|
| 1 | Delta win real vs. test-set selection | **Refuted (as selection mirage) / Refuted (on 5-seed bar passes)** | Learning over chance is real (`0.467` sine, `0.217` planted on 10 fresh seeds; `0.847` one-hot sine!), but planted drops below `0.22`, smooth-sine ties Markov-1 (`+0.006`, 5/10), and default `alpha=0.3` collapses planted (`0.079`). |
| 2 | Planted `0.221` marginal pass & `0.966` motif score | **Refuted** | `0.221` vs `0.22` is noise ($p=0.90$, drops to `0.217` on fresh seeds); `motif_acc=0.966` omits 1st-step motif transitions (`m1=0.628`, true all-motif `0.789`) and filler is sub-chance (`0.059`). |
| 3 | Byte-identical planted coactivity/burst controls | **Verified (no code bug), with clamp confound** | Driver/code has no shared state (`max |W_coact - W_burst| = 2.132`), but `norm_rows=False` pins `99.25%` / `97.29%` of entries at `+2.0`, locking both onto token 15 (`100%`). Restoring `norm_rows=True` still collapses both (`~0.041`). |
| 4 | Probe `0.917` vs `0.913` ("readout alignment") | **Verified (code parity) / Refuted (mechanism framing & ridge confound)** | Same data/config verified (3 seeds only), but fixed `lam=1.0` penalizes trained `d1` ~19× harder (`||d1||` drops 4.4×); invertible random `W_dist1` makes frozen=trained a Data Processing Inequality tautology, and delta performs online regression $y_2 \mapsto \hat{p}_1$, not orthogonal rotation. |
| 5 | Audit of `fix14` drivers for remaining mismatches | **Refuted (controls not matched to delta)** | Six defects: (1) `cov` uses spatial not temporal centering + ~1000× larger step + clamp (`44%` sine, `18%` planted); (2) `norm_rows=False` clamps `coact/burst`; (3) `mean_burst=0.000` unnoted; (4) `n_out=64` controls missing; (5) `m1` omitted; (6) missing `fix14a/b` results & post-crash `fix14c` edit. |
| 6 | Sine `0.484` vs Markov-1 (`0.459`) & Markov-2 (`0.802`) | **Verified (>1-step learned) / Refuted (1-step ceiling)** | Delta learned full phase (`99.4%` direction on ambiguous middle bins, `96.5%` within ±2 bins). Exact-bin `0.484` is capped by the **oracle `p1` prototype ceiling (`0.563`, $r=0.943$)** from `smooth_sigma=3.0` blur; `split_proto` jumps seed 0 `0.348 -> 0.760` and one-hot `alpha=1.0` hits **`0.847`**. |
| 7 | Untested tasks (var-sine, Lorenz, track) & "learns in noise" | **Refuted (claim premature)** | `sine-fixed` is noiseless; lag-1 planted is 1-step Markov and `<0.22` fresh. 1-seed checks show delta fails persistence on Lorenz (`0.438` vs `0.587`) and 1D track (`0.489` vs `0.776`), though lag-2 motif (`0.208` vs `M1 0.088`) is a genuine bright spot. |
| 8 | Does delta still count as Chapter 3's thesis theory? | **Refuted** | A24 is Urbanczik & Senn (2014) / normalized Widrow-Hoff LMS on a fixed ESN reservoir. All distinctive Chapter 3 mechanisms (burst-gated plasticity, confirmation, Layer V inhibition) are OFF or silent (`mean_burst=0.000`). |

---

## Detailed Checks (1–8)

### Check 1: Is the delta win real, or is it test-set selection?

**Verdict:** **Refuted** that the overall delta win is an artifact of test-set selection; **Refuted** that Round 14's 5-seed bar passes (`planted 0.221 >= 0.22` and `smooth-sine 0.484 > Markov-1 0.459`) survive a clean fresh-seed split.  
**References:** `simulation-results.md:283-299, 869-905, 983-988, 1965, 2019-2031, 2052-2058, 2073-2079`; `rpc_sine_fix14c.py:30-32, 68-82`; `rpc_planted_fix14c.py:28-30, 73-90`; `rpc_sine_fix14c_results.txt:1-3`; `rpc_planted_fix14c_results.txt:1-3`; `rpc_sine_diag_results.txt:1`.

First, correct a recurring misdescription in the brief and `simulation-results.md:1965, 2079`: on `sine-fixed`, Round 14 did **not** use `alpha=1.0`; `rpc_sine_fix14c.py:32` ran **`alpha=0.7, smooth_sigma=3.0`**. Both `smooth_sigma=3.0` (swept in sections 25–27) and `alpha=0.7` (swept in section 29) *were* selected on test seeds `0..4` (`seed+1000`), albeit under the *old* coactivity rule rather than delta. On `planted`, `alpha=1.0` was selected on the cycle task (`simulation-results.md:293-299`) and carried into planted (where `alpha=0.3/0.7/1.0` scored identically at `0.075/0.077/0.073` under burst, `simulation-results.md:379`). Meanwhile `n_out=256`, `full_code=True`, `norm_rows=False`, `s_inh=0.0`, and `eta_d=0.2/0.05` came from `agy-critique.md:221-226` (validated on 1-epoch seed 0/2 checks, not swept across seeds `0..4`).

With only 5 reused seeds (`0..4`), the reported numbers carry very little precision around the bars:
- **R14 `sine-fixed` (`0..4`):** `mean = 0.4844`, `SD = 0.1164`, `SEM = 0.0521`, 95% CI `[0.340, 0.629]` ($t_4 = +0.469, p = 0.6635$ vs `0.46`). Paired against matched Markov-1 on the exact same 5 seeds (`rpc_sine_diag_results.txt:1`: `[0.419, 0.549, 0.438, 0.405, 0.484]`): `delta - M1 = [-0.071, +0.080, +0.099, -0.022, +0.041]`, mean **`+0.0254`** (`SD = 0.0710`, `SEM = 0.0318`, 95% CI `[-0.063, +0.114]`, $t_4 = +0.800, p = 0.4685$; delta loses on 2/5 seeds).
- **R14 `planted` (`0..4`):** `mean = 0.2212`, `SD = 0.0202`, `SEM = 0.0090`, 95% CI `[0.196, 0.246]` ($t_4 = +0.133, p = 0.9006$ vs `0.22`).

To resolve Check 1 directly without waiting for a cancelled Round 15, I ran the exact `fix14c` configurations on **10 completely untouched fresh seeds (`100..109`)** and ran a hyperparameter ablation on fresh **seed `100`**:

| Diagnostic (exact `fix14c` config, 5 epochs, `n_out=256`) | $n$ | Mean | SD | 95% CI | vs. Bar / Markov-1 |
|---|---:|---:|---:|---|---|
| `sine-fixed` (`a=0.7, s=3.0`) — R14 seeds `0..4` | 5 | `0.484` | `0.116` | `[0.340, 0.629]` | `+0.025` vs M1 `0.459` ($p=0.47$, 3/5 wins) |
| **`sine-fixed` (`a=0.7, s=3.0`) — FRESH seeds `100..109`** | **10** | **`0.467`** | `0.085` | `[0.407, 0.528]` | **`+0.006`** vs M1 `0.461` ($t_9=0.37, p=0.72$, **5/10 wins**) |
| `sine-fixed` (`a=0.7, s=3.0`) — Pooled 15 seeds | 15 | `0.473` | `0.092` | `[0.422, 0.524]` | `+0.013` vs M1 `0.460` ($t_{14}=0.84, p=0.41$, 8/15 wins) |
| `planted` (`a=1.0`, one-hot) — R14 seeds `0..4` | 5 | `0.221` | `0.020` | `[0.196, 0.246]` | `+0.001` vs `0.22` ($p=0.90$, 3/5 $\ge 0.22$) |
| **`planted` (`a=1.0`, one-hot) — FRESH seeds `100..109`** | **10** | **`0.217`** | `0.016` | `[0.205, 0.229]` | **`-0.003` FAIL** vs `0.22` ($t_9=-0.60, p=0.56$, **4/10 $\ge 0.22$**) |
| `planted` (`a=1.0`, one-hot) — Pooled 15 seeds | 15 | **`0.218`** | `0.017` | `[0.209, 0.228]` | `-0.002` vs `0.22` ($t_{14}=-0.38, p=0.71$) |

*(Fresh per-seed scores: sine `100..109` = `[0.312, 0.520, 0.442, 0.478, 0.423, 0.597, 0.535, 0.365, 0.520, 0.479]`; matched M1 = `[0.410, 0.481, 0.461, 0.479, 0.466, 0.562, 0.473, 0.390, 0.459, 0.429]`; M2 = `0.795`. Planted `100..109` = `[0.205, 0.224, 0.186, 0.212, 0.215, 0.238, 0.231, 0.204, 0.217, 0.237]`; `motif_acc` mean `0.944` `[0.866..0.975]`; matched M1 = `0.277`.)*

Now look at the **hyperparameter ablation on untouched fresh seed `100`**—which delivers the single biggest surprise of this critique:

| Fresh Seed 100 (`delta-256`, 5 epochs) | `alpha` | `smooth_sigma` | Accuracy | Notes |
|---|---:|---:|---:|---|
| `sine-fixed` (v2 defaults!) | `0.3` | `0.0` (one-hot) | `0.269` | Slow reservoir smears `p1` prototypes |
| `sine-fixed` (R14 config) | `0.7` | `3.0` | `0.312` | **`smooth_sigma=3.0` actively hurts delta!** |
| `sine-fixed` | `0.3` | `3.0` | `0.160` | Both slow leak and smooth blur hurt |
| `sine-fixed` | `0.7` | `2.0` | `0.370` | Less blur $\to$ higher score |
| `sine-fixed` | `1.0` | `3.0` | `0.410` | Blur still caps accuracy |
| **`sine-fixed` (plain one-hot)** | **`0.7`** | **`0.0` (one-hot)** | **`0.741`** | **+0.429 jump from REMOVING `smooth_sigma=3.0`!** |
| **`sine-fixed` (plain one-hot)** | **`1.0`** | **`0.0` (one-hot)** | **`0.875`** | **3-fresh-seed (`100..102`) mean = `0.847` (`[0.875, 0.858, 0.810]`), beating Markov-2 (`0.795`)!** |
| `planted` (v2 default!) | `0.3` | `0.0` | **`0.079`** | **`motif_acc = 0.143` — total collapse at default `alpha=0.3`!** |
| `planted` | `0.7` | `0.0` | `0.210` | `motif_acc = 0.924` |
| `planted` (R14 config) | `1.0` | `0.0` | `0.205` | `motif_acc = 0.941` |

**Exact failure modes on a clean split:**
1. **Threshold winner's curse (already observed above):** With $n=5$, `planted = 0.221` sits `0.13` SEMs above `0.22` and immediately fails on 10 fresh seeds (`0.217 < 0.22`), while `smooth-sine = 0.484` regresses to `0.467` (statistically tied with Markov-1 `0.461`, `5/10` wins).
2. **Brittle dependence on high `alpha` (`0.7–1.0`):** `smooth_sigma=3.0` was *not* helping delta—it was a toxic legacy hyperparameter carried over from unsigned coactivity that actually cut delta's sine accuracy nearly in half (`0.847` one-hot vs `0.467` smooth). By contrast, **high `alpha` IS strictly load-bearing**: any clean protocol starting from the thesis/v2 slow-reservoir default (`alpha=0.3`) without validation-selecting `alpha >= 0.7` collapses planted completely to `0.079` (`motif_acc=0.143`) and cuts sine to `0.269`, because `p1`-prototype separation requires a fast reservoir (`simulation-results.md:283-287`).

---

### Check 2: Are the planted passes (`0.221` vs `0.22`, gain `+0.159` vs `+0.15`) real or noise?

**Verdict:** **Refuted**—`0.221` is statistically indistinguishable from `0.22`, fails replication on 10 fresh seeds (`0.217`), and the headline `motif_acc = 0.966` hides a major metric omission.  
**References:** `simulation-results.md:2050-2059`; `rpc_planted_fix14c.py:17-21, 59-70`; `rpc_planted_fix14c_results.txt:1-3`; `baselines.py:41-51`; `baselines_B.md:10, 29-37`.

1. **Statistical insignificance and failed replication:** On Round 14's 5 seeds (`0..4`), `delta-256` scores `[0.197, 0.230, 0.247, 0.205, 0.227]` (`mean = 0.2212`, `SEM = 0.0090`, 95% CI `[0.196, 0.246]`, $t_4 = +0.133, p = 0.9006$ vs `0.22`; 2/5 seeds fail the bar). Paired gain over frozen control (`[0.065, 0.068, 0.069, 0.052, 0.058]`) is `0.1588 ± 0.0079` SEM (95% CI `[0.137, 0.181]`, $t_4 = +1.120, p = 0.3254$ vs `+0.15`). On 10 fresh seeds (`100..109`), overall accuracy drops to **`0.2169`** (95% CI `[0.205, 0.229]`, **6/10 seeds below `0.22`**), and the pooled 15-seed mean is **`0.2183 < 0.220`**. The `+0.001` pass was pure finite-sample noise.
2. **Hidden metric flaw in `motif_acc` (`rpc_planted_fix14c.py:62-68`):** Each planted 3-token motif `(1,2,3)`, `(5,6,7)`, `(9,10,11)` contains **two** motif transitions:
   - **Transition `m1` (1st $\to$ 2nd token):** `1->2`, `5->6`, `9->10` (~135 test steps/seed), preceded by a random filler token, so $P(\text{motif succ} \mid \text{cue}_1) \approx 0.51$.
   - **Transition `m2` (2nd $\to$ 3rd token):** `(1,2)->3`, `(5,6)->7`, `(9,10)->11` (~124 test steps/seed), preceded by a consistent 2-token motif prefix, so $P(\text{motif succ} \mid \text{prefix}_2) \approx 0.96$.

   `run_planted_motif` (`rpc_planted_fix14c.py:62-68`) scores **ONLY `m2`** and silently drops `m1`! Running an exact 5-seed breakdown (`0..4`) comparing `delta-256` against matched Markov-1 on the identical data exposes **why** overall accuracy stalls at `0.221` instead of Markov-1's `0.281`:

| Seed (`0..4`) | `delta` acc | Matched M1 | `m1` (`1->2`) | `m2` (`12->3`) | All-motif (`m1+m2`) | `cue1` (`prev∈{1,5,9}`) | `cue2` (`prev∈{2,6,10}`) | `oth` (other 10 tokens, 53% of steps) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.197` | `0.273` | `0.522` (134) | `0.966` (119) | `0.731` | `0.274` | `0.460` | `0.046` |
| 1 | `0.230` | `0.286` | `0.627` (134) | `1.000` (123) | `0.805` | `0.311` | `0.505` | `0.061` |
| 2 | `0.247` | `0.292` | `0.748` (143) | `0.977` (130) | `0.857` | `0.394` | `0.507` | `0.064` |
| 3 | `0.205` | `0.274` | `0.598` (132) | `0.967` (122) | `0.776` | `0.307` | `0.460` | `0.045` |
| 4 | `0.227` | `0.281` | `0.642` (134) | `0.921` (127) | `0.778` | `0.321` | `0.471` | `0.081` |
| **Mean** | **`0.221`** | **`0.281`** | **`0.628`** | **`0.966`** | **`0.789`** | **`0.321`** | **`0.481`** | **`0.059`** |
| *M1 (seed 0)* | *—* | *`0.273`* | *`1.000`* | *`1.000`* | *`1.000`* | *`0.504`* | *`0.474`* | *`0.083`* |

**Mechanism of the `0.060` gap to Markov-1:**
- `delta-256` nails `m2` (`0.966`, `cue2 = 0.481` $\approx$ M1's `0.474`), because the deterministic 2-step prefix `(1,2)` stabilizes the recurrent reservoir state and has near-100% transition probability.
- It misses **37.2%** of `m1` transitions (`0.628` vs M1 `1.000`; `cue1 = 0.321` vs M1 `0.504`), because random preceding filler tokens perturb `r1/r2` via `W_res @ r` (active even at `alpha=1.0`, `rho=0.9`) while `1->2` occurs only ~51% of the time after `1`. That alone accounts for **~69%** of the seed-0 gap to Markov-1!
- On the remaining 53% non-cue steps (`oth`), `delta-256` scores **`0.059`** (below uniform chance `0.0625`, vs M1's `0.083`, which exploits the higher marginal frequency of motif-start tokens `1,5,9`), because on uninformative steps `d1` regresses to the grand mean `E[p1]` and cosine argmax locks onto whichever random non-motif prototype happens to align with `E[p1]` (e.g., token 15 on 18.5% of all steps on seed 0). That accounts for **~26%** of the gap.

**What is needed to call planted solved:**
1. At least **20 fresh unseen seeds** with the **lower 95% CI bound** strictly above `0.22` (and ideally reaching `>= 0.25`, i.e., ~90% of Markov-1 `0.278`, which may simply require smaller `eta_d` / LR decay or centered/prior-calibrated prototypes rather than `eta_d=0.05` SGD noise).
2. Separate reporting of **`m1 >= 0.85`** and **`m2 >= 0.90`** (not `m2` alone masquerading as "motif-only"), plus `oth >= 0.0625`.
3. Passing **motif lag-2** (where Markov-1 is capped near chance `~0.08–0.09`), proving multi-step sequence learning in noise rather than just 1-step transition lookup.

---

### Check 3: Why do planted coactivity and burst controls give byte-identical per-seed scores?

**Verdict:** **Verified** that the driver/code path is bug-free (no shared state, no dict leak, distinct weight matrices); **however**, the identical scores are an artifact of **`norm_rows=False` clamping both `W_dist1` matrices to `+2.0`** (>97% saturated).  
**References:** `rpc_planted_fix14d.py:20-34`; `rpc_planted_fix14c.py:32-44`; `rpc_planted_fix14d_results.txt:1-2`; `rpc_v2.py:422-435, 1036-1039, 1091-1116`; `simulation-results.md:2063-2067`.

**Code audit:** `rpc_planted_fix14d.py:24-31` passes distinct dicts `{"learn_rule": "coactivity", "eta_b": 0.05}` and `{"learn_rule": "burst", "eta_b": 0.05}` into `run_planted_motif(seed, **BASE, **extra)`. Inside `run_planted_motif` (`rpc_planted_fix14c.py:32-44`), both `torch.manual_seed(seed)` and `np.random.seed(seed)` are reset, `make_data` is regenerated identically, `merged = {**defaults, **kw}` allocates a fresh dictionary before `merged.pop(...)`, and a brand-new `RPCNetwork(**merged)` is constructed. There is no shared buffer, mutated default dict, or silent fallback.

**Direct forensic inspection of `W_dist1` on planted seed 0 (5 epochs, `n_out=256`):**

| Condition (seed 0, `n_out=256`) | `acc` | `m2` | `W == +2.0` | `Wmean` | `Wstd` | `uniq` preds | `top` pred (share) | `burst` | `||d1||` |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `coactivity` (`norm_rows=False`, R14) | `0.0400` | `0.0000` | **`99.25%`** | `1.991` | `0.119` | **`1`** | `tok 15 (100.0%)` | `0.1670` | `3254.7` |
| `burst` (`norm_rows=False`, R14) | `0.0400` | `0.0000` | **`97.29%`** | `1.952` | `0.301` | **`1`** | `tok 15 (100.0%)` | `0.1650` | `3222.5` |
| `coactivity` (`norm_rows=True`, A5 restored) | `0.0409` | `0.0000` | `0.00%` | `0.023` | `0.021` | `3` | `tok 15 (99.8%)` | `0.1670` | `48.6` |
| `burst` (`norm_rows=True`, A5 restored) | `0.0417` | `0.0000` | `0.00%` | `0.023` | `0.022` | `2` | `tok 15 (99.0%)` | `0.1650` | `48.1` |

- **Are the two networks identical?** **No:** `max |W_coact - W_burst| = 2.1322`, `np.array_equal == False`.
- **Why are all 5 per-seed scores byte-identical?** Because `norm_rows=False` disables A5 row-norm preservation (`rpc_v2.py:1114-1116`), leaving only `.clamp_(-2.0, 2.0)` (`line 1113`). Over 23,000 steps of strictly non-negative outer products with dense `full_code` (`mean_y2 ≈ 0.40`), `coactivity` pins **`99.25%`** of all 65,536 entries of `W_dist1` at `+2.0`. In `burst`, as soon as `W_dist1` becomes slightly positive, `||d1||` explodes (reaching `3222` against frozen pre-learning `sd_d1 = 0.3318`), so `d1 / sd_d1 > 1.0` holds on virtually every unit—collapsing `b1 = (p1n > 1) & (d1n > 1)` (`rpc_v2.py:1039`) into `(p1n > 1)` (`burst = 0.1650` vs `coact 0.1670`) and pinning **`97.29%`** of `W_dist1` at `+2.0` too. For any vector $y_2 \ge 0$, `2.0 * ones @ y2` points strictly along $(1,1,\dots,1)$, selecting the exact same constant prototype on 100.0% of test steps (`token 15` on seed 0, `acc = 0.0400`; and on seed 3, whichever single motif successor `3/7/11` has the largest prototype entry sum, yielding `macc = 0.295 ≈ 1/3`).
- **Does restoring `norm_rows=True` rescue them?** **No.** As shown above (and identically on sine seed 0: `coact_nrT = 0.0425`, `burst_nrT = 0.0214`), restoring A5 eliminates clamp saturation (`0.00%`) and breaks the byte-exact identity, but positive-orthant DC collapse still traps both rules at chance (`99.0–99.8%` constant prediction on token 15).

---

### Check 4: Audit of the ridge probe claim (`0.917` frozen vs `0.913` trained = "readout alignment")

**Verdict:** **Verified** on train/test data and training-config parity; **Refuted** on the "readout alignment" framing and **confounded** by an unstandardized scale mismatch in ridge regularization.  
**References:** `rpc_sine_fix14c.py:19-23, 30-65`; `rpc_sine_fix14c_probe.py:1-13`; `rpc_sine_fix14c_probe_results.txt:1-2`; `rpc_simple.py:60-68`; `rpc_v2.py:667-670, 676-734`; `simulation-results.md:2033-2040`.

1. **Data and config parity (`probe_delta` vs `run_task`) — VERIFIED:**
   - `probe_delta` (`rpc_sine_fix14c.py:34-65`) seeds torch/numpy identically (`lines 35-36`), generates `make_sine_fixed(50, seed=seed)` and `make_sine_fixed(20, seed=seed+1000)` (`lines 37-38`, matching `rpc_simple.py:64-65` since `length=200, K=32` are defaults), instantiates `RPCNetwork(K=32, n_out=256, ...)` with identical defaults (`theta_p=theta_d=1.0, eta_b=eta_c=0.05, n_res=200`), calls `calibrate(train)` and `build_prototypes(train, proto_n=50)`, and trains for 5 epochs with `use_confirmation=False` (`lines 44-49`).
   - **Calibration is NOT a leakage confound for frozen `d1`:** `calibrate(train)` never modifies `W_dist1` or any reservoir weight; its only touchpoint on `d1` is `sd_p2` inside `_y2_from_p2n(p2 / sd_p2)`, which cancels out inside `full_code`'s per-step standardization `(p2n - mean) / (std + 1e-6)` (`rpc_v2.py:668`).
   - **Minor reporting gap:** `probe_delta` ran only **3 seeds** (`range(3)`, `rpc_sine_fix14c_probe.py:10`), whereas the headline `0.484` is a 5-seed mean.
2. **Confound 1 — Scale-dependent ridge regularization (`rpc_sine_fix14c.py:63`):**
   `probe_delta` runs unstandardized, intercept-free ridge `Ftr.T @ Ftr + 1.0 * I` on raw `d1`, ignoring that delta training **shrinks `||d1||` by 4.4×** (from `5.47–5.77` frozen down to `1.23–1.48` trained, because `W_dist1` moves from random `0.5/sqrt(256)` init to match `p1`'s much smaller norm)! Thus fixed `lam=1.0` applies a **~19× heavier effective L2 penalty** ($\lambda / \|d_1\|^2$) to the trained drive than to the frozen drive! Sweeping `lam` and comparing standardized ridge+intercept across seeds `0..2` proves the confound directly:

| Seed | `||d1_fr||` vs `||d1_tr||` | `lam=1e-4` (`tr / fr / y2`) | `lam=1e-2` (`tr / fr / y2`) | **`lam=1.0` (R14: `tr / fr / y2`)** | `lam=100.0` (`tr / fr / y2`) | **Std+bias `lam=1.0` (`tr / fr / y2`)** | `cond(Wf)` | `rank99(Wt)` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `5.773` vs `1.228` (4.7×) | `0.889 / 0.890 / 0.889` | `0.889 / 0.887 / 0.889` | `0.884 / 0.890 / 0.887` | `0.493 / 0.753 / 0.870` | `0.888 / 0.889 / 0.889` | `903` | `164` |
| 1 | `5.550` vs `1.477` (3.8×) | `0.954 / 0.953 / 0.953` | `0.953 / 0.949 / 0.953` | `0.942 / 0.939 / 0.954` | `0.700 / 0.905 / 0.919` | `0.954 / 0.952 / 0.953` | `2855` | `167` |
| 2 | `5.468` vs `1.240` (4.4×) | `0.935 / 0.836 / 0.936` | `0.930 / 0.929 / 0.936` | `0.912 / 0.923 / 0.937` | `0.664 / 0.894 / 0.914` | `0.930 / 0.929 / 0.935` | `2184` | `160` |

   The apparent `-0.004` dip (`0.913` vs `0.917`) in `rpc_sine_fix14c_probe_results.txt` is purely this `4.4×` scale penalty at `lam=1.0` (watch `tr` plunge to `0.493–0.700` at `lam=100.0`); once standardized, trained slightly edges frozen (`0.924` vs `0.923`).
3. **Confound 2 — Why "readout alignment" / "rotates `d1`" is mathematically misleading (`simulation-results.md:2039`):**
   - Look at the `y2` column above: probing `y2_prev` directly gives **`0.889 / 0.953 / 0.936`**—identical to both frozen and trained `d1`! Because frozen $W_{\text{dist1}}^{(0)} \in \mathbb{R}^{256 \times 256}$ is an **almost surely invertible** Gaussian matrix, $\operatorname{span}(W_{\text{dist1}}^{(0)} y_2) = \operatorname{span}(y_2)$ identically. By the linear Data Processing Inequality, **no** linear matrix $W_{\text{dist1}}$ can *ever* exceed the linear decode of $y_2$ (modulo regularization conditioning)—so `rpc_sine_fix14c.py:22-23`'s pre-run prediction (*"the trained drive should decode well above the frozen drive"*) was mathematically impossible!
   - Calling what delta does merely *"readout alignment"* or *"rotating `d1` into prototype space"* (`simulation-results.md:2039`, `README.md:39-40`) mischaracterizes the geometry: delta does **not** perform an orthogonal rotation. It performs **online multivariate linear regression** $W_{\text{dist1}} \approx \arg\min_W \mathbb{E}[\|p_1(t) - W y_2(t-1)\|^2]$ (compressing norm 4.4× and effective 99%-energy rank from 256 to `160–167`, achieving $R^2(p_1) = 0.68–0.78$ and $\cos(d_1, p_1) = 0.914–0.953$). In the frozen probe, `W_dist1` predicts nothing; the **offline ridge probe itself** learns the entire predictive regression $W_{\text{ridge}} W_{\text{dist1}}^{(0)}$. Under delta, `W_dist1` actually **learns the predictive mapping** online so no ridge readout is needed.

---

### Check 5: Audit of the `fix14` drivers for remaining config mismatches and bugs

**Verdict:** **Refuted** that the delta cell differs from its controls *only* in the learning rule. Six concrete mismatches/defects remain in the Round 14 code and record.  
**References:** `rpc_v2.py:416-435, 1036-1039, 1060-1122`; `rpc_sine_fix14c.py:30-32`; `rpc_sine_fix14d.py:7-15, 20-29`; `rpc_planted_fix14c.py:28-30, 32-70, 78-88`; `rpc_planted_fix14d.py:6-15, 20-28`; `rpc_sine_fix14c_results.txt:1-11`.

1. **CRITICAL BUG — `cov` control (`rpc_v2.py:1076-1089`) is spatially centered (not temporally centered) AND ~1,000× oversized, causing heavy clamp saturation:**
   - Look at `rpc_v2.py:1082-1085`:
     ```python
     post = p1n - p1n.mean()
     pre = self.y2_prev - self.y2_prev.mean()
     self.W_dist1 += self.eta_b * torch.outer(post, pre)
     self.W_dist1.clamp_(-self.w_max, self.w_max)
     ```
     `p1n.mean()` and `y2_prev.mean()` subtract the **spatial mean across the 256 neurons on step $t$**, NOT each neuron's **temporal mean** across time ($\bar{p}_{1,i}, \bar{y}_{2,j}$, Sejnowski 1977)! Per-neuron temporal DC offsets remain completely uncentered, creating a huge rank-1 temporal DC matrix $(\bar{p}_1 - \langle\bar{p}_1\rangle)(\bar{y}_2 - \langle\bar{y}_2\rangle)^\top$.
   - Simultaneously, `delta` (`rpc_v2.py:1067-1070`) updates with raw `p1` (`sd ≈ 0.07–0.12`) divided by `||y2_prev||^2 + 1 ≈ 129`, giving per-entry step sizes `~1e-4` (`max |W| = 0.10–0.125`, **0.0% clamped**). By contrast, `cov` uses standardized `p1n` (8–14× larger than `p1`) with **unnormalized** `eta_b = 0.15` (sine) / `0.05` (planted)—roughly **1,000× larger** per-step updates—slamming **`43.95%`** of `W_dist1` into `±2.0` on sine seed 0 (`Wstd = 1.879`) and **`17.89%`** on planted seed 0 (`Wstd = 1.825`, `||d1|| = 690` vs delta's `1.65`)!
   - **Impact when fixed to true temporal covariance Hebbian learning** ($\Delta W = \eta (p_1 - \bar{p}_1)(y_2 - \bar{y}_2)^\top$, raw `p1`, no clamp, **zero error term**):
     - **Planted motifs (5 seeds `0..4`, `n_out=256`):** jumps from Round 14's `0.154` (`m2=0.645`) to **`0.199` overall (`[0.184, 0.211, 0.208, 0.202, 0.191]`) and `0.939` on `m2` (`[0.958, 0.992, 0.985, 0.975, 0.787]`)**—matching **90%** of delta's overall score (`0.221`) and **97%** of delta's `m2` score (`0.966`) with **NO error feedback at all!** (Delta's genuine advantage on planted shrinks to `m1`: `0.628` vs temporal-`cov` `0.449`, i.e., `+0.022` overall!)
     - **Sine-fixed (seed 0, `n_out=256`):** doubles from Round 14's `0.114` to **`0.222`**, still well below delta's `0.348` on seed 0—confirming that the error term's decorrelation ($\Sigma_{y_2 y_2}^{-1}$) *is* genuinely load-bearing on correlated continuous inputs, even though Round 14 overstated its margin on planted!
2. **`norm_rows=False` + `w_max=2.0` clamp asymmetry for `coactivity` and `burst` (`rpc_sine_fix14d.py:20`, `rpc_planted_fix14d.py:20`):** Setting `norm_rows=False` is a no-op for self-bounded `delta` (`0.0%` clamped), whereas it pins `72.9%` / `45.4%` (sine) and `99.25%` / `97.29%` (planted) of `coactivity`/`burst` weights at `+2.0`. While our check confirmed restoring `norm_rows=True` still leaves `coact`/`burst` at chance (`0.042/0.021` sine, `0.041/0.042` planted), the `fix14d` comparison was still confounded by hard clamp saturation.
3. **Unreported complete death of the burst mechanism under `delta` (`rpc_sine_fix14c_results.txt:1-2`):** Both `delta-nout64` and `delta-nout256` print **`mean_burst=0.000`** on sine (and `0.0008` in our planted seed-0 check), vs `0.024` frozen and `0.110–0.152` for controls. Cause: `rpc_v2.py:1036-1039` divides `d1` by frozen pre-learning `sd_d1 ≈ 0.34–0.36`, while delta shrinks `d1` to `p1`'s scale (`sd ≈ 0.07–0.09`), keeping `d1 / sd_d1` permanently below `theta_d = 1.0`. Section 43 never mentions that `delta` extinguishes bursting entirely.
4. **Incomplete "2×2 factorial" (`rpc_sine_fix14d.py:7`, `rpc_planted_fix14d.py:6`, `simulation-results.md:2048`):** Both docstrings and section 43.4 claim *"AGY's 2x2 factorial"* (`learn_rule` $\times$ `n_out ∈ {64, 256}`), yet `coactivity`, `burst`, `cov`, and `no-plasticity` were **never run at `n_out=64`** on either task—only `delta` ran at `n_out=64`.
5. **Metric omission and logging gaps in `rpc_planted_fix14c.py`:** `run_planted_motif` (`lines 62-68`) omits `m1` transitions, fails to log `mean_burst` and `mean_y2` (masking planted `mean_burst = 0.0008`), and confusingly names the `motif_n` list `bursts` (`lines 78, 83, 88`).
6. **File/provenance discrepancies:** `rpc_sine_fix14c_results.txt:4-11` contains the crash traceback from the pre-fix `rpc_sine_fix14c.py` (`RPCNetwork(..., proto_n=50, **DELTA)`), meaning `rpc_sine_fix14c.py` was modified after generating `rpc_sine_fix14c_results.txt` (though lines 1–3 came from `run_task`, which was unaffected by the `probe_delta` bug). Also `rpc_sine_fix14a_results.txt` and `rpc_sine_fix14b_results.txt` are missing from the review directory.

---

### Check 6: Sine `0.484` beats Markov-1 (`0.459`) but is far below Markov-2 (`0.802`) — one step of history, or more?

**Verdict:** **Verified** that delta learned **far more than 1 step of history (true multi-step phase and velocity)**; **Refuted** that `0.484` reflects a 1-step learning ceiling.  
**References:** `simulation-results.md:731-737, 2024-2032`; `rpc_sine_diag.py:21-34`; `rpc_sine_diag_results.txt:1-4`; `rpc_sine_fix14c_results.txt:2`; `rpc_v2.py:736-778, 854-864`.

Superficially, `0.484` vs Markov-1 `0.459` (and `0.467` vs `0.461` on 10 fresh seeds) looks like a 1-step model. **That impression is completely wrong.** Inspecting direction accuracy, error distance, and the **oracle `p1` ceiling** (feeding the *true noiseless* `p1(t)` directly into `predict_from_distal`) across all 5 Round 14 seeds (`0..4`) reveals what actually happened:

| Seed (`0..4`) | `delta` acc | **`oracle_p1`** (true `p1` $\to$ proto) | `cos(d1, p1)` / $R^2$ | `delta` `within-1` / `within-2` | `delta` MAE (bins) | **`delta` `mid_dir`** (bins 4–27) | `M1` `dir` / `within-1` | `adj_cos` (`p1` protos) | `M2` (full backoff) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.348` | **`0.415`** | `0.925` / `0.760` | `0.819` / `0.977` | `0.87` | **`0.998`** | `0.433` / `0.550` | `0.924` | `0.789` |
| 1 | `0.629` | **`0.671`** | `0.949` / `0.784` | `0.848` / `0.963` | `0.59` | **`0.993`** | `0.548` / `0.660` | `0.968` | `0.822` |
| 2 | `0.537` | **`0.644`** | `0.935` / `0.724` | `0.911` / `0.975` | `0.63` | **`0.987`** | `0.420` / `0.523` | `0.962` | `0.783` |
| 3 | `0.383` | **`0.417`** | `0.953` / `0.772` | `0.784` / `0.961` | `0.88` | **`0.996`** | `0.407` / `0.535` | `0.959` | `0.771` |
| 4 | `0.526` | **`0.670`** | `0.914` / `0.681` | `0.844` / `0.948` | `0.70` | **`0.996`** | `0.455` / `0.574` | `0.945` | `0.828` |
| **Mean** | **`0.485`** | **`0.563`** | **`0.935` / `0.744`** | **`0.841` / `0.965`** | **`0.73`** | **`0.994`** | **`0.453` / `0.568`** | **`0.952`** | **`0.799`** |

*(Note: `rpc_sine_diag.py:21-34` skips unseen contexts in `markov_acc`, yielding `M2 = 0.802`; proper backoff `baselines.markov_predict` gives `M2 = 0.799`, virtually identical.)*

1. **Delta learned full multi-step phase/velocity on EVERY seed:** Even on the worst seed (`seed 0`, exact-bin `0.348 < M1 0.419`), `delta-256` predicts the **correct direction on `99.8%` of ambiguous middle bins** (5-seed mean **`99.4%`**, overall `97.3%` vs Markov-1's `45.3%`), lands within ±1 bin **`84.1%`** of the time, and within ±2 bins **`96.5%`** (`MAE = 0.73` bins, vs Markov-1 `within-1 = 56.8%`). Paradoxically, on exact-bin accuracy in the middle bins (`4..27`), delta scores `0.350` vs Markov-1's `0.430` while getting the direction right **99.4%** of the time!
2. **Why exact-bin accuracy was capped at `0.484` with huge variance (`0.348–0.629`):** Look at `oracle_p1`—even the **true noiseless `p1(t)`** only scores **`0.563`** mean (`0.415` on seed 0, `0.417` on seed 3), correlating at **Pearson $r = 0.943$** with delta's exact-bin score! Delta actually reaches **`86.0%` of the theoretical oracle `p1`-prototype ceiling**. Why is `oracle_p1` so low? Because at `alpha=0.7, smooth_sigma=3.0`, adjacent bin prototypes have **`adj_cos = 0.952`**, AND `p1(t)` carries 30% previous state (`rising p1(k)` differs from `falling p1(k)`), so a single averaged bin prototype misclassifies even exact `p1(t)` by ±1 bin ~44% of the time!
3. **The settling tests (already executed, both decisive):**
   - **Test A (`split_proto=True` on worst seed 0, keeping `alpha=0.7, smooth_sigma=3.0`):** Splitting `p1` prototypes by rising/falling (`A15`) jumps `oracle_p1` from `0.415` to **`0.988`** and jumps the exact same delta-256 drive from **`0.348` to `0.760`** (nearly matching Markov-2's `0.789`)!
   - **Test B (Remove `smooth_sigma=3.0`, run plain one-hot `smooth_sigma=0.0, alpha=1.0`, unmodified single-prototype cosine readout):** Across 3 fresh unseen seeds (`100..102`), `delta-256` scores **`0.847` (`[0.875, 0.858, 0.810]`)**, decisively **beating Markov-2 (`0.795`)** because one-hot `alpha=1.0` keeps `p1` prototypes nearly orthogonal while `W_res @ r` (`rho=0.9`) still retains multi-step velocity history in `y2`!

---

### Check 7: Untested tasks (variable sine, Lorenz, 1D track) — is sine/planted enough to claim "learns in noise"?

**Verdict:** **Refuted.** Sine-fixed plus lag-1 planted motifs cannot support the broad claim that the model *"learns in noise"* (`README.md:27`), and exploratory checks on the untested tasks show real remaining failures.  
**References:** `simulation-results.md:2079`; `README.md:26-30, 243-247`; `rpc_spatial.py:21-55`; `rpc_track.py:27-49`; `baselines.py:54-74, 107-135`; `baselines_B.md:10-15`.

Why Round 14's two tasks are insufficient:
1. **`sine-fixed` has zero noise:** It is a noiseless, single-frequency deterministic periodic wave (`f=0.05`).
2. **`planted motifs` (lag 1) is noisy, but 1-step Markov and unresolved:** It falls below `0.22` on 10 fresh seeds (`0.217`), only requires 1-step transitions (`Markov-1 0.278 ≈ LSTM 0.283`), and true temporal-covariance Hebb already achieves `0.199` (`0.939 m2`) on it without any error rule.
3. **Exploratory 1-seed checks (`seed 0`, 5 epochs, `n_out=256`, `delta`) on ALL FOUR untested benchmarks** (strictly 1-seed diagnostics, not 5-seed verdicts):

| Untested Task (seed 0, 5 epochs, `n_out=256`) | `delta` (`a=0.7, s=3.0`) | `delta` (`a=1.0, s=0.0` one-hot) | Frozen ctrl | Persistence | Markov-1 | Markov-2 | Diagnostic readout |
|---|---:|---:|---:|---:|---:|---:|---|
| **Variable sine** (`rpc_spatial.py`) | `0.292` | `0.229` | `0.034` | `0.130` | `0.282` | `0.529` | Narrowly beats M1 with smooth; far below M2 |
| **Lorenz `dt=0.02`** (`rpc_spatial.py`) | `0.438` | `0.395` | `0.048` | **`0.587`** | `0.569` | `0.632` | Huge gain over old `0.093`, **FAILS persistence (`-0.149`)** |
| **1D track** (`rpc_track.py`) | `0.489` | `0.597` | `0.002` | **`0.776`** | `0.776` | `0.776` | Huge gain over old `0.098`, **FAILS persistence (`-0.179`)** |
| **Motif lag-2** (`1,X,2,X,3`, `baselines.py`) | `0.154` (`a=0.7` one-hot) | **`0.208`** (`a=1.0` one-hot) | — | — | **`0.088`** | `0.283` | **Strong bright spot: 2.4× Markov-1 in noise!** (`a=0.3` drops to `0.090`) |

**What this shows:**
- **Negative:** On the two slow continuous tasks (`Lorenz dt=0.02` and `1D track`), `delta-256` gains massively over frozen chance (`+0.39` to `+0.59`) yet **still fails to beat trivial persistence** (`0.438/0.395` vs `0.587`; `0.489/0.597` vs `0.776`), and on variable-frequency sine it barely ties Markov-1 (`0.292` vs `0.282`). Notice that `rpc_spatial.py`'s Lorenz (`dt=0.02`, 20 train trajectories) is both under-sampled in data and over-sampled in time (dominated by persistence), whereas `baselines.py`'s Lorenz (`dt_eff=0.10`, 200 train trajectories) has persistence `0.170` (`baselines_B.md:14`).
- **Positive:** **`motif-lag2` at `alpha=1.0` (`0.208` vs `M1 = 0.088`)** is actually a *far* stronger test of "learning multi-step structure in noise" than lag-1 planted motifs—because random filler separates every motif token, defeating Markov-1 completely while `W_res @ r` preserves the 2-step trace for `delta`!

**Cheapest next experiment that would change the verdict:** A single ~3-minute CPU run on **10 fresh seeds** of **`motif-lag2` (and `motif-lag5`)** plus **`baselines.py` Lorenz (`dt_eff=0.10`, 200 train)** and **variable sine**, comparing `delta-256` against **fixed temporal-covariance Hebb**, **frozen control**, **persistence**, **Markov-1/2**, and **ESN** on identical data (detailed in the final section).

---

### Check 8: The thesis question — does A24 still count as Chapter 3's theory?

**Verdict:** **Refuted.** A24 (`delta`) is **not** Chapter 3's random predictive coding theory; it is **local dendritic prediction-error learning (Urbanczik & Senn, 2014; normalized Widrow & Hoff, 1960) on top of a fixed random reservoir (Echo State Network)** wearing the RPC label.  
**References:** `ch3.txt:1352-1395, 1447-1561`; `rpc_v2.py:398-421, 1036-1075, 1123-1135`; `rpc_sine_fix14c.py:30-32`; `rpc_sine_fix14c_results.txt:1-2`; `simulation-results.md:2013-2016, 2079`.

Chapter 3 (`ch3.txt:1352-1561`) rests on four architectural/mechanistic pillars. Auditing each pillar against Round 14's winning A24 configuration gives a stark verdict:

| Chapter 3 Pillar (`ch3.txt`) | Status in A24 (`fix14c`) | Survives? |
|---|---|---|
| **1. Fixed random expansion + recurrent reservoir** (DG/L4 + CA3/L2/3 acting as an ESN/liquid state machine, explicitly citing Grigoryeva & Ortega, 2018; `ch3.txt:1352-1395`) | Kept 100% fixed random (with `n_out=256 >= n_res=200` and `full_code=True`). | **YES — survives**, but this is standard Reservoir Computing / ESN, not Chapter 3's novel contribution. |
| **2. Random distal collisions trigger Layer V dendritic bursts that drive Hebbian strengthening** (`ch3.txt:1447-1490`) | Replaced by signed error `err = p1 - d1` (`rpc_v2.py:1067-1070`). Worse, `b1` is not only unused—it is **literally extinguished (`mean_burst = 0.000` on sine, `0.0008` on planted)**! | **NO — dead.** Unsigned burst-gated learning collapses to chance at both `n_out=64` (`0.022`) and `n_out=256` (`0.028` sine, `0.059` planted, with or without `norm_rows`). |
| **3. Reciprocal EC $\to$ CA1 confirmation signal strengthening CA3 $\to$ CA1 synapses** (`ch3.txt:1500-1536`) | Explicitly turned **OFF** (`use_confirmation=False`) in every Round 14 delta run. | **NO — unverified / abandoned.** Never tested with reciprocal `W_conf = W_dist1.T`; turned off to make delta work. |
| **4. Layer V $\to$ Layer II/III predictive inhibition driving expectation suppression / omission responses & hierarchical routing** (`ch3.txt:1540-1561`) | Explicitly turned **OFF** (`s_inh=0.0`) in every Round 14 delta run; no omission response, no error population, no hierarchy. | **NO — absent.** |

**Exact mechanism comparison (Urbanczik & Senn, 2014 vs. A24 vs. Chapter 3):**
- In **Urbanczik & Senn (2014, *Neuron* 81(3):521–528)**, dendritic synapses minimize the local prediction error between somatic activity $S(t)$ (driven by proximal inputs) and the dendritic prediction $\phi(V_d^*)$: $\Delta w_i = \eta [S(t) - \phi(V_d^*)] \frac{\partial V_d}{\partial w_i}$.
- A24 (`rpc_v2.py:1067-1070`), $\Delta W_{\text{dist1}} = \frac{\eta_d}{\|y_2\|^2 + 1} (p_1 - d_1) y_2^\top$, is literally the linear rate-coded, input-normalized special case of Urbanczik & Senn (2014) (equivalently, normalized Widrow-Hoff LMS, 1960) on top of a fixed random two-reservoir basis.
- Note that Chapter 3 (`ch3.txt:1471-1489`) explicitly rejects requiring signed error populations during learning in favor of *random collisions confirmed by bursts*. Once you replace coincidence-burst potentiation with signed dendritic error `p1 - d1`, turn off confirmation, turn off Layer V inhibition, and silence bursting to `0.000`, **nothing distinctive to Chapter 3 remains in the plasticity mechanism**.

**Plain statement of what survives and what does not:**
- **Survives:** (a) A fixed random multi-module reservoir (`n_out >= n_res`, graded code, high leak `alpha >= 0.7`) preserves rich temporal features without learning `W_in`, `W_res`, `W_12`, or `W_prox`; (b) a **strictly local, non-backprop** synaptic rule on `W_dist1` alone can align the distal prediction `d1` onto the proximal drive `p1` (`cos(d1, p1) = 0.91–0.95`) so the unmodified `p1`-prototype cosine readout achieves high predictive accuracy (even `0.847` on one-hot `sine-fixed`, beating Markov-2!) with zero external trained readout weights.
- **Does not survive:** (a) Confirmation-only / burst-gated Hebbian sequence learning; (b) the claim that *"only the error term"* works on lag-1 planted motifs (true temporal covariance Hebb reaches `0.199` / `0.939 m2` with zero error feedback, though still using signed zero-mean LTD+LTP rather than unsigned bursts); (c) the EC–CA1 confirmation loop and Layer V predictive-routing/omission predictions, neither of which is active or validated in the working model; (d) slow-reservoir (`alpha=0.3`) prototype decoding.

---

## The Single Cheapest Next Experiment to Settle the Open Questions

If Reza wants one decisive, low-cost (~3–5 minutes CPU, single-threaded) experiment that settles **all remaining open empirical questions** (test-set selection, planted replication, temporal-covariance vs. delta, and multi-step learning in noise + chaotic dynamics) cleanly:

1. **Protocol (eliminates test-set selection and seed confounding):**
   - **Train / Validation / Test split by sequence** on **20 fresh seeds** (`200..219`, disjoint from `0..4` and `100..109`).
   - Select `(alpha ∈ {0.7, 1.0}, smooth_sigma ∈ {0.0, 2.0, 3.0}, eta_d ∈ {0.02, 0.05, 0.2})` **strictly on the validation split** (seeds `200..204` val sequences), freeze the winner per task, then evaluate **once** on the held-out test split across all 20 fresh seeds.
   - Report **mean, median, IQR, 95% CI, and paired per-seed differences** against matched baselines on the **identical** test sequences.
2. **Tasks (4 targeted benchmarks, ~3–5 min total):**
   - `planted lag-1` (200 train / 50 val / 50 test, `K=16`): report overall acc, **`m1`**, **`m2`**, and **`oth`**.
   - **`planted lag-2`** (`make_data_lag(..., lag=2)`, 200/50/50, `K=16`): the decisive multi-step-in-noise test where Markov-1 fails (`~0.08–0.09`) and our seed-0 check hit `0.208`.
   - **`sine-fixed`** (50/20/20, `K=32`): letting validation pick `smooth_sigma=0.0, alpha=1.0` instead of the sabotaging `sigma=3.0`.
   - **`baselines.py` Lorenz (`dt_eff=0.10`, 200/50/50, `K=32`) + variable sine**: report **both** exact-bin accuracy AND continuous MSE (and `within-1-bin` accuracy), compared against persistence (`0.170` on `dt_eff=0.10` Lorenz), Markov-1, Markov-2, and matched ESN.
3. **Matched Conditions (`n_out=256`, `full_code=True`, `use_confirmation=False`, `s_inh=0.0`, unmodified `p1`-prototype cosine readout):**
   - `delta` (`A24`)
   - **Fixed `temporal-cov`** ($\Delta W = \eta (p_1 - \bar{p}_1)(y_2 - \bar{y}_2)^\top$ using online EMA temporal means, raw `p1`, no clamp saturation) — cleanly isolating true covariance Hebbian (no error term) from error-driven whitening
   - `burst` with **`norm_rows=True`** (fair thesis control without clamp saturation)
   - `no-plasticity` (frozen control)
4. **Decisive Pass Bars (20-seed lower 95% CI bound, held-out test):**
   - `planted lag-1`: overall **lower 95% CI > `0.22`** (target `>= 0.25`), **`m1 >= 0.80`**, **`m2 >= 0.90`**, beating `temporal-cov` on paired seeds.
   - `planted lag-2`: overall **>= `0.20`** and **>= `+0.10` over Markov-1 (`~0.085`)**—this single result would establish genuine multi-step learning in noise.
   - `sine-fixed`: **>= `0.80`** (matching/beating Markov-2 `0.795` with unmodified `p1`-prototype cosine readout and no smoothing hack).
   - `Lorenz (dt_eff=0.10)` & `var-sine`: beat **both persistence and Markov-1** on exact-bin accuracy AND continuous MSE, with `>= +0.15` gain over frozen control.
5. *(Optional, only if attempting to test Chapter 3 itself rather than Urbanczik-Senn LMS):* Re-enable `W_conf = W_dist1.T` (reciprocal confirmation) and re-calibrate `sd_d1` dynamically (fixing `mean_burst = 0.000`), then test whether `s_inh > 0` at runtime produces an omission response (`ch3.txt:1540-1561`). Until that exists, the working model should be called **local dendritic LMS on a fixed random reservoir**, not Chapter 3's random predictive coding.

---

## References Verified

- **Chen, Y., Zhang, H., Cameron, M., & Sejnowski, T. J. (2024).** Predictive sequence learning in the hippocampal formation. *Neuron*, 112(15), 2645–2658.e4 (preprint bioRxiv, 2022). DOI: `10.1016/j.neuron.2024.05.024`.
- **Grigoryeva, L., & Ortega, J.-P. (2018).** Echo state networks are universal. *Neural Networks*, 108, 495–508.
- **Sejnowski, T. J. (1977).** Storing covariance with nonlinearly interacting neurons. *Journal of Mathematical Biology*, 4, 303–321.
- **Urbanczik, R., & Senn, W. (2014).** Learning by the dendritic prediction of somatic spiking. *Neuron*, 81(3), 521–528.
- **Widrow, B., & Hoff, M. E. (1960).** Adaptive switching circuits. *IRE WESCON Convention Record*, Part 4, 96–104.
