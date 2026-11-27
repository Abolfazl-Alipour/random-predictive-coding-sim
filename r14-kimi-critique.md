# Round-14 critique: the delta rule

**Reviewer:** Kimi  
**Scope:** Full record through section 43, `rpc_v2.py` A1-A24, the hunt report, all three earlier critiques, the fix14 drivers and outputs, and the relevant Chapter 3 text. The local copy of `external-critique-3.md` was absent, so I read that file at public commit `6b99f3ba`. I also ran seed-level statistical summaries, a planted-data denominator check, and a direct coactivity/burst weight-and-prediction diagnostic.

## Verdict

**The delta win stands as a strong pilot result within this implementation, but not as an independently validated pass, not as proof of learning beyond one-step history, and not as validation of Chapter 3's distinctive learning mechanism.** The trained-versus-frozen effects are large, the drivers contain no direct test-label leakage into training, and the signed error update genuinely changes the learned distal map. The evidence is much weaker for the exact headline scores `0.484` and `0.221` than for the qualitative conclusion that signed error-driven plasticity is better than the unsigned rules.

**The single strongest reason to doubt the headline is adaptive test-set selection without an independent replication.** Alpha, smoothing, and the preceding model/readout choices were developed on these same test seeds; the delta rule, width, and rates were then motivated by test diagnostics on the same continuing hunt. This creates selection bias or winner's curse: the reported configuration is partly the maximum of an adaptive search, so its score is optimistically biased on fresh data. On a clean split, the exact failure would be regression of the selected configuration toward a lower mean—plausibly below `0.46`/Markov-1 on sine and below `0.22` on planted—not weights having been trained directly on test labels.

One scope correction before the eight checks: the A23 measurements are real, but `simulation-results.md:2071` overstates their implication. A23 fits labelled class centroids from `d1` (`rpc_v2.py:780-800`), so it is a supervised nearest-centroid readout, not an unlearned fixed readout. Its failure rejects A23 as a rescue for the unsigned rules; it does not kill the whole family of fixed random features plus a fixed p1-prototype readout. If “version (a)” means fixed random non-distal wires with distal plasticity, A24 itself is a counterexample to declaring that family dead. If it literally means every wire is frozen, there is no associative learning left to test.

## 1. Is the delta win real, or test-set selection?

**Verdict: verified as a within-study pilot effect; refuted as a confirmatory pass.**

The third critique correctly identified adaptive selection of alpha and sigma (`simulation-results.md:1963-1966`), and the record already acknowledged the general test-reuse problem (`simulation-results.md:1790-1795`). The final delta scores are at `rpc_sine_fix14c_results.txt:2-3` and `rpc_planted_fix14c_results.txt:2-3`. There is no direct single-run leakage in the training path: calibration and prototypes use training data (`rpc_v2.py:1156-1176`), and test evaluation has `train=False` (`rpc_v2.py:1182-1191`). The problem is experiment-level selection, not a hidden test batch in an update.

Using the reported rounded seed values, descriptive Student-t intervals are:

- Sine delta accuracy: mean `0.4844`, approximate 95% interval `[0.340, 0.629]`; the `0.46` bar lies inside it.
- Sine paired gain: mean `0.4534`, interval `[0.306, 0.601]`; the `+0.40` bar lies inside it.
- Planted delta accuracy: mean `0.2212`, interval `[0.196, 0.246]`; the `0.22` bar lies inside it.
- Planted paired gain: mean `0.1588`, interval `[0.137, 0.181]`; the `+0.15` bar lies inside it.

These intervals are descriptive, not valid confirmatory confidence intervals after adaptive selection. Also, with only five paired seeds, an exact two-sided sign-flip test cannot return a p-value below `0.05`; its smallest possible value is `2/32 = 0.0625`. That is a limitation of the design, not evidence that the large gains are zero.

Accordingly, the sine gain and planted gain justify taking delta seriously. They do not justify treating the exact pass margins or the “first to beat Markov-1” statement as established generalisation results. Five seeds also confound data draws and initial weights because each seed sets both and regenerates the data (`rpc_simple.py:60-68`; `rpc_planted_fix14c.py:32-36`).

## 2. Is planted `0.221` meaningfully above `0.22`?

**Verdict: refuted. The accuracy and gain passes are threshold noise; “planted solved” remains unresolved.**

The unrounded mean from the printed seed values is only `0.2212`, just `0.0012` above the revised bar. Three of five seeds are above `0.22`; seeds 0 and 3 score `0.197` and `0.205` (`rpc_planted_fix14c_results.txt:2`). The accuracy interval above includes both sides of the bar, and the paired-gain interval includes `0.15`. The original strict bar is clearly not met: `0.221 < 0.2625` (`simulation-results.md:2052-2058`). The revised `0.22` bar may be scientifically better motivated, but it was chosen after extensive task results and should be preregistered before a fresh confirmation rather than treated like an untouched original threshold.

The `0.966` motif result is much stronger numerically, with every seed at least `0.921`, but its denominator is narrower than “all motif-prefix prediction contexts.” `run_planted_motif` counts a step only when the previous two tokens are a prefix **and** the current token actually is the motif successor (`rpc_planted_fix14c.py:62-68`). It excludes cases where a later planted motif overwrote that successor. In my direct check of the same five test draws, there were 6-15 such excluded prefix contexts per seed, against 119-130 included true completions. Thus `0.966` is true-completion recall, not accuracy over every presentation of a motif prefix. It should be accompanied by all-prefix accuracy, false positives on other contexts, and prediction precision.

Before calling planted solved, I would require:

1. A preregistered threshold and the exact frozen configuration evaluated on fresh data/weight seeds, with the accuracy and paired-gain intervals clearing their bars rather than point estimates barely crossing them.
2. Motif metrics over **all** prefix contexts, including overwritten successors, plus false-positive rates outside those contexts.
3. Markov-1/Markov-2 and majority baselines on the identical sequences and denominators, with the same unseen-context policy.
4. An explicit decision whether the target is the revised `0.22` criterion or the original strict `0.2625`; mixing a pass on the former with a claim against the latter is not justified.

## 3. Why are coactivity and burst byte-identical on planted?

**Verdict: verified as constant-output collapse, not shared state or a driver bug. The networks themselves are not identical.**

Each `run_planted_motif` call reseeds NumPy and Torch, creates fresh local `defaults` and `merged` dictionaries, and constructs a new `RPCNetwork` (`rpc_planted_fix14c.py:32-46`). The step arguments are popped from that call's own dictionary (`rpc_planted_fix14c.py:40-44`), and only scalar metrics are returned (`rpc_planted_fix14c.py:69-70`). `rpc_planted_fix14d.py:23-32` passes the distinct rule names correctly. There is no shared network, global `merged` dictionary, or result object through which coactivity could overwrite burst.

The update branches are genuinely different: burst uses `post=b1`, whereas coactivity uses thresholded p1 activity (`rpc_v2.py:1090-1113`). Both updates are non-negative and clamp at `w_max`, however. Once the large positive common mode dominates, cosine against the fixed prototypes selects one class regardless of the input.

I reproduced the exact five-epoch configuration and inspected the final matrices and predictions. On seeds 0-3, `97.9-99.3%` of coactivity entries and `89.8-97.3%` of burst entries were exactly `+2.0`. The two final matrices were **not** identical on any checked seed, yet both produced exactly one predicted class on all 1,150 scored test steps, and their predictions were identical. Their accuracies therefore reduce to that class's frequency in the shared test draw. This confirms the record's substantive explanation at `simulation-results.md:2062-2067`, with the refinement that near-total saturation and identical constant predictions—not identical networks—explain the bytes.

## 4. Does the ridge probe prove “no information; entirely readout alignment”?

**Verdict: probe configuration verified; the stronger interpretation is refuted.**

The earlier probe mismatches are fixed here:

- `probe_delta` uses `make_sine_fixed(50, seed=seed)` and `make_sine_fixed(20, seed=seed+1000)`, matching `run_task` (`rpc_sine_fix14c.py:34-39`; `rpc_simple.py:60-68`).
- It builds prototypes with `proto_n=50`, uses `n_out=256`, and keeps the same delta settings (`rpc_sine_fix14c.py:39-43`).
- It trains for five epochs with `use_confirmation=False`; frozen differs only by `plasticity=False` (`rpc_sine_fix14c.py:44-49`).
- The resume driver evaluates the same three seeds for trained and frozen (`rpc_sine_fix14c_probe.py:8-13`).

Calibration and p1-prototype construction on train data are legitimate and shared by both conditions, not a frozen-control confound. The ridge is also fit on train drives and scored on held-out test drives (`rpc_sine_fix14c.py:50-65`).

But the numerical result supports only this statement: **under this particular unstandardised, no-intercept, `lambda=1` ridge probe on three seeds, delta does not improve held-out linear next-token accuracy.** The paired trained-minus-frozen differences are `[-0.006, +0.002, -0.011]`, mean about `-0.004` (`rpc_sine_fix14c_probe_results.txt:1-2`). That is not proof of “no information,” in either an information-theoretic sense or across other decoders and regularisation choices. A ridge-sensitivity sweep and more seeds would be needed before even calling the decode difference identically zero.

“Readout alignment” is a plausible and well-motivated mechanism because the update explicitly minimises `p1-d1` (`rpc_v2.py:1060-1071`) while the readout compares against mean p1 patterns. It is not demonstrated **entirely** by two similar ridge scores. Delta could also change feature scaling, covariance, signal-to-noise ratio, or class geometry in ways that matter to cosine but not to this ridge. The remaining gap is large: fixed-prototype accuracy `0.484` versus ridge decode `0.913`. The clean claim is “consistent with converting already linearly decodable information into a p1-cosine-usable form.” Direct correct-prototype cosines, class-centroid scores, and covariance/alignment measurements on the same drives would test that mechanism.

## 5. Remaining fix14 configuration mismatches

**Verdict: the delta-versus-control comparison is mostly verified; the stated AGY specification is not fully implemented, and the width factorial is incomplete.**

The important settings match. Sine delta and its controls share `full_code=True`, `norm_rows=False`, `use_confirmation=False`, `s_inh=0`, `smooth_sigma=3`, `alpha=0.7`, `graded_conf=False`, and `n_out=256` (`rpc_sine_fix14c.py:30-32`; `rpc_sine_fix14d.py:20-32`). Planted likewise shares the full-code, width, leak, confirmation, inhibition, and normalisation settings (`rpc_planted_fix14c.py:28-30`; `rpc_planted_fix14d.py:20-31`). The corrected planted `merged` path passes alpha once and routes `use_confirmation`/`plasticity` to `step`, not the constructor (`rpc_planted_fix14c.py:37-51`). I found no remaining asymmetry in data, prototypes, calibration, test evaluation, or these settings that would give delta an extra advantage beyond its update rule.

Two qualifications matter:

1. **The implementation is clamped NLMS, not the originally specified unclamped or wide-clamp rule.** `w_max` defaults to `2.0` (`rpc_v2.py:503-506`), and every delta update is clamped to `[-w_max,w_max]` (`rpc_v2.py:1067-1075`). AGY's specification called for unclamped or a wide clamp of `10.0` (`agy-critique.md:180-188,219-231`). The clamp is shared by delta and controls, so it is not a delta-versus-control confound, but round 14 does not establish the behaviour of the unclamped rule. Clamp fractions should be logged for delta as well as for the unsigned controls.
2. **The requested rule-by-width factorial was not completed.** Delta ran at 64 and 256, but coactivity/burst/cov ran only at 256 (`rpc_sine_fix14c.py:67-75`; `rpc_sine_fix14d.py:24-32`; planted equivalents). The matched 256 controls adequately support the central rule comparison, but the complete width-by-rule interaction claimed in the design was not measured in this round.

The sine control rates also differ by design—delta `eta_d=0.2`, unsigned/cov `eta_b=0.15`—while planted uses `0.05` for both. Together with delta's presynaptic-norm denominator, these are parts of the respective update definitions, not hidden constructor mismatches. They do mean “only the rule changes” should not be read as identical update magnitude or dynamics.

## 6. Did delta learn one step of history or more?

**Verdict: unresolved. Beating the Markov-1 mean by `0.025` is numerical progress, not evidence of second-order learning.**

The paired values are more informative than the two means. Delta minus Markov-1 by seed is

`[-0.071, +0.080, +0.099, -0.022, +0.041]`,

using `rpc_sine_fix14c_results.txt:2` and `rpc_sine_diag_results.txt:1`. The mean difference is `+0.0254`; two seeds are below Markov-1, and the descriptive 95% paired interval is approximately `[-0.063,+0.114]`. Thus “first configuration to beat Markov-1” is true of the five-seed means (`simulation-results.md:2026-2032`) but is not a robust demonstrated superiority.

Markov-2 `0.802` is also not a ceiling and not scored on an identical denominator. `markov_acc` silently drops unseen contexts from both numerator and denominator (`rpc_sine_diag.py:21-34`), whereas RPC scores every noninitial test step. This prevents using the `0.318` gap to infer how much second-order structure delta failed to acquire.

The settling test is a matched order ablation: compare the full reservoir-driven delta network with a strictly first-order version whose distal pre-code is a fixed function of only the previous token, holding width, readout, training steps, and data fixed. On the same test steps, also report direction-conditional accuracy and paired delta-minus-Markov-1/Markov-2 scores, with a declared fallback for unseen Markov contexts. If the full model does not beat the strict first-order control specifically on rising/falling-ambiguous contexts, there is no evidence that it learned useful history beyond one step.

## 7. Is this enough to claim the theory “learns in noise”?

**Verdict: verified only as a narrow existence result on planted random filler; refuted as a general theory claim.**

Planted motifs do contain random filler, so the large gain and high true-completion recall are meaningful evidence that this algorithm can learn a specific association amid background noise. Fixed sine is deterministic given phase; random phase is not general observation noise. Neither result establishes variable-frequency tracking, chaotic dynamics, the 1D track, hierarchical prediction, or arbitrary temporal relationships. The record correctly lists these omissions at `simulation-results.md:2079`.

The cheapest generalisation experiment likely to change the verdict is **variable-frequency sine under the exact frozen delta-256 configuration, with no retuning**, on fresh sequences, with matched frozen and cov controls plus persistence and Markov baselines. It reuses the continuous-sine machinery while testing whether the fixed-sine result survives changing temporal context. Lorenz needs matched sampling/baselines, and TrackF's known two-step target misalignment should be fixed before it is used as evidence. Those are less clean next tests.

This generalisation test does not replace the more urgent independent replication of the existing sine and planted scores.

## 8. Does delta still count as the Chapter 3 thesis theory?

**Verdict: refuted as a validation of Chapter 3's distinctive burst-confirmation mechanism; verified as an RPC-inspired, local error-driven reservoir model.**

Chapter 3's specific proposal is that random feedback collides with newly active proximal cells, a successful prediction produces a burst, and that burst strengthens the corresponding synapses (`ch3.txt:1447-1453,1479-1489`). It then sends a confirmation signal back to strengthen the originating CA3-CA1 synapse (`ch3.txt:1500-1526`). A24 instead applies signed `p1-d1` error on every plastic step, independently of `b1`, with confirmation plasticity disabled and inhibition removed (`rpc_v2.py:1059-1075`; the fix14 `use_confirmation=False` and `s_inh=0` settings). This is Widrow-Hoff/normalised-LMS-style local error-driven plasticity, closely related to dendritic prediction-error learning, not the thesis's successful-prediction-only potentiation.

What survives, provisionally:

- Fixed random recurrent/projective features can supply useful temporal representations.
- A local plastic distal prediction pathway can align those features with a fixed p1-prototype cosine readout without supervised W_prox fitting or a trained ridge readout.
- Signed local prediction error is a much more effective learning signal in this architecture than unsigned coactivity or burst-gated potentiation.

What does not survive:

- The claim that the thesis's distinctive confirmation burst and reciprocal confirmation loop are sufficient or responsible for these wins; they are functionally absent from A24.
- The claim that all wires are fixed: `W_dist1` is learned, as it had to be. The narrower claim is that the **other** wires can remain fixed random.
- Any demonstration of hierarchical predictive coding, separate predicting/error populations, precision weighting, omission responses, or general noisy temporal learning. Chapter 3 explicitly lists prediction/error populations and hierarchy among predictive-coding assumptions (`ch3.txt:317-341`) and discusses precision weighting (`ch3.txt:653-681`); A24 does not implement those tests.

Calling A24 “the thesis theory works” would therefore be a different mechanism wearing the thesis label. The accurate statement is: **replacing the thesis's confirmation rule with local signed error plasticity rescues performance on two toy tasks while preserving fixed random features and the p1-match readout.** That is a valuable result, but it redirects the theory rather than confirming its original mechanism.

## Next experiment: fresh-seed, zero-tuning replication

Before adding more variants, run one preregistered confirmation of the **exact round-14 delta-256 configuration**, including the actual `w_max=2.0` clamp, on fixed sine and planted motifs:

- Use fresh sequence seeds and independently varied network seeds; do not inspect new test results until code, hyperparameters, bars, and analyses are frozen. Twenty paired replications are preferable; no validation split is needed if absolutely nothing is tuned.
- Run delta, its identical frozen control, and cov on the same data. Keep the old unsigned controls optional; they have already established the saturation failure.
- Report every seed, median/IQR and paired intervals for accuracy and gain. Log clamp fractions and constant-prediction rates.
- Score Markov-1/Markov-2 on identical test transitions with a preregistered unseen-context fallback. Add the strict first-order delta ablation and direction-conditional sine accuracy.
- On planted, report all-prefix accuracy, false-positive/precision metrics, and true-completion recall separately.

The decision rule should be fixed in advance: the numerical sine pass survives only if fresh performance clears the chosen accuracy/gain criteria and the Markov-1 comparison; planted survives only if its preregistered overall and gain criteria clear without relying on a one-point-in-the-third-decimal margin. If these fresh results regress below the bars, round 14 was selection plus marginal threshold noise. If they hold, then run the zero-tuning variable-sine generalisation test. That sequence resolves the main evidentiary doubt before making any broader claim about the theory.
