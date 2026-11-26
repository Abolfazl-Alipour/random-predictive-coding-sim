# Critique of the RPC simulation hunt

## Scope and bottom line

I read `RPC-HUNT-REPORT.md`, all 40 sections of `simulation-results.md`, `rpc_v2.py`, every `rpc_*.py` driver, and every `*_results.txt` file in this directory. I also read the RPC portion of `ch3.txt` and ran one seed-0, one-epoch diagnostic with one CPU thread. I did not check the literature claims or the private GitHub repository.

The positive result is real but narrow: this implementation learns the noiseless cycle and pure motif at high leak. The negative results are also real for this implementation: none of the tested versions learns the harder tasks to the stated bars. They do not yet justify the report's stronger root-cause story, and they do not falsify every model with fixed random reservoirs.

The main problems are not lack of experimentation. They are controls that do not test what their names say, comparisons between different configurations, and causal conclusions that do not follow from linear probes. In particular:

- “No-confirmation” does not remove the confirmation gate on distal learning.
- The burst does not test whether the predicted token is correct.
- The “orthogonal information” claim does not follow from the 31% energy measurement and is false as a statement about kNN.
- The supposedly clean planted stacking test is still confounded.
- A23 is **not** a mathematical no-op. It is a reasonable next diagnostic, but not yet a demonstrated fix.
- The thesis already includes learned synapses, including the CA3-to-CA1 proximal synapse. The question is which plastic rule works, not whether every wire must be fixed or learned.

## 1. Critique of the method and interpretation

### 1.1 The two most important control errors

**“No-confirmation” is misnamed and repeatedly misinterpreted.** The report defines it as “learning on but the confirmation gate removed” (`RPC-HUNT-REPORT.md:75-77`). In the code, distal learning happens at `rpc_v2.py:1026-1058`. The `use_confirmation` flag is checked only afterward, at `rpc_v2.py:1059-1068`, where it enables or disables the update to `mod2.W_prox`. Thus no-confirmation still has the same burst-gated distal update as the full model.

Consequently, statements such as “confirmation adds plus 0.088” on the pure motif (`simulation-results.md:325-332`) do not measure the value of gating distal learning. They measure the effect of also changing the module-2 proximal weights. The same correction applies to the claims that confirmation hurts on Lorenz and the track, and to the smooth-sine no-confirmation ablation in `rpc_sine_fix7_results.txt:1-4`.

The appropriate controls are separate: (a) distal plasticity on/off, (b) proximal confirmation plasticity on/off, and (c) the same distal post term with and without the distal factor of the burst. The present full versus no-confirmation comparison is only (b), with other learning left on.

**The burst is not a token-level correctness signal.** The report says the gate “only learns on steps where the prediction already matches” (`RPC-HUNT-REPORT.md:294-306`; also `rpc_sine_fix13.py:9-12`). The actual rule is

`b1[j] = 1` exactly when `p1[j] / sd_p1 > theta_p` and `d1[j] / sd_d1 > theta_d`,

at `rpc_v2.py:1003-1006`. The token prediction is computed separately at `rpc_v2.py:1013-1014` and is not used to form `b1`. A burst can therefore occur on a wrong-token prediction, and a correct-token prediction can fail to burst.

This distinction matters. The thesis describes a collision between a returning random transform and newly active proximal cells (`ch3.txt:1479-1489`), which is closer to the implemented per-cell coincidence than to a classifier-correctness gate. The report's cold-start and “learns only when already correct” explanations are about a different gate. To make those explanations, the study must measure burst probability conditional on correct and incorrect token predictions, not infer it from mean burst rate.

### 1.2 Several comparisons remain confounded

**The A18 burst/coactivity comparison changes graded confirmation too.** In `rpc_sine_fix13.py:77-84`, the coactivity base has `graded_conf=True`, while A18+burst explicitly sets `graded_conf=False`. The resulting 0.022 burst score and 0.105 coactivity score (`rpc_sine_fix13_results.txt:1-4`, together with `rpc_sine_fix10_results.txt:1-2`) therefore do not isolate the learning rule. The accompanying ridge probes have the same difference: compare `rpc_sine_fix13.py:47-48,98-99` with `rpc_sine_fix12.py:44-46`. The large difference is evidence about these two complete configurations, not clean evidence that the burst gate alone causes the failure.

**The planted stacking failure is still confounded.** The report correctly withdrew the first comparison because the learning rule changed (`simulation-results.md:1418-1433`), then reinstated it after round 13 (`RPC-HUNT-REPORT.md:410-416`). But A12 alone, which scored 0.113, explicitly used `use_confirmation=False` (`rpc_prox.py:79-87`). The round-13 A12+kNN+coactivity condition, which scored 0.075, does not set that flag (`rpc_planted_fix13.py:29-33`), so proximal confirmation is on and rewrites the fitted projection. The raw scores are in `rpc_prox_results.txt:3` and `rpc_planted_fix13_results.txt:3`.

Thus 0.113 versus 0.075 changes both the readout and proximal confirmation. The clean stacking test still needs A12 with confirmation off plus kNN, compared with A12 with confirmation off plus cosine. The report's “real stacking failure” conclusion is not supported yet.

**The extraction ratios mix different drives.** The per-seed ratios in `simulation-results.md:1473-1489` divide A18+read2 accuracies by ridge decodes from plain A18 without read sharpening. See `rpc_sine_fix12_results.txt:1` versus line 8. Read sharpening changes the drive, so these are not ratios on the same representation. For example, plain A18 seed 2 scores 0.028, not the 0.035 used in the narrative (`rpc_sine_fix10_results.txt:1`). The broad observation—high ridge decode but low prototype score on that seed—survives, but the quoted ratios do not.

**The 0.556–0.673 ridge range is not “on the same drive” as the best cosine, dot, and kNN scores.** The 0.556 result is one-hot, alpha 1.0, three seeds (`rpc_sine_fix_results.txt:15`). The 0.673 value is the mean of three frozen smooth-2.0 drive probes (`rpc_sine_fix3_results.txt:9-11`). The best cosine/dot/kNN results instead use smooth 3.0, alpha 0.7, and graded coactivity. The matched top-6 ridge probe for that configuration is 0.507 (`rpc_sine_fix12_results.txt:7`). The report repeatedly combines these as though they were one readout comparison, for example `RPC-HUNT-REPORT.md:237-248,358-365`.

This does not erase the large prototype/ridge gap. It does mean that the exact size of the gap and the claim that the frozen control beats learning on the best configuration still need a matched run.

### 1.3 A simpler target/readout mismatch was skipped

The Hebbian rule does not learn to reconstruct the raw proximal prototypes used by the readout. Coactivity uses a binary post term, `(p1n > theta_p)`, at `rpc_v2.py:1027-1034`; burst learning uses the binary `b1` at lines 1034-1036. The update is an outer product of that binary vector with the previous code at lines 1042-1052. In contrast, prototypes are means of raw, unthresholded `p1` vectors (`rpc_v2.py:703-744`), and prediction compares raw `d1` with those means (`rpc_v2.py:939-976`).

So the learning target is a thresholded, and in the burst case prediction-dependent, binary pattern, while the evaluation target is a raw class-mean pattern. This alone can produce a drive that is linearly informative but badly aligned with prototype cosine. It is a more direct explanation than an unmeasured orthogonal-information story, and it should be tested first by comparing prototypes of the actual binary post targets with the raw-p1 prototypes.

The full-code interventions do not address this mismatch. A20 changes the pre-term and A21 changes the drive-read code; neither makes the post target equal to the readout target (`rpc_v2.py:589-610,1037-1058`).

### 1.4 The code-versus-learning and readout conclusions are too strong

The stage measurements are useful, but they are measurements of linear decode accuracy under particular probes, not fractions of information in an information-theoretic sense.

- The planted stage results support a drop in held-out linear decodability from 0.984 at `r2` to 0.711 at `p2` and 0.332 at top-6 `y2` (`rpc_probe2_results.txt:4-6`). Calling this “a third” and then “half” of the information is an approximation of accuracy differences, not a measurement of mutual information. Dimension, sparsity, ridge penalty, and class geometry all differ between stages.
- The full-code sine probe supports higher token decode from `y2` and lower next-token decode from `d1` (`rpc_sine_fix10_results.txt:7-8`). It does not by itself locate the cause in either the read or write.
- A20 only tests two particular power sharpenings with an L2 renorm. It does not exhaust the dilution hypothesis. A21 introduces a train/serve mismatch: learning uses the raw code while the drive reads a sharpened code (`rpc_v2.py:589-610,1042-1048`). Their failures do not prove that the degradation “sits in the read” (`simulation-results.md:1405-1414`).
- Wide256 does not report a new `p2` or `y2` decode. Its unchanged end-to-end score (`rpc_wide_results.txt:1-2`) does not prove that the wider projection failed to improve the code, nor that dimensionality is irrelevant. Other stages could remain limiting.

There is also a direct contradiction in the narrative. Section 24 calls learning “not the bottleneck” because a frozen random distal map preserves probe information (`simulation-results.md:738-745`). Section 25 then finds that plasticity lowers ridge accuracy from 0.606 to 0.556 (`rpc_sine_fix_results.txt:15-16`). A random matrix preserving some linear information says nothing about whether the learned matrix improves the representation. On those measurements, learning is indeed a problem.

Similarly, three low-scoring, closely related p1-similarity readouts do not show that “the readout is no longer the binding constraint” (`simulation-results.md:1352-1357`). Cosine and dot use the same class means; kNN uses samples from the same proximal representation. Their similar scores show a shared limitation of that readout family. The much higher linear-probe scores are evidence that readout/extraction remains binding, alongside the learning and code problems.

### 1.5 Some reported experimental conditions are wrong

**The variable-sine and Lorenz “10ep” runs actually used five epochs.** Both names and record text say ten epochs (`rpc_sine_fix5b_results.txt:1,5`; `simulation-results.md:1005-1010`), but `rpc_sine_fix5b.py:19-25` calls `run_one(..., epochs=5, ...)`. The duplicated block in `rpc_sine_fix5.py:50-58` also uses five. The printed accuracies are valid five-epoch results; the ten-epoch interpretation is not.

**TrackF is not the same next-step task as the token track.** Each TrackF item at time `t` contains `[F(t), F(t-1)]` but is labelled with position bin `t+1` (`rpc_track_F.py:43-57`). At item `i`, `d1` is computed before the current input and therefore only has the previous item's code (`rpc_v2.py:993-1004`), yet `run_vec` scores it against the current item's `y` (`rpc_v2.py:1149-1157`). From the last input available to `d1`, the scored target is two position steps ahead, not one. Plasticity meanwhile teaches with the current proximal input at `rpc_v2.py:1026-1048`.

For a token-style one-step test, the item label should be the current position bin `t`; then the previous code predicts the current bin and the current proximal code represents that same target. The present TrackF scores cannot be compared directly with the token-track scores or used to close the thesis-faithful input question.

**The reported “z-scores” are not z-scores.** Calibration divides all entries by one global standard deviation over the stacked drive array (`rpc_v2.py:661-662`, similarly lines 679 and 698). It neither subtracts a mean nor computes a separate standard deviation per cell. Thus `theta_p=1` is not a per-cell standard-normal threshold. This is an allowed modelling assumption, but it should be described as global scale calibration, not standardisation into z-scores.

### 1.6 The task bars and benchmarks need a fairer design

**The Markov numbers are baselines, not ceilings.** `markov_acc` fits finite training counts and, when a test context is unseen, omits that step from the denominator (`rpc_sine_diag.py:21-34`). Therefore Markov-1 0.459, Markov-2 0.802, and Markov-3 0.833 (`rpc_sine_diag_results.txt:1-3`) are empirical scores of those estimators, not mathematical upper bounds on all one-, two-, or three-step models. The same terminology problem applies to the planted 0.36/0.39 estimates and to the claim that a delay line has an expected gain of exactly about 0.03. The correction that three planted motifs are present is right; the ceiling language is not.

**The accuracy bars are not equally meaningful across tasks.** Uniform `1/K` is not the best no-history predictor when class frequencies are nonuniform. More importantly, the fixed-sine bar 0.231 is far below the report's own Markov-1 score 0.459 and Markov-2 score 0.802. Passing it would not demonstrate useful phase-history learning. The position-track bar 0.23 is far below persistence 0.776 (`rpc_track_results.txt:1`). A large gain over a weak prototype-readout control can also coexist with a poor model, as the sine results show. Keep the preregistered bars for continuity, but add majority/persistence and matched Markov/ESN comparisons as primary evidence.

**The baselines are not evaluated on identical RPC sequences.** Baseline data use `np.random.seed(1000 + seed)` (`baselines.py:360-374`), whereas RPC planted runs use `np.random.seed(seed)` (`rpc_planted_alpha.py:23-27`). The distributions match, but the sequences do not. The baseline means are useful references, not paired comparisons on the same examples.

**The test sets have been used for model selection.** Many variants and hyperparameters were screened on the same seeds, followed by five-seed “confirmations” on those same seeds. That is adaptive test-set reuse, not an untouched final evaluation. Use a validation split for selection and fresh data and weight seeds for the final run. Report paired per-seed differences and uncertainty rather than calling differences of about 0.01 “noise” without a paired analysis.

**A seed confounds data and initialisation.** `run_task` seeds both torch and numpy with the same value and regenerates train/test data for every seed (`rpc_simple.py:60-68`). Thus the A18 collapsed seeds may reflect data draws, initial weights, or their interaction; calling them an initialisation instability alone is unwarranted. Cross data seeds with network seeds to identify the bimodality.

Finally, the no-plasticity control still learns calibration scales and prototypes from training data (`rpc_v2.py:1105-1107`). It is a no-synaptic-plasticity control, not a completely learning-free model. That is fine if named precisely, especially for A23, whose control must also fit drive prototypes from the frozen drive.

## 2. Audit of the corrections listed in report section 5

1. **V1 scale bug:** the correction is right. V1 can have exactly zero distal drive because `y2` never fires (`rpc_toy.py:106-120`). V2 calibrates scales and produces nonzero learning. The record also correctly notes that the proximal threshold was a second scale failure.
2. **Prototype misstatement:** the correction is right. `build_prototypes` sums `p1`, not `d1` (`rpc_v2.py:703-744`).
3. **Dilution explanation:** retracting the original explanation is right. Replacing it first with “the read” and then with orthogonal information is not yet justified; A20/A21 do not isolate those alternatives, and the binary-post/raw-prototype mismatch remains untested.
4. **Planted stacking:** withdrawing the first confounded claim was right. Reinstating it after round 13 is not right because proximal confirmation also changed, as detailed above.
5. **A22 design error:** the no-op correction is right for cosine and raw dot against class-mean prototypes. The further claim that the removed energy is exactly the ridge-only information does not follow.
6. **One-epoch smoke tests:** the warning is right. My one-epoch A23 check below is therefore an implementation/algebra check only, not a performance verdict.
7. **Planted ceiling:** recognising all three motifs is right. The empirical Markov scores still should not be called ceilings, and one failed delay implementation does not prove memory span is never a bottleneck.
8. **Smooth 1.5 overstatement:** the correction matches `rpc_sine_fix2_results.txt:9-10` and is right.
9. **Graded confirmation:** the five-seed corrections match `rpc_sine_fix6b_results.txt` and `rpc_sine_fix6c_results.txt`. The separate full/no-confirmation interpretation still needs the control correction in section 1.1.

## 3. The root-cause chain, step by step

### Step 1: the reservoir holds the token

Supported for these tasks and probes. The planted `r1` decode is 1.000 (`rpc_probe_results.txt:1-2`), and the sine frozen `r1` next-bin decode is also 1.000 (`rpc_sine_diag_results.txt:9`). This is evidence about linear availability of information, not proof that every relevant nonlinear feature is retained.

### Step 2: the fixed proximal projection loses about a third

The cited numbers are correct for held-out ridge decode: 0.984 at `r2` and 0.711 at `p2` (`rpc_probe2_results.txt:4-5`). It is fair to say the random 64-unit projection plus readout loses linearly accessible token information in this configuration. It is not fair to generalise that to a measured one-third information loss in the theory. A12 shows that a supervised fitted projection can preserve the 0.984 decode (`rpc_prox_results.txt:1-2`), not that fixed random projections in general cannot work with a different downstream decoder.

### Step 3: top-6 loses more, and the full code changes the drive

The top-6 drop to 0.332 is supported (`rpc_probe2_results.txt:6`). The full-code sine numbers, 0.709 for `y2` and 0.457 for `d1` versus 0.516 and 0.509 for top-6, are also supported (`rpc_sine_fix10_results.txt:7-8`). Note that these figures come from different tasks/configurations than the planted 0.332 figure and should not be concatenated into one numerical loss chain.

The causal word “then” is the weak part. Full code simultaneously changes the distal read, the Hebbian pre-term, sparsity, and normalisation. The cause of the lower drive decode remains unresolved.

### Step 4: the drive is informative, largely in the orthogonal complement

The first half is supported: ridge decodes the next token well above the network score, including on A18 collapsed seed 2 (`rpc_sine_fix12_results.txt:8`). The second half is not supported by the cited measurement.

If `P` projects onto the prototype span, observing that `(I-P)d` contains 31% of the energy only measures vector length. Energy is not class information. A22's unchanged accuracy proves only that class-mean cosine and dot scores are algebraically insensitive to `(I-P)d`; it does not prove that the ridge probe gets its accuracy there rather than from `Pd`.

My one-epoch seed-0 A18 coactivity check illustrates the missing measurement. Using the reported smooth-3.0, alpha-0.7, graded configuration and the original p1-prototype projector, I obtained:

- full-drive ridge accuracy: 0.377387;
- ridge on `Pd` only: 0.345226;
- ridge on `(I-P)d` only: 0.349246;
- orthogonal energy fraction: 0.561463.

Both components carried substantial information in this check. This is not a reproduction of the five-epoch result, and I did not rerun the two-epoch A22 diagnostic because of the one-epoch limit. It shows why a split probe is required before saying the orthogonal part is “exactly” the ridge-only information. The original 31% figure itself is recorded in `simulation-results.md:1553-1567`, but there is no separate `rpc_proj_check` output file here for me to verify independently.

### Step 5: no fixed comparison against p1 patterns can see the orthogonal part

This is true only for dot/cosine against the **class-mean prototypes that define that particular span**. It is false for the kNN bank. The bank contains individual raw p1 vectors (`rpc_v2.py:709-732`), which generally have components outside the span of the class means. For a bank vector `b`, `(d_perp · b)` need not be zero. The report itself acknowledges that A22 is nontrivial for kNN (`RPC-HUNT-REPORT.md:315-319`), which contradicts the broader wording in section 4.

The kNN failures therefore do not corroborate the orthogonal-span explanation. They could reflect within-class variation, distance geometry, or the binary-target/raw-pattern mismatch. A22+kNN was not run.

### Step 6: the burst gate makes the misalignment worse

The A18+burst configuration does have low network accuracy, 0.022, while its drive has mean ridge decode 0.462 (`rpc_sine_fix13_results.txt:1,7`). That is evidence of poor extraction in that configuration. It is not clean evidence that the gate causes the worsening, because graded confirmation also changed, and the gate is not a token-correctness signal. The statement is a plausible hypothesis requiring the factorial controls described above.

### Step 7: smooth input supplies direction information

The association is supported: in the frozen smooth-2.0 probe, direction decode rises relative to one-hot while the reservoir remains informative (`simulation-results.md:910-916`; raw values in `rpc_sine_fix3_results.txt:9-11`). Smoothing also changes input geometry, recurrent states, prototypes, and distal dynamics, so “which is why the score rises” is a mechanism hypothesis, not an isolated causal proof. It is currently one of the better-supported hypotheses in the report.

### The simpler chain I would test instead

The evidence is consistent with a less exotic sequence:

1. Random recurrent states contain substantial history.
2. Sparsification and projection reduce what a simple linear probe can recover, but do not remove all of it.
3. Distal plasticity learns binary thresholded coactivity patterns, not raw p1 prototypes and not a prediction-error objective.
4. Row-norm competition and purely potentiating updates can produce seed-dependent, idiosyncratic linear codes.
5. A flexible linear readout can decode those codes, while raw-p1 class means often cannot.

This explains high ridge/low cosine without assuming that the useful information lives in a special orthogonal subspace. It also directly suggests the next modelling change: make the distal update error-correcting in the same representation used for readout.

## 4. A23: linear algebra and verdict

Let the old p1 prototypes be rows `c_k`, and let `P` be the orthogonal projector onto their span. A22 replaces `d` with `Pd`. Because `Pc_k=c_k` and `P` is symmetric,

`(Pd) · c_k = d · (Pc_k) = d · c_k`.

Cosine also removes the common factor `||Pd||`, and raw dot preserves every numerator. A22 is therefore exactly prediction-invariant for those two readouts. That correction is sound.

A23 instead constructs new templates

`m_k = mean { d_t : the token at step t is k }`

and scores `d · m_k / (||d|| ||m_k||)`. Decompose `d=d_span+d_perp` and `m_k=a_k+b_k` with `a_k` in the old span and `b_k` orthogonal to it. Then

`d · m_k = d_span · a_k + d_perp · b_k`.

The second term is generally nonzero, and the new template norms can also differ. A23 is therefore **not** a no-op. It can use information that A22 removes, but nearest class centroids are still much more restricted than a full ridge classifier, so there is no guarantee that it will rescue the collapsed seeds.

The implementation builds these templates from training sequences only, after plasticity, and excludes each sequence's first step (`rpc_v2.py:747-768`). It swaps them in after training (`rpc_v2.py:1113-1117`). That is a legitimate held-out diagnostic. It is also a fitted readout: “no trained weights” only means no additional matrix regression, not no learned parameters. Contrary to the A23 comment at `rpc_v2.py:378-380`, the current burst and graded gate do not use prototype predictions; they use the raw drives at `rpc_v2.py:1003-1006,1021-1025`. A23 is therefore strictly an evaluation-readout change in this implementation.

My paired one-epoch seed-0 check found:

- original p1-cosine accuracy: 0.233668;
- A23 drive-cosine accuracy on the same states: 0.198241;
- fraction of test predictions changed by A23: 0.794975.

This confirms execution and non-equivalence. The lower smoke score must not be used as the A23 verdict, especially given the report's own full-code smoke-test warning.

**Recommendation:** A23 is the right immediate diagnostic, not another A22. Run it because it discriminates between comparison-space and centroid hypotheses, not because the algebra predicts a pass.

## 5. How to do it right, and the single next experiment

### The single most promising next experiment

Run **A23 on the fixed-sine A18 coactivity configuration**, with the measurements needed to make it decisive:

- Task/config: fixed sine, smooth 3.0, alpha 0.7, graded coactivity, `eta_b=0.15`, full code, five epochs, the same five seeds as rounds 10–13.
- Controls: paired no-plasticity networks (all synaptic plasticity off) with every other setting identical, including drive-prototype fitting. Do not compare with ridge controls from other input/leak configurations.
- Readouts on identical saved train/test drive vectors: original p1 cosine, A23 drive cosine, and ridge. Fit all readouts on train only. Report each seed, especially seeds 2–4, and paired full-minus-control differences.
- Required decomposition: using the original p1-prototype projector `P`, fit separate ridge probes on `d`, `Pd`, and `(I-P)d`. Also report class-mean cosine scores in those components. This replaces the energy inference with an information measurement.
- Bars: retain accuracy 0.231 and paired plasticity gain 0.10 for continuity, but do not call 0.231 temporal mastery; it remains below the reported Markov-1 baseline. The diagnostic succeeds if it identifies why extraction fails even when no pass bar is met.

The outcomes are informative. A23 rescue only in the full network would support comparison-space misalignment. Equal rescue in the frozen control would support a readout improvement but not RPC learning. High ridge with failed A23 would show that class centroids are insufficient. Strong decoding from `Pd` would kill the claim that the ridge-only information is specifically orthogonal.

After that diagnostic, the most promising **model** change is a local error-correcting distal update, not another input or readout tweak. For pre-activity `y=y2_prev`, raw proximal target `p=p1`, and prediction `d=W_dist1 y`, test

`W_dist1 <- W_dist1 + eta * outer(p - d, y)`

with a validation-selected rate and the same row-norm policy across compared conditions. Both `p` and `d` are locally available at module 1. This directly trains the representation used by the p1-prototype readout and allows depression as well as potentiation. It is a small change to the already-plastic distal pathway, although it departs from a strict confirmation-only rule. Test it first on the planted one-step task against Markov-1, then on fixed sine with the same smooth input given to all relevant controls.

### General experimental standard

1. Freeze a precise formal model before further sweeps. Specify the target representation, burst definition, readout, and which synapses are plastic. Do not call the current qualitative-theory implementation “the theory” without qualification.
2. Use separate ablations for distal plasticity, distal gating, and proximal confirmation plasticity. Keep graded confirmation and all learning rates matched when comparing rules.
3. Use matched data for RPC, Markov, ESN, and controls; separate data seeds from weight seeds; select on validation data; report final fresh seeds with paired uncertainty.
4. Include majority, persistence, Markov-1/2, and a frozen-random-features control with the identical readout. For a temporal-learning claim, require more than the low uniform-chance-plus-0.20 bar.
5. Measure the actual learning dynamics: burst rates conditional on correct/incorrect token predictions, post-target/prototype alignment, and information gains from plasticity on the same drive. Stop using ridge accuracy alone as evidence of useful RPC learning; the frozen controls are essential.
6. Correct TrackF's target alignment and the false ten-epoch labels before including those conditions in a final boundary table.

## 6. Does the theory need learned wires, and where?

A completely fixed set of wires cannot learn new input-output associations. Some associative weights must change. That does not contradict the thesis when the thesis is read carefully: it proposes random nonlinear transforms followed by selective strengthening. It explicitly says the confirmation signal strengthens the originating CA3-to-CA1 synapse (`ch3.txt:1500-1506`), and it also describes strengthening the feedback association that produced a successful collision (`ch3.txt:1479-1489`). Thus proximal plasticity in module 2 is not itself a new betrayal of the thesis. A12's **supervised ridge fit** is the departure.

The present results do not prove that a learned proximal projection is necessary:

- Random `W_prox` reduces linear decode, but a downstream learned map could in principle use a distributed code that the simple probe handles poorly.
- The full graded code increases token decode yet lowers end-to-end performance, so the fixed projection is not the only binding problem.
- A12 improves planted accuracy only to 0.113, still far below the bar (`rpc_prox_results.txt:3`), and its comparison against confirmation is not a test of every possible local proximal learning rule.
- Fixed random reservoirs with a trained readout already perform well on the continuous tasks, although that demonstrates reservoir computing rather than the RPC confirmation rule (`rpc_readout_gen_results.txt:1-6`).

The minimal version worth preserving is therefore: **keep the input, expansion, recurrent, and inter-module random feature wires fixed; make the distal associative weights genuinely error-correcting; retain module-2 proximal plasticity if following the thesis text.** If a supervised proximal fit becomes necessary after that test, the strong “random wires are enough” claim must be narrowed to “random recurrent features are enough when combined with learned input/proximal and associative projections.” If only the distal and confirmation synapses learn, the thesis's random-transform claim remains intact.

The current experiments do not establish that no local, fixed-random-feature RPC can work. They establish that the particular thresholded Hebbian/coactivity rules and p1-similarity readouts tested here do not work beyond simple deterministic mappings. That is a narrower and more useful conclusion.

## Final assessment

Muse did a large amount of useful falsification and several honest corrections. The cycle/motif boundary, the code-probe measurements, the frozen-readout controls, and the A22 algebra are valuable. The next step should not be another broad sweep. It should be a small, fully matched experiment that separates representation, learning objective, and readout. A23 qualifies because it is mathematically different and can falsify the current comparison-space story. The orthogonal-energy explanation, the burst-correctness explanation, and the claimed clean stacking failure should be withdrawn until the matched measurements are done.
