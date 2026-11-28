# Rounds 17–18 critique: the clamp and the continuous wall

**Reviewer:** Kimi  
**Scope:** `simulation-results.md` sections 44–53, the round-17/18/19 protocols, A1–A27 in `rpc_v2.py`, both drivers and analyzers, both analysis reports, all 200 round-17 and 360 round-18 test records, and the two round-14 critiques. The local pack omitted `rpc_round15.py` and `rpc_round16_omission.py`; for the original scoring path and diagnostic reproductions I used the public files at commit [`b2543c73`](https://github.com/Abolfazl-Alipour/random-predictive-coding-sim/tree/b2543c73).

I recomputed the paired intervals from the JSONLs and ran two small diagnostics. They are not new preregistered results: (1) seed 241 reproduced the round-17 record exactly, then varied gain only at test time on the same gain-0.1-trained net; (2) Lorenz seed 262 reproduced the recorded ridge score and compared raw and column-standardized ridge over lambda `{1e-6, 1e-4, 1e-2, 1, 1e2, 1e4}`, selecting lambda independently for each condition on its validation split. I also recomputed persistence after quantizing its output to a bin centre.

## Lead verdict

**Experiment B's failure stands only in this precise form: the validation-selected, fixed random additive A26 wire at gain 0.1 fails the preregistered lag-1 conjunction C1–C4.** The broader headline “the wire carries no causal role” is too strong. The lag-1 on-minus-cut omission effect is small but its exact 95% CI is above zero (`+0.004936`, CI `[+0.000118,+0.009754]`), and the final training error is not identical: wire-on minus wire-off is `+0.002031`, CI `[+0.001574,+0.002488]`, positive on 19/20 seeds. There is a tiny causal effect, but no evidence that the prediction's content does useful work: shuffle does not differ from on, and the 0.10 practical-effect bars fail decisively. **The strongest reason to doubt the general feedback conclusion is the winner pick itself:** normal accuracy selected the smallest positive gain in a tie with no wire, zero was not an eligible winner, and neither omission performance nor a cut gap was measured during validation.

**The continuous-wall verdict needs to be split.** The original Lorenz C1 prototype-readout MSE wall is verified and large. H0 also fails exactly as the frozen conjunction requires. But the var-sine MSE wall is unresolved rather than established: D3 has the wrong point estimate but includes zero. More importantly, “delta does not learn continuous information” is refuted as a blanket interpretation. Delta massively improves continuous MSE over frozen nets with the same prototype readout, while D8 asks a narrower question about a raw, fixed-lambda linear probe. **The strongest reason to doubt the broad continuous-learning conclusion is that D8 repeats the round-14 scale confound:** on a directly reproduced seed, trained drives are 4–5.5 times smaller; the frozen advantage at lambda 1 disappears under validation-selected or standardized ridge. Round 18 establishes that delta does not improve *that particular ridge probe on these two low-order tasks*, not that it learns no continuous information.

## 1. Is round 17's wire live, and is it inert by construction?

**Verdict: verified live; refuted that gain 0.1 cannot matter by construction; the “99.8 percent identical” wording is misleading.**

The wire is computed from the pre-input `d1` and passed into module 1 at `rpc_v2.py:1122-1143`. `ReservoirModule.forward` adds it to `W_in @ e_sparse + W_res @ r - inhib`, then applies `tanh`, then the leak update (`rpc_v2.py:554-574`). Thus it enters **before the nonlinearity and before the leak**. Round 17 uses `alpha=1.0` (`rpc_round17_clamp.py:77-86`), so the leak supplies no attenuation at all. Feedback bypasses the input expansion/top-k stage, and there is no post-feedback normalization of `r1` or `p1`; `p1` is simply `W_prox @ r1` (`rpc_v2.py:570-574`). Module 2's later standardization (`rpc_v2.py:761-763`) cannot retroactively erase the current `p1` change.

The remaining attenuation is ordinary `tanh` sensitivity, not a hard threshold or a normalization that makes the term a no-op. The observed `r1_max` stays below 0.92 across the wire-on test records, so this is not a hard-clipping explanation. The reported 6.5% is also a ratio to the **bottom-up** norm, not to total recurrent-plus-input drive (`rpc_v2.py:1281-1288`). It therefore does not license an inference of zero effect.

`p1_cos` is a cosine between state vectors (`rpc_round17_clamp.py:130-145`), not a percentage of identical cells or a task-relevant effect size. Cosine 0.998 corresponds to an angle of about 3.6 degrees; a small orthogonal perturbation can leave cosine near one while changing decisions near prototype boundaries. Conversely, a large cosine change is not necessary for every possible causal effect. C3 is a deliberately demanding state-perturbation bar, not a general test of causality.

The direct gain diagnostic removes any doubt that the implementation can respond:

| Test-time gain on the same seed-241, gain-0.1-trained net | p1 cosine on vs cut | normal accuracy on | normal accuracy cut |
|---:|---:|---:|---:|
| 0.1 | 0.997982 | 0.221739 | 0.229565 |
| 0.3 | 0.982217 | 0.213043 | 0.229565 |
| 1.0 | 0.843305 | 0.167826 | 0.229565 |
| 3.0 | 0.441319 | 0.122609 | 0.229565 |

Gain 1.0 therefore passes C3's `<0.90` state-change threshold in this diagnostic. It is causal and harmful to ordinary accuracy, not inert. This is consistent with validation accuracy falling at gain 1.0; it does not show that stronger feedback cannot pass a clamp test. Training at each gain could change the result, so this sweep is not a substitute for that experiment.

The defensible conclusion is “gain 0.1 has a very small effect under this random additive wiring.” For a real test of useful predictive feedback, use a measured state-effect gate (gain 1.0 is sufficient on the diagnostic seed) and separately test utility. To test Rao–Ballard-style explanation away rather than arbitrary recurrent perturbation, the deeper code change is an **aligned or learned feedback map**, ideally with an explicit error pathway. A random additive reservoir drive is not itself the canonical computation.

Finally, the error-magnitude result is not null. It rises slightly with the wire, opposite the protocol's explain-away expectation (`PROTOCOL-R17-CLAMP.md:145-148`; `r17/r17_analysis.md:41-47`). “Never building the wire is the same as building it” (`simulation-results.md:2970-2972`) is therefore too strong even about training.

## 2. Is the lag-2 C1 pass meaningful?

**Verdict: refuted as a protocol-licensed pass; the practical effect is negligible; experiment B is unchanged either way.**

The frozen protocol defines C1 as a cut degradation of **at least 0.10 with CI above zero**, says lag-1 is primary and lag-2 is reported, and never specifies a separate floor-free lag-2 rule (`PROTOCOL-R17-CLAMP.md:154-177`). The analyzer, however, applies the 0.10 floor only to lag-1 and silently changes lag-2 C1/C2 to `ci_lo > 0` (`r17/r17_analyze.py:169-191`). The weaker lag-2 bar is therefore an analyzer invention, not a threshold stated in the frozen protocol.

Numerically, lag-2 C1 is `+0.004194`, exact CI `[+0.000400,+0.007988]`. It would fail the 0.10 floor by a factor of about 24. The lag-1 effect is also statistically above zero but fails the same practical floor. These are tiny effects on a coarse omission accuracy measure, not evidence of useful top-down prediction.

This does not change experiment B: its verdict is determined by lag-1 C1–C4, and C2, C3, and C4 fail regardless. The reported “2/9” also counts C5, which the protocol calls a guard rather than a pass bar. Under the frozen C1 definition, lag-2 C1 should not be counted as a pass at all.

## 3. Does the winner pick bias the clamp toward a null?

**Verdict: verified as a design bias toward a weak or utility-free wire; whether another selection metric would choose another gain is unresolved.**

Validation optimizes ordinary accuracy only (`PROTOCOL-R17-CLAMP.md:99-108`). The analyzer considers only gains `>0` as winners; gain 0 is used solely for the guard (`r17/r17_analyze.py:98-116`). Thus the procedure must select a positive wire even if every positive gain is worse than no wire. The selected values are `0.2260869565` at gain 0.1 and `0.2266086957` at gain 0 (`r17/winner.json:1-5`): gain 0.1 wins among positive gains while being 0.000522 worse than no wire.

The validation driver records no omission score, no on-versus-cut gap, and no state cosine (`rpc_round17_clamp.py:335-357`). It cannot deliberately select a causally potent or omission-relevant wire. This does not mathematically force a null—causal but harmful feedback could still produce a large clamp effect—but it biases against finding beneficial feedback and makes the round's general feedback language unwarranted.

Whether validation on omission accuracy would change the pick is unknowable from the saved record because those measurements were not taken. Selecting on the **validation** cut gap is more directly aligned with the causal question, but it needs a stability guard, a shuffle/content check, and a state-effect measurement so that the largest noisy or destabilizing perturbation does not win. Five validation seeds would also produce winner's curse, especially because those seeds are reused in the reported 20-seed test (`PROTOCOL-R17-CLAMP.md:244-251`). A clean rerun should use disjoint validation and test seeds.

The observation that gain 1.0 lowers ordinary validation accuracy is not evidence against its causal potency. The diagnostic above shows the opposite about state perturbation. Utility and causal effect must be separate selection criteria.

## 4. Are D2/D3 too borderline, and is the var-sine MSE wall established?

**Verdict: H0 failure is verified under the frozen rule; the var-sine MSE wall itself is unresolved.**

D2b is a genuine pass under the preregistered normal-approximation interval, not an exact tie: C1 minus Markov-1 is `0.010808`, exact CI `[0.000312,0.021304]`, with 13/20 seeds positive (`r18/r18_analysis.md:54-56`; `r18/r18_analyze.py:196-207`). D3 is not evidence of equivalence or of a established persistence advantage: persistence minus C1 MSE is `-0.008275`, exact CI `[-0.018325,+0.001775]`, with 13/20 seeds in the wrong direction (`r18/r18_analysis.md:58-60`). D2 and D3 can diverge because exact-bin accuracy ignores the distance of errors, whereas MSE is dominated by distant-bin mistakes.

H0 passes only if D1, D2, and D3 all pass (`PROTOCOL-R18-CONTINUOUS.md:240-253`), so **H0 FAIL is correct**. That is a failure to demonstrate the model's superiority, not positive proof that persistence is superior. The record should say “no demonstrated var-sine MSE advantage” rather than treating the var-sine wall as settled.

A 40-seed rerun cannot be known from the existing 20 seeds. If the present means and standard deviations happened to remain unchanged, simply doubling n would give approximately:

- D2b: `0.010808`, CI `[0.003386,0.018230]` — still a pass.
- D3: `-0.008275`, CI `[-0.015382,-0.001169]` — a failure that now also excludes zero in persistence's favor.

Those are precision projections, not predictions about fresh draws. The appropriate rerun is fresh seeds with no retuning, ideally 40 independent data/network seeds. A preregistered equivalence or non-inferiority margin is needed if the desired claim is “no practically important MSE difference,” rather than merely “the model failed to prove a win.”

By contrast, Lorenz D4 is decisive: persistence minus C1 is `-9.784519`, CI `[-12.028579,-7.540459]`, and all 20 seeds favor persistence. The robust wall is specifically the **original Lorenz C1 prototype configuration's MSE wall**, not every tested continuous configuration.

## 5. Does D8 show “delta adds nothing” or “delta actively harms the ridge”?

**Verdict: implementation parity is verified; frozen superiority under the raw lambda-1 probe is verified; both broad mechanistic interpretations are unresolved after the scale sensitivity check.**

The comparison is otherwise matched. Trained and frozen nets use the same seed, data, initialization, calibration, and prototype construction; the condition changes only plasticity (`rpc_round18_continuous.py:123-180`). The same collection code gathers `d1(t)` with target `v(t)` for `t>=1`, and the same centred ridge with intercept and lambda 1 is fit on train only (`rpc_round18_continuous.py:188-213,231-246,316-373`). There is no test-label fit and no trained/frozen probe-procedure mismatch.

But the features are **not standardized**, and the penalty is `D_c.T @ D_c + lambda I` (`rpc_round18_continuous.py:238-246`). Centering fixes the round-14 no-intercept issue, not the scale-dependent regularization issue. Using the same lambda as the ESN does not make penalties comparable across differently scaled features.

On the 20 preregistered seeds, the result is worse than “no gain”: frozen minus trained MSE is `-0.584112`, CI `[-0.734631,-0.433592]`, for one-hot, and `-0.180136`, CI `[-0.236163,-0.124109]`, for bump. Frozen wins on 19/20 seeds in both comparisons. So delta **actively harms this fixed-lambda raw ridge** on Lorenz. The var-sine analogue is essentially tied (`r18/r18_analysis.md:74-82`).

The seed-262 sensitivity diagnostic shows why that statement cannot be generalized to information content:

| Lorenz seed 262 | Mean train `||d1||`, trained / frozen | Test MSE at raw lambda 1, trained / frozen | Validation-selected raw ridge, trained / frozen | Validation-selected standardized ridge, trained / frozen |
|---|---:|---:|---:|---:|
| One-hot | 1.025 / 5.651 | 14.904 / 14.073 | 13.982 / 14.005 | 13.997 / 14.008 |
| Continuous bump | 1.297 / 5.609 | 5.326 / 5.123 | 4.848 / 4.841 | 4.849 / 4.841 |

Thus the frozen advantage at lambda 1 is associated with a 4.3–5.5-fold drive-scale difference and disappears when regularization is validation-selected or features are standardized. This one seed does not establish a 20-seed tie; it does refute using the fixed-lambda result as clean evidence about information. D8 remains a preregistered FAIL, but its mechanism needs a fresh, validation-tuned probe replication.

There is also a structural limit. Frozen `W_dist1` is a full random linear transform of `y2_prev`; a ridge on frozen `d1` can already learn the composite linear map from that pre-code. A learned linear `W_dist1` cannot create new information out of the same pre-code, though it can change conditioning, discard dimensions, or make information accessible to the fixed cosine readout. A direct `y2_prev` ridge, feature singular-value/rank measurements, and independently validated regularization are needed. This is essentially the round-14 warning (`r14-kimi-critique.md:56-71`; `r14-agy-critique.md:147-169`), still applicable here.

Most importantly, the other round-18 conditions directly contradict the blanket wording “delta does not learn continuous information.” Frozen-minus-trained prototype MSE is `178.863` on Lorenz one-hot, `250.111` on Lorenz bump, `0.798` on var-sine one-hot, and `0.840` on var-sine bump, positive on 20/20 seeds in all four cases (`r18/r18_analysis.md:14-21,33-40`). Delta clearly learns a mapping that makes continuous prediction vastly better under the prototype readout. What D8 does not show is an **additional linearly ridge-decodable advantage over the frozen pre-code**. Those are different claims.

The clean interpretation is therefore: delta converts existing reservoir information into a prototype-usable prediction, while the raw lambda-1 ridge on trained drives is conditioned worse. “No added continuous information under one probe” may survive a fair follow-up; “no continuous learning” does not.

## 6. Is the round-18 scoring correction complete?

**Verdict: verified for temporal alignment; no remaining one-step shift found in the round-18 numbers. Other fairness caveats remain.**

In `rpc_v2.py`, `d1` is computed from `y2_prev` before the current input reaches either module (`rpc_v2.py:1122-1143`), and `run()` scores `pred(t)` against token `t` for `t>0` (`rpc_v2.py:1330-1340`). Round 18's `collect` follows that same convention for prototype and ridge targets (`rpc_round18_continuous.py:188-213`); `proto_metrics` compares the predicted bin and decoded value with the same-time target (`rpc_round18_continuous.py:216-228`), and `ridge_metrics` does likewise (`rpc_round18_continuous.py:249-262`).

The original bug is exactly as described: public-commit `rpc_round15.py:205-223` collects `pred(t)` and sends it to scorers that compare index `t` with target `t+1`; `baselines.py:307-330` implements that shifted comparison. Round 18 no longer calls those functions for model predictions. Markov predictions use tokens through `t` to predict `t+1` (`baselines.py:167-193`), and the ESN state after token `t` predicts `t+1` (`baselines.py:218-255`). Public-commit `rpc_round15.py:229-249`, reused by round 18, scores those baselines on positions 1 through the end. This is the same one-step target and denominator as round 18's model scores. I found no residual shift in C1–C6, C2f/C4f, Markov-1/2, persistence, or ESN.

Three qualifications do not involve timing:

1. Persistence MSE uses the exact previous continuous value (`baselines.py:333-340`), whereas prototype/Markov/ESN predictions are decoded bin centres. That is an information/output mismatch, not a shift. My output-matched diagnostic changes pooled Lorenz persistence MSE only from `12.605169` to `12.827181`, and var-sine from `0.055928` to `0.055900`; it does not explain D3 or D4.
2. The label `C1_onehot` is inaccurate on var-sine: the frozen round-15 winner has `smooth_sigma=2.0`, and `net_kw` applies it whenever `bump=False` (`rpc_round18_continuous.py:70-73,123-130`). C1 is bin-centred smooth input without within-bin information, not literal one-hot input.
3. C4's continuous bump gives it information the ESN baseline does not receive. Its paired mean advantage over ESN is real on these runs, but it is not a same-input architecture comparison. D7's preregistered bar concerns C2/C3, not ESN (`PROTOCOL-R18-CONTINUOUS.md:234-238`).

## 7. Does Markov-2 show that the tasks, rather than the model, create the wall?

**Verdict: verified that the tasks strongly favor two-bin lookup; refuted that this makes D4/D8 wholly uninformative; it sharply limits their scope.**

Variable sine is noiseless within a sequence, with fixed frequency and phase (`rpc_round15.py:59-75` at the public commit). A continuous sine obeys a second-order recurrence, so two past values nearly determine the next one. Two past bins retain enough local velocity information for a strong lookup. Lorenz is only partially observed, but over a short smooth step two past x bins likewise approximate local velocity (`baselines.py:95-135`). Markov-2's means—Lorenz accuracy `0.556` / MSE `1.416`, var-sine accuracy `0.531` / MSE `0.005294`—are therefore diagnostic of low-order task structure, not merely an extra baseline inconvenience.

This does not make D4 uninformative: it decisively rejects the original C1 readout/configuration on Lorenz. It also does not show that a one-step reservoir cannot do continuous prediction: ESN reaches Lorenz MSE `4.308`, and C4 reaches `3.795`. It does show that these tasks cannot support a broad claim about learning long-range, stochastic, or hidden-state continuous dynamics. D8 is even narrower because it combines easy low-order tasks with the linear-probe limitations above. Markov-2 was not a frozen bar, so it does not change the formal H0 verdict; it should change the next task choice.

A discriminating continuous task should make two-bin lookup inadequate while keeping history useful: for example, a stochastic switching dynamical system with a latent regime persisting over many steps, noisy partial observations, and regime-dependent continuous dynamics. Regime changes and observation noise should prevent exact phase/velocity recovery from two bins; an oracle-regime condition and a noise ceiling should verify that long-history integration can outperform lookup. All models should receive identical continuous observations and use comparable continuous outputs. Include Markov-1/2 with declared backoff, quantized and exact persistence, direct autoregressive ridge, ESN with the same bump input, and delta with a continuous target. MSE should be supplemented by probabilistic scoring or an irreducible-noise estimate so that noise is not mislabeled as model failure.

## 8. Does A-pass plus B-fail strengthen the predictive-coding claim?

**Verdict: refuted as support for causal Rao–Ballard predictive coding; verified only as support for a learned prospective code.**

A-pass shows that a trained prediction exists, is stimulus-specific, survives a missing current input, and supports continuation. B-fail means the tested prediction pathway does not do substantial causal top-down work. That combination strengthens the narrow claim “the network contains useful predictions,” but weakens the claim that those predictions implement hierarchical predictive coding. It is fully compatible with feedforward sequence prediction from a fixed reservoir.

The omission result is especially non-unique at time `t`: `d1(t)` is computed before the blank, so omission decode equals normal decode by construction (`simulation-results.md:2430-2443,2582-2590`). Any correctly aligned one-step predictor has that property. Continuation after the blank is more informative about recurrent state, but a plain reservoir with NLMS is precisely the operative mechanism here: the update is signed normalized LMS on `W_dist1` (`rpc_v2.py:1168-1184`). The frozen control does not supply a trained plain-reservoir/NLMS omission comparator; delta is that comparator's learning rule. ESN's stronger task scores further show that the phenomenon is not evidence of a distinctive mechanism.

The record's statement that “a fixed random feedback wire cannot deliver a stimulus-specific template” (`simulation-results.md:3011-3014`) is also incorrect as stated. `W_fb @ d1` remains stimulus-specific when `d1` is stimulus-specific; a random projection can preserve discriminative information. What the fixed random wire lacks is a demonstrated **alignment** to the lower-level representation or an error-computation pathway. That is the problem to test, not the existence of stimulus specificity.

What survives of the thesis claim is:

- fixed random recurrent features plus a strictly local signed-error update can learn discrete sequential predictions;
- the learned distal state is stimulus-specific and persists across omissions;
- the same local learning can make continuous information accessible to the fixed prototype readout, even when it does not improve an offline ridge decode.

What does not survive as demonstrated is Chapter 3's burst-confirmation mechanism, causal top-down explanation away, distinct error populations, precision weighting, or a unique advantage over reservoir plus Widrow–Hoff/NLMS. The accurate current name remains **local error-driven prediction learning on fixed random reservoirs**, with an omission-compatible predictive representation—not validated predictive coding.

## 9. Does round 19 repeat round 17's null by construction?

**Verdict: verified as a serious risk; the design measures feedback drive but not sufficient test-time state efficacy.**

Round 19 still uses a fixed random `W_fb_l` with the A26 additive form (`PROTOCOL-R19-HIERARCHY.md:67-79`). It is not a learned feedback matrix; only its input `d_l` is learned. Thus section 53's suggestion that hierarchy automatically supplies a non-random feedback wire is wrong.

Validation contains only feedback gains 0.1 and 0.3, selects on overall accuracy, and has no zero-feedback candidate for the slow candidate architecture (`PROTOCOL-R19-HIERARCHY.md:196-210`). Overall accuracy is dominated by positions other than the late context-dependent m1 events. H8 then tests on-minus-cut **overall** accuracy rather than the late-m1 mechanism the feedback is meant to support (`PROTOCOL-R19-HIERARCHY.md:292-302`). This reproduces round 17's insensitive selection and adds an insensitive outcome measure.

The protocol does record per-wire `fb_rms_ratio` (`PROTOCOL-R19-HIERARCHY.md:250-257`), so it measures input magnitude. But p1 cosine is only a smoke check that the value is below 1, with no minimum effect and no per-level test-phase state gate (`PROTOCOL-R19-HIERARCHY.md:332-344`). A null H8 at an unknown or tiny effective perturbation would again be uninterpretable. The slower level-2 leak also means round-17 gain magnitudes cannot simply be transferred between levels.

There is no universal gain that “matters”; it must be measured per level against total state change and content-specific behavior. Round 17 supplies a concrete starting point: gain 0.1 yielded cosine 0.998, while gain 1.0 yielded 0.843 on the diagnostic net. The cheapest real repair is to **replace round-19 validation with a causal, task-relevant selection**: include gains `{0,0.1,0.3,1.0}`, require per-wire on/cut cosine below a frozen threshold such as 0.90 with a stability guard, and select on validation late-m1 on-minus-cut (with shuffle as the content control), not overall accuracy. H8 should likewise be judged on late m1, with overall accuracy secondary. If no stable gain passes the state gate, stop and report an underpowered wire rather than a feedback null.

If the scientific question is useful predictive coding rather than merely demonstrable perturbation, the next step after that minimal repair is an aligned/learned feedback map. Increasing the gain of an unaligned random map mainly produces larger harmful interference, as the round-17 diagnostic illustrates.

## 10. Precision (C) or generative (D) next?

**Verdict: C would settle more, but only after a real causal feedback pathway is in place. D is next under the round-18 scope rule, not the stronger discriminator.**

A generative rollout can be produced by an ordinary autoregressive NLMS model. Round 16 already demonstrates forward continuation after an omission. Another D pass would show generative competence, but would not distinguish predictive coding from local error learning unless the rollout is driven by an internal top-down pathway and cut/shuffle abolishes content-specific generation. Feeding an external argmax readout back as the next input would demonstrate an autoregressive decoder, not the missing Rao–Ballard mechanism.

Precision gets closer to a defining computation, but only if it is more than a variable NLMS learning rate. To change the verdict, C would need all of the following:

- an explicit sensory/reliability manipulation independent of token identity and prior expectation;
- precision-weighted prediction error that affects ongoing state or learning, with a measured dose-response;
- an expectation-by-precision interaction on stimulus-specific error/omission responses, not merely higher average update rates;
- a live, state-effective top-down pathway whose cut and content shuffle remove the precision-dependent effect;
- matched plain-reservoir/NLMS controls with the same average update magnitude, plus fresh held-out seeds.

Without the causal feedback requirement, a C pass would establish precision-modulated local error learning, which is still compatible with adaptive NLMS. A C failure in the current no-feedback architecture would also be nearly uninformative about hierarchical predictive coding.

A meaningful D would need frozen-weight, multi-step free running without teacher forcing; correct context-conditional joint sequence statistics rather than one-step accuracy; continuous as well as discrete targets where relevant; Markov-2, ESN, and plain-NLMS generative controls; and on/cut/shuffle conditions showing that internally generated content depends on the learned top-down state. Even then, a pass would show a learned generative model, not by itself uniquely predictive coding.

## Next experiment

Run a **precision-weighted causal-feedback experiment (B′ + C), not an unchanged-architecture D rollout**. First validate an aligned/learned or at least demonstrably state-effective feedback wire on disjoint validation seeds, using per-level state change and content-specific late/omission effects rather than overall accuracy. Then freeze it and manipulate sensory precision orthogonally to expectation on fresh test seeds. The verdict changes only if high versus low precision produces the predicted error-response interaction, on-versus-cut and on-versus-shuffle show substantial stimulus-specific effects, and matched ordinary NLMS controls fail under the same measurements. If the wire cannot pass the state/content gate, the honest result remains local error learning; if D-style free generation passes without that gate, it is evidence of autoregressive generation, not a resolution of the predictive-coding question.
