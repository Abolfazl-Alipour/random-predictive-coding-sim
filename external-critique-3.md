# External Critique 3: Random Predictive Coding Simulation Report

Critic: External reviewer
Basis: Chapter3-RPC-Predictive-Coding.pdf, the supplied report, and the stated simulation-results.md / rpc_v2.py record.
Limitation: I could not run the code or inspect line numbers in this environment. Numerical claims below are taken from the report unless they are algebraic derivations I can check directly. I do not invent results.

## 1. Executive verdict

The work is unusually honest and systematic. The negative boundary you report is credible for the implementation you built: a confirmation-gated Hebbian sequence learner with a fixed random proximal projection, top-6 code, and prototype-cosine readout. That implementation fails on noisy/ambiguous tasks, and the failure is not merely a tuning issue.

But the report overstates what this means for predictive coding and for the thesis claim. The model as formalised does not test the predictive-coding assumptions listed in Chapter 3. It lacks an explicit prediction-error population, hierarchical top-down prediction, precision weighting, and omission-response behaviour. Its central gate learns only when the prediction is already correct, which is the opposite of an error-driven predictive-coding update. So the correct conclusion is:

Confirmation-gated RPC with these readout and code assumptions fails on the tested tasks.

It is not yet:

Random predictive coding cannot work.

The most important next steps are not more variants of the same readout. They are: (1) test A23 properly with controls, (2) probe whether the "orthogonal part" actually carries next-token information, and (3) replace or supplement the match-gated burst with an error-driven update. If the theory forbids error-driven updating, that is a serious theoretical problem, not a simulation detail.

## 2. Method critique

### 2.1 Conceptual mismatch with predictive coding

Chapter 3 lists standard predictive-coding assumptions: prediction errors scale inversely with predictions; top-down signals carry predictions; separate predicting and error populations; hierarchy; alpha/beta vs gamma routing. RPC v2 implements none of these directly. It has one module predicting the next token through a learned distal matrix, and a burst gate that fires on confirmation. That is a confirmation-gated associative learner, not canonical predictive coding.

This matters because your results do not adjudicate the theory reviewed in Chapter 3. They adjudicate a specific and rather unusual formalisation. The report should say this explicitly. If the thesis claim is "random predictive coding," the model needs to map onto at least one predictive-coding signature, e.g. omission response or error-driven updating. The current burst-on-match rule is closer to "reinforce correct guesses" than to "minimise prediction error."

### 2.2 Hyperparameter selection on the test set

Alpha 1.0 was selected after seeing that lower alpha collapsed the cycle prototypes. Smooth-input sigma was selected after seeing the sine fail. That is model selection on the test data. The pass bars were fixed in advance, which is good, but alpha and sigma were not. You need a validation split, choose alpha/sigma/learning rate on validation, then run held-out test once.

This matters most for the cycle result. At alpha 1.0 the reservoir tracks the current token, so the cycle is nearly a fixed token-to-token map. Markov-1 scores 1.000; RPC scores 0.843. That is not evidence for temporal predictive coding. It is evidence that the burst rule can learn a trivial deterministic map when the code separates. You already state this scope limitation; keep it central, not buried.

### 2.3 Readout assumptions are not theory

The prototype-cosine readout, top-6 code, and row-norm preservation are your assumptions. You acknowledge this, but the report still sometimes treats "three theory-faithful readouts" as if cosine, dot, and kNN were equivalent tests of the theory. They are not. kNN stores training exemplars; dot weights prototypes by norm; cosine normalises. Only the prototype-cosine readout is the one you originally specified. Agreement among three similar fixed-space readouts does not show the readout is not binding; the trained ridge result shows the opposite: the drive contains linearly decodable information that the fixed readouts miss.

The full graded code is more faithful than top-6, but it is bimodal: seeds 0 and 1 work, seeds 2-4 collapse to chance. A mean over five seeds is not a valid summary of a bimodal result. Report per-seed values, median/IQR, and paired comparisons. Five seeds is too few for a bimodal condition.

### 2.4 Controls and baselines are good but incomplete

No-plasticity and no-confirmation controls are the right start. Add:

- random fixed W_dist1 with the same row norms;
- label-shuffled training for the readout/prototypes;
- the same A23 centroid procedure applied to the no-plasticity and no-confirmation controls;
- validation-selected hyperparameters;
- paired seed tests and confidence intervals.

The baselines are informative. Markov-1 beats RPC on planted motifs (0.278 vs 0.113), and Markov-1 scores 1.000 on the cycle. Passing a bar set at 0.2625 is not meaningful if a one-step baseline scores 0.278. The bar should be baseline-relative, not just above chance.

### 2.5 "Loses a third" is accuracy, not information

The stage probe says token decode falls from 0.984 to 0.711 through the random proximal projection. That is a drop in linear-decode accuracy, not a loss of one third of the information. For a 32-way or 16-way task, 0.711 is still far above chance. You need mutual information or decoding error, not just accuracy, before saying a third of the information is destroyed.

### 2.6 A22 correction is right, with one refinement

Your A22 correction is essentially correct. If P is the orthogonal projection onto the span of the prototypes p_i, then for cosine ranking over prototypes:

cos(Pd,p_i) = <Pd,p_i> / (||Pd|| ||p_i||) = <d,p_i> / (||Pd|| ||p_i||).

The denominator ||Pd|| is common across i, so the argmax is identical to the unprojected cosine ranking. Dot product is also unchanged because <Pd,p_i> = <d,p_i>. So A22 is a no-op for accuracy/ranking with cosine and dot. It is not a no-op for score values or thresholds, and it is not a no-op for kNN because bank vectors need not lie in the prototype span. Your report should state this precisely.

### 2.7 The kNN contradiction in section 4

Section 4 says "no fixed comparison against p1 patterns can see the orthogonal part" and then includes kNN among the failures. But kNN bank vectors are individual proximal-drive vectors, not necessarily in the span of the averaged prototypes. A22 is non-trivial for kNN exactly because of this. So the statement is too strong. The correct statement is: cosine and dot against the averaged prototypes cannot see the orthogonal component; kNN can see some of it, but in your runs it did not help enough.

## 3. Root-cause chain audit

Step 1: module-1 reservoir holds the current token perfectly (1.000).
Probably true for the probe condition, but alpha is not specified. At alpha 1.0 this is almost tautological. At alpha 0.3 it is a stronger claim. Report the alpha used.

Step 2: fixed random proximal projection loses a third (0.984 to 0.711).
Supported as an accuracy drop, not as information loss. A random projection can preserve linear separability if the dimension is high enough; the drop may be due to the 64-dimensional bottleneck, not randomness per se.

Step 3: top-6 selection loses another half (0.711 to 0.332).
Supported for that readout, but top-6 is your assumption. The full graded code recovers token decode to 0.709, so top-6 is a real bottleneck. However, the full code then makes the learned drive carry less (0.457 vs 0.509). That is a separate effect of learning, not just selection.

Step 4: the learned drive holds next-token information but it is orthogonal to the prototype span.
This is the weakest link. The trained ridge probe reads the drive at 0.408-0.507, while fixed cosine reads it poorly. That shows the information is not aligned with the averaged prototypes. It does not show that the 31% orthogonal energy is the informative part. Orthogonal energy can be noise. You need to decode from Pd and (I-P)d separately. If the orthogonal part decodes next token well and the projected part does not, the claim is supported. If both are weak, the "orthogonal part" story is incomplete.

Step 5: no fixed comparison against p1 patterns can see the orthogonal part.
True for cosine/dot against the averaged prototypes. Not true in general for kNN. Also, a fixed comparison against drive-space centroids (A23) can see it. So this step is readout-specific, not a fundamental limitation.

Step 6: the burst gate makes misalignment worse.
Plausible and important. The gate only learns when the prediction already matches. On noisy tasks the early predictions are poor, so the gate rarely opens or opens on idiosyncratic matches. Measure burst rate per task and per seed. If burst rate is very low, the gate cannot bootstrap. This is a structural problem, not a tuning problem.

Step 7: smooth input helps because it encodes direction.
Supported by the direction-decode increase from 0.748 to about 0.915 and the sine improvement. But smooth input was selected after seeing the failure. It is a legitimate representational change, but it is not a test of the original one-hot theory.

Simpler explanation: the fixed random proximal projection plus top-6 code plus prototype-cosine readout is a low-capacity, misaligned classifier. The drive may contain linearly decodable next-token information, but not in the directions the fixed readout uses. The burst gate then entrenches an idiosyncratic encoding. This explanation does not require a special "orthogonal information" mechanism; it requires a readout that lives in the drive space and an update that corrects errors.

## 4. Is A23 the right next test?

A23 is not another A22. Work through the algebra. After training, let W be the distal matrix and c the module-2 code. The drive is d=Wc. A23 builds, for each next token t,

q_t = (1/N_t) sum_{s: next(s)=t} d_s = W m_t,

where m_t is the mean module-2 code for training steps whose next token is t. The A23 score is cos(d,q_t). This is a nearest-centroid classifier in the drive space with metric G = W^T W:

cos(d,q_t) = <c, W^T W m_t> / (sqrt(c^T W^T W c) sqrt(m_t^T W^T W m_t)).

It can see directions in the drive that are not in the span of the original proximal prototypes p_i. So it is not a mathematical no-op like A22. It is worth one run.

But it is confounded in two ways. First, q_t is built using labels and the trained network's own drive. That is a supervised nearest-centroid readout, even though no gradients are trained. Second, on ambiguous tasks like the sine, q_t averages over rising and falling contexts that share the same next token. If those contexts produce different drives, the centroid may sit between them and match neither. That is exactly the ambiguity that split prototypes tried to address.

So run A23, but only with:

- the same A23 procedure applied to the no-plasticity and no-confirmation controls;
- per-seed results, not just a mean;
- Euclidean and cosine centroids;
- a mixture of several centroids per token if the single centroid is near-parallel;
- the orthogonal-probe diagnostic in the same batch.

If trained A23 beats control A23, learning is adding drive-space class structure. If trained A23 ties control A23, the readout was the bottleneck. If both are near chance, the drive lacks separable structure or the gate is preventing it. A23 alone will not resolve the root cause.

## 5. The correct next experiment

Do not run another readout variant in isolation. Run one clean batch:

Model changes: keep the random proximal projection for the test of the thesis claim. Use the full graded code, not top-6. Keep row-norm preservation. Replace the match-gated burst with an error-gated or delta-rule update for one condition, while keeping the confirmation-gated condition as the thesis-faithful control. The error-driven condition is the single most promising change because it can correct a misaligned map; the confirmation gate cannot.

Tasks: planted motifs and fixed sine first. Add variable sine and Lorenz only after one of them passes. Use 20 seeds. Split train/validation/test by sequence. Select alpha, sigma, learning rate, and readout shrinkage on validation only.

Baselines: Markov-1, Markov-2, Markov-3, echo state network with the same reservoir, LSTM. Report all.

Controls: no plasticity, no confirmation, random fixed W_dist1, label-shuffled readout, A23 readout built on each control.

Readouts: p1 cosine, p1 dot, kNN, A23, A23 on control, trained ridge as a diagnostic only, and linear probes on Pd and (I-P)d.

Bars: for planted motifs, beat the one-step ceiling of about 0.36 and at least match Markov-1/ESN. For sine, at least match Markov-2 (0.802 in your report) or clearly beat the ESN with the same reservoir. A pass bar of 0.231 is too weak when Markov-2 is 0.802.

Reporting: per-seed table, median/IQR, paired seed differences, and burst rate. Do not report a mean over a bimodal condition without the per-seed values.

## 6. Random wires: what the results do and do not show

The fixed random proximal projection loses accuracy, but random projections can preserve information if the expansion is high-dimensional and the readout is appropriate. Your loss is confounded by the 64-dimensional bottleneck and top-6 selection. So the results do not prove that random wires are insufficient. They prove that this random-wire architecture with this readout and gate is insufficient.

There are three possible versions of the thesis claim:

1. All wires random, readout fixed: current evidence is negative. A23 is the last reasonable fixed-readout test; if it fails, this version is in trouble.
2. Proximal wires random, distal wires learned, readout learned/centroid: this is close to reservoir computing. The trained-ridge result already shows the reservoir contains the information; the RPC gate must then add something beyond the ESN. Your current gain-over-control results are not enough to show that.
3. Proximal wires must be learned: if the learned proximal projection is the only thing that fixes the code, then the thesis claim "fixed random wires are enough" is false for this architecture. The smallest learned component could be the readout/prototypes rather than the wires, which would preserve the random-wire claim. But if learning the proximal projection is necessary, say so plainly.

## 7. Errors and corrections audit

Your section 5 is mostly right. Specific notes:

- V1 scale bug: correct to fix; not a theoretical issue.
- Prototype misstatement: correcting "distal" to "proximal" matters; good.
- Dilution explanation: correctly withdrawn. A20 and A21 show concentration is not the issue.
- Planted stacking: withdrawing then reinstating on clean same-rule evidence is correct.
- A22: correction is right for ranking; refine that score values/thresholds and kNN are not invariant.
- Smoke tests: correct to stop trusting 1-epoch verdicts.
- Ceiling error: correct; three motifs change the one-step ceiling.
- Smooth-input 1.5 overstatement: correct to regress at 5 seeds.
- Graded confirmation: correct that same-LR hurts and 5x-LR only ties binary.

Add these to the error list: "loses a third of information" should be "drops decode accuracy"; "orthogonal energy" is not "orthogonal information"; section 4's kNN statement contradicts A22; alpha/sigma selection on test is a model-selection error.

## 8. Bottom line

The simulation record is credible as a negative result for the current RPC implementation. The root cause is best stated as: the fixed random proximal projection plus top-6 code plus p1-prototype readout creates a drive whose next-token information is not aligned with the comparison vectors, and the match-gated burst cannot correct that misalignment because it only learns from already-correct predictions.

A23 is not a no-op and is worth one controlled run. But the decisive next test is an error-driven update that keeps the random proximal wires. If that fails, and A23 with controls fails, then the theory as formalised needs either a learned proximal projection or a learned readout. That would directly qualify the thesis claim that fixed random wires are enough.
