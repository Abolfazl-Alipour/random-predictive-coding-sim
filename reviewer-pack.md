# Reviewer pack: random predictive coding (RPC)

## What to send to another AI reviewer

Copy the block in section 1 exactly, then paste the full text of Chapter 3 in ONE message, in the same chat, right after it. Do not split Chapter 3 into parts. Do not paste only the RPC section. The reviewer needs the full predictive coding review, the model reviews, and the RPC section together, because the claims depend on that context.

Thesis: Abolfazl Alipour, Investigating the Computational Principles of Neural Dynamics Using Deep Learning Techniques, PhD thesis, Indiana University, August 2023. Chapter 3, pages 64-98. RPC section, pages 89-98. Chapter 4, pages 99-132, is a separate sparse autoencoder paper, published as Alipour et al., Journal of Computational Neuroscience 53:131-162 (2025).

## 1. Copy-paste prompt for other AI reviewers

```
You are an adversarial peer reviewer in computational neuroscience. Do not praise the work. Do not summarize it back to me in soft language. Test every claim against the literature and against logic, and say plainly where it fails.

I will paste Chapter 3 of my 2023 PhD thesis in ONE message after this prompt. The chapter proposes a new model called random predictive coding, RPC, on pages 89-98.

Your job, in order:

1. Restate RPC in your own words in 10 lines or less, with no new terms. Name every step that has no equation, no number, and no defined time scale. List each missing number: collision threshold, delay in milliseconds, learning rate, number of random transforms needed, and the exact synapse that strengthens.

2. Prior art check. For each item below, say whether it already contains RPC or a part of it, with a full citation and a date, and quote the decisive sentence:
   - Echo state networks / liquid state machines, Jaeger 2001, Maass et al. 2002, Grigoryeva and Ortega 2018
   - Feedback alignment, Lillicrap et al. 2016, and direct feedback alignment, Nokland 2016
   - Maoz et al., PNAS 2020, random sparse projections that compute surprise with a local rule
   - Hierarchical Temporal Memory, Cui et al. 2016, Hawkins and Ahmad 2016
   - Chen, Zhang, Cameron and Sejnowski, bioRxiv 19 May 2022, Neuron 2024, hippocampal predictive sequence learning with CA3 prediction and CA1 error
   - O'Reilly et al. 2017 and 2020-21, temporal difference predictive learning, 100 ms alpha step, layer 5 bursting
   - Rao and Ballard 1999, Friston free energy 2010, Spratling PC/BC-DIM 2017, Heeger 2017, PredNet Lotter et al. 2016/2017 (note: the thesis cites Lotter 2020 for the architecture; check the date)

3. Dendritic burst check. RPC says a distal feedback signal that collides with a new proximal feedforward signal causes a dendritic burst, and that burst selectively strengthens the one matching random transform. Test this against:
   - Bittner et al. 2015 and 2017, behavioral timescale synaptic plasticity, BTSP: seconds-wide window, potentiation before and after the plateau, bidirectional effects
   - Larkum et al. 1999 and Shai et al. 2015: distal plus proximal coincidence, burst, millisecond window
   - Question: can a cell-wide plateau potentiate one synapse out of hundreds active at the same time? If not, say so and name the missing selection mechanism.

4. Circuit mapping check. RPC maps dentate gyrus to layer IV, CA3 to layer II/III, CA1 to layer V. Test the homology claim, not just the functional analogy. Check the claim that allocortex is three-layered and that developmental work equates it with layers 1, 5 and 6, not layer IV. Check that dentate gyrus input is cortical (entorhinal cortex) while layer IV input is thalamic.

5. Timing check. The thesis says an input at t=1 returns through the trisynaptic path and collides with a new input at t>1. The thesis never defines how long one time step t lasts. Use the 15-20 ms trisynaptic versus monosynaptic delay (Yeckel and Berger 1990) and the return leg CA1 to subiculum to entorhinal cortex. Ask: does the return fit inside one step for a theta cycle (about 100 ms), a video frame (about 33 ms), and a behavioral event (seconds)? State which case works and which case breaks the model.

6. Confirmation signal check. RPC needs a signal back to the originating CA1 neuron to strengthen the CA3 to CA1 synapse that started the prediction. Read Naber et al. 2001, which the thesis cites for this. Naber reports re-entry loops are only rarely observed and only a small percentage of CA1 neurons return to their originating entorhinal area. Say whether that supports or undermines the confirmation signal.

7. Acetylcholine check. The thesis footnote 10 says acetylcholine lets CA3 alone drive CA1 in the end module. Test that against Hasselmo 2006. Hasselmo supports general modulation and encoding versus retrieval phases, not an all-or-none CA3-alone switch. Say what evidence would be needed.

8. Sign-of-burst check. RPC says bursting marks a successful prediction. HTM and BTSP say bursting marks an unpredicted or new event. These predict opposite slopes: RPC predicts bursting rises with transition probability, HTM predicts bursting falls with transition probability. State the experiment that decides this, with the exact prediction for each side.

9. Chapter 4 check. Chapter 4 trains a sparse autoencoder to reconstruct leaky integrator inputs, with velocity as a modulator, not as an input. Its training is invariant to temporal order: the text says temporal order of the samples did not impact training. It has no loop, no prediction, no random reservoir, no burst, and no confirmation signal. Its best thesis count is 70 of 100 place cells with local plus shared weights, versus 45 simple, 22 sparse, 28 local only. The journal version reports sweep means, highest about 65.33, not 70. Answer: does Chapter 4 test RPC at all? Answer with yes or no first, then the reason. Also note: local plus shared weights use only 19 trainable parameters, a shared derivative kernel, and the thesis discussion says this may be genetically coded, not learned.

10. Critique check. The thesis itself concedes that repetition suppression, expectation suppression, omission response, and precision weighting can be explained by Hebbian and spike-timing-dependent plasticity, with no need for prediction. Check Heilbron and Chait 2018, Walsh et al. 2020, Feuerriegel et al. 2021, May 2021, and Furutachi et al. 2024 (prediction errors amplify unexpected input, not subtract). Say whether RPC escapes these critiques or inherits them.

11. Kill tests. Design six decisive tests, one per claim above, each with: the exact manipulation, the control, the metric, the pass bar set in advance, and the result that would falsify RPC. Include at minimum: (a) implement RPC on next-step prediction versus an echo state network readout, HTM, and PredNet; (b) distal selectivity test, pattern A versus equally active pattern A prime; (c) burst versus transition probability; (d) rate tuning across inter-stimulus intervals; (e) lesion of the CA1 to entorhinal return; (f) acetylcholine block.

12. Verdict. Give one verdict for the label, one for the mechanism, and one for Chapter 4 as evidence for RPC. Use these words only: supported, partially supported, contested, unsupported, novel. Then list the three strongest counter-arguments in order, and the single experiment you would run first if you had one month.

Rules for you as reviewer: cite only sources you actually checked, with links. Flag every claim that rests on an index search only. Do not invent a citation. If a paper is paywalled, say so and use only its abstract or a verified secondary quote. Do not soften a fail to keep the author happy.
```

## 2. How to run the review in practice

1. Open a fresh chat with the reviewer AI, so it has no prior praise for the thesis in its context.
2. Send the prompt in section 1 first, alone.
3. Paste the full Chapter 3 text in the next message, in one message. Do not split it. Splitting burns context and lets the reviewer miss the concession on pages 87-89, where the thesis says Hebbian rules can explain the phenomena with no need for prediction.
4. Ask the reviewer to return its verdict table first, then its kill tests. Do not ask for a rewrite.
5. Run the same prompt on at least two different reviewer AIs. Compare only their verdicts and their citations. If they disagree on a citation date, check the primary source yourself before you trust either one.
6. Do not send Chapter 4 as evidence for RPC. Send it only with the question in item 9 above, so the reviewer judges whether it tests RPC at all.

## 3. What my own checks already found (so reviewers cannot surprise you)

1. Label novel, idea not novel. Exact phrase "random predictive coding" had no neuroscience use before or after 2023 in web-index searches on 2026-11-25, only an unrelated 2014 legal use. Every ingredient was published earlier, mostly in quantitative form.
2. Closest prior art: Hierarchical Temporal Memory, 2016, covers distal equals prediction, proximal equals drive, bursting, local Hebbian learning, and fixed random proximal connectivity at 50 percent. RPC adds a reservoir and a loop, and flips the burst sign.
3. Most damaging prior art: Chen et al., bioRxiv 19 May 2022, Neuron 2024. CA3 predicts the next input, CA1 signals prediction error against direct entorhinal input, with the trisynaptic versus monosynaptic delay as the timing mechanism. The preprint predates the thesis, August 2023. The thesis does not cite it.
4. Confirmation signal is the weakest link. Naber et al. 2001, cited for it, says re-entry loops are only rarely observed.
5. Burst selectivity is unproven. BTSP potentiates inputs seconds before and after a plateau, so it cannot pick one matching random transform out of hundreds.
6. Time step t is never defined. That makes the collision timing claim untestable as stated.
7. Chapter 4 does not test RPC. It reconstructs the current input, with shuffled temporal order, by backprop (Adam), with no loop and no burst.
8. Simulation check, run 2026-11-25, file rpc_toy.py in this folder: a two-module reservoir model built from the thesis text gave burst rate 0.000 in all runs and accuracy 0.049 against chance 0.062. No learning fired at all. Cause: distal drive starts near 0.05 in size but must pass a 0.5 threshold. That is a scale mismatch in the first formal version, not a proof the idea is dead. The next step is a threshold sweep, not video work.

## 4. Files in this folder

1. thesis.txt: full thesis text extracted from the PDF.
2. ch3.txt: Chapter 3 only.
3. rpc_toy.py: runnable PyTorch toy simulation, CPU, with the burst rule, confirmation rule, and ablations. Run with ./venv/bin/python rpc_toy.py from this folder.
4. reviewer-pack.md: this file.
