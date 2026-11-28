# Round 22 analysis: the precision test (experiment C)

Test records: 480 (expect 480: 8 configs x 3 tasks x 20 seeds).
planted1: aligned-on=20, aligned-cut=20, aligned-shuffle=20, random-on=20, random-cut=20, random-shuffle=20, wire-off=20, frozen-aligned=20
planted2: aligned-on=20, aligned-cut=20, aligned-shuffle=20, random-on=20, random-cut=20, random-shuffle=20, wire-off=20, frozen-aligned=20
sine: aligned-on=20, aligned-cut=20, aligned-shuffle=20, random-on=20, random-cut=20, random-shuffle=20, wire-off=20, frozen-aligned=20

## Floors (from winner.json, set by the interaction validation phase)
planted1: c_low=0.25 floor_P1=0.0270 floor_P2=0.1208 floor_P3=0.0431 floor_W1=0.0200 floor_W2=0.0200 floor_W3=0.0200 floor_W4=0.0200
planted2: c_low=0.35 floor_P1=0.0200 floor_P2=0.1463 floor_P3=0.0200 floor_W1=0.0200 floor_W2=0.0200 floor_W3=0.0200 floor_W4=0.0200
sine: c_low=0.25 floor_P1=0.0791 floor_P2=0.1358 floor_P3=0.1082 floor_W1=0.0395 floor_W2=0.0480 floor_W3=0.0378 floor_W4=0.0200

## Planted lag-1 (K=16)
P1 interaction (aligned-on): mean=0.0425 median=0.0397 IQR=[0.0281,0.0487] 95% CI=[0.0329,0.0520] n=20 floor=0.0270 PASS
P2 spec interaction (aligned-on): mean=0.2355 median=0.2313 IQR=[0.2181,0.2544] 95% CI=[0.2239,0.2470] n=20 floor=0.1208 PASS
P3 gap0 trained - frozen: mean=0.0918 CI=[0.0779,0.1058] n=20 floor=0.0431 PASS
P4 spec0: mean=0.5487 median=0.5610 IQR=[0.5233,0.5807] 95% CI=[0.5242,0.5733] n=20 (floor 0.25, CI > 0.15); decE_VIO: mean=1.0000 median=1.0000 IQR=[1.0000,1.0000] 95% CI=[1.0000,1.0000] n=20 (floor 0.5, CI > 0.3) PASS
W1 interaction on - cut: mean=0.0246 CI=[0.0170,0.0321] n=20 floor=0.0200 PASS
W2 interaction on - shuffle: mean=0.0316 CI=[0.0236,0.0396] n=20 floor=0.0200 PASS
W3 interaction aligned - random: mean=0.0198 CI=[0.0129,0.0267] n=20 floor=0.0200 FAIL
W4 interaction on - off: mean=0.0079 CI=[0.0012,0.0146] n=20 floor=0.0200 FAIL
W1s (diagnostic) interaction_cut: mean=0.0179 median=0.0215 IQR=[0.0093,0.0277] 95% CI=[0.0094,0.0264] n=20 floor=0.0200 PERSISTS cut
W5 gate fbdr: mean=0.2520 median=0.2530 IQR=[0.2444,0.2582] 95% CI=[0.2461,0.2579] n=20; p1_cos: mean=0.8410 median=0.8447 IQR=[0.8318,0.8494] 95% CI=[0.8363,0.8457] n=20 PASS
W6 normal_acc on vs 0.8*off: mean diff=0.0603 CI=[0.0566,0.0640] n=20; mean ratio=1.072 PASS
DIAG interaction_raw (aligned-on): mean=0.0723 median=0.0673 IQR=[0.0479,0.0792] 95% CI=[0.0555,0.0891] n=20
DIAG interaction_raw (random-on): mean=0.0522 median=0.0525 IQR=[0.0268,0.0750] 95% CI=[0.0374,0.0670] n=20
S1 decP1 EXP-UNEXP wire-off: mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20 bound=0.03 PASS
S2 p1n EXP-UNEXP wire-off: mean=0.0027 median=0.0004 IQR=[-0.0142,0.0150] 95% CI=[-0.0091,0.0145] n=20 bound=0.05 PASS
S3 (measurement only) wire-on: decP1 diff mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20; p1n diff mean=-0.0448 median=-0.0463 IQR=[-0.0627,-0.0287] 95% CI=[-0.0578,-0.0317] n=20

## Planted lag-2 (K=16)
P1 interaction (aligned-on): mean=-0.0052 median=-0.0088 IQR=[-0.0229,0.0191] 95% CI=[-0.0159,0.0054] n=20 floor=0.0200 FAIL
P2 spec interaction (aligned-on): mean=0.3105 median=0.3045 IQR=[0.2924,0.3481] 95% CI=[0.2904,0.3306] n=20 floor=0.1463 PASS
P3 gap0 trained - frozen: mean=0.0038 CI=[-0.0071,0.0147] n=20 floor=0.0200 FAIL
P4 spec0: mean=0.5723 median=0.5765 IQR=[0.5445,0.6030] 95% CI=[0.5449,0.5998] n=20 (floor 0.25, CI > 0.15); decE_VIO: mean=1.0000 median=1.0000 IQR=[1.0000,1.0000] 95% CI=[1.0000,1.0000] n=20 (floor 0.5, CI > 0.3) PASS
W1 interaction on - cut: mean=-0.0053 CI=[-0.0142,0.0036] n=20 floor=0.0200 FAIL
W2 interaction on - shuffle: mean=-0.0022 CI=[-0.0096,0.0052] n=20 floor=0.0200 FAIL
W3 interaction aligned - random: mean=-0.0045 CI=[-0.0111,0.0021] n=20 floor=0.0200 FAIL
W4 interaction on - off: mean=-0.0046 CI=[-0.0109,0.0017] n=20 floor=0.0200 FAIL
W1s (diagnostic) interaction_cut: mean=0.0001 median=0.0011 IQR=[-0.0189,0.0079] 95% CI=[-0.0117,0.0119] n=20 floor=0.0200 REMOVED by cut
W5 gate fbdr: mean=0.2537 median=0.2546 IQR=[0.2484,0.2584] 95% CI=[0.2486,0.2588] n=20; p1_cos: mean=0.8448 median=0.8436 IQR=[0.8385,0.8525] 95% CI=[0.8399,0.8497] n=20 PASS
W6 normal_acc on vs 0.8*off: mean diff=0.0521 CI=[0.0470,0.0572] n=20; mean ratio=1.032 PASS
DIAG interaction_raw (aligned-on): mean=-0.0088 median=-0.0149 IQR=[-0.0397,0.0325] 95% CI=[-0.0271,0.0095] n=20
DIAG interaction_raw (random-on): mean=-0.0012 median=-0.0062 IQR=[-0.0289,0.0336] 95% CI=[-0.0189,0.0165] n=20
S1 decP1 EXP-UNEXP wire-off: mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20 bound=0.03 PASS
S2 p1n EXP-UNEXP wire-off: mean=0.0032 median=0.0049 IQR=[-0.0259,0.0348] 95% CI=[-0.0147,0.0212] n=20 bound=0.05 PASS
S3 (measurement only) wire-on: decP1 diff mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20; p1n diff mean=-0.0313 median=-0.0267 IQR=[-0.0570,-0.0147] 95% CI=[-0.0448,-0.0177] n=20

## Sine (K=32)
P1 interaction (aligned-on): mean=0.1643 median=0.1752 IQR=[0.1251,0.1987] 95% CI=[0.1416,0.1871] n=20 floor=0.0791 PASS
P2 spec interaction (aligned-on): mean=0.2423 median=0.2359 IQR=[0.2177,0.2673] 95% CI=[0.2239,0.2608] n=20 floor=0.1358 PASS
P3 gap0 trained - frozen: mean=0.2485 CI=[0.2174,0.2797] n=20 floor=0.1082 PASS
P4 spec0: mean=0.5777 median=0.5791 IQR=[0.5449,0.6405] 95% CI=[0.5434,0.6121] n=20 (floor 0.25, CI > 0.15); decE_VIO: mean=0.8950 median=0.9500 IQR=[0.8500,0.9625] 95% CI=[0.8403,0.9497] n=20 (floor 0.5, CI > 0.3) PASS
W1 interaction on - cut: mean=0.0805 CI=[0.0624,0.0986] n=20 floor=0.0395 PASS
W2 interaction on - shuffle: mean=0.0955 CI=[0.0753,0.1157] n=20 floor=0.0480 PASS
W3 interaction aligned - random: mean=0.0874 CI=[0.0643,0.1106] n=20 floor=0.0378 PASS
W4 interaction on - off: mean=-0.0605 CI=[-0.0772,-0.0437] n=20 floor=0.0200 FAIL
W1s (diagnostic) interaction_cut: mean=0.0838 median=0.0836 IQR=[0.0741,0.0932] 95% CI=[0.0743,0.0934] n=20 floor=0.0395 PERSISTS cut
W5 gate fbdr: mean=0.2792 median=0.2785 IQR=[0.2677,0.2870] 95% CI=[0.2720,0.2863] n=20; p1_cos: mean=0.7912 median=0.7991 IQR=[0.7734,0.8070] 95% CI=[0.7704,0.8119] n=20 PASS
W6 normal_acc on vs 0.8*off: mean diff=-0.0687 CI=[-0.0992,-0.0382] n=20; mean ratio=0.720 FAIL
DIAG interaction_raw (aligned-on): mean=0.1754 median=0.1841 IQR=[0.1452,0.2074] 95% CI=[0.1549,0.1959] n=20
DIAG interaction_raw (random-on): mean=0.1563 median=0.1492 IQR=[0.1269,0.1762] 95% CI=[0.1371,0.1755] n=20

## Bar tally
24 of 34 bars pass.
PASS P1 planted1: interaction mean=0.0425 median=0.0397 IQR=[0.0281,0.0487] 95% CI=[0.0329,0.0520] n=20 floor=0.0270
PASS P2 planted1: spec_interaction mean=0.2355 median=0.2313 IQR=[0.2181,0.2544] 95% CI=[0.2239,0.2470] n=20 floor=0.1208
PASS P3 planted1: gap0 trained-frozen 0.0918 CI=[0.0779,0.1058] n=20 floor=0.0431
PASS P4 planted1: spec0 mean=0.5487 median=0.5610 IQR=[0.5233,0.5807] 95% CI=[0.5242,0.5733] n=20; decE_VIO mean=1.0000 median=1.0000 IQR=[1.0000,1.0000] 95% CI=[1.0000,1.0000] n=20
PASS W1 planted1: interaction on-cut 0.0246 CI=[0.0170,0.0321] n=20 floor=0.0200 causal-but-small=False
PASS W2 planted1: interaction on-shuffle 0.0316 CI=[0.0236,0.0396] n=20 floor=0.0200
FAIL W3 planted1: interaction aligned-random 0.0198 CI=[0.0129,0.0267] n=20 floor=0.0200
FAIL W4 planted1: interaction on-off 0.0079 CI=[0.0012,0.0146] n=20 floor=0.0200
PASS W5 planted1: fbdr mean=0.2520 median=0.2530 IQR=[0.2444,0.2582] 95% CI=[0.2461,0.2579] n=20 (> 0.05); p1_cos mean=0.8410 median=0.8447 IQR=[0.8318,0.8494] 95% CI=[0.8363,0.8457] n=20 (< 0.9)
PASS W6 planted1: normal_acc on - 0.8*off 0.0603 CI=[0.0566,0.0640] n=20; mean ratio on/off=1.072
PASS S1 planted1: decP1 EXP-UNEXP (wire-off) mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20 bound=0.03
PASS S2 planted1: p1n EXP-UNEXP (wire-off) mean=0.0027 median=0.0004 IQR=[-0.0142,0.0150] 95% CI=[-0.0091,0.0145] n=20 bound=0.05
FAIL P1 planted2: interaction mean=-0.0052 median=-0.0088 IQR=[-0.0229,0.0191] 95% CI=[-0.0159,0.0054] n=20 floor=0.0200
PASS P2 planted2: spec_interaction mean=0.3105 median=0.3045 IQR=[0.2924,0.3481] 95% CI=[0.2904,0.3306] n=20 floor=0.1463
FAIL P3 planted2: gap0 trained-frozen 0.0038 CI=[-0.0071,0.0147] n=20 floor=0.0200
PASS P4 planted2: spec0 mean=0.5723 median=0.5765 IQR=[0.5445,0.6030] 95% CI=[0.5449,0.5998] n=20; decE_VIO mean=1.0000 median=1.0000 IQR=[1.0000,1.0000] 95% CI=[1.0000,1.0000] n=20
FAIL W1 planted2: interaction on-cut -0.0053 CI=[-0.0142,0.0036] n=20 floor=0.0200 causal-but-small=False
FAIL W2 planted2: interaction on-shuffle -0.0022 CI=[-0.0096,0.0052] n=20 floor=0.0200
FAIL W3 planted2: interaction aligned-random -0.0045 CI=[-0.0111,0.0021] n=20 floor=0.0200
FAIL W4 planted2: interaction on-off -0.0046 CI=[-0.0109,0.0017] n=20 floor=0.0200
PASS W5 planted2: fbdr mean=0.2537 median=0.2546 IQR=[0.2484,0.2584] 95% CI=[0.2486,0.2588] n=20 (> 0.05); p1_cos mean=0.8448 median=0.8436 IQR=[0.8385,0.8525] 95% CI=[0.8399,0.8497] n=20 (< 0.9)
PASS W6 planted2: normal_acc on - 0.8*off 0.0521 CI=[0.0470,0.0572] n=20; mean ratio on/off=1.032
PASS S1 planted2: decP1 EXP-UNEXP (wire-off) mean=0.0000 median=0.0000 IQR=[0.0000,0.0000] 95% CI=[0.0000,0.0000] n=20 bound=0.03
PASS S2 planted2: p1n EXP-UNEXP (wire-off) mean=0.0032 median=0.0049 IQR=[-0.0259,0.0348] 95% CI=[-0.0147,0.0212] n=20 bound=0.05
PASS P1 sine: interaction mean=0.1643 median=0.1752 IQR=[0.1251,0.1987] 95% CI=[0.1416,0.1871] n=20 floor=0.0791
PASS P2 sine: spec_interaction mean=0.2423 median=0.2359 IQR=[0.2177,0.2673] 95% CI=[0.2239,0.2608] n=20 floor=0.1358
PASS P3 sine: gap0 trained-frozen 0.2485 CI=[0.2174,0.2797] n=20 floor=0.1082
PASS P4 sine: spec0 mean=0.5777 median=0.5791 IQR=[0.5449,0.6405] 95% CI=[0.5434,0.6121] n=20; decE_VIO mean=0.8950 median=0.9500 IQR=[0.8500,0.9625] 95% CI=[0.8403,0.9497] n=20
PASS W1 sine: interaction on-cut 0.0805 CI=[0.0624,0.0986] n=20 floor=0.0395 causal-but-small=False
PASS W2 sine: interaction on-shuffle 0.0955 CI=[0.0753,0.1157] n=20 floor=0.0480
PASS W3 sine: interaction aligned-random 0.0874 CI=[0.0643,0.1106] n=20 floor=0.0378
FAIL W4 sine: interaction on-off -0.0605 CI=[-0.0772,-0.0437] n=20 floor=0.0200
PASS W5 sine: fbdr mean=0.2792 median=0.2785 IQR=[0.2677,0.2870] 95% CI=[0.2720,0.2863] n=20 (> 0.05); p1_cos mean=0.7912 median=0.7991 IQR=[0.7734,0.8070] 95% CI=[0.7704,0.8119] n=20 (< 0.9)
FAIL W6 sine: normal_acc on - 0.8*off -0.0687 CI=[-0.0992,-0.0382] n=20; mean ratio on/off=0.720

## Verdict (protocol section 8)
Verdict: C STRONG PASS

Thesis-level: no outcome of round 22 establishes predictive coding. Round 20 failed B' and D; a C pass establishes only that the model produces precision-like behaviour from input noise plus error selection.
