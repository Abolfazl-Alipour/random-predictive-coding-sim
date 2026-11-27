# Round 15 analysis (preregistered replication)

Protocol: PROTOCOL-R15.md. Seeds 201..220 (20 seeds).
CI = mean +/- 1.96 * sample-std / sqrt(20).

## Planted lag-1

- delta: acc mean=0.223 median=0.220 IQR=[0.213,0.235] 95% CI=[0.216,0.230] n=20 clamp_frac=0.0000 | m1 mean=0.680 m2 mean=0.956 all-motif mean=0.811
- tempcov: acc mean=0.070 median=0.077 IQR=[0.061,0.081] 95% CI=[0.063,0.078] n=20 clamp_frac=0.0036 | m1 mean=0.116 m2 mean=0.096 all-motif mean=0.106
- burst: acc mean=0.071 median=0.074 IQR=[0.056,0.082] 95% CI=[0.062,0.080] n=20 clamp_frac=0.0000 | m1 mean=0.087 m2 mean=0.149 all-motif mean=0.116
- frozen: acc mean=0.059 median=0.059 IQR=[0.050,0.065] 95% CI=[0.054,0.064] n=20 clamp_frac=0.0000 | m1 mean=0.055 m2 mean=0.054 all-motif mean=0.054
- paired delta-tempcov: mean diff=+0.152 95% CI=[+0.141,+0.164]
- paired delta-frozen: mean diff=+0.164 95% CI=[+0.155,+0.173]
- baselines (identical test seqs): Markov-1=0.274 Markov-2=0.267 persistence=0.046 ESN=0.275

- BAR lower 95% CI > 0.22: FAIL
- BAR target mean >= 0.25: FAIL
- BAR m1 mean >= 0.80: FAIL
- BAR m2 mean >= 0.90: PASS
- BAR beats tempcov paired (CI above 0): PASS

## Planted lag-2

- delta: acc mean=0.222 median=0.221 IQR=[0.213,0.232] 95% CI=[0.216,0.227] n=20 clamp_frac=0.0000 | m1 mean=0.264 m2 mean=0.050 all-motif mean=0.261
- tempcov: acc mean=0.070 median=0.072 IQR=[0.051,0.088] 95% CI=[0.060,0.079] n=20 clamp_frac=0.0033 | m1 mean=0.156 m2 mean=0.100 all-motif mean=0.162
- burst: acc mean=0.069 median=0.071 IQR=[0.049,0.081] 95% CI=[0.059,0.079] n=20 clamp_frac=0.0000 | m1 mean=0.068 m2 mean=0.000 all-motif mean=0.066
- frozen: acc mean=0.062 median=0.061 IQR=[0.057,0.069] 95% CI=[0.058,0.066] n=20 clamp_frac=0.0000 | m1 mean=0.046 m2 mean=0.000 all-motif mean=0.044
- paired delta-tempcov: mean diff=+0.152 95% CI=[+0.141,+0.163]
- paired delta-frozen: mean diff=+0.159 95% CI=[+0.153,+0.166]
- baselines (identical test seqs): Markov-1=0.084 Markov-2=0.267 persistence=0.057 ESN=0.261

- BAR mean >= 0.20: PASS
- BAR mean >= Markov-1 + 0.10: PASS

## Sine-fixed

- delta: acc mean=0.868 median=0.872 IQR=[0.849,0.893] 95% CI=[0.851,0.886] n=20 clamp_frac=0.0000
- tempcov: acc mean=0.083 median=0.076 IQR=[0.049,0.114] 95% CI=[0.066,0.100] n=20 clamp_frac=0.0063
- burst: acc mean=0.048 median=0.034 IQR=[0.022,0.062] 95% CI=[0.032,0.063] n=20 clamp_frac=0.0000
- frozen: acc mean=0.036 median=0.033 IQR=[0.023,0.044] 95% CI=[0.028,0.044] n=20 clamp_frac=0.0000
- paired delta-tempcov: mean diff=+0.785 95% CI=[+0.758,+0.812]
- paired delta-frozen: mean diff=+0.832 95% CI=[+0.813,+0.852]
- baselines (identical test seqs): Markov-1=0.456 Markov-2=0.798 persistence=0.124 ESN=0.940

- BAR mean >= 0.80: PASS

## Lorenz

- delta: acc mean=0.164 median=0.160 IQR=[0.153,0.178] 95% CI=[0.156,0.171] n=20 clamp_frac=0.0000 | within1 mean=0.402 mse mean=66.141
- tempcov: acc mean=0.042 median=0.040 IQR=[0.033,0.047] 95% CI=[0.034,0.050] n=20 clamp_frac=0.0379 | within1 mean=0.132 mse mean=201.778
- burst: acc mean=0.052 median=0.057 IQR=[0.043,0.071] 95% CI=[0.041,0.063] n=20 clamp_frac=0.0000 | within1 mean=0.161 mse mean=167.748
- frozen: acc mean=0.037 median=0.035 IQR=[0.028,0.042] 95% CI=[0.032,0.042] n=20 clamp_frac=0.0000 | within1 mean=0.117 mse mean=215.119
- paired delta-tempcov: mean diff=+0.121 95% CI=[+0.112,+0.131]
- paired delta-frozen: mean diff=+0.127 95% CI=[+0.119,+0.135]
- baselines (identical test seqs): Markov-1=0.205 Markov-2=0.565 persistence=0.173 ESN=0.398
- baseline MSE: Markov-1=12.326 persistence=12.598 ESN=4.407

- BAR beats persistence on acc: FAIL
- BAR beats Markov-1 on acc: FAIL
- BAR beats persistence on MSE: FAIL
- BAR beats Markov-1 on MSE: FAIL
- BAR gain >= +0.15 over frozen: FAIL

## Var sine

- delta: acc mean=0.150 median=0.154 IQR=[0.127,0.166] 95% CI=[0.138,0.161] n=20 clamp_frac=0.0000 | within1 mean=0.324 mse mean=0.172
- tempcov: acc mean=0.043 median=0.028 IQR=[0.022,0.047] 95% CI=[0.028,0.057] n=20 clamp_frac=0.0000 | within1 mean=0.108 mse mean=0.779
- burst: acc mean=0.053 median=0.041 IQR=[0.031,0.055] 95% CI=[0.038,0.067] n=20 clamp_frac=0.0000 | within1 mean=0.120 mse mean=0.795
- frozen: acc mean=0.043 median=0.036 IQR=[0.031,0.052] 95% CI=[0.035,0.052] n=20 clamp_frac=0.0000 | within1 mean=0.098 mse mean=0.857
- paired delta-tempcov: mean diff=+0.107 95% CI=[+0.092,+0.122]
- paired delta-frozen: mean diff=+0.107 95% CI=[+0.092,+0.121]
- baselines (identical test seqs): Markov-1=0.283 Markov-2=0.528 persistence=0.125 ESN=0.385
- baseline MSE: Markov-1=0.108 persistence=0.054 ESN=0.044

- BAR beats persistence on acc: PASS
- BAR beats Markov-1 on acc: FAIL
- BAR beats persistence on MSE: FAIL
- BAR beats Markov-1 on MSE: FAIL
- BAR gain >= +0.15 over frozen: FAIL

## Summary: 6/18 bars pass