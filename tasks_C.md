# Worker C: spatial-temporal tasks for RPC testing

## 1. Files

1. `tasks.py` is the only file this worker created for code. It uses numpy only.
2. `tasks_C.md` is this file: the interface spec, the chance levels, and the learnability numbers.
3. No RPC is implemented here. This worker only builds the tasks and proves they are learnable by a simple control.
4. `rpc_toy.py` was not touched. `rpc_v2.py` was not touched (it was created by another worker while this work ran).
5. All work ran on this VM in the local venv. No home server was used, no SSH was opened, and no Agescape files were touched.

## 2. Reproduce

1. Run this command from `~/workspace/thesis-rpc/`:
   `~/workspace/thesis-rpc/venv/bin/python tasks.py`
2. The full run takes about 60 seconds on this VM. It uses Python 3.12 and numpy 2.5.4.
3. The run prints the integrator check, then the ridge versus persistence numbers for all four tasks, as means over seeds 0, 1, and 2.
4. Exit code is 0. The exact output of the final run is saved at `/tmp/tasks_run.txt` for reference.

## 3. Interface for the coordinator

1. Every dataset builder returns the same 5-tuple:
   `(train_X, train_y, test_X, test_y, meta)`.
2. `meta` is a dict. It always has `task`, `target`, `target_description`, `input_dim`, `output_dim`, `output_kind`, `n_classes`, `n_train`, `n_test`, `chance_level`, and `chance_description`.
3. The four builders are `make_track_dataset`, `make_sine_dataset`, `make_lorenz_dataset`, and `make_dot_dataset`.
4. The dispatcher is `get_dataset(name, seed=0, **kwargs)`. Name is one of `track`, `sine`, `lorenz`, `dot`.
5. Low-level track trajectories come from `make_track_data(track_len=180, n_trials=100, n_steps=1800, seed=0)`. It returns a dict with `positions`, `velocities`, `s`, and `F` (or `None`, see section 7).
6. `leaky_integrators(positions)` returns F for any positions, using F(s, t) = exp(-s * x(t)).
7. `make_track_generalisation_dataset` returns `(X, y, meta)` for the 300 cm track, in the same format as the test split.
8. `decode_position_from_F(F, track_len=180)` decodes position in cm from an F vector by grid search over 0.5 cm steps.
9. `run_learnability_checks(seeds=(0, 1, 2))` reruns the whole control check and returns the numbers as a dict.
10. `fit_ridge`, `predict_ridge`, and `select_alpha` are the control helpers. They are numpy only.

## 4. Task specs

1. Task 1 is the 1D track (thesis Chapter 4 style). Track length is 180 cm. There are 100 leaky integrators with s log-spaced from 0.001 to 0.1.
2. The integrator equation is dF/dt = -s * v(t) * F. Decay follows distance travelled, not time. The closed form is F(s, x) = exp(-s * x) for a trial that starts at x = 0 with F = 1.
3. Train has 6000 trials of 1800 steps. Test has 1000 fresh trials of 1800 steps. Pairs are subsampled at 20 random steps per trial (see section 7).
4. Track input X is [F(t), F(t-1)], 200 numbers. The lagged copy carries velocity, since dx = -(log F(t) - log F(t-1)) / s.
5. Track target y is the next F vector F(t+1), 100 numbers in [0, 1]. The next position in cm is in `meta["test_y_pos_cm"]`.
6. Track shapes at seed 0 are train_X (120000, 200), train_y (120000, 100), test_X (20000, 200), test_y (20000, 100), all float32.
7. The 300 cm generalisation set has 1000 fresh trials of 1800 steps, 20 pairs each, so X is (20000, 200) and y is (20000, 100). It was never trained on.
8. Task 2 is the sine wave. Each sequence has its own frequency, uniform in [0.02, 0.08] cycles per step, and its own phase, uniform in [0, 2*pi).
9. Sine value is x(t) = sin(2*pi*f*t + phase). Length is 200 steps. Train has 600 sequences, test has 200 sequences.
10. Sine input X is the previous 10 continuous values. Target y is the next-step bin index, 0 to 31, over [-1, 1]. The next continuous value is in `meta["test_y_cont"]`.
11. Sine shapes at seed 0 are train_X (114000, 10), train_y (114000,), test_X (38000, 10), test_y (38000,).
12. Task 3 is the Lorenz system. Parameters are sigma = 10, rho = 28, beta = 8/3, dt = 0.02, RK4 integration, with a 500-step burn-in.
13. Lorenz uses the x coordinate only. Train has 40 trajectories of 300 steps, test has 15 trajectories of 300 steps.
14. Lorenz input X is the previous 10 x values. Target y is the next-step bin index, 0 to 31, over [-20, 20]. The next continuous value is in `meta["test_y_cont"]`.
15. Lorenz shapes at seed 0 are train_X (11600, 10), train_y (11600,), test_X (4350, 10), test_y (4350,).
16. Task 4 is the 2D moving dot. The grid is 32 by 32. Speed is 1.5 cells per step. Turn probability is 0.1 per step, with a turn uniform in [-90, 90] degrees. Walls reflect.
17. Dot input X is the previous 10 (x, y) positions, each divided by 31, flattened to 20 numbers. That is 10 frames in, predicting frame 11.
18. Dot target y is the next cell index, 0 to 1023, row-major (cell = y_cell * 32 + x_cell). The next coordinates are in `meta["test_y_xy"]`.
19. Dot train has 800 trajectories of 40 steps, test has 300 trajectories of 40 steps. Shapes at seed 0 are train_X (24000, 20), test_X (9000, 20).

## 5. Chance levels

1. Track chance is the mean predictor: predict the train-mean F vector for every test pair. Chance F-MSE is 0.0396. Chance position RMSE (predict the train-mean position) is 52.3 cm.
2. Sine chance is random-guess accuracy, 1/32 = 0.03125. Chance continuous MSE (predict the train-mean value) is 0.4999.
3. Lorenz chance is random-guess accuracy, 1/32 = 0.03125. Chance continuous MSE (predict the train-mean x) is 63.4.
4. Dot chance is random-guess accuracy, 1/1024 = 0.000977. Chance coordinate MSE (predict the train-mean position) is 83.0.

## 6. Learnability numbers (mean +/- std over seeds 0, 1, 2)

1. The control is ridge regression, fit on train only. The penalty alpha is chosen from (0.03, 0.1, 0.3, 1, 3, 10, 30, 100) on a train-internal split, then the model is refit on all of train and scored on held-out test.
2. The baseline is persistence: predict the next value or cell equals the current one.
3. Chosen alphas were the same on all three seeds: track 0.03, log-F track 0.03, sine 0.03, Lorenz 0.03, dot 0.1.
4. Track, target next F vector: ridge F-MSE 9.18e-05 +/- 3.1e-06; persistence F-MSE 2.33e-04 +/- 7.4e-06; chance 0.0396.
5. Track, decoded position: ridge RMSE 1.04 +/- 0.005 cm; persistence RMSE 1.89 +/- 0.009 cm; chance 52.3 cm. Log-F ridge gives F-MSE 9.59e-05 +/- 2.8e-06 and position RMSE 1.03 +/- 0.004 cm.
6. Sine, continuous next value: ridge MSE 1.43e-06 +/- 1.7e-07; persistence MSE 0.0507 +/- 0.0016; chance 0.4999.
7. Sine, binned accuracy: ridge 0.987 +/- 0.001; persistence 0.130 +/- 0.002; chance 0.0313.
8. Lorenz, continuous next x: ridge MSE 0.00112 +/- 1.8e-05; persistence MSE 0.770 +/- 0.02; chance 63.4.
9. Lorenz, binned accuracy: ridge 0.980 +/- 0.003; persistence 0.472 +/- 0.009; chance 0.0313.
10. Dot, coordinates: ridge MSE 0.178 +/- 0.001; persistence MSE 1.09 +/- 0.001; chance 83.0.
11. Dot, cell accuracy: ridge 0.665 +/- 0.008; persistence 0.0092 +/- 0.0006; chance 0.00098.
12. Verdict: ridge beats persistence on the primary metric for all four tasks on held-out test. Each task is learnable by a simple control.

## 7. Generalisation: the 300 cm track

1. A ridge model trained on the 180 cm track was scored on fresh 300 cm trials, with no training on 300 cm.
2. On 300 cm, ridge F-MSE is 5.54e-04 +/- 4.4e-05, log-F ridge is 1.59e-04 +/- 6.8e-06, and persistence is 1.84e-04 +/- 1.3e-05.
3. On 300 cm, decoded position RMSE is 8.54 +/- 0.74 cm for ridge, 10.73 +/- 0.29 cm for log-F ridge, and 1.88 +/- 0.02 cm for persistence.
4. The linear controls therefore lose to persistence on 300 cm. This is expected, not a task failure.
5. Ridge shrinks predictions toward the 180 cm train mean, and F is exponential in position, so a linear readout cannot extrapolate to a longer track.
6. The bar for RPC on this probe is to beat persistence on 300 cm (F-MSE below 1.84e-04 and position RMSE below 1.88 cm). That would show the model learned the exponential code itself, not a linear shortcut.
7. The 300 cm result is a generalisation probe. It is not part of the learnability proof in section 6.

## 8. Deviations from the thesis, and why

1. The thesis sampled each new location independently from a uniform distribution. Velocity was the difference of two independent uniform draws, and order did not matter because Chapter 4 only reconstructed the current input.
2. Independent uniform locations make next-step prediction impossible by construction. The next location is independent of the past.
3. This worker therefore uses a smooth AR(1) velocity process (rho = 0.85, noise sigma = 1.0, clipped to +/-6 cm per step) with reflecting walls. The integrator equation, the s range, the 100 units, the 180 cm track, and the 6000 by 1800 scale stay thesis-faithful.
4. Full trajectories are generated at thesis scale (6000 train trials by 1800 steps), but pairs are subsampled at 20 random steps per trial.
5. The full pair matrix would be 10.8 million rows, and the full F matrix would be 6000 by 1800 by 100 numbers (about 4.3 GB in float32). Subsampling keeps the arrays in memory. Sampled steps are uniform over a trial, so the subsample is unbiased over time.
6. F is computed from position by the closed form exp(-s * x), not by stepping 100 integrators per trial. The two agree: the stepwise update F <- F * exp(-s * dx) matches the closed form to a max abs error of 4.62e-08 on a 200-step trial. That check runs in `tasks.py` `__main__`.
7. `make_track_data` returns the full F array only when `return_F=True`. The default is False, because the full F array at 6000 by 1800 is about 4.3 GB. Use `leaky_integrators(positions)` for F in chunks instead.

## 9. Method notes and limits

1. For the binned tasks (sine, Lorenz, dot), the primary control is ridge regression to the continuous next value or coordinates, with the prediction then binned or floored to a cell.
2. A one-hot ridge classifier is reported as a secondary number only. On sine it scores 0.226 +/- 0.001 and on Lorenz 0.242 +/- 0.024.
3. The one-hot classifier is weak here because bin membership is a narrow threshold function of a linear predictor, and the fit collapses toward the majority bin. On sine, the most frequent bin alone has frequency 0.226. The regression control is the learnability proof, not the one-hot classifier.
4. Ridge alpha is chosen on a train-internal split. Rows are ordered by sequence or trial, so the first 80 percent of rows are earlier sequences and the last 20 percent are later ones. The split is by sequence, not by shuffled time step.
5. Position decoding from F uses grid search, not log least squares. Log least squares blows up when a predicted F is near zero for large s.
6. Lorenz dt = 0.02 was chosen deliberately. At dt = 0.01, persistence already scores about 56 percent binned accuracy, because the step is small next to the bin width. At dt = 0.02, persistence is about 47 percent and ridge reaches 98 percent.
7. Sine needs a 10-step window, not 2 steps. Two lags cannot identify the per-sequence frequency, while 10 lags can. The same window of 10 is used for Lorenz and for the dot.
8. All four tasks are deterministic given the seed. Seeds 0, 1, and 2 give the means in section 6. The test seeds are offset by 1000 from the train seeds inside each builder, so train and test never share a random stream.
