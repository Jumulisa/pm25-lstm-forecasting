# pm25-lstm-forecasting

Forecasting hourly PM2.5 in Beijing using recurrent neural networks (LSTM/GRU). I framed this as a next-hour prediction problem: given the last L hours of features, predict PM2.5 one hour ahead. I started with a simple baseline and improved it with longer context windows, time features, and a bidirectional LSTM.

Project overview

Goal: Predict next-hour PM2.5 (µg/m³) to support alerts and planning.

Approach: Sequence modeling with RNNs (Keras). Inputs are sliding windows over weather + time features; output is the next PM2.5 value.

What changed: Extending the history window (24 → 48 → 72 hours) and using a bidirectional LSTM on the first layer.

Best submission: submission_r006.csv (public RMSE 4428.2966; leaderboard position shown ~8/36 at submission time).

Data

From the course Kaggle competition: train.csv, test.csv, sample_submission.csv.

Shapes seen: train ≈ 30,676 × 12, test ≈ 13,148 × 11.

Target: pm2.5 (train only).

Cleaning: parsed datetime, forward/backward-filled missing feature values (weather & one-hot wind direction).

EDA (short): clear seasonality and sharp winter spikes; motivated giving the model more history (longer sequence length).

Note: The dataset is not included in this repo. Download it from the competition and keep it outside version control.

Features & preprocessing

Time features: hour, weekday, month + cyclical encodings (sin/cos) for hour and month so 23:00 ≈ 00:00 and Dec wraps to Jan.

Scaling: StandardScaler fit on the train split only; applied to validation and test.

Windows: For a window length L, each sample is (L, n_features) predicting the next hour.

Validation split: last 20% of the training timeline (no shuffle) to mimic real forecasting.

Models

Best architecture (Keras):

Input (L, n_features)
→ Bidirectional LSTM(128, return_sequences=True)
→ Dropout(0.2)
→ LSTM(64)
→ Dense(16, activation='relu')
→ Dense(1)


Optimizer: Adam (lr = 5e-4)

Batch size: 64

Loss: MSE (we report RMSE)

Callbacks: EarlyStopping (patience=5, restore best), ReduceLROnPlateau

Notes: LSTM gates help with vanishing gradients; LR scheduling stabilized training. Gradient clipping is available if needed.

Results (summary)

Validation RMSE (time-based split):

Baseline 24h window: 78.59

48h + bidirectional LSTM: 75.18 → 75.13 → 74.81 (batch/LR tweaks)

72h window + slightly larger model (best): ≈ 70.79

Kaggle (public leaderboard):

submission_r006.csv → 4428.2966 (public RMSE), rank shown 8/36 at the time of submission.

What improved things

Longer history captured slower build-ups before spikes → less lag, less peak flattening.

Smaller batch (64) + slightly smaller LR (5e-4) → steadier validation.

Repository contents
.
├─ air_quality_forecasting_starter_code_FIXED.ipynb   # main notebook (clean run)
├─ experiment_log.csv                                 # experiments and validation RMSEs
├─ val_rmse_by_run.png                                # validation trend across runs
├─ submission_r006.csv                                # best submission uploaded to Kaggle
├─ submission_r007.csv                                # additional submission
├─ Air_Quality_Forecasting_Report_FULL.pdf            # final report (pdf)
├─ README.md
├─ requirements.txt                                   # minimal deps
└─ .gitignore                                         # excludes data files, etc.


Setup
# Python 3.10+ recommended
pip install -r requirements.txt


Minimal requirements.txt:

tensorflow
pandas
numpy
scikit-learn
matplotlib

How to run

Place data (not tracked):
Put train.csv, test.csv, sample_submission.csv in a local data/ folder OR adjust paths in the notebook where you load them.

Open the notebook:
air_quality_forecasting_starter_code_FIXED.ipynb in Jupyter/VS Code.

Run top-to-bottom.

The notebook builds features, creates sequences, trains models, and evaluates validation RMSE.

It also saves submissions and appends each run to experiment_log.csv.

Reproduce my best run

Use 72-hour windows with the bidirectional LSTM (128→64), Dropout=0.2, Adam lr=5e-4, batch=64, EarlyStopping + ReduceLROnPlateau.
The notebook has a tiny Config + an Experiment Runner cell so you can switch settings by editing a few lines, then re-run.

Experiment log

All runs (IDs like r002, r003, …) are captured in experiment_log.csv with key parameters and validation RMSE.
You can sort the file by val_RMSE to see what helped most. The figure val_rmse_by_run.png shows the trend.

Kaggle submission

Go to the competition page → Submit Predictions.

Upload your CSV (e.g., submission_r006.csv).

Check your Submissions tab for the Public RMSE and make sure the file is selected among your top submissions.

CSV format: exactly two columns → row ID, pm2.5, 13148 rows (no index column).

RMSE (definition)​

I used RMSE on a time-based split for model selection. Kaggle uses its own test set; the public score may differ from validation, but the trend was consistent while tuning.


Acknowledgements / references

S. Hochreiter, J. Schmidhuber, “Long short-term memory,” Neural Computation, 1997.

I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016 (Sequence modeling).

F. Chollet, Deep Learning with Python, 2nd ed., Manning, 2021 (Keras RNNs).
