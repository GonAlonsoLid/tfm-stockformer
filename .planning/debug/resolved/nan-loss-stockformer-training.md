---
status: resolved
trigger: "nan-loss-stockformer-training — Training collapses with loss nan and all metrics = 0.0000 on epoch 1"
created: 2026-03-12T00:00:00Z
updated: 2026-03-12T00:00:00Z
---

## Current Focus

hypothesis: CONFIRMED — NaN values in feature CSVs propagate through the model causing NaN loss
test: Apply fix in feature_engineering.py (ffill per ticker) and StockDataset (safety net fillna)
expecting: After fix, feature CSVs and bonus_X will be NaN-free, loss will be finite
next_action: Apply fix in feature_engineering.py and StockDataset, regenerate feature CSVs

## Symptoms

expected: Model trains with decreasing loss and non-zero mae/rmse/mape metrics
actual: After epoch 1 (93 batches, ~78 min), loss=nan, mae=0.0000, rmse=0.0000, mape=0.0000. Accuracy ~0.497 (random).
errors: |
  epoch 1, lr 0.001000, loss nan, time 4694.7 sec
  step 1, acc: 0.4992, mae: 0.0000, rmse: 0.0000, mape: 0.0000
  step 2, acc: 0.4962, mae: 0.0000, rmse: 0.0000, mape: 0.0000
  average, acc: 0.4977, mae: 0.0000, rmse: 0.0000, mape: 0.0000
  Epoch 1: New best mae: 0.0000, Model saved.
reproduction: caffeinate -i python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf --max_epoch 2
started: First training run on SP500 dataset. Custom config created for this project.

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-03-12T00:10:00Z
  checked: flow.npz, trend_indicator.npz, corr_adj.npy, 128_corr_struc2vec_adjgat.npy
  found: All four files are NaN-free and Inf-free. flow.npz values in [-0.54, 0.75], well-scaled.
  implication: The NaN does not originate from the primary traffic or adjacency data.

- timestamp: 2026-03-12T00:15:00Z
  checked: features/ directory (69 feature CSVs)
  found: 92,419 total NaN values across 10 feature files. VOL_norm_252.csv alone has 91,298 NaN.
  implication: bonus_X (the auxiliary feature input) contains NaN values that are fed into the model.

- timestamp: 2026-03-12T00:20:00Z
  checked: Training sample NaN coverage after bonus_seq2instance with T1=20
  found: 982 of 1066 training samples (92.1%) contain at least 1 NaN in bonus_X. Only 84 samples are clean.
  implication: Nearly every training batch contains NaN in the input features.

- timestamp: 2026-03-12T00:25:00Z
  checked: NaN propagation through attention (softmax with NaN input)
  found: NaN in any element of a softmax input poisons the entire output row (all NaN). Same for matmul.
  implication: A single NaN in bonus_X for a single stock corrupts attention outputs for ALL stocks in that batch -> NaN loss.

- timestamp: 2026-03-12T00:30:00Z
  checked: NaN sources in feature CSVs
  found: Two root causes:
    (1) VOL_norm_252 uses a 252-day rolling window but build_feature_matrix only drops 60 warmup rows -> 192 NaN rows at dataset start.
    (2) SW (731 zero-volume days) and AMCR (232 zero-volume days) produce NaN in VOL_ratio_* and STOCH_* features when rolling mean=0 or High-Low range=0.
  implication: NaN is baked into the saved CSV files and persists into training.

- timestamp: 2026-03-12T00:35:00Z
  checked: feature_engineering.py NaN handling
  found: compute_features() intentionally leaves NaN (docstring says "caller handles that"). build_feature_matrix drops first 60 rows (warmup for 60-day window) but this is insufficient for the 252-day VOL_norm_252 feature. _cross_sectional_normalize() computes mean/std with skipna=True but leaves NaN values as NaN in the output.
  implication: The pipeline has no NaN-filling step before saving CSVs.

## Resolution

root_cause: |
  NaN values in the feature CSVs (data/Stock_SP500_2018-01-01_2024-01-01/features/) propagate
  through the model causing NaN loss. Two sources:
  (1) VOL_norm_252 has a 252-day warmup but only 60 rows are dropped, leaving 192 NaN rows at
      the start of the dataset (~13% of that feature's values).
  (2) Stocks SW and AMCR have hundreds of zero-volume trading days, causing VOL_ratio_* and
      STOCH_* features to produce NaN throughout training (not just at warmup).
  NaN enters bonus_X (shape [samples, T1, 478, 69]) in 92.1% of training samples.
  Since attention softmax with any NaN input produces all-NaN output, a single NaN stock
  corrupts ALL 478 nodes in the batch -> NaN loss -> NaN gradients -> NaN weights.

fix: |
  Two-layer fix:
  1. feature_engineering.py: After computing raw features per ticker, forward-fill then
     backward-fill NaN values before building the wide matrix. This ensures zero-volume
     gaps and warmup-period NaN are filled with the nearest valid value.
  2. StockDataset.__init__: After concatenating the feature arrays into bonus_all,
     fill any residual NaN with 0.0 as a safety net. Since all features are already
     cross-sectionally normalized (z-score), 0.0 is the cross-sectional mean.

verification: |
  Self-verified:
  - After regenerating feature CSVs with the fix: total NaN across all 69 CSVs = 0 (was 92,419)
  - bonus_all NaN after CSV fix = 0 (was 1,754,967 in the training window)
  - safety net in StockDataset also shows 0 NaN
  - Awaiting human confirmation that training loss is finite on next run.
files_changed:
  - data_processing_script/sp500_pipeline/feature_engineering.py
  - lib/Multitask_Stockformer_utils.py
