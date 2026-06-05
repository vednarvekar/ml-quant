# CNN Model Training Log

This file tracks model/training-loop changes, why they were made, and what the results were.
Whenever we change `models/cnn/model.py`, `models/cnn/train.py`, or the data/label logic in `script/process_data.py`, add a new entry here.

## Current Setup

- Data: `data/processed/5M`
- Input shape per sample: `(60, 12)` before training, converted to `(12, 60)` for `Conv1d`
- Labels:
  - `0 = HOLD`
  - `1 = BUY`
  - `2 = SELL`
- Model: small 1D CNN in `models/cnn/model.py`
- Main metric to watch: validation macro F1
- Accuracy baseline: around `45%` if always predicting the majority `SELL` class

## 2026-06-05 - Baseline With Plain Cross Entropy

### Change

- Simplified `models/cnn/train.py`.
- Removed class weights, label smoothing, scheduler, early stopping clutter, and extra checks.
- Used plain `CrossEntropyLoss`.
- Kept loss/accuracy reporting.

### Result

Final reported run:

```text
Train | loss 0.9492 | acc 0.528 | macro F1 0.485
  HOLD precision 0.505 | recall 0.356 | F1 0.418
  BUY  precision 0.560 | recall 0.323 | F1 0.410
  SELL precision 0.527 | recall 0.773 | F1 0.627

Val   | loss 1.1475 | acc 0.412 | macro F1 0.355
  HOLD precision 0.358 | recall 0.253 | F1 0.296
  BUY  precision 0.318 | recall 0.184 | F1 0.233
  SELL precision 0.452 | recall 0.653 | F1 0.535

Test  | loss 1.0410 | acc 0.458 | macro F1 0.293
  HOLD precision 0.430 | recall 0.159 | F1 0.232
  BUY  precision 0.391 | recall 0.020 | F1 0.038
  SELL precision 0.462 | recall 0.901 | F1 0.610
```

### Interpretation

- Accuracy looked acceptable only because `SELL` is the majority class.
- The model mostly collapsed toward predicting `SELL`.
- `BUY` recall was almost zero on test, so the model was not useful for balanced trading decisions.

## 2026-06-05 - Class Weights And Macro-F1 Early Stopping

### Change

- Added class-weighted `CrossEntropyLoss`.
- Saved best checkpoint by validation macro F1 instead of validation loss.
- Made patience track validation macro F1.
- Added per-class precision/recall/F1 report for every epoch.

### Result

Final reported run:

```text
Train | loss 0.9297 | acc 0.509 | macro F1 0.508
  HOLD precision 0.482 | recall 0.509 | F1 0.495
  BUY  precision 0.440 | recall 0.696 | F1 0.539
  SELL precision 0.665 | recall 0.387 | F1 0.489

Val   | loss 1.2231 | acc 0.357 | macro F1 0.357
  HOLD precision 0.348 | recall 0.375 | F1 0.361
  BUY  precision 0.306 | recall 0.491 | F1 0.377
  SELL precision 0.459 | recall 0.263 | F1 0.334

Test  | loss 1.0779 | acc 0.364 | macro F1 0.365
  HOLD precision 0.361 | recall 0.485 | F1 0.414
  BUY  precision 0.310 | recall 0.469 | F1 0.373
  SELL precision 0.471 | recall 0.228 | F1 0.308
```

### Interpretation

- The model stopped collapsing to `SELL`.
- Raw accuracy dropped, but macro F1 improved.
- This is a healthier model than the previous one because it attempts all three classes.
- Generalization is still weak: train macro F1 is around `0.51`, while validation/test macro F1 are around `0.36`.

## 2026-06-05 - Train-Only Feature Normalization

### Change

- Added feature normalization in `models/cnn/train.py`.
- Mean/std are computed from a sample of `50,000` train windows.
- Train, validation, and test all use the same train statistics.

### Why

- Feature scales were very different.
- Price returns had tiny standard deviation, around `0.001-0.002`.
- Volume features could range from `-1` to `10`.
- Without normalization, large volume features can dominate smaller price-return features.

### Result

Final reported run:

```text
Train | loss 0.9333 | acc 0.530 | macro F1 0.530
  HOLD precision 0.458 | recall 0.634 | F1 0.531
  BUY  precision 0.512 | recall 0.551 | F1 0.530
  SELL precision 0.637 | recall 0.450 | F1 0.528

Val   | loss 1.1893 | acc 0.372 | macro F1 0.372
  HOLD precision 0.351 | recall 0.457 | F1 0.397
  BUY  precision 0.313 | recall 0.368 | F1 0.338
  SELL precision 0.459 | recall 0.324 | F1 0.380

Test  | loss 1.0526 | acc 0.399 | macro F1 0.394
  HOLD precision 0.369 | recall 0.631 | F1 0.465
  BUY  precision 0.360 | recall 0.338 | F1 0.349
  SELL precision 0.485 | recall 0.298 | F1 0.369
```

### Interpretation

- Test macro F1 improved from about `0.365` to `0.394`.
- Validation macro F1 remained weak, around `0.37`.
- Training macro F1 reached `0.53`, so the model learns training patterns but generalizes poorly.
- This points toward data/label quality and market-regime shift more than a broken model.

### Correction After Audit

- The first version used the first `50,000` train windows for normalization stats.
- Because processed data is stored stock-by-stock, that sample mostly came from the first stock chunk.
- Changed normalization to use a fixed random sample across the full train set.
- Next run should test this corrected normalization.

## 2026-06-05 - Data Processing Audit

### Findings

- `bankNifty`, `finnifty`, and `nifty50` have `100%` zero volume in `data/raw/5M`.
- Stocks have real volume, but indices do not, so volume features mean different things for different instruments.
- Labels are biased toward `SELL` because `LOWER_MULT = 1.0` and `UPPER_MULT = 1.5`.
- Per-stock label distributions are similar but consistently sell-heavy.
- A few OHLC rows have tiny data-quality anomalies, but not enough to explain the weak model.

### Interpretation

- The preprocessing is usable for a baseline, but it is not how a strong institutional research pipeline would stop.
- Mixing zero-volume indices with real-volume stocks can confuse volume features.
- Asymmetric barriers make `SELL` easier to label than `BUY`.
- The current features may not represent chart shape strongly enough for a CNN.

## 2026-06-05 - Raw Nifty Files Removed

### Finding

- Raw `data/raw/5M` now contains 11 stock files only.
- The remaining stock files have usable volume data.
- `data/processed/5M` is still stale from the old raw set:
  - current processed train shape: `[880265, 60, 12]`
  - expected stock-only train shape after regeneration: `[686128, 60, 12]`

### Interpretation

- Training will still include old Nifty/index data until `script/process_data.py` is rerun.
- After regenerating, rerun training and record the new result here.

## 2026-06-05 - Processed Data Regenerated After Nifty Removal

### Finding

- `script/process_data.py` was rerun after removing Nifty/index raw files.
- Processed shapes now match the 11 stock-only raw set:
  - train: `[686128, 60, 12]`
  - validation: `[85757, 60, 12]`
  - test: `[85757, 60, 12]`

### Label Distribution

```text
train: HOLD 28.19% | BUY 28.32% | SELL 43.48%
val:   HOLD 28.30% | BUY 27.28% | SELL 44.42%
test:  HOLD 27.84% | BUY 27.30% | SELL 44.86%
```

### Interpretation

- Old zero-volume index data is no longer in the processed arrays.
- Labels are still sell-heavy because barriers are still asymmetric:
  - `UPPER_MULT = 1.5`
  - `LOWER_MULT = 1.0`
- Next training run should use this regenerated dataset before changing label logic.

## Model Assessment

The current model is not "bad", but it is very basic.

Good parts:

- `Conv1d` is appropriate for `(features, time)` data.
- The model is small enough to train quickly.
- Pooling reduces the 60-step window to a compact representation.

Weak parts:

- It flattens the final feature map directly, which can overfit to positions inside the 60-candle window.
- It has no dropout, so it may memorize training patterns.
- It has no global pooling, so it may not generalize well across slightly shifted chart patterns.
- It only sees engineered features, not a richer normalized price path.

Recommended next model change:

- Replace the flatten-heavy classifier with adaptive average pooling plus adaptive max pooling.
- Add small dropout before the final classifier.
- Keep the model small; the current results point more to noisy data/features/labels than to needing a huge model.

## Next Experiments

1. Run the current train-normalized setup and record the result here.
2. If validation macro F1 does not improve, make labels symmetric in `script/process_data.py`:

```python
UPPER_MULT = 1.0
LOWER_MULT = 1.0
```

or:

```python
UPPER_MULT = 1.5
LOWER_MULT = 1.5
```

3. Add normalized price-path features so the CNN can see chart shape more directly.
4. Test a model with adaptive pooling and dropout.
