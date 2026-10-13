# Nepal Earthquake Building Damage Prediction

## Project Overview

This machine-learning project predicts the level of building damage caused by the 2015 Nepal earthquake using building characteristics from the Nepal Earthquake dataset.

The target variable is `damage_grade`:

- **Grade 1:** Low damage
- **Grade 2:** Medium damage
- **Grade 3:** Severe damage

The project covers exploratory data analysis, model training and evaluation, feature importance, and predictions for the supplied test dataset.

## Dataset

The notebook expects four CSV files to be attached as a Kaggle dataset input:

| File | Purpose |
|---|---|
| `train_values.csv` | Features for labelled training buildings |
| `train_labels.csv` | Damage-grade labels |
| `test_values.csv` | Features for buildings requiring predictions |
| `submission_format.csv` | Template for prediction output |

Expected dimensions:

- `train_values.csv`: 260,601 × 39
- `train_labels.csv`: 260,601 × 2
- `test_values.csv`: 86,868 × 39
- `submission_format.csv`: 86,868 × 2

The notebook's setup cell uses:

```python
DATA_DIR = "/kaggle/input/datasets/ggo4411/nepal-earthquake-datasets"
```

If Kaggle shows a different mount path, update `DATA_DIR`. Files under `/kaggle/input/` are read-only.

## Workflow

The notebook `nepal_earthquake_ml_cleaned_labeled.ipynb` includes:

1. Loading libraries and all four CSV files.
2. Checking dataset structure and aligning labels to features using `building_id`.
3. EDA of damage-grade distribution, data types, missing values, foundation type, and building age.
4. A stratified 80:20 training-validation split.
5. Preprocessing with one-hot encoding for text and geographic identifier features.
6. A baseline Random Forest model with classification metrics and a confusion matrix.
7. A fair comparison of Random Forest and Extra Trees using the same sampled training rows and validation set.
8. Random Forest feature importance aggregated by original input feature.
9. Final model fitting on all labelled examples and test prediction export.

## Models and Evaluation

The baseline Random Forest uses:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`

The notebook reports accuracy, macro F1-score, weighted F1-score, per-class precision/recall/F1, and a confusion matrix. Macro F1 is important because the damage grades are imbalanced.

The Random Forest and Extra Trees sampled comparison uses the same 25% sample of the training split and the same validation set. Treat those sampled results separately from the full-training-split baseline because the training sizes differ.

Previously recorded baseline validation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7319 |
| Macro F1-score | 0.6801 |
| Weighted F1-score | 0.7277 |

These are results from the earlier baseline run; exact results can vary with data, package versions, and execution environment.

## Requirements

Libraries used:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Kaggle Notebooks usually includes these. For a local environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run on Kaggle

1. Open Kaggle Notebooks.
2. Attach the Nepal Earthquake dataset containing the four CSV files.
3. Import `nepal_earthquake_ml_cleaned_labeled.ipynb`.
4. Confirm `DATA_DIR` matches the path shown in Kaggle's Input panel.
5. Run notebook cells from top to bottom.
6. Wait for training and evaluation to finish.
7. After the final prediction cell runs, locate the output CSV in `/kaggle/working/`.

## Output

The notebook saves predictions to:

```text
/kaggle/working/nepal_earthquake_submission.csv
```

The predictions are aligned to test `building_id` values and written using the expected submission columns. Download the CSV from Kaggle's output/files panel.

The provided test data has no ground-truth damage labels, so local test accuracy cannot be calculated from these files alone.

## Notes and Limitations

- The target classes are imbalanced; accuracy should not be the only metric considered.
- Some recorded building ages are unusually high, including `995`. They are retained rather than removed without verification.
- Duplicate feature rows are not automatically removed because different buildings can share the same recorded characteristics.
- Feature importance describes model reliance; it does not prove causation.
- This educational model should not replace structural engineering inspection or official damage assessment.

## Suggested Repository Structure

```text
your-repository/
├── README.md
├── nepal_earthquake_ml_cleaned_labeled.ipynb
└── (optional: report, presentation, and result figures)
```

Avoid committing large dataset files if they are already hosted separately. Instead, document where the data can be attached or obtained.

## Reproducibility

The notebook uses `random_state=42` for the split and model configurations. Results and runtimes may still vary with library versions and compute resources.
