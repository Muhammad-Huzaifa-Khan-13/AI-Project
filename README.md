# Intelligent Reading Comprehension and Quiz Generation System

AI course project (BSCS Spring 2026) for building a modular reading comprehension and quiz generation platform using the RACE dataset, traditional ML models, and Streamlit.

## Project Goals

- Model A: question/answer verification with multiple ML baselines, unsupervised learning, and ensemble comparison
- Model B: distractor generation and hint generation
- UI: Streamlit application with article input, quiz view, hint panel, and analytics dashboard

## Technology Stack

- Python
- pandas
- scikit-learn
- Streamlit
- TF-IDF / One-Hot Encoding
- Cosine similarity
- Logistic Regression, SVM, KMeans
- XGBoost (optional)

## Dataset

RACE dataset (Kaggle): [https://www.kaggle.com/datasets/ankitdhiman7/race-dataset](https://www.kaggle.com/datasets/ankitdhiman7/race-dataset)

Expected schema:

- `id`
- `article`
- `question`
- `A`
- `B`
- `C`
- `D`
- `answer` (must be one of `A/B/C/D`)

## Current Implemented Foundation

Phase 1 data pipeline is implemented:

- `src/config.py`: central paths, schema, labels, split names, deterministic seed
- `src/utils.py`: logging, reproducibility helpers, JSON saving, directory helpers
- `src/data_loader.py`: split loading + schema checks + label checks + null-row handling
- `src/preprocessing.py`: CLI preprocessing pipeline and report generation

### Preprocessing Features

- lowercasing
- punctuation removal
- whitespace normalization
- cleaned text columns:
  - `clean_article`, `clean_question`, `clean_A`, `clean_B`, `clean_C`, `clean_D`
- derived columns:
  - `verifier_input`
  - `question_context`

Outputs:

- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/val_processed.csv`
- `data/processed/reports/preprocessing_report.json`

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Preprocessing

```bash
python -m src.preprocessing
```

Optional custom seed:

```bash
python -m src.preprocessing --seed 42
```

## Raw Data File Names

Place raw files in `data/raw/` as:

- `train.csv`
- `test.csv`
- `val.csv` (preferred)

If `val.csv` is not present, pipeline automatically falls back to `dev.csv` for the validation split.

## Project Structure

```bash
AI-Project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── report/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── utils.py
├── tests/
├── ui/
├── .gitignore
├── PROJECT_CONTEXT.md
├── requirements.txt
└── README.md
```

## Next Steps

- Add `notebooks/EDA.ipynb` with rubric-aligned exploratory analysis
- Train Model A baselines (Logistic Regression, SVM)
- Add unsupervised track (KMeans) and evaluation tables
- Implement Model B distractor + hint pipeline
- Integrate all components into Streamlit UI

## Run Model A Baselines

After preprocessing, train Model A baseline classifiers:

```bash
python -m src.model_a_train
```

Evaluate on both validation and test splits:

```bash
python -m src.model_a_train --evaluate-test {venv\Scripts\python -m src.model_a_train --evaluate-test}
```

Artifacts generated:

- `models/model_a/traditional/logistic_regression.joblib`
- `models/model_a/traditional/linear_svm.joblib`
- `models/model_a/reports/baseline_metrics.json`

## Run Model A Unsupervised (KMeans)

Train unsupervised baseline and compare with supervised report:

```bash
python -m src.model_a_unsupervised
```

Include test evaluation in addition to validation:

```bash
python -m src.model_a_unsupervised --evaluate-test
```

Artifacts generated:

- `models/model_a/traditional/kmeans_label_mapped.joblib`
- `models/model_a/reports/unsupervised_metrics.json`

## Run Model A Ensemble

Train hard-voting ensemble (Logistic Regression + Linear SVM):

```bash
python -m src.model_a_ensemble
```

Include test evaluation:

```bash
python -m src.model_a_ensemble --evaluate-test
```

Artifacts generated:

- `models/model_a/traditional/ensemble_hard_vote_lr_svm.joblib`
- `models/model_a/reports/ensemble_metrics.json`

## Run Model A Stacking Ensemble

Train stacking ensemble (LR + SVM base learners, Logistic Regression meta learner):

```bash
python -m src.model_a_stacking
```

Include test evaluation:

```bash
python -m src.model_a_stacking --evaluate-test
```

Artifacts generated:

- `models/model_a/traditional/stacking_lr_svm_meta_lr.joblib`
- `models/model_a/reports/stacking_metrics.json`

## Run Model B Distractor Generation

Generate top-3 distractors using TF-IDF cosine ranking:

```bash
python -m src.model_b_distractors --split val
```

Quick sample run:

```bash
python -m src.model_b_distractors --split val --sample-size 200
```

Artifacts generated:

- `models/model_b/outputs/distractors_<split>.csv`
- `models/model_b/reports/distractors_report_<split>.json`

## Run Model B Progressive Hints

Generate easy -> medium -> hard extractive hints from article sentences:

```bash
python -m src.model_b_hints --split val
```

Quick sample run:

```bash
python -m src.model_b_hints --split val --sample-size 200
```

Artifacts generated:

- `models/model_b/outputs/hints_<split>.csv`
- `models/model_b/reports/hints_report_<split>.json`

## run app

  .\venv\Scripts\python -m streamlit run ui\app.py                                        