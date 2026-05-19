# Building Transparent Loan Approval Prediction System

This project builds a **loan approval prediction** model and then explains its behavior using multiple **explainable AI (XAI)** and **feature relevance** techniques.

The workflow is implemented in a single notebook that:
- Loads and explores the dataset
- Cleans missing / inconsistent values
- Encodes categorical features and engineers additional features
- Trains a baseline **Random Forest** classifier
- Compares multiple interpretability methods (filter, wrapper, embedded, model-specific, and model-agnostic SHAP)

## Dataset

The dataset file is included in this repo as `LoanPredictionProblemDataset.csv`.

- Target column: `Loan_Status` (approval outcome)
- The notebook removes `Loan_ID` and performs preprocessing before training.

## Project Structure

- `main.ipynb` — end-to-end pipeline (EDA → preprocessing → modeling → interpretability)
- `LoanPredictionProblemDataset.csv` — input dataset
- `README.md` — project overview and setup

## Setup (macOS)

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

Notes:
- If `shap` installation fails on your machine, try upgrading pip first (`python -m pip install -U pip`) and reinstalling.

## Run

### Option A — VS Code

1. Open `main.ipynb` in VS Code.
2. Select the `.venv` Python kernel.
3. Run the notebook cells from top to bottom.

### Option B — Jupyter

```bash
jupyter lab
```

Then open `main.ipynb` and run all cells.

## What’s Inside the Notebook

- **EDA**: distributions and categorical counts
- **Cleaning**: missing value handling and basic value fixes
- **Preprocessing**: label encoding + one-hot encoding
- **Feature engineering**: `TotalIncome`, `EMI`, `IncomeToLoan`
- **Model**: `RandomForestClassifier` + accuracy / classification report
- **Interpretability / feature relevance**:
  - Filter methods: Chi-square, ANOVA F-test, Mutual Information
  - Wrapper method: RFE (Logistic Regression)
  - Embedded method: Lasso
  - Model-specific: RF feature importance (MDI) + permutation importance
  - Model-agnostic: SHAP global + local explanations
  - Comparative plots and correlation heatmap across methods

## Author

Name: Nahom Zenebe
