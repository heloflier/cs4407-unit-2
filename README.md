# Sales Prediction Using Regression

Programming assignment: predicting weekly retail sales revenue from advertising
spend, store size, customer counts, and promotional activity, using linear,
multiple linear, and polynomial regression models built with `scikit-learn`.

## Contents

- `sales_regression.py` — main analysis script (data loading, preprocessing,
  model building, evaluation)
- `requirements.txt` — Python dependencies

## Workflow

The script is built incrementally, corresponding to the assignment's four
learning outcomes:

1. Data loading
2. Preprocessing (missing values, categorical encoding, feature scaling)
3. Model implementation (simple linear, multiple linear, polynomial regression)
4. Model interpretation and evaluation (predictions, plots, RMSE/MAE/R²,
   Ridge and Lasso regularization)

## Running

```bash
pip install -r requirements.txt
python sales_regression.py
```
