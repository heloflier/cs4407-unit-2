"""
Sales Prediction Using Regression
CS 4407 - Machine Learning

Predicts weekly retail sales revenue from advertising spend, store size,
customer counts, and promotional activity using linear, multiple linear,
and polynomial regression models.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------------------

data = {
    "Advertising_Spend": [2000, 3000, 2500, 4000, 3500, None, 5000, 4500, 3000, 3800],
    "Store_Size": [1500, 2000, 1800, 2200, None, 2100, 2500, 2400, 2000, 2300],
    "Customers": [200, 250, 230, 300, 280, 260, 320, 310, 270, None],
    "Promotion": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes"],
    "Sales": [40000, 50000, 45000, 60000, 52000, 48000, 65000, 63000, 51000, 59000]
}

df = pd.DataFrame(data)
 
print("Dataset preview:")
print(df)
print("\nDataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isna().sum())

# ---------------------------------------------------------------------------
# Step 2: Handle missing values
# ---------------------------------------------------------------------------
# Using median instead of mean since it's more robust to outliers, and with
# only 10 rows here, one extreme value could throw off the mean easily.

numeric_columns = ["Advertising_Spend", "Store_Size", "Customers"]
 
print("\nRows with missing values (before imputation):")
print(df[df[numeric_columns].isna().any(axis=1)])
 
for column_name in numeric_columns:
    column_median = df[column_name].median()
    df[column_name] = df[column_name].fillna(column_median)
    print(f"\nImputed '{column_name}' missing value with median = {column_median}")
 
print("\nMissing values per column (after imputation):")
print(df.isna().sum())
 
print("\nDataset after handling missing values:")
print(df)