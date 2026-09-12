"""
Sales Prediction Using Regression
CS 4407 - Machine Learning

Predicts weekly retail sales revenue from advertising spend, store size,
customer counts, and promotional activity using linear, multiple linear,
and polynomial regression models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 1: LOAD DATASET")
print("=" * 70)

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
# Step 2 / Question 2a: Handle missing values
# ---------------------------------------------------------------------------
# Using median instead of mean since it's more robust to outliers, and with
# only 10 rows here, one extreme value could throw off the mean easily.

print("\n" + "=" * 70)
print("QUESTION 2a: HANDLE MISSING VALUES")
print("=" * 70)

numeric_columns = ["Advertising_Spend", "Store_Size", "Customers"]
missing_rows_index = df[df[numeric_columns].isna().any(axis=1)].index
 
print("\nRows with missing values (before imputation):")
print(df.loc[missing_rows_index])
 
for column_name in numeric_columns:
    column_median = df[column_name].median()
    df[column_name] = df[column_name].fillna(column_median)
    print(f"\nImputed '{column_name}' missing value with median = {column_median}")
 
print("\nMissing values per column (after imputation):")
print(df.isna().sum())
 
print("\nSame rows after imputation:")
print(df.loc[missing_rows_index])

# ---------------------------------------------------------------------------
# Step 3 / Question 2b: Encode the categorical Promotion column
# ---------------------------------------------------------------------------
# Promotion only has two values (Yes/No), so a simple binary mapping is
# enough - no need for one-hot encoding on a single binary column.

print("\n" + "=" * 70)
print("QUESTION 2b: ENCODE PROMOTION COLUMN")
print("=" * 70)
 
promotion_mapping = {"Yes": 1, "No": 0}
 
print("\nPromotion column before encoding:")
print(df["Promotion"].head())
 
df["Promotion"] = df["Promotion"].map(promotion_mapping)
 
print("\nPromotion column after encoding:")
print(df["Promotion"].head())

# ---------------------------------------------------------------------------
# Step 4 / Question 2c: Apply feature scaling
# ---------------------------------------------------------------------------
# Scaling puts Advertising_Spend, Store_Size, and Customers on comparable
# ranges - matters most for polynomial regression later, where squaring or
# cubing large raw values can cause numerical issues.

print("\n" + "=" * 70)
print("QUESTION 2c: APPLY FEATURE SCALING")
print("=" * 70)

feature_columns_to_scale = ["Advertising_Spend", "Store_Size", "Customers"]
scaler = StandardScaler()
 
print("\nFeature values before scaling:")
print(df[feature_columns_to_scale].head())
 
df[feature_columns_to_scale] = scaler.fit_transform(df[feature_columns_to_scale])
 
print("\nFeature values after scaling:")
print(df[feature_columns_to_scale].head())

# ---------------------------------------------------------------------------
# Step 5 / Question 3.i.a: Simple linear regression (Advertising_Spend only)
# ---------------------------------------------------------------------------
# Built on the full preprocessed dataset, as instructed in Question 3. A
# proper train/test split comes later in Question 4 for performance
# evaluation.

print("\n" + "=" * 70)
print("QUESTION 3.i.a: SIMPLE LINEAR REGRESSION")
print("=" * 70)

X_simple = df[["Advertising_Spend"]]
y = df["Sales"]

simple_linear_model = LinearRegression()
simple_linear_model.fit(X_simple, y)

print(f"\nIntercept: {simple_linear_model.intercept_:.2f}")
print(f"Coefficient (Advertising_Spend): {simple_linear_model.coef_[0]:.2f}")

plt.scatter(df["Advertising_Spend"], y, label="Actual")
plt.plot(df["Advertising_Spend"], simple_linear_model.predict(X_simple), color="red", label="Predicted")
plt.xlabel("Advertising Spend (scaled)")
plt.ylabel("Sales")
plt.title("Simple Linear Regression Preview")
plt.legend()
plt.savefig("simple_linear_preview.png")
# plt.show()

# ---------------------------------------------------------------------------
# Step 6 / Question 3.i.b: Multiple linear regression (all features)
# ---------------------------------------------------------------------------
# With more than one predictor, there's no single x-axis to plot a fit line
# against, so the preview below uses actual vs. predicted instead, with a
# diagonal reference line marking perfect predictions.
 
print("\n" + "=" * 70)
print("QUESTION 3.i.b: MULTIPLE LINEAR REGRESSION")
print("=" * 70)
 
feature_columns = ["Advertising_Spend", "Store_Size", "Customers", "Promotion"]
X_multiple = df[feature_columns]
 
multiple_linear_model = LinearRegression()
multiple_linear_model.fit(X_multiple, y)
 
print(f"\nIntercept: {multiple_linear_model.intercept_:.2f}")
for feature_name, coefficient in zip(feature_columns, multiple_linear_model.coef_):
    print(f"Coefficient ({feature_name}): {coefficient:.2f}")
 
predicted_sales = multiple_linear_model.predict(X_multiple)
 
plt.figure()
plt.scatter(y, predicted_sales, label="Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Multiple Linear Regression Preview")
plt.legend()
plt.savefig("multiple_linear_preview.png")
# plt.show()
 
 # ---------------------------------------------------------------------------
# Step 7 / Question 3.i.c: Polynomial regression (degree 2)
# ---------------------------------------------------------------------------
# Built on Advertising_Spend alone, same predictor as the simple linear
# model, to isolate the effect of adding a squared term. Extending
# polynomial terms across all four features would produce more parameters
# than data points (10 rows), guaranteeing an overfit.
 
print("\n" + "=" * 70)
print("QUESTION 3.i.c: POLYNOMIAL REGRESSION (DEGREE 2)")
print("=" * 70)
 
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_transformer.fit_transform(X_simple)
 
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly, y)
 
print(f"\nIntercept: {polynomial_model.intercept_:.2f}")
print(f"Coefficient (Advertising_Spend): {polynomial_model.coef_[0]:.2f}")
print(f"Coefficient (Advertising_Spend^2): {polynomial_model.coef_[1]:.2f}")
 
# Sort by Advertising_Spend so the curve draws cleanly left to right
sort_order = df["Advertising_Spend"].argsort()
advertising_spend_sorted = df["Advertising_Spend"].values[sort_order]
predicted_sales_sorted = polynomial_model.predict(X_poly)[sort_order]
 
plt.figure()
plt.scatter(df["Advertising_Spend"], y, label="Actual")
plt.plot(advertising_spend_sorted, predicted_sales_sorted, color="red", label="Predicted")
plt.xlabel("Advertising Spend (scaled)")
plt.ylabel("Sales")
plt.title("Polynomial Regression Preview")
plt.legend()
plt.savefig("polynomial_preview.png")
# plt.show()
 