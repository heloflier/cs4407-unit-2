"""
Sales Prediction Using Regression
CS 4407 - Machine Learning

Predicts weekly retail sales revenue from advertising spend, store size,
customer counts, and promotional activity using linear, multiple linear,
and polynomial regression models.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
df.info()
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

multiple_predicted_sales = multiple_linear_model.predict(X_multiple)

plt.figure()
plt.scatter(y, multiple_predicted_sales, label="Predictions")
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

# ---------------------------------------------------------------------------
# Step 8 / Question 3.ii.a: Predict sales for a new data point
# ---------------------------------------------------------------------------
# The new point goes through the same preprocessing as the training data:
# same Promotion mapping, and the same fitted scaler (transform, not
# fit_transform, so it's scaled using the training data's mean/std, not
# its own).

print("\n" + "=" * 70)
print("QUESTION 3.ii.a: PREDICT SALES FOR A NEW DATA POINT")
print("=" * 70)
 
new_data_point = pd.DataFrame({
    "Advertising_Spend": [4200],
    "Store_Size": [2100],
    "Customers": [290],
    "Promotion": ["Yes"]
})

new_data_point["Promotion"] = new_data_point["Promotion"].map(promotion_mapping)
new_data_point[feature_columns_to_scale] = scaler.transform(new_data_point[feature_columns_to_scale])

print("\nNew data point after preprocessing:")
print(new_data_point)
 
new_point_simple = new_data_point[["Advertising_Spend"]]
new_point_multiple = new_data_point[feature_columns]
new_point_poly = poly_transformer.transform(new_point_simple)

simple_prediction = simple_linear_model.predict(new_point_simple)[0]
multiple_prediction = multiple_linear_model.predict(new_point_multiple)[0]
polynomial_prediction = polynomial_model.predict(new_point_poly)[0]

print(f"\nSimple linear regression prediction: {simple_prediction:.2f}")
print(f"Multiple linear regression prediction: {multiple_prediction:.2f}")
print(f"Polynomial regression prediction: {polynomial_prediction:.2f}")

# ---------------------------------------------------------------------------
# Step 9 / Question 3.ii.b: Plot actual vs predicted for all models
# ---------------------------------------------------------------------------
# All three models plotted on the same actual-vs-predicted axes so they
# can be compared directly, rather than as three separate figures.

print("\n" + "=" * 70)
print("QUESTION 3.ii.b: PLOT ACTUAL VS PREDICTED (ALL MODELS)")
print("=" * 70)

simple_predicted_sales = simple_linear_model.predict(X_simple)
polynomial_predicted_sales = polynomial_model.predict(X_poly)

plt.figure()
plt.scatter(y, simple_predicted_sales, label="Simple Linear")
plt.scatter(y, multiple_predicted_sales, label="Multiple Linear")
plt.scatter(y, polynomial_predicted_sales, label="Polynomial")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - All Models")
plt.legend()
plt.savefig("all_models_actual_vs_predicted.png")
print("\nSaved plot: all_models_actual_vs_predicted.png")
# plt.show()

# ---------------------------------------------------------------------------
# Step 10 / Question 4a: Train/test split and evaluation metrics
# ---------------------------------------------------------------------------
# Models are refit on the training split only, rather than reusing the
# Question 3 models (which were fit on the full dataset) - evaluating a
# model on data it already saw during training would defeat the purpose
# of a train/test split. With only 10 rows total, even a 3-row test set
# is small enough that these metrics should be read as indicative, not
# precise (see write-up notes for a comparison against a 2-row split).

print("\n" + "=" * 70)
print("QUESTION 4a: TRAIN/TEST SPLIT AND EVALUATION METRICS")
print("=" * 70)

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(train_df)} rows")
print(f"Test set size: {len(test_df)} rows")

y_train = train_df["Sales"]
y_test = test_df["Sales"]

def evaluate_model(model_name, model, X_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R^2: {r2:.4f}")
    return model, rmse, mae, r2

evaluate_model(
    "Simple Linear Regression", LinearRegression(),
    train_df[["Advertising_Spend"]], test_df[["Advertising_Spend"]]
)

evaluate_model(
    "Multiple Linear Regression", LinearRegression(),
    train_df[feature_columns], test_df[feature_columns]
)

poly_eval_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_eval_transformer.fit_transform(train_df[["Advertising_Spend"]])
X_test_poly = poly_eval_transformer.transform(test_df[["Advertising_Spend"]])

evaluate_model(
    "Polynomial Regression", LinearRegression(),
    X_train_poly, X_test_poly
)

# ---------------------------------------------------------------------------
# Step 11 / Question 4b: Ridge and Lasso regression
# ---------------------------------------------------------------------------
# Applied to the same four features as multiple linear regression, since
# that's the model showing multicollinearity (the flipped Advertising_Spend
# sign in Question 3.i.b) - regularization is the standard fix for exactly
# this kind of coefficient instability.
 
print("\n" + "=" * 70)
print("QUESTION 4b: RIDGE AND LASSO REGRESSION")
print("=" * 70)
 
ridge_model, *_ = evaluate_model(
    "Ridge Regression", Ridge(alpha=1.0),
    train_df[feature_columns], test_df[feature_columns]
)
for feature_name, coefficient in zip(feature_columns, ridge_model.coef_):
    print(f"  Coefficient ({feature_name}): {coefficient:.2f}")
 
lasso_model, *_ = evaluate_model(
    "Lasso Regression", Lasso(alpha=1.0),
    train_df[feature_columns], test_df[feature_columns]
)
for feature_name, coefficient in zip(feature_columns, lasso_model.coef_):
    print(f"  Coefficient ({feature_name}): {coefficient:.2f}")
    