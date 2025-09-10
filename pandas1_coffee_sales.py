# -*- coding: utf-8 -*-
"""Coffee Sales Prediction using Gradient Boosting"""

# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import shap

# -----------------------------
# 2. Load Datasets
# -----------------------------
df1 = pd.read_csv('/kaggle/input/coffee-sales/index_1.csv')
df2 = pd.read_csv('/kaggle/input/coffee-sales/index_2.csv')

print(f'df1 shape: {df1.shape}')
print(f'df2 shape: {df2.shape}')

# Drop unnecessary column from df1
df1 = df1.drop(['card'], axis=1)

# -----------------------------
# 3. Inspect Data
# -----------------------------
df1.info()
df1.describe()
df1.head()
df2.head()

# -----------------------------
# 4. Combine Datasets
# -----------------------------
df_all = pd.concat([df1, df2], ignore_index=True)
df_all.head()
df_all.isnull().sum()

# -----------------------------
# 5. Feature Engineering
# -----------------------------
# Convert date column to datetime
df_all['date'] = pd.to_datetime(df_all['date'])

# Extract date features
df_all['sale_year'] = df_all['date'].dt.year
df_all['sale_month'] = df_all['date'].dt.month
df_all['sale_day'] = df_all['date'].dt.day
df_all['dayofweek'] = df_all['date'].dt.dayofweek
df_all['is_weekend'] = df_all['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
df_all['week_of_month'] = df_all['sale_day'].apply(lambda x: (x - 1) // 7 + 1)

# Coffee-specific features
df_all['has_milk'] = df_all['coffee_name'].str.contains('milk', case=False).astype(int)
df_all['coffee_length'] = df_all['coffee_name'].str.len()
df_all['money_range'] = pd.qcut(df_all['money'], q=3, labels=['Low', 'Medium', 'High'])

# Drop original date columns
df_all = df_all.drop(['date','datetime'], axis=1)
df_all.head()

# -----------------------------
# 6. Explore Data (Optional Visuals)
# -----------------------------
numeric_cols = df_all.select_dtypes(include='number').columns
skew_vals = df_all[numeric_cols].apply(skew).sort_values(ascending=False)

# Plot distributions
for col in numeric_cols:
    sns.histplot(df_all[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Boxplot for money
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_all['money'])
plt.title('Box Plot of Money')
plt.xlabel('Money')
plt.show()

# -----------------------------
# 7. Encoding Features
# -----------------------------
# Encode categorical columns
df_all['cash_type'] = df_all['cash_type'].map({'card':0,'cash':1})
df_all['money_range'] = df_all['money_range'].map({'Medium':0,'Low':1,'High':2})

# Frequency encoding for coffee_name
freq_encoding = df_all['coffee_name'].value_counts(normalize=True)
df_all['coffee_name'] = df_all['coffee_name'].map(freq_encoding)
df_all.head()

# -----------------------------
# 8. Pivot Tables (Optional Analysis)
# -----------------------------
pivot_mean = df_all.pivot_table(
    values='money',
    index='cash_type',
    columns='coffee_name',
    aggfunc='mean',
    fill_value=0
)
print("Pivot Table (Mean of money):")
print(pivot_mean)

pivot_count = df_all.pivot_table(
    values='money',
    index='cash_type',
    columns='coffee_name',
    aggfunc='count',
    fill_value=0
)
print("\nPivot Table (Count of records):")
print(pivot_count)

# -----------------------------
# 9. Optional: Random Forest + SHAP
# -----------------------------
X = df_all.drop(columns=['money'])
y = df_all['money']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Explain model predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# -----------------------------
# 10. Gradient Boosting Pipeline
# -----------------------------
feature_cols = ['cash_type', 'coffee_name', 'sale_year', 'sale_month', 'sale_day',
                'dayofweek', 'is_weekend', 'week_of_month', 'has_milk', 'coffee_length']
target_col = 'money'

X = df_all[feature_cols]
y = df_all[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: scale numeric features
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features)
], remainder='passthrough')

# Pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# -----------------------------
# 11. Cross-Validation
# -----------------------------
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print("Gradient Boosting - Cross-Validation R2 Scores:", np.round(cv_scores, 4))
print("Mean R2 Score:", round(cv_scores.mean(), 4))
print("Std Dev of R2:", round(cv_scores.std(), 4))

# -----------------------------
# 12. Evaluate on Test Set
# -----------------------------
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print("Test Set R² Score:", round(test_score, 4))

# Save predictions
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results_df.to_csv("/kaggle/working/prediction_results.csv", index=False)
print("✅ Saved prediction_results.csv to Kaggle working directory.")

# -----------------------------
# 13. README.txt Creation
# -----------------------------
readme_text = """
Coffee Sales Prediction with Gradient Boosting
=============================================

Project Overview:
-----------------
Predicts money spent per coffee order using features like coffee type, date, and order details.
Gradient Boosting Regressor used with pipeline and cross-validation.

Steps Performed:
----------------
1. Data loading and cleaning
2. Feature engineering and encoding
3. Exploratory analysis
4. Gradient Boosting pipeline with cross-validation
5. Final evaluation on test set
6. Saving predictions as CSV

Results:
--------
- Cross-validation Mean R² Score: ~0.98
- Test Set R² Score: ~0.98

Files:
------
- prediction_results.csv : predicted vs actual values
- README.txt : project documentation

Usage:
------
Run the script in a Jupyter/Kaggle notebook environment.
Ensure datasets are placed in correct paths.

Author:
-------
Zahraa Alrubaie
"""

with open("README.txt", "w") as f:
    f.write(readme_text)

print("✅ README.txt created and saved.")
