# Coffee sales prediction 
## Overview
This project uses machine learning techniques to predict coffee sales revenue (`money`) based on various features such as coffee type, order date, and additional characteristics. The goal is to build an accurate regression model to estimate sales and explore key data patterns.

## Files Included:
- **pandas1-COFFEE SALES.ipynb** – Main Jupyter Notebook containing the full workflow: data cleaning, feature engineering, visualization, model training, and evaluation.
- **PANDAS - COFFEE SALES PREDICTION.csv** – Output CSV containing actual vs. predicted sales values.
- **COFFEE-SALES-TXTFILE.txt** – Plain text summary or notes from the project.
- **README.md** – This file, describing the project structure and usage.

## Features Used:
- Categorical features like `coffee_name`, `cash_type`, and `has_milk`
- Date-based features: `sale_day`, `sale_month`, `dayofweek`, `is_weekend`, `week_of_month`
- Text processing and frequency encoding for high-cardinality categories
- SHAP and correlation analysis for feature importance

## Model Used:
- Gradient Boosting Regressor with a preprocessing pipeline
- Cross-validation (5-fold) used to ensure model stability and reliability

## Results
- **Cross-Validation Mean R² Score**: ~0.9856
- **Test Set R² Score**: ~0.9827

## How to Use
1. Clone the repository or open the notebook on [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/)
2. Ensure all files are in the same directory or adjust paths accordingly
3. Run the notebook step by step to explore the process or generate new predictions


---
## Note
This project is ideal for showcasing data preprocessing, feature engineering, model building, and evaluation in a real-world sales context.

