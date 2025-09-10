# Coffee Sales Prediction ☕💰

## Overview
This project uses machine learning to predict coffee sales revenue (money) based on various features such as coffee type, order date, and other characteristics.  
The goal is to build an accurate regression model to estimate sales and explore key data patterns.

## Files Included
- **pandas1-COFFEE_SALES.ipynb** – Main Jupyter Notebook containing data cleaning, feature engineering, visualization, model training, and evaluation.  
- **PANDAS-COFFEE_SALES_PREDICTION.csv** – Output CSV with actual vs. predicted sales values.  
- **COFFEE-SALES-TXTFILE.txt** – Plain text summary or notes from the project.  
- **README.md** – Project description, structure, and usage instructions.  

## Features Used
- **Categorical features**: `coffee_name`, `cash_type`, `has_milk`  
- **Date-based features**: `sale_day`, `sale_month`, `dayofweek`, `is_weekend`, `week_of_month`  
- **Text processing** and **frequency encoding** for high-cardinality categories  
- **SHAP** and correlation analysis for feature importance  

## Model Used
- **Gradient Boosting Regressor** with a preprocessing pipeline  
- **5-fold cross-validation** for model stability and reliability  

## Results
- **Cross-Validation Mean R² Score**: ~0.9856  
- **Test Set R² Score**: ~0.9827  

## How to Use
1. Clone the repository or open the notebook on Kaggle or Google Colab.  
2. Ensure all files are in the same directory or adjust paths accordingly.  
3. Run the notebook step by step to explore the process or generate new predictions.  

## Project Structure
````
coffee-sales-prediction/
│
├── pandas1-COFFEE_SALES.py # Main notebook with full workflow
├── PANDAS-COFFEE_SALES_PREDICTION.csv # Predicted vs actual sales CSV
├── COFFEE-SALES-TXTFILE.txt # Project notes or summary
└── README.md # Project documentation
````
## Notes
This project demonstrates **data preprocessing, feature engineering, model building, and evaluation**.  
It is ideal for showcasing skills in machine learning pipelines and regression modeling.  

## License
This project is licensed under the **MIT License** 
