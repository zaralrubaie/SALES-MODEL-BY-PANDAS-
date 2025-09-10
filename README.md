# Coffee Sales Prediction 

## Overview
This project uses machine learning to predict coffee sales revenue (money) based on various features such as coffee type, order date, and other characteristics.  
The goal is to build an accurate regression model to estimate sales and explore key data patterns.

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
├── pandas1_coffee_sales.py                # Main Python script with full workflow
├── PANDAS - COFFEE SALES PREDICTION.csv   # Output CSV with actual vs predicted values
├── README.md                               # Project description and usage
├── requirements.txt                        # Project dependencies
└── LICENSE                                 # Optional: license file
                   
````
## Notes
This project demonstrates **data preprocessing, feature engineering, model building, and evaluation**.  
It is ideal for showcasing skills in machine learning pipelines and regression modeling.  

## License
This project is licensed under the **MIT License** 
