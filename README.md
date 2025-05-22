# Employee-Retention-Analysis

## Project Objective
Predict whether an employee is likely to leave the company using machine learning techniques in a semi-supervised setup. Designed as part of a final course project to simulate real-world attrition prediction using limited labeled data.


## Project Description
A machine learning system that predicts employee attrition risk using:
- Semi-supervised learning (SME-guided clustering)
- Feature engineering
- Ensemble modeling (Random Forest classifier turned out to be the best)

Course: DS602 - Intro to Data Analysis and Machine Learning 
Institution: University of Maryland, Baltimore County  
Instructor: Professor Mehmet Sarica

## Dataset Attribution
All data files were provided for educational purposes by Professor Sarica at UMBC:

## Data Sources
- Training data: [employee_departure_dataset_X.csv](https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/final/employee_departure_dataset_X.csv)
- Test data: [employee_departure_dataset_X_prod.csv](https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/final/employee_departure_dataset_X_prod.csv)
- _X_prod.csv`  Production test features (https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/final/employee_departure_dataset_X_prod.csv) 
- _y_prod.csv`  Production test labels  (https://raw.githubusercontent.com/msaricaumbc/DS_data/master/ds602/final/employee_departure_dataset_y_prod.csv)

Key Features
Semi-supervised approach: Combines SME knowledge with machine learning

Feature engineering: Creates meaningful predictors like SalaryGrowthPerYear and WorkloadScore

Cluster-based labeling: Uses K-means to group similar employees before prediction

Multiple model comparison: Evaluates Random Forest, Gradient Boosting, and Logistic Regression

Production-ready pipeline: Includes data cleaning, feature engineering, and model deployment

## Technical Approach
1. **Data Preparation**:
   - Cleaned inconsistent distance metrics
   - Engineered predictive features (SalaryGrowthPerYear, WorkloadScore)
   - Handled missing values

2. **Modeling**:
   - K-means clustering for employee segmentation
   - SME-guided cluster labeling (500 queries)
   - Random Forest classifier (99.18% accuracy)

## Repository Structure
/project-root
│── /notebooks
│ ├── employee_retention_analysis.ipynb # Main analysis notebook
│── /src
│ ├── production.py # Production prediction script
│── README.md
│── requirements.txt
│── best_model.pkl # Trained model


## Installation
''' bash
git clone https://github.com/thedatasage/employee-retention-analysis.git
cd employee-attrition-prediction
pip install -r requirements.txt

## Usage
1. Training:
# In Jupyter notebook:
%run notebooks/employee_retention_analysis.ipynb

2. Production Prediction:
from src.production import production
production(X_path, y_path)  

## Dependencies
Python 3.8+

pandas

scikit-learn

numpy

matplotlib

jupyter

> © 2025 Shiva Bhavya Sree Muttireddy. All rights reserved.  
> This project is part of a course submission and is shared only for academic purposes.  
> Reproduction, reuse, or modification of any content is not permitted.

