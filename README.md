# ML Assignment 2 : Vedant Milind Gokhale - 2025AB05211

## Cardiovascular Disease Dataset - [Link](https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset/data)
## Deployment on streamlit cloud - [Link](https://2025ab05211.streamlit.app/)

## Model Performance Summary

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.985 | 0.998 | 0.983 | 0.991 | 0.987 | — |
| Decision Tree | 0.980 | 0.981 | 0.991 | 0.974 | 0.983 | — |
| kNN | 0.920 | 0.971 | 0.939 | 0.922 | 0.930 | — |
| Naive Bayes | 0.945 | 0.992 | 0.934 | 0.974 | 0.954 | — |
| Random Forest (Ensemble) | 0.985 | 0.998 | 0.983 | 0.991 | 0.987 | — |
| XGBoost (Ensemble) | **0.990** | **0.9997** | **0.991** | **0.991** | **0.991** | — |


---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves very high accuracy and AUC, indicating that the dataset is close to linearly separable. Provides strong baseline performance with excellent interpretability. |
| Decision Tree | Performs well but slightly below ensemble models. Susceptible to overfitting and variance, which explains marginally lower AUC and recall. |
| kNN | Lowest overall performance among tested models. Sensitive to feature scaling and local noise, leading to reduced accuracy and F1 compared to other approaches. |
| Naive Bayes | Shows strong AUC despite independence assumptions. Performs better than kNN but slightly below tree-based ensembles in overall accuracy. |
| Random Forest (Ensemble) | Matches Logistic Regression in accuracy while improving robustness through ensemble averaging. Demonstrates strong generalization and stability. |
| XGBoost (Ensemble) | **Best performing model** across all metrics. Extremely high AUC and balanced precision-recall indicate superior ability to capture complex nonlinear relationships. |

---

## Project Overview

This project applies multiple **machine learning classification algorithms** to predict the presence of **heart disease** from clinical attributes.

A **Streamlit interactive dashboard** enables:

- Loading **pretrained models**
- Editing **test patient data**
- Viewing **evaluation metrics & confusion matrix**
- Exploring **training insights and model comparison**

---

## Models Implemented

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## Evaluation Metrics Used

- Accuracy  
- AUC (Area Under ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## How to Run the Application

```bash
pip install -r requirements.txt
streamlit run app.py
