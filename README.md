# ❤️ Heart Disease Prediction System

## 📌 Overview

The Heart Disease Prediction System is a Machine Learning-based healthcare application designed to predict the likelihood of cardiovascular disease using patient medical parameters.

The project uses the UCI Heart Disease Dataset and compares multiple machine learning algorithms to identify the most effective model for disease prediction. The final solution is deployed through an interactive Streamlit web application that allows users to enter patient details and receive real-time risk assessments.

---

## 🎯 Objectives

* Predict the risk of heart disease using patient health records.
* Compare the performance of multiple machine learning algorithms.
* Build an interactive and user-friendly web application.
* Provide interpretable predictions and risk categorization.
* Assist users in understanding cardiovascular risk factors.

---

## 🗂 Dataset

**Dataset:** UCI Heart Disease Dataset

### Features Used

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age of patient                    |
| sex      | Gender                            |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Serum cholesterol                 |
| fbs      | Fasting blood sugar               |
| restecg  | Resting ECG results               |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise-induced angina           |
| oldpeak  | ST depression induced by exercise |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels           |
| thal     | Thalassemia                       |
| target   | Heart disease presence            |

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Missing Value Handling
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Model Comparison
9. Streamlit Dashboard Development
10. Deployment

---

## 🤖 Machine Learning Models

The following models were implemented and compared:

### Logistic Regression

* Accuracy: 86.89%

### Random Forest (Best Model)

* Accuracy: 90.16%
* ROC-AUC: 95.45%

### XGBoost

* Accuracy: 83.61%

---

## 📊 Performance Metrics

### Random Forest Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 90.16% |
| Precision | 84%    |
| Recall    | 96%    |
| F1 Score  | 90%    |
| ROC-AUC   | 95.45% |

---

## 🚀 Features

### Machine Learning

* Data preprocessing pipeline
* Multiple model comparison
* Model persistence using Joblib
* Performance evaluation

### Dashboard Features

* Real-time prediction
* Risk probability estimation
* Risk categorization
* Patient summary
* Health recommendations
* Feature importance visualization

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Joblib
* Matplotlib
* Streamlit

### Development Tools

* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
heart_disease_prediction/
│
├── data/
│   ├── processed.cleveland.data
│   └── heart_cleaned.csv
│
├── models/
│   ├── heart_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── results/
│   └── feature_importance.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   ├── evaluate_rf.py
│   ├── feature_importance.py
│   └── shap_explain.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ Installation & Execution

### Clone Repository

```bash
git clone https://github.com/vidhigoyal28/heart-disease-prediction-system.git
cd heart-disease-prediction-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Future Enhancements

* SHAP-based Explainable AI
* PDF Health Report Generation
* Advanced Risk Visualization
* Cloud Deployment
* Real-Time Health Monitoring Integration
* Deep Learning Models for Prediction

---

## 👩‍💻 Author

**Vidhi Goyal**

B.Tech Student | Machine Learning Enthusiast | Software Developer

GitHub: https://github.com/vidhigoyal28

---

## ⭐ Project Highlights

* End-to-end Machine Learning Pipeline
* 90.16% Prediction Accuracy
* 95.45% ROC-AUC Score
* Interactive Streamlit Dashboard
* Healthcare-focused AI Application
* Production-ready Project Structure
