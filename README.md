# 🍽️ Mess Demand & Food Waste Prediction System

## 📌 Project Overview
This project is developed as part of **Hackathon-3: Development of Pipelines and Maintenance of Models**.  
The objective is to build a **production-ready machine learning pipeline** for predicting mess demand in a university setting, with emphasis on **engineering robustness**, **model lifecycle**, **SQL data storage**, **dashboarding**, and **version control**.

The system predicts mess demand levels (**Low / Medium / High**) using operational and contextual features such as:
- Meal type and day
- Menu composition (primary dish abstraction)
- Hostel occupancy
- Academic calendar (Regular / Exams / Holidays)
- Availability of dessert, fruit, and drinks

The focus of this project is **pipeline survivability and maintainability**, not model accuracy alone.

---

## 🎯 Problem Statement
University mess operations frequently face:
- Food wastage due to over-preparation
- Long queues and shortages due to under-preparation
- Poor anticipation of demand changes caused by menu and academic schedules

This project aims to:
- Predict mess demand in advance
- Support better food quantity planning
- Reduce food wastage
- Improve operational efficiency

---

## 🧠 Solution Approach
The solution is implemented as a complete ML pipeline consisting of:

1. **Synthetic Data Generation**
   - Rule-based generation simulating realistic mess behavior
   - Factors include meal timing, menu type, occupancy, and semester phase

2. **SQL Data Storage**
   - All data stored in a SQLite database
   - Enables structured querying and future updates

3. **Model Training**
   - Traditional machine learning models (no neural networks)
   - Feature engineering to generalize menu impact
   - Models evaluated and serialized for reuse

4. **Prediction-Ready Pipeline**
   - Models loaded outside notebooks
   - Supports real-time predictions

5. **Interactive Dashboard**
   - Built using Streamlit
   - Provides prediction, EDA, data overview, and model lifecycle visibility

6. **Model Lifecycle & Retraining**
   - New data triggers retraining
   - Each retraining creates a new model version
   - Older models are preserved using a model registry

---

## 🧪 Data Description
- The dataset is **synthetically generated** using rule-based logic
- Simulates real university mess operations
- Stored in a **SQLite database**
- Includes features such as:
  - Meal type
  - Day of week
  - Primary dish abstraction
  - Hostel occupancy
  - Semester phase
  - Demand level (target variable)

---

## 🤖 Machine Learning Models

- **Problem Type:** Multi-class Classification  
- **Target Variable:** Demand Level (Low / Medium / High)  

### 🧪 Models Trained & Evaluated

During experimentation and retraining phases, the following models were trained:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting Classifier  
- XGBoost Classifier  

Each model was evaluated using:

- Accuracy  
- Precision (Macro)  
- Recall (Macro)  
- F1-Score (Macro)  

The final deployed model for each version was selected based on **Macro F1-Score**, ensuring balanced performance across all demand classes.

---

---

## 🔄 Model Lifecycle & Update History

This project follows a structured retraining and versioning workflow.

Whenever new synthetic data is generated and appended to the database:

1. The model is retrained on the updated dataset.
2. Performance metrics are recalculated.
3. A new model version is saved.
4. The `model_registry.json` file is updated.
5. The dashboard automatically loads the latest model version.

---

### 📅 Model Training Timeline

| Version | Date        | Event              | Description |
|---------|------------|--------------------|-------------|
| v1      | 08 Feb 2025 | Initial Training   | Baseline model trained on initial dataset |
| v2      | 20 Feb 2025 | First Retraining   | Additional data appended and model retrained |
| v3      | 28 Feb 2025 | Second Retraining  | Data regenerated and model re-evaluated |

Each retraining includes:

- Updated dataset size  
- Updated evaluation metrics  
- Updated training timestamp  
- Version increment in registry  

---

## 📊 Model Registry

The file `models/model_registry.json` maintains:

- Model version  
- Model type  
- Training date  
- Number of records used  
- Accuracy  
- Precision (Macro)  
- Recall (Macro)  
- F1-Score (Macro)  

This ensures transparency, reproducibility, and traceability of model evolution.
---

## 📊 Dashboard Features
The Streamlit dashboard provides:

### 🔮 Demand Prediction
- Inputs:
  - Meal type
  - Primary dish
  - Day of week
  - Hostel occupancy
  - Semester phase
  - Dessert / Fruit / Drink indicators
- Output:
  - Predicted demand level (Low / Medium / High)

### 📊 EDA & Insights
- Overall demand distribution
- Demand by meal type
- Demand by day of week
- Hostel occupancy vs demand
- Impact of drinks on demand
- Demand variation across semester phases

### 📁 Data Overview
- Dataset statistics
- Sample records
- Data description

### ℹ️ Model Info
- Current model version
- Training date
- Records used
- Algorithm type

---

