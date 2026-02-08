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

## 🗂️ Project Structure

Mess-demand-prediction-hackathon3/
│
├── app.py # Streamlit dashboard
│
├── database/
│ └── mess.db # SQLite database (included intentionally)
│
├── models/
│ ├── model_v1.pkl # Trained ML model
│ ├── encoders.pkl # Label encoders
│ └── model_registry.json # Model version history
│
├── scripts/
│ ├── generate_mess_data.py # Data generation script
│ ├── train_model.py # Initial model training
│ └── retrain_model.py # Model retraining & versioning
│
├── README.md
└── .gitignore


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

## 🤖 Machine Learning Model
- **Problem Type:** Classification  
- **Target Variable:** Demand Level (Low / Medium / High)  
- **Models Used:**
  - Logistic Regression
  - Decision Tree
  - Random Forest (final selected model)
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

Model performance is evaluated to ensure reasonable predictive behavior, with emphasis on **deployment readiness** rather than aggressive optimization.

---

## 🔄 Model Lifecycle & Maintenance
This project demonstrates a complete **model lifecycle pipeline**:

- New data is added to the database
- Retraining script is executed
- A new model version is created
- Model registry is updated
- Dashboard automatically loads the latest model

Example versioning:
model_v1.pkl → model_v2.pkl → model_v3.pkl


The model registry (`model_registry.json`) stores:
- Model version
- Training timestamp
- Number of records used for training

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

## 🧾 Database and Model Files (Important Note)
The SQLite database and trained model files are **intentionally included** in this repository to allow **immediate reproducibility and evaluation**.

All artifacts are fully reproducible using the provided scripts:
- `scripts/generate_mess_data.py`
- `scripts/train_model.py`
- `scripts/retrain_model.py`

In a production environment, such generated files would typically be excluded from version control.

---


