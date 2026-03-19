# Mess Demand & Food Waste Management System

> A production-ready machine learning pipeline for predicting university mess demand and estimating food waste — built for Hackathon-3: ML Pipeline Development & Model Maintenance.

---

## Overview

University mess operations regularly face two opposing problems: over-preparation leads to food waste, while under-preparation causes shortages and long queues. This system addresses both by predicting demand levels in advance and estimating food waste per meal using a fully automated ML pipeline.

The system classifies demand as **Low**, **Medium**, or **High** based on operational and contextual inputs such as meal type, menu composition, hostel occupancy, and academic calendar phase. It also calculates estimated food waste and cost impact per meal to support better planning decisions.

---

## Features

- **Demand prediction** — Multi-class classification (Low / Medium / High) using 6 ML models
- **Food waste estimation** — Rule-based formula combining demand predictions with occupancy and meal factors
- **Automated retraining** — Model retrains automatically when 100+ new records are added to the database
- **Data drift detection** — PSI, Chi-Square, and Mean Shift checks on key features
- **Model registry** — Full version tracking with metrics, timestamps, and record counts for every trained model
- **Dual dashboards** — React + FastAPI (primary) and Streamlit (secondary)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Languages** | Python, TypeScript, SQL |
| **API** | FastAPI, Uvicorn, Pydantic |
| **ML** | Scikit-learn, XGBoost, Pandas, NumPy |
| **Database** | SQLite (`database/mess.db`) |
| **Model storage** | Joblib (`.pkl`), JSON (model registry) |
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, Recharts |
| **Alt dashboard** | Streamlit, Matplotlib, Seaborn, Plotly |
| **Dev tools** | Git, Node.js, npm, ESLint, Vitest, VS Code |

---

## ML Models

Six models are trained and evaluated on every run. The best-performing model is selected automatically based on **Macro F1-Score**.

| # | Model | Library |
|---|---|---|
| 1 | Logistic Regression | scikit-learn |
| 2 | Decision Tree | scikit-learn |
| 3 | Random Forest | scikit-learn |
| 4 | Gradient Boosting | scikit-learn |
| 5 | XGBoost Classifier | xgboost |
| 6 | Support Vector Machine (SVM) | scikit-learn |

**Evaluation metrics:** Accuracy · Precision (Macro) · Recall (Macro) · F1-Score (Macro)

---

## Food Waste Estimation

Since real consumption data requires IoT sensors, waste is estimated using domain-informed factors applied to the predicted demand level and current occupancy.

```
Recommended Portions  = Occupancy × 3 × Meal Factor
Expected Consumption  = Recommended × Demand Factor
Estimated Waste       = Recommended − Expected Consumption
Cost of Waste         = Waste Portions × ₹45
```

| | Breakfast | Lunch | Dinner |
|---|---|---|---|
| **Meal Factor** | 0.6 | 0.9 | 0.75 |

| | High | Medium | Low |
|---|---|---|---|
| **Demand Factor** | 0.92 | 0.75 | 0.55 |

---

## System Architecture

```
Data Generator (Python)
        ↓
SQLite Database
        ↓                    ←──────────────────────────┐
ML Model Training                              Data Drift Detection
        ↓                                    (PSI · Chi-Square · Mean Shift)
Model Registry (JSON)
        ↓
FastAPI Backend
        ↓
React Dashboard (Frontend)
        ↓
User Interaction & Predictions
```

---

## Pipeline Scripts

| Script | Description |
|---|---|
| `generate_mess_data.py` | Generates synthetic mess records across Regular, Exam, and Holiday phases |
| `train_model.py` | Trains all 6 models, selects best by Macro F1, saves model and encoders |
| `retrain_model.py` | Checks 100-record threshold and triggers retraining if met |
| `detect_drift.py` | Runs PSI, Chi-Square, and Mean Shift checks; outputs a formatted report |
| `verify_data.py` | Validates data integrity in the database |
| `create_database.py` | Initialises the SQLite database and schema |

---

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Demand prediction + food waste estimation |
| `GET` | `/eda` | Chart data for the EDA tab |
| `GET` | `/waste-stats` | Waste metrics, charts, and reduction insights |
| `GET` | `/data` | Dataset preview and summary statistics |
| `GET` | `/model-info` | Metrics, feature importance, confusion matrix |
| `GET` | `/model-history` | All registry versions for the model evolution chart |
| `GET` | `/pipeline-status` | Record counts and retrain readiness |
| `POST` | `/add-data` | Generate and insert new synthetic records |
| `POST` | `/retrain` | Threshold check, retrain model, update registry |

---

## Data Drift Detection

Run drift detection at any time:

```bash
python detect_drift.py
```

The script splits records into reference (first 70%) and current (last 30%) and checks for distribution shifts using three methods:

| Method | Feature | Threshold |
|---|---|---|
| **PSI** (Population Stability Index) | `hostel_occupancy_pct` | PSI > 0.20 → retrain recommended |
| **Chi-Square Test** | `meal_type`, `semester_phase`, `day_of_week` | p-value < 0.05 → drift detected |
| **Mean Shift Analysis** | `has_dessert`, `is_weekend` | Shift > 5% → flagged |

---

## Automated Retraining

Whenever new data is added to the database, the system checks if the retraining threshold has been met:

1. New records added via `/add-data` or manually
2. System checks if **100+ new records** exist since last training
3. If threshold is met — model retrains on the full updated dataset
4. Metrics recalculated on a held-out test split
5. New model version saved to `models/`
6. `model_registry.json` updated automatically
7. Dashboard loads the latest model version

To retrain manually without the threshold check:

```bash
python retrain_model.py
```

---

## Model Registry

Every trained model version is tracked in `models/model_registry.json` with:

- Model name and file path
- Training date and timestamp
- Number of records used
- Accuracy, Precision, Recall, F1-Score (Macro)
- Training time in seconds

| Version | Date | Records | Notes |
|---|---|---|---|
| v1 | 2026-02-20 | 600 | Initial training |
| v2 | 2026-02-20 | 600 | First retraining |
| v3 | 2026-02-28 | 1,020 | second retraining |
| v4 | 2026-03-08 | 1,020 | Third retraining |
| v5 | 2026-03-18 | 1300 | Fourth retraining |

---

## Dataset

The dataset is synthetically generated to simulate real university mess operations and stored in SQLite (`database/mess.db`), table: `mess_records`.

| Feature | Type | Description |
|---|---|---|
| `meal_type` | Categorical | Breakfast / Lunch / Dinner |
| `day_of_week` | Categorical | Monday – Sunday |
| `primary_item` | Categorical | Main dish served |
| `menu_demand_tier` | Categorical | High / Medium / Low |
| `has_paneer` | Binary | Paneer served |
| `has_chicken` | Binary | Chicken served |
| `has_egg` | Binary | Egg served |
| `has_dessert` | Binary | Dessert served |
| `has_special_cuisine` | Binary | Special cuisine served |
| `has_drink` | Binary | Drink served |
| `has_fruit` | Binary | Fruit served |
| `hostel_occupancy_pct` | Numerical | % of hostel occupied |
| `semester_phase` | Categorical | Regular / Exams / Holidays |
| `is_weekend` | Binary | Weekend flag |
| `previous_meal_demand` | Categorical | Demand level of previous meal |
| `demand_level` | Categorical | **Target** — Low / Medium / High |

---

## Dashboards

### React Dashboard (Primary) — `http://localhost:8080`

Built with React + TypeScript + Recharts, connected to the FastAPI backend.

- **Demand Prediction** — Select inputs, get real-time prediction + waste estimate
- **Waste Management** — Total waste, trends by meal type, weekly charts, cost impact
- **EDA & Insights** — Demand distribution, occupancy vs demand, semester phase impact
- **Data Overview** — Live dataset preview and summary statistics
- **Model Info & Pipeline** — Metrics, feature importance, confusion matrix, model evolution chart, pipeline controls (Add Data / Retrain)

### Streamlit Dashboard (Secondary)

```bash
streamlit run app.py
```

Covers demand prediction, EDA, data overview, and model info.

---

## Project Structure

```
├── app/                  # Streamlit dashboard
├── Backend/              # FastAPI app and API logic
├── database/             # SQLite database (mess.db)
├── Frontend/             # React + TypeScript dashboard
├── models/               # Trained model files and registry
├── scripts/              # ML pipeline scripts
├── src/                  # Source utilities
└── README.md
```

---

## Author

Developed by **Utkarsh Somvanshi** as part of **Hackathon-3 — ML Pipeline Development & Model Maintenance**.