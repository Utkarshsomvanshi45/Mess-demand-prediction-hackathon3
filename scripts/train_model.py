import sqlite3
import os
import pandas as pd
import joblib
import json
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------
DB_PATH = os.path.join("database", "mess.db")
MODEL_DIR = "models"
REGISTRY_PATH = os.path.join(MODEL_DIR, "model_registry.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM mess_records", conn)
    conn.close()
    return df


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess(df):
    df = df.drop(columns=["id", "meal_date", "primary_item"])

    cat_cols = [
        "day_of_week",
        "meal_type",
        "menu_demand_tier",
        "semester_phase",
        "previous_meal_demand"
    ]

    encoders = {}

    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    df["demand_level"] = target_encoder.fit_transform(df["demand_level"])
    encoders["demand_level"] = target_encoder

    X = df.drop(columns=["demand_level"])
    y = df["demand_level"]

    return X, y, encoders


# --------------------------------------------------
# Train & Evaluate
# --------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 3)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    return acc, precision, recall, f1, train_time


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    df = load_data()
    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM (RBF)": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    }

    results = []
    best_model = None
    best_f1 = 0
    best_model_name = None

    print("\n========== MODEL TRAINING ==========")

    for name, model in models.items():
        print(f"\nTraining: {name}")

        acc, precision, recall, f1, train_time = train_and_evaluate(
            model, X_train, X_test, y_train, y_test
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        print(f"F1 Score (Macro): {f1:.4f}")
        print(f"Train Time: {train_time}s")

        results.append([name, acc, precision, recall, f1, train_time])

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision(Macro)", "Recall(Macro)", "F1(Macro)", "Train Time"]
    ).sort_values(by="F1(Macro)", ascending=False)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df)

    # --------------------------------------------------
    # Save Best Model as Version 1
    # --------------------------------------------------
    version = 1
    model_filename = f"model_v{version}.pkl"

    joblib.dump(best_model, os.path.join(MODEL_DIR, model_filename))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    print(f"\nBest Model Selected: {best_model_name}")
    print(f"Saved as: {model_filename}")

    # --------------------------------------------------
    # Initialize Registry
    # --------------------------------------------------
    registry = {
        "current_version": version,
        "models": [
            {
                "version": version,
                "model_name": best_model_name,
                "model_file": model_filename,
                "trained_on_records": df.shape[0],
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "f1_macro": round(float(best_f1), 4)
            }
        ]
    }

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print("\nRegistry initialized successfully.")