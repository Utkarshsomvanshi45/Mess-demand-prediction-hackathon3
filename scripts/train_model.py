import sqlite3
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DB_PATH = os.path.join("database", "mess.db")
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM mess_records", conn)
    conn.close()
    return df

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
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["demand_level"])
    y = df["demand_level"]

    return X, y, encoders


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report

if __name__ == "__main__":
    df = load_data()
    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier()
    }

    best_model = None
    best_acc = 0

    for name, model in models.items():
        acc, report = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        print(f"\n Model: {name}")
        print(f"Accuracy: {acc:.2f}")
        print(report)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model_v1.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

    print("\n Best model saved as model_v1.pkl")
