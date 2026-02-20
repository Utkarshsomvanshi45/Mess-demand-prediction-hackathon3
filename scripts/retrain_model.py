import sqlite3
import pandas as pd
import joblib
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------
DB_PATH = "database/mess.db"
MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")


# --------------------------------------------------
# Load Registry
# --------------------------------------------------
with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

current_version = registry["current_version"]
previous_model_info = registry["models"][-1]
model_type = previous_model_info["model_name"]

print(f"Previous Model Type: {model_type}")


# --------------------------------------------------
# Initialize Same Model Type
# --------------------------------------------------
def initialize_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier()
    elif model_name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier()
    elif model_name == "SVM (RBF)":
        return SVC()
    elif model_name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    else:
        raise ValueError("Unsupported model type in registry.")


model = initialize_model(model_type)


# --------------------------------------------------
# Load Data
# --------------------------------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM mess_records", conn)
conn.close()

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

# Encode target
target_encoder = LabelEncoder()
df["demand_level"] = target_encoder.fit_transform(df["demand_level"])
encoders["demand_level"] = target_encoder

X = df.drop(columns=["demand_level"])
y = df["demand_level"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
model.fit(X_train, y_train)

preds = model.predict(X_test)
f1_macro = f1_score(y_test, preds, average="macro")

print(f"New F1 Macro: {f1_macro:.4f}")

# --------------------------------------------------
# Save New Version
# --------------------------------------------------
new_version = current_version + 1
model_filename = f"model_v{new_version}.pkl"

joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))

# --------------------------------------------------
# Update Registry
# --------------------------------------------------
registry["current_version"] = new_version

registry["models"].append({
    "version": new_version,
    "model_name": model_type,
    "model_file": model_filename,
    "trained_on_records": df.shape[0],
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "f1_macro": round(float(f1_macro), 4)
})

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=4)

print(f"Model retrained successfully: {model_filename}")