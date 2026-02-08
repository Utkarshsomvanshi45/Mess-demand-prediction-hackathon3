import sqlite3
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Paths
# --------------------------------------------------
DB_PATH = "database/mess.db"
MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")

# --------------------------------------------------
# Load data
# --------------------------------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM mess_records", conn)
conn.close()

# --------------------------------------------------
# Preprocessing (same logic as initial training)
# --------------------------------------------------
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

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------------------------------
# Load registry
# --------------------------------------------------
with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

new_version = registry["current_version"] + 1
model_filename = f"model_v{new_version}.pkl"

# --------------------------------------------------
# Save model & encoders
# --------------------------------------------------
joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))

# --------------------------------------------------
# Update registry
# --------------------------------------------------
registry["current_version"] = new_version
registry["models"].append({
    "version": new_version,
    "model_file": model_filename,
    "trained_on_records": df.shape[0],
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=4)

print(f" Model retrained successfully: {model_filename}")
