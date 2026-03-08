import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.metrics import confusion_matrix
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Mess Demand Prediction", layout="wide")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DB_PATH = os.path.join("database", "mess.db")
MODELS_DIR = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoders.pkl")

# --------------------------------------------------
# LOAD LATEST MODEL FROM REGISTRY
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    latest_version = registry["current_version"]

    model_file = None
    for m in registry["models"]:
        if m["version"] == latest_version:
            model_file = m["model_file"]
            break

    model_path = os.path.join(MODELS_DIR, model_file)

    model = joblib.load(model_path)
    encoders = joblib.load(ENCODER_PATH)

    return model, encoders, latest_version

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM mess_records", conn)
    conn.close()
    return df

df = load_data()
model, encoders, version = load_model()

# --------------------------------------------------
# MENU LISTS
# --------------------------------------------------
BREAKFAST_ITEMS = [
    "Paratha","Idli Vada","Misal Pav","Dhokla",
    "Uttapam","Sabudana Vada","Poha","Vada Pav",
    "Bombay Sandwich","Coleslaw Sandwich",
    "Besan Chilla","Vermicelli Upma","Rava Upma"
]

LUNCH_ITEMS = [
    "Kadhi Pakoda","Bhindi Kurkure","Chole","Baingan Bharta",
    "Paneer","Soya 65","Aloo Jeera","Aloo Bhindi","Aloo Capsicum",
    "Sev Tamatar","Methi","Baingan Masala","Chana Masala",
    "Mix Veg","Soya Masala","Matki","Tendli","Cabbage",
    "Lauki","Karela","Turai","Chawali","Rajma","Capsicum"
]

DINNER_ITEMS = [
    "Chicken","Paneer","Biryani","Pav Bhaji","Egg","Chole",
    "Mexican","Mix Veg","Soya Masala","Chinese","Tendli",
    "Cabbage","Lauki","Karela","Turai","Chawali","Rajma","Capsicum"
]

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🍽️ Mess Demand & Food Waste Prediction System")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Demand Prediction", "📊 EDA & Insights", "🗂️ Data Overview", "ℹ️ Model Info"]
)

# ==================================================
# TAB 1 — PREDICTION
# ==================================================
with tab1:
    st.subheader("Predict Mess Demand")

    c1, c2 = st.columns(2)

    with c1:
        meal_type = st.selectbox("Meal Type", ["Breakfast","Lunch","Dinner"])

        if meal_type == "Breakfast":
            primary_item = st.selectbox("Primary Dish", BREAKFAST_ITEMS)
        elif meal_type == "Lunch":
            primary_item = st.selectbox("Primary Dish", LUNCH_ITEMS)
        else:
            primary_item = st.selectbox("Primary Dish", DINNER_ITEMS)

        day = st.selectbox(
            "Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )

    with c2:
        occupancy = st.slider("Hostel Occupancy (%)", 30, 100, 80)
        semester_phase = st.selectbox("Semester Phase", ["Regular","Exams","Holidays"])
        dessert = st.checkbox("Dessert Served")
        fruit = st.checkbox("Fruit Served")
        drink = st.checkbox("Drink Served")

    if st.button("Predict Demand"):
        input_df = pd.DataFrame([{
            "day_of_week": day,
            "meal_type": meal_type,
            "menu_demand_tier": "Medium",
            "has_paneer": int("Paneer" in primary_item),
            "has_chicken": int("Chicken" in primary_item),
            "has_egg": int("Egg" in primary_item),
            "has_dessert": int(dessert),
            "has_special_cuisine": int(primary_item in ["Chinese","Mexican","Pav Bhaji","Biryani"]),
            "has_drink": int(drink),
            "has_fruit": int(fruit),
            "hostel_occupancy_pct": occupancy,
            "semester_phase": semester_phase,
            "is_weekend": int(day in ["Saturday","Sunday"]),
            "previous_meal_demand": "Medium"
        }])

        for col in encoders:
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])

        pred = model.predict(input_df)[0]

        if hasattr(model, "classes_"):
            pred = model.classes_[pred]

        st.success(f"📌 **Predicted Demand Level: {pred}**")

# ==================================================
# TAB 2 — EDA
# ==================================================
with tab2:
    st.subheader("Exploratory Data Analysis")

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Records", df.shape[0])
    k2.metric("Avg Occupancy", f"{int(df['hostel_occupancy_pct'].mean())}%")
    k3.metric("High Demand Count", df[df["demand_level"]=="High"].shape[0])
    k4.metric("Weekend Records", df[df["is_weekend"]==1].shape[0])

    st.divider()

    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="demand_level", color="demand_level",
                                     title="Demand Distribution", height=260),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df, x="meal_type", color="demand_level",
                                     barmode="group", title="Meal Type vs Demand", height=260),
                        use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.box(df, x="demand_level", y="hostel_occupancy_pct",
                               color="demand_level", title="Occupancy vs Demand", height=260),
                        use_container_width=True)
    with c4:
        st.plotly_chart(px.histogram(df, x="semester_phase", color="demand_level",
                                     title="Semester Phase Impact", height=260),
                        use_container_width=True)

# ==================================================
# TAB 3 — DATA OVERVIEW
# ==================================================
with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)
    st.write("### Summary Statistics")
    st.write(df.describe())

# ==================================================
# TAB 4 — MODEL INFO
# ==================================================
with tab4:
    st.subheader("Model Information")

    st.metric("Model Used", type(model).__name__)
    st.metric("Model Version", f"v{version}")
    st.metric("Training Records", df.shape[0])

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        features = model.feature_names_in_
        fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)

        fig_fi = px.bar(fi_df.head(15), x="Importance", y="Feature",
                        orientation="h", height=400)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    X = df.drop(columns=["id","meal_date","primary_item","demand_level"])
    y_true = df["demand_level"]

    for col in encoders:
        if col in X.columns:
            X[col] = encoders[col].transform(X[col])

    y_pred = model.predict(X)

    if hasattr(model, "classes_"):
        y_pred = model.classes_[y_pred]

    cm = confusion_matrix(y_true, y_pred, labels=["Low","Medium","High"])

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Low","Medium","High"],
        y=["Low","Medium","High"],
        colorscale="Blues"
    ))
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)