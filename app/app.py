import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Mess Demand Prediction",
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DB_PATH = os.path.join("database", "mess.db")
MODEL_PATH = os.path.join("models", "model_v3.pkl")
ENCODER_PATH = os.path.join("models", "encoders.pkl")


# --------------------------------------------------
# LOADERS
# --------------------------------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM mess_records", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

df = load_data()
model, encoders = load_model()

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

        # Encode categorical
        for col in encoders:
            if col in input_df.columns:
                input_df[col] = encoders[col].transform(input_df[col])

        pred = model.predict(input_df)[0]

        # If encoded numeric → decode
        if hasattr(model, "classes_"):
            pred = model.classes_[pred]

        st.success(
            f"📌 **Predicted Demand Level: {pred}**\n\n"
            f"🍽️ Dish: {primary_item}"
        )

# ==================================================
# TAB 2 — EDA (COMPACT & PROFESSIONAL)
# ==================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis & Key Insights")

    # KPI ROW
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Records", df.shape[0])
    k2.metric("Avg Occupancy", f"{int(df['hostel_occupancy_pct'].mean())}%")
    k3.metric("High Demand Days", df[df["demand_level"]=="High"].shape[0])
    k4.metric("Weekend Records", df[df["is_weekend"]==1].shape[0])

    st.divider()

    # ROW 1
    c1,c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(df, x="demand_level", color="demand_level",
                            title="Overall Demand Distribution", height=260)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(df, x="meal_type", color="demand_level",
                            barmode="group", title="Demand by Meal Type", height=260)
        st.plotly_chart(fig2, use_container_width=True)

    # ROW 2
    c3,c4 = st.columns(2)

    with c3:
        fig3 = px.box(df, x="demand_level", y="hostel_occupancy_pct",
                      color="demand_level", title="Occupancy vs Demand", height=260)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.histogram(df, x="semester_phase", color="demand_level",
                            title="Demand Across Semester Phases", height=260)
        st.plotly_chart(fig4, use_container_width=True)

    # ROW 3
    c5,c6 = st.columns(2)

    with c5:
        fig5 = px.histogram(df, x="has_drink", color="demand_level",
                            title="Impact of Drinks on Demand", height=260)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        top_items = (
            df.groupby("primary_item")["demand_level"]
            .count().sort_values(ascending=False)
            .head(10).reset_index()
        )
        fig6 = px.bar(top_items, x="demand_level", y="primary_item",
                      orientation="h", title="Top 10 Most Served Dishes", height=260)
        st.plotly_chart(fig6, use_container_width=True)

# ==================================================
# TAB 3 — DATA OVERVIEW
# ==================================================
with tab3:
    st.subheader("🗂️ Dataset Overview")

    st.dataframe(df.head(20), use_container_width=True)

    st.write("### Dataset Summary")
    st.write(df.describe())

# ==================================================
# TAB 4 — MODEL INFO
# ==================================================
with tab4:
    st.subheader("ℹ️ Model Information")

    st.metric("Model Used", type(model).__name__)
    st.metric("Features Used", len(model.feature_names_in_))
    st.metric("Total Records Trained On", df.shape[0])

    st.info(
        "Model predicts mess demand levels using meal, occupancy, "
        "menu features and semester patterns. Retraining improves "
        "accuracy as new data is generated."
    )