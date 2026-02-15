import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import plotly.express as px
import json

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Mess Demand & Food Waste Prediction",
    layout="wide"
)

# --------------------------------------------------
# Load Model Registry
# --------------------------------------------------
@st.cache_data
def load_model_registry():
    with open("models/model_registry.json", "r") as f:
        return json.load(f)

registry = load_model_registry()
current_model = registry["models"][-1]


# --------------------------------------------------
# Paths
# --------------------------------------------------
DB_PATH = os.path.join("database", "mess.db")
import json

with open("models/model_registry.json") as f:
    registry = json.load(f)

latest_model_file = registry["models"][-1]["model_file"]
MODEL_PATH = os.path.join("models", latest_model_file)

ENCODER_PATH = os.path.join("models", "encoders.pkl")

# --------------------------------------------------
# Loaders
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
# Menu Definitions
# --------------------------------------------------
BREAKFAST_HIGH = ["Paratha", "Idli Vada", "Misal Pav", "Dhokla"]
BREAKFAST_MID = ["Uttapam", "Sabudana Vada", "Poha", "Vada Pav"]
BREAKFAST_LOW = [
    "Bombay Sandwich", "Coleslaw Sandwich",
    "Besan Chilla", "Vermicelli Upma", "Rava Upma"
]

LUNCH_HIGH = ["Kadhi Pakoda", "Bhindi Kurkure", "Chole", "Baingan Bharta", "Paneer", "Soya 65"]
LUNCH_MID = [
    "Aloo Jeera", "Aloo Bhindi", "Aloo Capsicum",
    "Sev Tamatar", "Methi", "Baingan Masala",
    "Chana Masala", "Mix Veg", "Soya Masala", "Matki"
]
LUNCH_LOW = ["Tendli", "Cabbage", "Lauki", "Karela", "Turai", "Chawali", "Rajma", "Capsicum"]

DINNER_HIGH = ["Chicken", "Paneer", "Biryani", "Pav Bhaji", "Egg", "Chole"]
DINNER_MID = ["Mexican", "Mix Veg", "Soya Masala"]
DINNER_LOW = ["Chinese", "Tendli", "Cabbage", "Lauki", "Karela", "Turai", "Chawali", "Rajma", "Capsicum"]

def get_menu_tier(meal, item):
    if meal == "Breakfast":
        if item in BREAKFAST_HIGH: return "High"
        if item in BREAKFAST_MID: return "Medium"
        return "Low"
    if meal == "Lunch":
        if item in LUNCH_HIGH: return "High"
        if item in LUNCH_MID: return "Medium"
        return "Low"
    if meal == "Dinner":
        if item in DINNER_HIGH: return "High"
        if item in DINNER_MID: return "Medium"
        return "Low"
                                                        

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🍽️ Mess Demand & Food Waste Prediction System")

# --------------------------------------------------
# Tabs (ORDER FIXED)
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Demand Prediction", "📊 EDA & Insights", "📁 Data Overview", "ℹ️ Model Info"]
)

# ==================================================
# TAB 1: DEMAND PREDICTION
# ==================================================
with tab1:
    st.subheader("Predict Mess Demand")

    col1, col2 = st.columns(2)

    with col1:
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner"])

        if meal_type == "Breakfast":
            primary_item = st.selectbox("Primary Dish", BREAKFAST_HIGH + BREAKFAST_MID + BREAKFAST_LOW)
        elif meal_type == "Lunch":
            primary_item = st.selectbox("Primary Dish", LUNCH_HIGH + LUNCH_MID + LUNCH_LOW)
        else:
            primary_item = st.selectbox("Primary Dish", DINNER_HIGH + DINNER_MID + DINNER_LOW)

        day = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    with col2:
        occupancy = st.slider("Hostel Occupancy (%)", 30, 100, 80)
        semester_phase = st.selectbox("Semester Phase", ["Regular", "Exams", "Holidays"])

        dessert = st.checkbox("Dessert Served (Dinner only)")
        fruit = st.checkbox("Fruit Served (Breakfast only)")
        drink = st.checkbox("Drink Served (Lunch only)")

    if st.button("Predict Demand"):
        menu_tier = get_menu_tier(meal_type, primary_item)

        input_df = pd.DataFrame([{
            "day_of_week": day,
            "meal_type": meal_type,
            "menu_demand_tier": menu_tier,
            "has_paneer": int("Paneer" in primary_item),
            "has_chicken": int("Chicken" in primary_item),
            "has_egg": int("Egg" in primary_item),
            "has_dessert": int(dessert),
            "has_special_cuisine": int(primary_item in ["Chinese", "Mexican", "Pav Bhaji", "Biryani"]),
            "has_drink": int(drink),
            "has_fruit": int(fruit),
            "hostel_occupancy_pct": occupancy,
            "semester_phase": semester_phase,
            "is_weekend": int(day in ["Saturday", "Sunday"]),
            "previous_meal_demand": "Medium"
        }])

        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]

        st.success(
            f"📌 **Predicted Demand Level:** {prediction}\n\n"
            f"🍽️ Dish: {primary_item}\n"
            f"📊 Menu Tier: {menu_tier}"
        )

# ==================================================
# TAB 2: EDA & INSIGHTS (STRUCTURED & PROFESSIONAL)
# ==================================================
with tab2:
    st.subheader("Exploratory Data Analysis & Insights")

    # ---- Row 1 ----
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            df,
            x="demand_level",
            color="demand_level",
            title="Overall Demand Distribution",
            height=320
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df,
            x="meal_type",
            color="demand_level",
            barmode="group",
            title="Demand by Meal Type",
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Row 2 ----
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            df,
            x="day_of_week",
            color="demand_level",
            title="Demand by Day of Week",
            height=320
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            df,
            x="demand_level",
            y="hostel_occupancy_pct",
            color="demand_level",
            title="Hostel Occupancy vs Demand",
            height=320
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ---- Row 3 ----
    col5, col6 = st.columns(2)

    with col5:
        fig5 = px.histogram(
            df,
            x="has_drink",
            color="demand_level",
            title="Impact of Drinks on Demand",
            height=320
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        fig6 = px.histogram(
            df,
            x="semester_phase",
            color="demand_level",
            title="Demand Across Semester Phases",
            height=320
        )
        st.plotly_chart(fig6, use_container_width=True)

# ==================================================
# TAB 3: DATA OVERVIEW
# ==================================================
with tab3:
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Target Variable", "demand_level")

    st.markdown("### Sample Records")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown(
        """
        **Dataset Description**  
        This dataset is synthetically generated using rule-based logic to simulate
        real university mess operations. It captures menu composition, hostel
        occupancy, academic calendar effects, and meal-specific demand behavior.
        """
    )

# ==================================================
# TAB 4: MODEL INFO
# ==================================================
with tab4:
    st.subheader("Model Information & Lifecycle")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Version", f"v{current_model['version']}")
    col2.metric("Model File", current_model["model_file"])
    col3.metric("Training Records", current_model["trained_on_records"])

    st.markdown("### Training Details")
    st.write(f"**Last Trained On:** {current_model['training_date']}")
    st.write(f"**Algorithm Used:** {type(model).__name__}")
    st.write(f"**Number of Features:** {len(model.feature_names_in_)}")

    st.info(
        "This model is part of a versioned lifecycle. "
        "Each retraining cycle creates a new model version "
        "while preserving historical models for traceability."
    )

