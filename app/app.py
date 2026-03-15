import streamlit as st
import sqlite3
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Mess Demand Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS — dark, modern, production-grade
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0f1117; color: #e8eaf0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1400px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 60%, #0d1f0f 100%);
    border: 1px solid #2a2f3e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 3rem; }
.hero-title { font-size: 1.8rem; font-weight: 700; color: #f0f2f8; margin: 0; letter-spacing: -0.02em; }
.hero-subtitle { font-size: 0.9rem; color: #6b7280; margin: 0.25rem 0 0 0; }
.hero-badge {
    margin-left: auto;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    white-space: nowrap;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2f3e;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b7280;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #22c55e !important;
    color: #0f1117 !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Cards */
.card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #4b5563; margin-bottom: 0.5rem; }
.card-value { font-size: 2rem; font-weight: 700; color: #f0f2f8; font-family: 'DM Mono', monospace; line-height: 1; }
.card-sub { font-size: 0.78rem; color: #6b7280; margin-top: 0.3rem; }

/* Prediction box */
.pred-box { border-radius: 14px; padding: 2rem; text-align: center; margin-top: 1rem; border: 2px solid; }
.pred-box.high  { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.4); }
.pred-box.medium{ background: rgba(234,179,8,0.08);  border-color: rgba(234,179,8,0.4); }
.pred-box.low   { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.4); }
.pred-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; color: #6b7280; margin-bottom: 0.5rem; }
.pred-value { font-size: 3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.pred-value.high   { color: #ef4444; }
.pred-value.medium { color: #eab308; }
.pred-value.low    { color: #22c55e; }
.pred-desc { font-size: 0.85rem; color: #9ca3af; margin-top: 0.75rem; }

/* Section heading */
.section-heading {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e8eaf0;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #2a2f3e;
    margin-left: 0.5rem;
}

/* Tier badge */
.tier-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; font-family: 'DM Mono', monospace; }
.tier-High   { background: rgba(239,68,68,0.15);  color: #ef4444; }
.tier-Medium { background: rgba(234,179,8,0.15);  color: #eab308; }
.tier-Low    { background: rgba(34,197,94,0.15);  color: #22c55e; }

/* Button */
.stButton > button {
    background: #22c55e !important;
    color: #0f1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #16a34a !important;
    box-shadow: 0 8px 24px rgba(34,197,94,0.25) !important;
}

/* Model stat */
.model-stat { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 10px; padding: 1rem 1.25rem; text-align: center; }
.model-stat-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #4b5563; font-weight: 600; }
.model-stat-value { font-size: 1.4rem; font-weight: 700; color: #22c55e; font-family: 'DM Mono', monospace; margin-top: 0.25rem; }

hr { border-color: #2a2f3e !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
DB_PATH      = os.path.join("database", "mess.db")
MODELS_DIR   = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
ENCODER_PATH  = os.path.join(MODELS_DIR, "encoders.pkl")

# --------------------------------------------------
# LOAD MODEL & DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    latest = registry["current_version"]
    info = next(m for m in registry["models"] if m["version"] == latest)
    model    = joblib.load(os.path.join(MODELS_DIR, info["model_file"]))
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders, latest, info, registry

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM mess_records", conn)
    conn.close()
    return df

df = load_data()
model, encoders, version, model_info, registry = load_model()

# --------------------------------------------------
# MENUS
# --------------------------------------------------
BREAKFAST_ITEMS = ["Paratha","Idli Vada","Misal Pav","Dhokla","Uttapam",
                   "Sabudana Vada","Poha","Vada Pav","Bombay Sandwich",
                   "Coleslaw Sandwich","Besan Chilla","Vermicelli Upma","Rava Upma"]
LUNCH_ITEMS     = ["Kadhi Pakoda","Bhindi Kurkure","Chole","Baingan Bharta","Paneer",
                   "Soya 65","Aloo Jeera","Aloo Bhindi","Aloo Capsicum","Sev Tamatar",
                   "Methi","Baingan Masala","Chana Masala","Mix Veg","Soya Masala",
                   "Matki","Tendli","Cabbage","Lauki","Karela","Turai","Chawali","Rajma","Capsicum"]
DINNER_ITEMS    = ["Chicken","Paneer","Biryani","Pav Bhaji","Egg","Chole","Mexican",
                   "Mix Veg","Soya Masala","Chinese","Tendli","Cabbage","Lauki",
                   "Karela","Turai","Chawali","Rajma","Capsicum"]

ITEM_TIER_MAP = {
    "Paratha":"High","Idli Vada":"High","Misal Pav":"High","Dhokla":"High",
    "Uttapam":"Medium","Sabudana Vada":"Medium","Poha":"Medium","Vada Pav":"Medium",
    "Bombay Sandwich":"Low","Coleslaw Sandwich":"Low","Besan Chilla":"Low",
    "Vermicelli Upma":"Low","Rava Upma":"Low",
    "Kadhi Pakoda":"High","Bhindi Kurkure":"High","Chole":"High","Baingan Bharta":"High",
    "Paneer":"High","Soya 65":"High",
    "Aloo Jeera":"Medium","Aloo Bhindi":"Medium","Aloo Capsicum":"Medium",
    "Sev Tamatar":"Medium","Methi":"Medium","Baingan Masala":"Medium",
    "Chana Masala":"Medium","Mix Veg":"Medium","Soya Masala":"Medium","Matki":"Medium",
    "Tendli":"Low","Cabbage":"Low","Lauki":"Low","Karela":"Low",
    "Turai":"Low","Chawali":"Low","Rajma":"Low","Capsicum":"Low",
    "Chicken":"High","Biryani":"High","Pav Bhaji":"High","Egg":"High",
    "Mexican":"Medium","Chinese":"Low",
    "Paneer & Chicken":"High","Paneer & Egg":"High",
}

DEMAND_DESCRIPTIONS = {
    "High":   "Expect heavy footfall. Prepare maximum quantity. Consider extra staff.",
    "Medium": "Moderate turnout expected. Prepare standard quantity.",
    "Low":    "Light attendance expected. Reduce preparation to minimise waste.",
}

CHART_COLORS = {"High":"#ef4444","Medium":"#eab308","Low":"#22c55e"}
CHART_LAYOUT = dict(
    paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
    font_color="#9ca3af", title_font_color="#e8eaf0",
    legend=dict(font=dict(color="#9ca3af")),
    xaxis=dict(gridcolor="#2a2f3e"),
    yaxis=dict(gridcolor="#2a2f3e"),
    margin=dict(t=50,b=10,l=10,r=10)
)

# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(f"""
<div class="hero">
  <div class="hero-icon">🍽️</div>
  <div>
    <p class="hero-title">Mess Demand &amp; Food Waste Prediction</p>
    <p class="hero-subtitle">ML-powered demand forecasting for university mess operations</p>
  </div>
  <div class="hero-badge">v{version} · {model_info.get("model_name", type(model).__name__)}</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Demand Prediction",
    "📊  EDA & Insights",
    "🗂️  Data Overview",
    "🧠  Model Info"
])

# ══════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════
with tab1:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<p class="section-heading">📋 Meal Parameters</p>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1: meal_type = st.selectbox("Meal Type", ["Breakfast","Lunch","Dinner"])
        with r2: day = st.selectbox("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        items_map = {"Breakfast": BREAKFAST_ITEMS, "Lunch": LUNCH_ITEMS, "Dinner": DINNER_ITEMS}
        primary_item = st.selectbox("Primary Dish", items_map[meal_type])

        tier = ITEM_TIER_MAP.get(primary_item, "Medium")
        st.markdown(
            f'<p style="font-size:0.8rem;color:#6b7280;margin-top:-0.5rem;">'
            f'Menu tier: <span class="tier-badge tier-{tier}">{tier}</span></p>',
            unsafe_allow_html=True
        )

        r3, r4 = st.columns(2)
        with r3: semester_phase = st.selectbox("Semester Phase", ["Regular","Exams","Holidays"])
        with r4: occupancy = st.slider("Hostel Occupancy (%)", 30, 100, 80)

        st.markdown('<p class="section-heading">🍰 Menu Add-ons</p>', unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3)
        with ac1: dessert = st.checkbox("🍮 Dessert")
        with ac2: fruit   = st.checkbox("🍎 Fruit")
        with ac3: drink   = st.checkbox("🥤 Drink")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ Predict Demand", use_container_width=True)

    with right:
        st.markdown('<p class="section-heading">🎯 Prediction Result</p>', unsafe_allow_html=True)
        if predict_clicked:
            input_df = pd.DataFrame([{
                "day_of_week": day, "meal_type": meal_type,
                "menu_demand_tier": tier,
                "has_paneer": int("Paneer" in primary_item),
                "has_chicken": int("Chicken" in primary_item),
                "has_egg": int("Egg" in primary_item),
                "has_dessert": int(dessert),
                "has_special_cuisine": int(primary_item in ["Chinese","Mexican","Pav Bhaji","Biryani"]),
                "has_drink": int(drink), "has_fruit": int(fruit),
                "hostel_occupancy_pct": occupancy,
                "semester_phase": semester_phase,
                "is_weekend": int(day in ["Saturday","Sunday"]),
                "previous_meal_demand": "Medium"
            }])

            for col in encoders:
                if col in input_df.columns:
                    input_df[col] = encoders[col].transform(input_df[col])

            pred_encoded = model.predict(input_df)[0]
            # ✅ FIX: inverse_transform decodes numeric → "High"/"Medium"/"Low"
            pred_label = encoders["demand_level"].inverse_transform([pred_encoded])[0]
            css = pred_label.lower()

            st.markdown(f"""
            <div class="pred-box {css}">
                <p class="pred-label">Predicted Demand Level</p>
                <p class="pred-value {css}">{pred_label}</p>
                <p class="pred-desc">{DEMAND_DESCRIPTIONS[pred_label]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-heading" style="margin-top:1.5rem;">📌 Input Summary</p>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Parameter": ["Meal","Day","Dish","Tier","Occupancy","Phase","Dessert","Fruit","Drink"],
                "Value": [meal_type, day, primary_item, tier, f"{occupancy}%",
                          semester_phase, "Yes" if dessert else "No",
                          "Yes" if fruit else "No", "Yes" if drink else "No"]
            }), hide_index=True, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#1a1f2e;border:1px dashed #2a2f3e;border-radius:14px;
                        padding:3.5rem;text-align:center;margin-top:1rem;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">🎯</div>
                <p style="font-weight:600;color:#6b7280;margin:0;">Fill in parameters and click</p>
                <p style="font-size:0.8rem;color:#4b5563;margin:0.4rem 0 0 0;">
                    Predict Demand to see results
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════
with tab2:
    k1,k2,k3,k4 = st.columns(4)
    for col, label, value, sub in zip(
        [k1,k2,k3,k4],
        ["Total Records","Avg Occupancy","High Demand","Weekend Records"],
        [df.shape[0], f"{int(df['hostel_occupancy_pct'].mean())}%",
         df[df["demand_level"]=="High"].shape[0], df[df["is_weekend"]==1].shape[0]],
        ["meals in dataset","hostel fill rate","high demand meals","weekend entries"]
    ):
        with col:
            st.markdown(f"""<div class="card">
                <p class="card-title">{label}</p>
                <p class="card-value">{value}</p>
                <p class="card-sub">{sub}</p>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        counts = df["demand_level"].value_counts().reset_index()
        counts.columns = ["demand_level","count"]
        fig = px.pie(counts, values="count", names="demand_level",
                     title="Demand Distribution", color="demand_level",
                     color_discrete_map=CHART_COLORS, hole=0.5, height=320)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="meal_type", color="demand_level", barmode="group",
                           title="Demand by Meal Type", color_discrete_map=CHART_COLORS, height=320)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="demand_level", y="hostel_occupancy_pct", color="demand_level",
                     title="Hostel Occupancy vs Demand", color_discrete_map=CHART_COLORS, height=320)
        fig.update_layout(**{**CHART_LAYOUT, "showlegend": False})
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.histogram(df, x="semester_phase", color="demand_level",
                           title="Semester Phase vs Demand",
                           color_discrete_map=CHART_COLORS, height=320)
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-heading">📅 Demand by Day of Week</p>', unsafe_allow_html=True)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_df = df.groupby(["day_of_week","demand_level"]).size().reset_index(name="count")
    fig = px.bar(day_df, x="day_of_week", y="count", color="demand_level",
                 barmode="group", color_discrete_map=CHART_COLORS,
                 category_orders={"day_of_week": day_order}, height=300)
    fig.update_layout(**{**CHART_LAYOUT, "margin": dict(t=10,b=10,l=10,r=10)})
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════
# TAB 3 — DATA OVERVIEW
# ══════════════════════════════════════════════════
with tab3:
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""<div class="card">
            <p class="card-title">Total Records</p>
            <p class="card-value">{df.shape[0]}</p>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="card">
            <p class="card-title">Input Features</p>
            <p class="card-value">{df.shape[1]-2}</p>
            <p class="card-sub">columns</p>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""<div class="card">
            <p class="card-title">Target Column</p>
            <p class="card-value" style="font-size:1.2rem;color:#22c55e;">demand_level</p>
            <p class="card-sub">Low / Medium / High</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-heading">🗃️ Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(30), use_container_width=True, height=380)

    st.markdown('<p class="section-heading">📐 Summary Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("""
    <div style="background:#1a1f2e;border:1px solid #2a2f3e;border-radius:10px;
                padding:1rem 1.5rem;margin-top:1.5rem;">
        <span style="color:#22c55e;font-weight:700;font-size:0.85rem;">ℹ️ Data Source Note</span>
        <p style="color:#6b7280;font-size:0.82rem;margin:0.4rem 0 0 0;">
        Synthetically generated using rule-based logic to simulate real university mess operations
        across Regular, Exam, and Holiday semester phases.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ══════════════════════════════════════════════════
with tab4:
    m1,m2,m3,m4 = st.columns(4)
    for col, label, val in zip(
        [m1,m2,m3,m4],
        ["Algorithm","Version","Accuracy","F1 Score"],
        [model_info.get("model_name", type(model).__name__),
         f"v{version}",
         f"{model_info.get('accuracy',0)*100:.1f}%",
         f"{model_info.get('f1_macro',0)*100:.1f}%"]
    ):
        with col:
            st.markdown(f"""<div class="model-stat">
                <p class="model-stat-label">{label}</p>
                <p class="model-stat-value">{val}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<p class="section-heading">⚖️ Feature Importance</p>', unsafe_allow_html=True)
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(12)
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", height=380)
            fig.update_traces(marker_color="#22c55e", marker_line_width=0)
            fig.update_layout(**{**CHART_LAYOUT,
                "xaxis": dict(gridcolor="#2a2f3e", title=""),
                "yaxis": dict(gridcolor="rgba(0,0,0,0)", title=""),
                "margin": dict(t=10,b=10,l=10,r=10)})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with right:
        st.markdown('<p class="section-heading">🎯 Confusion Matrix</p>', unsafe_allow_html=True)
        st.caption("Held-out 20% test split — not training data.")

        X_all = df.drop(columns=["id","meal_date","primary_item","demand_level"])
        y_all = df["demand_level"]
        for col in encoders:
            if col in X_all.columns:
                X_all[col] = encoders[col].transform(X_all[col])

        _, X_test_cm, _, y_test_cm = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        y_pred_raw = model.predict(X_test_cm)
        # ✅ FIX: decode numeric predictions → labels before confusion matrix
        y_pred_cm = encoders["demand_level"].inverse_transform(y_pred_raw)

        labels = ["Low","Medium","High"]
        cm = confusion_matrix(y_test_cm, y_pred_cm, labels=labels)

        fig = go.Figure(data=go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,"#1a1f2e"],[0.5,"#14532d"],[1,"#22c55e"]],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            showscale=False
        ))
        fig.update_layout(
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
            font_color="#9ca3af",
            xaxis=dict(title="Predicted", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Actual",    gridcolor="rgba(0,0,0,0)"),
            height=380, margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model evolution chart
    st.markdown('<p class="section-heading">📈 Model Evolution</p>', unsafe_allow_html=True)
    if len(registry["models"]) > 1:
        evo_df = pd.DataFrame(registry["models"])
        fig = go.Figure()
        for metric, color, name in [
            ("accuracy","#22c55e","Accuracy"),
            ("f1_macro","#3b82f6","F1 Score"),
            ("precision_macro","#f59e0b","Precision"),
        ]:
            fig.add_trace(go.Scatter(
                x=evo_df["version"], y=evo_df[metric],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color)
            ))
        fig.update_layout(
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
            font_color="#9ca3af",
            xaxis=dict(title="Model Version", gridcolor="#2a2f3e", tickprefix="v"),
            yaxis=dict(title="Score", gridcolor="#2a2f3e", range=[0,1]),
            legend=dict(font=dict(color="#9ca3af")),
            height=300, margin=dict(t=10,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#1a1f2e;border:1px dashed #2a2f3e;border-radius:10px;
                    padding:2rem;text-align:center;color:#4b5563;">
            Retrain the model to see performance evolution across versions.
        </div>
        """, unsafe_allow_html=True)

    # Registry table
    st.markdown('<p class="section-heading">📋 Registry History</p>', unsafe_allow_html=True)
    reg_df = pd.DataFrame(registry["models"])[
        ["version","model_name","trained_on_records","accuracy","f1_macro","training_date"]
    ].rename(columns={
        "version":"Version","model_name":"Model","trained_on_records":"Records",
        "accuracy":"Accuracy","f1_macro":"F1","training_date":"Trained On"
    })
    reg_df["Version"]  = reg_df["Version"].apply(lambda x: f"v{x}")
    reg_df["Accuracy"] = reg_df["Accuracy"].apply(lambda x: f"{x*100:.1f}%")
    reg_df["F1"]       = reg_df["F1"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(reg_df, hide_index=True, use_container_width=True)