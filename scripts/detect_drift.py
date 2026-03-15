
import sqlite3
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from datetime import datetime


# --------------------------------------------------
# PATHS
# --------------------------------------------------
DB_PATH       = os.path.join("database", "mess.db")
REGISTRY_PATH = os.path.join("models", "model_registry.json")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM mess_records ORDER BY id ASC", conn)
    conn.close()
    return df


# --------------------------------------------------
# SPLIT INTO REFERENCE vs CURRENT
# --------------------------------------------------
def split_reference_current(df, reference_pct=0.70):
    """
    Reference data = first 70% of records (what the model was trained on)
    Current data   = last 30% of records  (new incoming data)
    """
    split_idx = int(len(df) * reference_pct)
    reference = df.iloc[:split_idx].copy()
    current   = df.iloc[split_idx:].copy()
    return reference, current


# --------------------------------------------------
# PSI — Population Stability Index (numerical features)
# --------------------------------------------------
def calculate_psi(reference_col, current_col, bins=10):
    """
    PSI measures how much a numerical distribution has shifted.
    Higher PSI = more drift.
    """
    # Create bins based on reference distribution
    breakpoints = np.linspace(
        min(reference_col.min(), current_col.min()),
        max(reference_col.max(), current_col.max()),
        bins + 1
    )

    ref_counts, _  = np.histogram(reference_col, bins=breakpoints)
    curr_counts, _ = np.histogram(current_col,   bins=breakpoints)

    # Convert to proportions, avoid division by zero
    ref_pct  = ref_counts  / len(reference_col)
    curr_pct = curr_counts / len(current_col)

    ref_pct  = np.where(ref_pct  == 0, 1e-6, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(float(psi), 4)


def interpret_psi(psi):
    if psi < 0.10:
        return " No Drift",       "green"
    elif psi < 0.20:
        return "  Moderate Drift", "yellow"
    else:
        return " Significant Drift", "red"


# --------------------------------------------------
# CHI-SQUARE TEST (categorical features)
# --------------------------------------------------
def calculate_chi2_drift(reference_col, current_col):
    """
    Chi-square test checks if the category distribution has shifted.
    p-value < 0.05 means the distribution changed significantly.
    """
    all_categories = set(reference_col.unique()) | set(current_col.unique())

    ref_counts  = reference_col.value_counts().reindex(all_categories, fill_value=0)
    curr_counts = current_col.value_counts().reindex(all_categories,   fill_value=0)

    contingency_table = pd.DataFrame({"reference": ref_counts, "current": curr_counts})

    # Need at least 2 categories to run test
    if len(all_categories) < 2:
        return 1.0, 0.0

    chi2, p_value, _, _ = chi2_contingency(contingency_table.T)
    return round(float(p_value), 4), round(float(chi2), 4)


def interpret_chi2(p_value):
    if p_value >= 0.05:
        return " No Drift",          "green"
    elif p_value >= 0.01:
        return "  Moderate Drift",  "yellow"
    else:
        return " Significant Drift", "red"


# --------------------------------------------------
# DEMAND DISTRIBUTION SHIFT (target variable drift)
# --------------------------------------------------
def check_target_drift(reference, current):
    ref_dist  = reference["demand_level"].value_counts(normalize=True).sort_index()
    curr_dist = current["demand_level"].value_counts(normalize=True).sort_index()

    all_labels = sorted(set(ref_dist.index) | set(curr_dist.index))
    ref_dist   = ref_dist.reindex(all_labels, fill_value=0)
    curr_dist  = curr_dist.reindex(all_labels, fill_value=0)

    return ref_dist, curr_dist


# --------------------------------------------------
# PRINT HELPERS
# --------------------------------------------------
def print_header(title):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

def print_row(feature, metric_value, interpretation, extra=""):
    print(f"  {feature:<28} {str(metric_value):<10} {interpretation}  {extra}")


# --------------------------------------------------
# MAIN DRIFT REPORT
# --------------------------------------------------
def run_drift_detection():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   MESS DEMAND PREDICTION — DATA DRIFT DETECTION REPORT  ║")
    print("╚" + "═"*58 + "╝")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()
    reference, current = split_reference_current(df, reference_pct=0.70)

    print(f"\n   Reference set : {len(reference):,} records (first 70%)")
    print(f"   Current set   : {len(current):,} records (last 30%)")

    drift_flags = []  # collect features with significant drift

    # ── NUMERICAL FEATURES — PSI ──────────────────────────
    print_header("NUMERICAL FEATURES — Population Stability Index (PSI)")
    print(f"  {'Feature':<28} {'PSI':<10} Status")
    print("  " + "-"*55)

    numerical_features = ["hostel_occupancy_pct"]

    for feat in numerical_features:
        psi = calculate_psi(reference[feat], current[feat])
        label, color = interpret_psi(psi)
        print_row(feat, psi, label)
        if color in ("yellow", "red"):
            drift_flags.append((feat, "PSI", psi, label))

    # ── CATEGORICAL FEATURES — CHI-SQUARE ────────────────
    print_header("CATEGORICAL FEATURES — Chi-Square Test (p-value)")
    print(f"  {'Feature':<28} {'p-value':<10} Status")
    print("  " + "-"*55)

    categorical_features = [
        "day_of_week",
        "meal_type",
        "menu_demand_tier",
        "semester_phase",
        "previous_meal_demand",
    ]

    for feat in categorical_features:
        p_val, chi2_stat = calculate_chi2_drift(reference[feat], current[feat])
        label, color = interpret_chi2(p_val)
        print_row(feat, p_val, label, f"(χ²={chi2_stat})")
        if color in ("yellow", "red"):
            drift_flags.append((feat, "Chi2", p_val, label))

    # ── TARGET VARIABLE DRIFT ────────────────────────────
    print_header("TARGET VARIABLE — Demand Level Distribution Shift")
    ref_dist, curr_dist = check_target_drift(reference, current)

    print(f"  {'Label':<12} {'Reference':>12} {'Current':>12} {'Δ Shift':>12}")
    print("  " + "-"*50)

    target_drifted = False
    for label in ref_dist.index:
        ref_pct  = ref_dist[label]
        curr_pct = curr_dist[label]
        delta    = curr_pct - ref_pct
        flag     = " ⚠️" if abs(delta) > 0.05 else ""
        print(f"  {label:<12} {ref_pct*100:>11.1f}% {curr_pct*100:>11.1f}% {delta*100:>+11.1f}%{flag}")
        if abs(delta) > 0.05:
            target_drifted = True

    if target_drifted:
        drift_flags.append(("demand_level", "Distribution", "-", "⚠️  Moderate Drift"))

    # ── BINARY FEATURE DRIFT ─────────────────────────────
    print_header("BINARY FEATURES — Mean Shift Check")
    print(f"  {'Feature':<28} {'Ref Mean':>10} {'Curr Mean':>10} {'Δ':>8} Status")
    print("  " + "-"*65)

    binary_features = ["has_paneer","has_chicken","has_egg",
                       "has_dessert","has_special_cuisine","has_drink","has_fruit","is_weekend"]

    for feat in binary_features:
        ref_mean  = reference[feat].mean()
        curr_mean = current[feat].mean()
        delta     = curr_mean - ref_mean
        if abs(delta) > 0.05:
            status = "  Shifted"
            drift_flags.append((feat, "Mean Shift", round(delta,3), "⚠️  Moderate Drift"))
        else:
            status = " Stable"
        print(f"  {feat:<28} {ref_mean:>10.3f} {curr_mean:>10.3f} {delta:>+8.3f}  {status}")

    # ── SUMMARY & RECOMMENDATION ─────────────────────────
    print_header("DRIFT SUMMARY & RECOMMENDATION")

    if not drift_flags:
        print("   No significant drift detected across all features.")
        print("   Recommendation: No retraining needed at this time.")
        retrain_needed = False
    else:
        print(f"   Drift detected in {len(drift_flags)} feature(s):\n")
        for feat, method, value, status in drift_flags:
            print(f"     • {feat:<28} [{method}]  {status}")

        significant = [f for f in drift_flags if "Significant" in f[3]]
        if significant:
            print(f"\n   Recommendation: RETRAIN RECOMMENDED")
            print(f"     {len(significant)} feature(s) show significant drift.")
            retrain_needed = True
        else:
            print(f"\n   Recommendation: MONITOR CLOSELY")
            print(f"     Drift is moderate. Retrain if more data confirms the trend.")
            retrain_needed = False

    # Load registry for context
    try:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        latest = registry["models"][-1]
        print(f"\n   Current model  : {latest['model_name']} v{latest['version']}")
        print(f"   Last trained   : {latest['training_date']}")
        print(f"   Trained on     : {latest['trained_on_records']:,} records")
        print(f"   Accuracy       : {latest['accuracy']*100:.1f}%")
        print(f"   F1 Score       : {latest['f1_macro']*100:.1f}%")
    except Exception:
        pass

    print("\n" + "═"*60 + "\n")
    return retrain_needed


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    retrain_needed = run_drift_detection()
    if retrain_needed:
        print("    Run retrain_model.py to update the model.\n")