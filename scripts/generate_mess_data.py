import sqlite3
import random
import os
from datetime import datetime, timedelta

# Database path
DB_PATH = os.path.join("database", "mess.db")

def get_db_connection():
    return sqlite3.connect(DB_PATH)


# Menu options based on demand tiers and meal types
BREAKFAST_HIGH = [
    "Paratha", "Idli Vada", "Misal Pav", "Dhokla"
]

BREAKFAST_MID = [
    "Uttapam", "Sabudana Vada", "Poha", "Vada Pav"
]

BREAKFAST_LOW = [
    "Bombay Sandwich", "Coleslaw Sandwich",
    "Besan Chilla", "Vermicelli Upma", "Rava Upma"
]

LUNCH_HIGH = [
    "Kadhi Pakoda", "Bhindi Kurkure", "Chole",
    "Baingan Bharta", "Paneer", "Soya 65"
]

LUNCH_MID = [
    "Aloo Jeera", "Aloo Bhindi", "Aloo Capsicum",
    "Sev Tamatar", "Methi", "Baingan Masala",
    "Chana Masala", "Mix Veg", "Soya Masala", "Matki"
]

LUNCH_LOW = [
    "Tendli", "Cabbage", "Lauki", "Karela",
    "Turai", "Chawali", "Rajma", "Capsicum"
]

DINNER_HIGH = [
    "Chicken", "Paneer", "Biryani", "Pav Bhaji", "Egg", "Chole"
]

DINNER_MID = [
    "Mexican", "Mix Veg", "Soya Masala"
]

DINNER_LOW = [
    "Chinese", "Tendli", "Cabbage", "Lauki",
    "Karela", "Turai", "Chawali", "Rajma", "Capsicum"
]

# Function to get day of week and weekend status
def get_day_info(date_obj):
    day = date_obj.strftime("%A")
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
    return day, is_weekend


# Function to simulate hostel occupancy based on semester phase and weekend
def get_hostel_occupancy(semester_phase, is_weekend):
    if semester_phase == "Holidays":
        return random.randint(30, 60)
    if semester_phase == "Exams":
        return random.randint(70, 85)
    if is_weekend:
        return random.randint(60, 80)
    return random.randint(80, 95)


# Function to select menu items based on meal type, day, and demand tier
def select_menu(meal_type, day, is_weekend):
    has_paneer = has_chicken = has_egg = 0
    has_dessert = has_special_cuisine = 0
    has_drink = has_fruit = 0

    # ---------- BREAKFAST ----------
    if meal_type == "Breakfast":
        if day == "Sunday":
            tier = "High"
            primary_item = random.choice(BREAKFAST_HIGH)
        elif day == "Saturday":
            tier = random.choice(["Low", "Medium"])
            primary_item = random.choice(
                BREAKFAST_LOW if tier == "Low" else BREAKFAST_MID
            )
        else:
            tier = random.choice(["Medium", "High"])
            primary_item = random.choice(
                BREAKFAST_HIGH if tier == "High" else BREAKFAST_MID
            )

        # Fruit only twice a week & mostly with low demand
        has_fruit = 1 if tier == "Low" and random.random() < 0.5 else 0

    # ---------- LUNCH ----------
    elif meal_type == "Lunch":
        has_drink = 1
        if not is_weekend:
            tier = "High"
            primary_item = random.choice(LUNCH_HIGH)
        else:
            tier = random.choice(["Low", "Medium"])
            primary_item = random.choice(
                LUNCH_LOW if tier == "Low" else LUNCH_MID
            )

    # ---------- DINNER ----------
    else:
        # Fixed rules
        if day in ["Wednesday", "Sunday"]:
            has_paneer = 1
            has_chicken = 1
            tier = "High"
            primary_item = "Paneer & Chicken"

        elif day == "Friday":
            has_paneer = 1
            has_egg = 1
            tier = "High"
            primary_item = "Paneer & Egg"

        else:
            tier = random.choice(["Low", "Medium", "High"])
            if tier == "High":
                primary_item = random.choice(DINNER_HIGH)
            elif tier == "Medium":
                primary_item = random.choice(DINNER_MID)
            else:
                primary_item = random.choice(DINNER_LOW)

        # Monthly special cuisine
        if primary_item in ["Chinese", "Mexican", "Pav Bhaji", "Biryani"]:
            has_special_cuisine = 1

        # Dessert twice a week
        has_dessert = 1 if random.random() < 0.3 else 0

    return {
        "primary_item": primary_item,
        "menu_demand_tier": tier,
        "has_paneer": has_paneer,
        "has_chicken": has_chicken,
        "has_egg": has_egg,
        "has_dessert": has_dessert,
        "has_special_cuisine": has_special_cuisine,
        "has_drink": has_drink,
        "has_fruit": has_fruit
    }

# Function to calculate demand level based on menu tier, occupancy, and dessert
def calculate_demand(menu_tier, occupancy, has_dessert):
    score = 0

    if menu_tier == "High":
        score += 2
    elif menu_tier == "Medium":
        score += 1

    if occupancy > 80:
        score += 2
    elif occupancy > 60:
        score += 1

    if has_dessert:
        score += 2

    if score >= 4:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"


# Main function to generate and insert data into the database
def generate_data(start_date, days=30, semester_phase="Regular"):
    conn = get_db_connection()
    cursor = conn.cursor()

    previous_demand = None

    for i in range(days):
        current_date = start_date + timedelta(days=i)
        day, is_weekend = get_day_info(current_date)

        for meal in ["Breakfast", "Lunch", "Dinner"]:
            occupancy = get_hostel_occupancy(semester_phase, is_weekend)
            menu = select_menu(meal, day, is_weekend)
            demand = calculate_demand(
                menu["menu_demand_tier"], occupancy, menu["has_dessert"]
            )

            cursor.execute("""
                INSERT INTO mess_records (
                    meal_date, day_of_week, meal_type,
                    primary_item, menu_demand_tier,
                    has_paneer, has_chicken, has_egg,
                    has_dessert, has_special_cuisine,
                    has_drink, has_fruit,
                    hostel_occupancy_pct, semester_phase,
                    is_weekend, previous_meal_demand, demand_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                current_date.date(), day, meal,
                menu["primary_item"], menu["menu_demand_tier"],
                menu["has_paneer"], menu["has_chicken"], menu["has_egg"],
                menu["has_dessert"], menu["has_special_cuisine"],
                menu["has_drink"], menu["has_fruit"],
                occupancy, semester_phase,
                is_weekend, previous_demand, demand
            ))

            previous_demand = demand

    conn.commit()
    conn.close()

# Run the data generation
if __name__ == "__main__":
    start = datetime.today()
    generate_data(start_date=start, days=30, semester_phase="Regular")
    generate_data(start_date=start + timedelta(days=30), days=20, semester_phase="Exams")
    generate_data(start_date=start + timedelta(days=50), days=20, semester_phase="Holidays")

    print(" Mess data generated and inserted successfully")
