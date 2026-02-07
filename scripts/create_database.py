import sqlite3
import os

# Path to database file
db_path = os.path.join("database", "mess.db")

# Connect to SQLite (creates file if not exists)
conn = sqlite3.connect(db_path)

# Create a cursor (used to run SQL commands)
cursor = conn.cursor()

# SQL command to create table
create_table_query = """
CREATE TABLE IF NOT EXISTS mess_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    meal_date DATE NOT NULL,
    day_of_week TEXT NOT NULL,

    meal_type TEXT NOT NULL,
    primary_item TEXT NOT NULL,

    menu_demand_tier TEXT NOT NULL,

    has_paneer INTEGER NOT NULL,
    has_chicken INTEGER NOT NULL,
    has_egg INTEGER NOT NULL,

    has_dessert INTEGER NOT NULL,
    has_special_cuisine INTEGER NOT NULL,

    has_drink INTEGER NOT NULL,
    has_fruit INTEGER NOT NULL,

    hostel_occupancy_pct INTEGER NOT NULL,
    semester_phase TEXT NOT NULL,
    is_weekend INTEGER NOT NULL,

    previous_meal_demand TEXT,
    demand_level TEXT NOT NULL
);
"""

# Execute SQL
cursor.execute(create_table_query)

# Save changes
conn.commit()

# Close connection
conn.close()

print(" Database and table created successfully!")
