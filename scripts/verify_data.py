import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join("database", "mess.db")

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM mess_records", conn)
    conn.close()
    return df

if __name__ == "__main__":
    df = load_data()

    print(" Total rows:", df.shape[0])
    print(" Total columns:", df.shape[1])

    print("\n🔹 Sample data:")
    print(df.head())

    print("\n🔹 Demand distribution:")
    print(df["demand_level"].value_counts())

    print("\n🔹 Meal-wise distribution:")
    print(df["meal_type"].value_counts())
