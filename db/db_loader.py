# db/db_loader.py

import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

def insert_dataframe(df, filename):
    df["filename"] = filename

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                try:
                    cur.execute("""
                        INSERT INTO transactions (date, description, amount, category, direction, source, filename)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """, (
                        row["date"],
                        row["description"],
                        row["amount"],
                        row["category"],
                        row.get("direction", "outflow"),
                        row["source"],
                        row["filename"]
                    ))
                except Exception as e:
                    print(f"❌ Insert error: {e}")
    print(f"✅ Inserted rows from: {filename}")
