#!/usr/bin/env python3

import os 
import pandas as pd
import sqlite3
import re



# Connect to SQLite database
conn = sqlite3.connect("orlando_housing.db")
cursor = conn.cursor()

# Define schema 
create_table_template = """
CREATE TABLE IF NOT EXISTS orlando_{zip} (
    city TEXT,
    state TEXT,
    county TEXT,
    address TEXT,
    zipcode TEXT,
    longitude REAL,
    latitude REAL,
    sale_price INTEGER,
    date_sold TEXT,
    home_type INTEGER,
    age REAL,
    living_area REAL,
    bedrooms REAL,
    bathrooms REAL,
    levels REAL,
    parking_spaces INTEGER,
    has_garage INTEGER,
    private_pool INTEGER,
    hoa INTEGER,
    gated INTEGER,
    recreational_facilities INTEGER,
    park INTEGER,
    playground INTEGER,
    greenbelt INTEGER,
    above_flood_plain INTEGER,
    city_lot INTEGER,
    historic_district INTEGER,
    view INTEGER,
    water_view INTEGER,
    avg_distance_to_schools REAL,
    avg_schools_rating REAL,
    min_distance_highway REAL,
    sqft_per_bedroom REAL
);
"""

house_data = pd.read_csv('Data/house_data_full.csv', low_memory=False)

# Iterate through each ZIP code group and create/populate the table
for zip_code, df_zip in house_data.groupby("zipcode"):
    table_name = f"orlando_{zip_code}"
    df_zip.to_sql(table_name, conn, if_exists="replace", index=False)
    
# Commit 
conn.commit()


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

csv_folder = "Data/"

zipfile_pattern = re.compile(r"^orlando_(\d{5})\.csv$")   # exactly 5 digits

conn = sqlite3.connect("orlando_housing.db")

for filename in os.listdir(csv_folder):
    m = zipfile_pattern.match(filename)
    if not m:                # ignore orlando_house_data.csv, SeriesReport.csv, etc.
        continue

    zip_code   = m.group(1)                     # the 5‑digit ZIP
    table_name = f"orlando_{zip_code}"
    csv_path   = os.path.join(csv_folder, filename)

    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"Table '{table_name}' populated with {len(df)} rows.")

conn.commit()
conn.close()


