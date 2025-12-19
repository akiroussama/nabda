
import duckdb
import pandas as pd

try:
    conn = duckdb.connect("data/jira.duckdb")
    print("Connected to database")
    
    # Check issues table schema
    print("\nIssues Table Schema:")
    schema = conn.execute("DESCRIBE issues").fetchall()
    for col in schema:
        print(f"{col[0]} ({col[1]})")

    # Sample data
    print("\nSample Data (First 1 row):")
    sample = conn.execute("SELECT * FROM issues LIMIT 1").fetchone()
    print(sample)

except Exception as e:
    print(f"Error: {e}")
