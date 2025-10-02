import psycopg2

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname="mydb",
    user="melvint",
    password="MelvinGeorgi",
    host="localhost",
    port="5432"
)

# Create a cursor to run SQL commands
cur = conn.cursor()
#cur.execute("DELETE FROM users;")

# Example: create a table
cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name TEXT,
        age INT
    );
""")

# Insert a row
cur.execute("INSERT INTO users (name, age) VALUES (%s, %s)", ("Alice", 25))

# Commit changes
conn.commit()

# Query data
cur.execute("SELECT * FROM users;")
rows = cur.fetchall()
for row in rows:
    print(row)

# Close everything
cur.close()
conn.close()
