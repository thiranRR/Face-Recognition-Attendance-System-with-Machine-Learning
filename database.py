import sqlite3
import pickle
from datetime import datetime

DB_NAME = "attendance_system.db"

def init_db():
    """Creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Users Table (Stores Name and the 128-d Face Embedding as a BLOB)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    face_encoding BLOB NOT NULL,
                    created_at TEXT
                )''')

    # 2. Attendance Table (With 'Status' for Late Logic)
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT,
                    date TEXT,
                    time TEXT,
                    status TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
    
    conn.commit()
    conn.close()

def add_user(name, face_encoding):
    """Saves a new user and their face encoding (numpy array)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Convert numpy array to binary for storage
    encoding_blob = pickle.dumps(face_encoding)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO users (name, face_encoding, created_at) VALUES (?, ?, ?)", 
              (name, encoding_blob, date_str))
    conn.commit()
    conn.close()

def get_all_users():
    """Retrieves all users and converts BLOBs back to numpy arrays"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, face_encoding FROM users")
    rows = c.fetchall()
    conn.close()
    
    users = []
    for r in rows:
        user_id, name, blob = r
        encoding = pickle.loads(blob)
        users.append({"id": user_id, "name": name, "encoding": encoding})
    return users

def mark_attendance(user_id, name, late_threshold="09:15"):
    """Marks attendance in SQL and calculates Late Status"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if already present today
    c.execute("SELECT * FROM attendance WHERE user_id=? AND date=?", (user_id, date_str))
    if c.fetchone():
        conn.close()
        return "Already Marked"

    # Late Logic
    threshold_time = datetime.strptime(late_threshold, "%H:%M").time()
    current_time = now.time()
    status = "Late" if current_time > threshold_time else "On Time"
    
    c.execute("INSERT INTO attendance (user_id, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
              (user_id, name, date_str, time_str, status))
    conn.commit()
    conn.close()
    return f"Marked ({status})"

def get_today_report():
    """Fetches today's records for the report tab"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, time, status FROM attendance WHERE date=?", (date_str,))
    rows = c.fetchall()
    conn.close()
    return rows

# Initialize DB immediately when this module is imported
init_db()