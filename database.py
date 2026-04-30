import sqlite3
import pandas as pd
from config import COLUMNS, MAX_HISTORY

DB_SQLITE = "stroke_history.db"

def get_connection():
    return sqlite3.connect(DB_SQLITE)

def init_db():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clinical_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                filename TEXT,
                date TEXT,
                time TEXT,
                model TEXT,
                verdict TEXT,
                hemisphere TEXT,
                hu TEXT,
                area TEXT,
                confidence TEXT,
                speed TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON clinical_history(created_at DESC)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON clinical_history(patient_id)')

def load_last_records(limit=MAX_HISTORY):
    with get_connection() as conn:
        cur = conn.execute('''
            SELECT patient_id, filename, date, time, model, verdict, hemisphere, hu, area, confidence, speed
            FROM clinical_history
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        return cur.fetchall()

def save_history(record):
    with get_connection() as conn:
        conn.execute('''
            INSERT INTO clinical_history 
            (patient_id, filename, date, time, model, verdict, hemisphere, hu, area, confidence, speed)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ''', record)

def get_history_dataframe(limit=MAX_HISTORY):
    records = load_last_records(limit)
    return pd.DataFrame(records, columns=COLUMNS)

def get_max_patient_id():
    with get_connection() as conn:
        cur = conn.execute("SELECT COALESCE(MAX(CAST(patient_id AS INTEGER)), 0) FROM clinical_history")
        return cur.fetchone()[0]

init_db()