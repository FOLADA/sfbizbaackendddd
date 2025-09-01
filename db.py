import sqlite3
from pathlib import Path

DB_PATH = Path("businesses.db")

def init_db() -> None:
    """Create the businesses table if not exists"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS businesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                services TEXT,
                service_pricing TEXT,
                location TEXT,
                latitude REAL,
                longitude REAL,
                image_url TEXT,
                rating REAL DEFAULT 0,
                total_reviews INTEGER DEFAULT 0,
                socials TEXT,
                owner_id INTEGER,
                business_hours TEXT,
                dynamic_pricing_config TEXT,
                ai_analysis_data TEXT,
                market_position TEXT DEFAULT 'competitive',
                revenue_potential_score REAL DEFAULT 0.7,
                FOREIGN KEY (owner_id) REFERENCES users (id) ON DELETE SET NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS business_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                image_url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (business_id) REFERENCES businesses (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS business_hours (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                day_of_week INTEGER NOT NULL CHECK (day_of_week >= 0 AND day_of_week <= 6),
                open_time TEXT,
                close_time TEXT,
                is_closed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (business_id) REFERENCES businesses (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                name TEXT,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                text TEXT NOT NULL,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (business_id) REFERENCES businesses (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS service_pricing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                service_name TEXT NOT NULL,
                current_price REAL,
                recommended_price REAL,
                pricing_strategy TEXT,
                confidence_score REAL DEFAULT 0.8,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (business_id) REFERENCES businesses (id) ON DELETE CASCADE
            )
        """)
        
        # Add all necessary columns to existing businesses table
        columns_to_add = [
            ("owner_id", "INTEGER REFERENCES users(id)"),
            ("business_hours", "TEXT"),
            ("dynamic_pricing_config", "TEXT"),
            ("total_reviews", "INTEGER DEFAULT 0"),
            ("latitude", "REAL"),
            ("longitude", "REAL"),
            ("service_pricing", "TEXT"),
            ("ai_analysis_data", "TEXT"),
            ("market_position", "TEXT DEFAULT 'competitive'"),
            ("revenue_potential_score", "REAL DEFAULT 0.7")
        ]
        
        for column_name, column_def in columns_to_add:
            try:
                conn.execute(f"ALTER TABLE businesses ADD COLUMN {column_name} {column_def}")
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        conn.commit()

def get_db():
    """Return a SQLite connection with dict-like row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
