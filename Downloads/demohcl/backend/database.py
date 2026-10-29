import sqlite3

from pathlib import Path
from datetime import datetime, timezone


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = (
    BASE_DIR /
    "data" /
    "iot_fault_diagnosis.db"
)


def get_connection():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_sensor_data (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                device_id TEXT NOT NULL,

                voltage_magnitude REAL NOT NULL,

                current_magnitude REAL NOT NULL,

                voltage_imbalance REAL NOT NULL,

                current_imbalance REAL NOT NULL,

                power REAL NOT NULL,

                power_variation REAL NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                raw_data_id INTEGER NOT NULL UNIQUE,

                analyzed_at TEXT NOT NULL,

                predicted_fault TEXT NOT NULL,

                confidence REAL NOT NULL,

                anomaly TEXT NOT NULL,

                anomaly_score REAL NOT NULL,

                severity TEXT NOT NULL,

                evidence TEXT,

                FOREIGN KEY(raw_data_id)
                REFERENCES raw_sensor_data(id)
            )
        """)

        conn.commit()


def save_raw_data(data):

    with get_connection() as conn:

        cursor = conn.execute("""
            INSERT INTO raw_sensor_data (

                timestamp,
                device_id,

                voltage_magnitude,
                current_magnitude,

                voltage_imbalance,
                current_imbalance,

                power,
                power_variation

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            data["timestamp"],
            data["device_id"],

            float(data["voltage_magnitude"]),
            float(data["current_magnitude"]),

            float(data["voltage_imbalance"]),
            float(data["current_imbalance"]),

            float(data["power"]),
            float(data["power_variation"])
        ))

        conn.commit()

        return cursor.lastrowid


def save_analysis(
    raw_data_id,
    predicted_fault,
    confidence,
    anomaly,
    anomaly_score,
    severity,
    evidence
):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO analysis_results (

                raw_data_id,
                analyzed_at,

                predicted_fault,
                confidence,

                anomaly,
                anomaly_score,

                severity,
                evidence

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            raw_data_id,

            datetime.now(
                timezone.utc
            ).isoformat(),

            predicted_fault,

            float(confidence),

            anomaly,

            float(anomaly_score),

            severity,

            evidence
        ))

        conn.commit()


if __name__ == "__main__":

    initialize_database()

    print("SQLite database initialized successfully.")

    print(
        "Database:",
        DB_PATH
    )