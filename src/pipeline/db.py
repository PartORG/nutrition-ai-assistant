# db_setup/user_model.py

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List
from typing import Optional

DB_FILE = "users.db"


#DATA MODELS

@dataclass
class User:
    name: str
    surname: str
    preferences: str
    restrictions: str
    health_condition: str
    caretaker: str
    created_at: str
    updated_at: str
    deleted_at: str
    age: int
    gender: str  # "Female", "Male", or "Other"
    id: Optional[int] = None


@dataclass
class MedicalAdvice:
    health_condition: str
    medical_advice: str
    user_id: int
    id: Optional[int] = None


@dataclass
class Authentication:
    login: str
    password: str
    role: str
    created_at: str
    deleted_at: str
    updated_at: str
    user_id: int
    id: Optional[int] = None


#DATABASE HANDLER

class UserDBHandler:
    def __init__(self, db_file: str = DB_FILE):
        self.db_file = db_file

    def connect(self):
        return sqlite3.connect(self.db_file)

    #USERS TABLE 
    def create_users_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                surname TEXT,
                preferences TEXT,
                restrictions TEXT,
                health_condition TEXT,
                caretaker TEXT,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
                age INTEGER,
                gender TEXT
            )
        """)
        conn.commit()
        conn.close()

    def insert_user(self, user: User):
        conn = self.connect()
        cursor = conn.cursor()
        user_dict = asdict(user)
        # remove id so SQLite can autoincrement it
        user_dict.pop("id", None)
        columns = ", ".join(user_dict.keys())
        placeholders = ", ".join(["?"] * len(user_dict))
        print(f"[DB] Inserting user: {user_dict}")
        cursor.execute(
            f"INSERT INTO users ({columns}) VALUES ({placeholders})",
            tuple(user_dict.values())
        )
        conn.commit()
        # return the auto-generated user id
        user_id = cursor.lastrowid
        conn.close()
        return user_id  

    def read_users(self) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        return users
    
    def read_user(self, name: str, surname: str):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE name = ? AND surname = ? ORDER BY updated_at DESC LIMIT 1",
            (name, surname)
        )
        result = cursor.fetchone()
        conn.close()
        return result

    def update_user(self, user_id: int, field: str, new_value):
        allowed_fields = {
            "name", "surname", "preferences", "restrictions",
            "health_condition", "caretaker", "age", "gender"
        }
        if field not in allowed_fields:
            raise ValueError("Invalid field name")

        if isinstance(new_value, list):
            new_value = ", ".join(map(str, new_value))

        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE users SET {field} = ?, updated_at = ? WHERE id = ?",
            (new_value, datetime.now().isoformat(), user_id)
        )
        conn.commit()
        conn.close()

    def get_users_by_condition(self, condition: str) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE health_condition = ?",
            (condition,)
        )
        results = cursor.fetchall()
        conn.close()
        return results

    def soft_delete_user(self, user_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET deleted_at = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id)
        )
        conn.commit()
        conn.close()

    #MEDICAL ADVICE TABLE

    def create_medical_advice_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_advice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                health_condition TEXT,
                medical_advice TEXT,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()

    def insert_medical_advice(self, advice: MedicalAdvice):
        conn = self.connect()
        cursor = conn.cursor()
        if isinstance(advice.medical_advice, list):
            advice.medical_advice = "\n".join(advice.medical_advice)
        cursor.execute("""
            INSERT INTO medical_advice (health_condition, medical_advice, user_id)
            VALUES (?, ?, ?)
        """, (advice.health_condition, advice.medical_advice, advice.user_id))
        conn.commit()
        advice_id = cursor.lastrowid
        conn.close()
        return advice_id

    def get_medical_advice_by_user(self, user_id: int) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM medical_advice WHERE user_id = ?",
            (user_id,)
        )
        results = cursor.fetchall()
        conn.close()
        return results

    # AUTHENTICATION TABLE

    def create_authentication_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authentication (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT,
                password TEXT,
                role TEXT,
                created_at TEXT,
                deleted_at TEXT,
                updated_at TEXT,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()


#MAIN: TEST EVERYTHING

if __name__ == "__main__":
    db = UserDBHandler()

    #create tables
    db.create_users_table()
    db.create_medical_advice_table()
    db.create_authentication_table()

    now = datetime.now().isoformat()

    #insert test user
    user = User(
        id=1,
        name="Adam",
        surname="Levin",
        preferences="Low Sugar",
        restrictions="Rental",
        health_condition="Kidney Disease",
        caretaker="None",
        created_at=now,
        updated_at=now,
        deleted_at="",
        age=67,
        gender="Other"
    )

    db.insert_user(user)

    print("All users:")
    print(db.read_users())

    #Update user
    db.update_user(1, "preferences", "Low Sugar")

    print("\nUsers with Kidney Disease:")
    print(db.get_users_by_condition("Kidney Disease"))

    #Insert medical advice
    advice = MedicalAdvice(
        id=1,
        health_condition="Kidney Disease",
        medical_advice="Avoid high phosphorus and potassium foods.",
        user_id=1
    )

    db.insert_medical_advice(advice)

    print("\nMedical advice for user 1:")
    print(db.get_medical_advice_by_user(1))

    #Soft delete user
    db.soft_delete_user(1)

    print("\nAfter soft delete:")
    print(db.read_users())
