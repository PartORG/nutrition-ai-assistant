# db_setup/user_model.py

import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional

from huggingface_hub import login

DB_FILE = "users.db"

# DATA MODELS

@dataclass
class User:
    name: str
    surname: str
    user_name: str
    caretaker: str
    created_at: str
    updated_at: str
    deleted_at: str
    age: int
    gender: str
    id: Optional[int] = None

@dataclass
class MedicalAdvice:
    health_condition: str
    medical_advice: str
    created_at: str
    updated_at: str
    deleted_at: str
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

@dataclass
class UserProfileHistory:
    preferences: str
    user_id: int
    health_condition: str
    restrictions: str
    created_at: str
    updated_at: str
    deleted_at: str
    id: Optional[int] = None

@dataclass
class RecipeHistory:
    user_id: int
    recipe_id: int
    servings: int
    ingredients: str
    prep_time: str
    created_at: str
    updated_at: str
    deleted_at: str
    restrictions: str
    id: Optional[int] = None

@dataclass
class NutritionHistory:
    user_id: int
    recipe_id: int
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: float
    sugar: float
    sodium: float
    created_at: str
    updated_at: str
    deleted_at: str
    id: Optional[int] = None

# DATABASE HANDLER

class UserDBHandler:
    def __init__(self, db_file: str = DB_FILE): #, db_file: str = "users.db"
        self.db_file = db_file

    def connect(self):                      #   Connect to the SQLite database and enable foreign key support
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def create_users_table(self):           #  Create the users table if it doesn't exist
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                surname TEXT,
                user_name TEXT UNIQUE,
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

    def insert_user(self, user: User):          # Insert a new user into the users table and return the new user's ID
        conn = self.connect()
        cursor = conn.cursor()
        user_dict = asdict(user)
        user_dict.pop("id", None)
        columns = ", ".join(user_dict.keys())
        placeholders = ", ".join(["?"] * len(user_dict))
        print(f"[DB] Inserting user: {user_dict}")
        cursor.execute(
            f"INSERT INTO users ({columns}) VALUES ({placeholders})",
            tuple(user_dict.values())
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id

    def read_users(self) -> List[tuple]:        # Retrieve all users from the users table and return them as a list of tuples
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        return users

    def read_user(self, name: str, surname: str):           # Retrieve a user by name and surname, ordered by updated_at in descending order, and return the most recent match as a tuple
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM users 
            WHERE name = ? AND surname = ? 
            ORDER BY updated_at DESC 
            LIMIT 1
            """,
            (name, surname)
        )
        result = cursor.fetchone()
        conn.close()
        return result

    def update_user(self, user_id: int, field: str, new_value):         # Update a specific field of a user identified by user_id with a new value, and update the updated_at timestamp. Only allows certain fields to be updated.
        allowed_fields = {
            "name", "surname", "user_name",
            "caretaker", "age", "gender"
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

    def soft_delete_user(self, user_id: int):           # Soft delete a user by setting the deleted_at timestamp to the current time for the user identified by user_id
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET deleted_at = ? WHERE id = ?",
            (datetime.now().isoformat(), user_id)
        )
        conn.commit()
        conn.close()

    # CRUD for MedicalAdvice ---

    def create_medical_advice_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_advice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                health_condition TEXT,
                medical_advice TEXT,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
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
            INSERT INTO medical_advice 
            (health_condition, medical_advice, created_at, updated_at, deleted_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            advice.health_condition,
            advice.medical_advice,
            advice.created_at,
            advice.updated_at,
            advice.deleted_at,
            advice.user_id
        ))

        conn.commit()
        advice_id = cursor.lastrowid
        conn.close()
        return advice_id

    def update_medical_advice(self, advice_id: int, new_advice: str):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE medical_advice SET medical_advice = ?, updated_at = ? WHERE id = ?",
            (new_advice, datetime.now().isoformat(), advice_id)
        )
        conn.commit()
        conn.close()

    def delete_medical_advice(self, advice_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM medical_advice WHERE id = ?", (advice_id,))
        conn.commit()
        conn.close()


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
    
    def read_medical_advice(self, advice_id: int):          # Retrieve a specific medical advice entry by its ID and return it as a tuple
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM medical_advice WHERE id = ?", (advice_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    def read_all_medical_advice(self) -> List[tuple]:   # Retrieve all medical advice entries from the medical_advice table and return them as a list of tuples
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM medical_advice")
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    
    # --- CRUD for Authentification ---

    def create_authentication_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authentication (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login TEXT UNIQUE,
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

    def insert_authentication(self, auth: Authentication) -> int:
        conn = self.connect()
        cursor = conn.cursor()

        auth_dict = asdict(auth)
        auth_dict.pop("id", None)

        columns = ", ".join(auth_dict.keys())
        placeholders = ", ".join(["?"] * len(auth_dict))

        cursor.execute(
            f"INSERT INTO authentication ({columns}) VALUES ({placeholders})",
            tuple(auth_dict.values())
        )
        conn.commit()
        auth_id = cursor.lastrowid
        conn.close()
        return auth_id

    def read_authentication_by_id(self, auth_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authentication WHERE id = ?", (auth_id,))
        row = cursor.fetchone()
        conn.close()
        return row
    
    def read_authentication_by_user(self, user_id: int) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authentication WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def read_authentication_by_login(self, login: str):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authentication WHERE login = ?", (login,))
        row = cursor.fetchone()
        conn.close()
        return row
    
    def update_authentication(self, auth_id: int, field: str, new_value):
        allowed_fields = {"login", "password", "role"}
        if field not in allowed_fields:
            raise ValueError("Invalid field")

        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE authentication SET {field} = ?, updated_at = ? WHERE id = ?",
            (new_value, datetime.now().isoformat(), auth_id)
        )
        conn.commit()
        conn.close()

    def soft_delete_authentication(self, auth_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE authentication SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), datetime.now().isoformat(), auth_id)
        )
        conn.commit()
        conn.close()

    # --- CRUD for UserProfileHistory ---
    
    def create_user_profile_history_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preferences TEXT,
                user_id INTEGER,
                health_condition TEXT,
                restrictions TEXT,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()

    def insert_user_profile_history(self, history: UserProfileHistory) -> int:
        conn = self.connect()
        cursor = conn.cursor()

        hist_dict = asdict(history)
        hist_dict.pop("id", None)
        columns = ", ".join(hist_dict.keys())
        placeholders = ", ".join(["?"] * len(hist_dict))

        cursor.execute(
            f"INSERT INTO user_profile_history ({columns}) VALUES ({placeholders})",
            tuple(hist_dict.values())
        )
        conn.commit()
        hist_id = cursor.lastrowid
        conn.close()
        return hist_id

    def read_user_profile_history_by_user(self, user_id: int) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user_profile_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def update_user_profile_history(self, history_id: int, field: str, new_value):
        allowed_fields = {"preferences", "health_condition", "restrictions"}
        if field not in allowed_fields:
            raise ValueError("Invalid field")
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE user_profile_history SET {field} = ?, updated_at = ? WHERE id = ?",
            (new_value, datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()
        
    def soft_delete_user_profile_history(self, history_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_profile_history SET deleted_at = ? WHERE id = ?",
            (datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()

    # --- CRUD for Recipe History ---   

    def create_recipe_history_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                recipe_id INTEGER,
                restrictions TEXT,
                servings INTEGER,
                ingredients TEXT,
                prep_time TEXT,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()

    def insert_recipe_history(self, history: RecipeHistory) -> int:
        conn = self.connect()
        cursor = conn.cursor()

        hist_dict = asdict(history)
        hist_dict.pop("id", None)

        columns = ", ".join(hist_dict.keys())
        placeholders = ", ".join(["?"] * len(hist_dict))
        cursor.execute(
            f"INSERT INTO recipe_history ({columns}) VALUES ({placeholders})",
            tuple(hist_dict.values())
        )
        conn.commit()
        hist_id = cursor.lastrowid
        conn.close()
        return hist_id

    def read_recipe_history_by_user(self, user_id: int) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM recipe_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def update_recipe_history(self, history_id: int, field: str, new_value):
        allowed_fields = {"restrictions", "servings", "ingredients", "prep_time"}
        if field not in allowed_fields:
            raise ValueError("Invalid field")
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE recipe_history SET {field} = ?, updated_at = ? WHERE id = ?",
            (new_value, datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()

    def soft_delete_recipe_history(self, history_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE recipe_history SET deleted_at = ? WHERE id = ?",
            (datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()

    # --- CRUD for Nutrition History ---

    def create_nutrition_history_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nutrition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                recipe_id INTEGER,
                calories REAL,
                protein REAL,
                fat REAL,
                carbohydrates REAL,
                fiber REAL,
                sugar REAL,
                sodium REAL,
                created_at TEXT,
                updated_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()

    def insert_nutrition_history(self, history: NutritionHistory) -> int:
        conn = self.connect()
        cursor = conn.cursor()

        hist_dict = asdict(history)
        hist_dict.pop("id", None)

        columns = ", ".join(hist_dict.keys())
        placeholders = ", ".join(["?"] * len(hist_dict))
        cursor.execute(
            f"INSERT INTO nutrition_history ({columns}) VALUES ({placeholders})",
            tuple(hist_dict.values())
        )
        conn.commit()
        hist_id = cursor.lastrowid
        conn.close()
        return hist_id

    def read_nutrition_history_by_user(self, user_id: int) -> List[tuple]:
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM nutrition_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows

    def update_nutrition_history(self, history_id: int, field: str, new_value):
        allowed_fields = {"calories", "protein", "fat", "carbohydrates", "fiber", "sugar", "sodium"}
        if field not in allowed_fields:
            raise ValueError("Invalid field")
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE nutrition_history SET {field} = ?, updated_at = ? WHERE id = ?",
            (new_value, datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()

    def soft_delete_nutrition_history(self, history_id: int):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE nutrition_history SET deleted_at = ? WHERE id = ?",
            (datetime.now().isoformat(), history_id)
        )
        conn.commit()
        conn.close()

# MAIN TEST

if __name__ == "__main__":
    db = UserDBHandler()

    # Create tables
    db.create_users_table()
    db.create_medical_advice_table()
    db.create_authentication_table()
    db.create_user_profile_history_table()
    db.create_recipe_history_table()
    db.create_nutrition_history_table()

    now = datetime.now().isoformat()    #     Current timestamp in ISO format for created_at and updated_at fields

    # -------------------
    # USER
    # -------------------
    user = User(
        name="Adam",
        surname="Levin",
        user_name="AdamL",
        caretaker="None",
        created_at=now,
        updated_at=now,
        deleted_at="",
        age=67,
        gender="Other"
    )

    user_id = db.insert_user(user)

    print("All users:")
    print(db.read_users())

    db.update_user(user_id, "user_name", "AdamUpdated")

    print("\nAfter update:")
    print(db.read_users())

   # -------------------
    #  Medical Advice
    # -------------------

    # advice = MedicalAdvice(
    #     health_condition="Kidney Disease",
    #     medical_advice="Avoid high phosphorus and potassium foods.",
    #     created_at=now,
    #     updated_at=now,
    #     deleted_at="",
    #     user_id=user_id
    # )

    # advice_id = db.insert_medical_advice(advice)

    # print("\nMedical advice for user:")
    # print(db.get_medical_advice_by_user(user_id))

    # -------------------
    # AUTHENTICATION
    # -------------------
    # auth = Authentication(
    #     login="adaml@example.com",
    #     password="securepassword123",
    #     role="user",
    #     created_at=now,
    #     deleted_at="",
    #     updated_at=now,
    #     user_id=user_id
    # )

    # auth_id = db.insert_authentication(auth)

    # print("\nAuthentication:")
    # print(db.read_authentication_by_id(auth_id))

    # -------------------
    # USER PROFILE HISTORY
    # -------------------
    # profile_history = UserProfileHistory(
    #     preferences="High Protein",
    #     user_id=user_id,
    #     health_condition="Kidney Disease",
    #     restrictions="Low Sodium",
    #     created_at=now,
    #     updated_at=now,
    #     deleted_at=""
    # )

    # profile_id = db.insert_user_profile_history(profile_history)

    # print("\nUser profile history:")
    # print(db.read_user_profile_history_by_user(user_id))

    # -------------------
    # RECIPE HISTORY
    # -------------------
    # recipe_history = RecipeHistory(
    #     user_id=user_id,
    #     recipe_id=123,
    #     servings=2,
    #     ingredients="Chicken, Rice, Broccoli",
    #     prep_time="30 minutes",
    #     created_at=now,
    #     updated_at=now,
    #     deleted_at="",
    #     restrictions="No dairy"
    # )

    # recipe_hist_id = db.insert_recipe_history(recipe_history)

    # print("\nRecipe history:")
    # print(db.read_recipe_history_by_user(user_id))

    # -------------------
    # NUTRITION HISTORY
    # -------------------
    # nutrition_history = NutritionHistory(
    #     user_id=user_id,
    #     recipe_id=123,
    #     calories=520.5,
    #     protein=45.2,
    #     fat=18.3,
    #     carbohydrates=40.6,
    #     fiber=5.1,
    #     sugar=3.4,
    #     sodium=230.0,
    #     created_at=now,
    #     updated_at=now,
    #     deleted_at=""
    # )

    # nutrition_hist_id = db.insert_nutrition_history(nutrition_history)

    # print("\nNutrition history:")
    # print(db.read_nutrition_history_by_user(user_id))

    # -------------------
    # SOFT DELETE USER
    # -------------------
    db.soft_delete_user(user_id)

    print("\nAfter soft delete:")
    print(db.read_users())