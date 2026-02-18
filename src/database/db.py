"""
SQLite database handler for the nutrition-ai-assistant.

Provides UserDBHandler with CRUD methods for all 6 tables:
users, medical_advice, authentication, user_profile_history,
recipe_history, nutrition_history.

Uses a context-manager pattern for connection lifecycle to
prevent leaked connections and ensure commits/rollbacks.
"""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Generator, List, Optional

from database.models import (
    User,
    MedicalAdvice,
    Authentication,
    UserProfileHistory,
    RecipeHistory,
    NutritionHistory,
)

logger = logging.getLogger(__name__)

DB_FILE = "users.db"


class UserDBHandler:
    """Central handler for all SQLite database operations.

    Usage:
        db = UserDBHandler()
        db.create_all_tables()
        user_id = db.insert_user(User(name="A", surname="B", ...))
    """

    def __init__(self, db_file: str = DB_FILE):
        """Initialize the handler with a path to the SQLite database file.

        Args:
            db_file: Path to the SQLite database file. Defaults to 'users.db'.
        """
        self.db_file = db_file

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a SQLite connection with foreign keys enabled.

        Commits on successful exit, rolls back on exception, and always
        closes the connection.

        Yields:
            sqlite3.Connection: An open database connection.

        Raises:
            sqlite3.Error: Propagated after rollback if a DB error occurs.
        """
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            logger.exception("Database operation failed, transaction rolled back.")
            raise
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Table creation
    # ----------------------------------------------------------------

    def create_all_tables(self) -> None:
        """Create all 6 tables if they do not already exist.

        Convenience method that calls each individual create_*_table method.
        Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
        """
        self.create_users_table()
        self.create_medical_advice_table()
        self.create_authentication_table()
        self.create_user_profile_history_table()
        self.create_recipe_history_table()
        self.create_nutrition_history_table()
        logger.info("All tables created (or already exist).")

    def create_users_table(self) -> None:
        """Create the 'users' table if it does not already exist."""
        with self._get_connection() as conn:
            conn.execute("""
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

    def create_medical_advice_table(self) -> None:
        """Create the 'medical_advice' table if it does not already exist.

        Has a foreign key reference to users(id).
        """
        # TODO: add new columns: limit, avoid, constraints - completed!
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS medical_advice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    health_condition TEXT,
                    medical_advice TEXT,
                    dietary_limit TEXT,
                    avoid TEXT,
                    dietary_constraints TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT,
                    user_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

    def create_authentication_table(self) -> None:
        """Create the 'authentication' table if it does not already exist.

        Has a UNIQUE constraint on 'login' and a foreign key to users(id).
        """
        with self._get_connection() as conn:
            conn.execute("""
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

    def create_user_profile_history_table(self) -> None:
        """Create the 'user_profile_history' table if it does not already exist.

        Has a foreign key reference to users(id).
        """
        with self._get_connection() as conn:
            conn.execute("""
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

    def create_recipe_history_table(self) -> None:
        """Create the 'recipe_history' table if it does not already exist.

        Has a foreign key reference to users(id).
        """
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recipe_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    recipe_id INTEGER,
                    recipe_name TEXT,
                    cook_instructions TEXT,
                    servings INTEGER,
                    ingredients TEXT,
                    prep_time TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    deleted_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

    def create_nutrition_history_table(self) -> None:
        """Create the 'nutrition_history' table if it does not already exist.

        Has a foreign key reference to users(id).
        """
        with self._get_connection() as conn:
            conn.execute("""
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

    # ----------------------------------------------------------------
    # Users CRUD
    # ----------------------------------------------------------------

    def insert_user(self, user: User) -> int:
        """Insert a new user and return the auto-generated row ID.

        Args:
            user: A User dataclass instance. The 'id' field is ignored.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            user_dict = asdict(user)
            user_dict.pop("id", None)
            columns = ", ".join(user_dict.keys())
            placeholders = ", ".join(["?"] * len(user_dict))
            logger.info("Inserting user: %s", user_dict)
            cursor = conn.execute(
                f"INSERT INTO users ({columns}) VALUES ({placeholders})",
                tuple(user_dict.values()),
            )
            return cursor.lastrowid

    def read_users(self) -> List[tuple]:
        """Return all rows from the users table as a list of tuples."""
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM users").fetchall()

    def read_user(self, name: str, surname: str) -> Optional[tuple]:
        """Find the most recently updated user matching name and surname.

        Args:
            name:    User's first name (exact match).
            surname: User's last name (exact match).

        Returns:
            A single row tuple, or None if no match is found.
        """
        with self._get_connection() as conn:
            return conn.execute(
                """SELECT * FROM users
                   WHERE name = ? AND surname = ?
                   ORDER BY updated_at DESC LIMIT 1""",
                (name, surname),
            ).fetchone()

    def update_user(self, user_id: int, field: str, new_value) -> None:
        """Update one field on a user row and refresh its updated_at timestamp.

        Args:
            user_id:   The row ID of the user to update.
            field:     Column name to update. Must be one of: name, surname,
                       user_name, caretaker, age, gender.
            new_value: The new value. Lists are joined with ', '.

        Raises:
            ValueError: If field is not in the allowed set.
        """
        allowed_fields = {"name", "surname", "user_name", "caretaker", "age", "gender"}
        if field not in allowed_fields:
            raise ValueError(
                f"Invalid field '{field}'. Allowed: {allowed_fields}"
            )
        if isinstance(new_value, list):
            new_value = ", ".join(map(str, new_value))

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE users SET {field} = ?, updated_at = ? WHERE id = ?",
                (new_value, datetime.now().isoformat(), user_id),
            )

    def soft_delete_user(self, user_id: int) -> None:
        """Soft-delete a user by setting deleted_at to the current timestamp.

        Args:
            user_id: The row ID of the user to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE users SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), user_id),
            )

    # ----------------------------------------------------------------
    # MedicalAdvice CRUD
    # ----------------------------------------------------------------

    def insert_medical_advice(self, advice: MedicalAdvice) -> int:
        """Insert a medical advice record and return its row ID.

        If advice.medical_advice is a list, items are joined with newlines.

        Args:
            advice: A MedicalAdvice dataclass instance.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            if isinstance(advice.medical_advice, list):
                advice.medical_advice = "\n".join(advice.medical_advice)
            cursor = conn.execute(
                """INSERT INTO medical_advice
                   (health_condition, medical_advice, dietary_limit, avoid, dietary_constraints, created_at, updated_at, deleted_at, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (advice.health_condition, advice.medical_advice,
                 advice.dietary_limit, advice.avoid, advice.dietary_constraints,
                 advice.created_at, advice.updated_at, advice.deleted_at, advice.user_id),
            )
            return cursor.lastrowid

    def update_medical_advice(self, advice_id: int, new_advice: str) -> None:
        """Update the medical_advice text for a given advice record.

        Args:
            advice_id: The row ID of the advice to update.
            new_advice: The new medical advice text.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE medical_advice SET medical_advice = ?, updated_at = ? WHERE id = ?",
                (new_advice, datetime.now().isoformat(), advice_id),
            )

    def soft_delete_medical_advice(self, advice_id: int) -> None:
        """Soft-delete a medical advice record by setting deleted_at.

        Args:
            advice_id: The row ID of the advice to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE medical_advice SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), advice_id),
            )

    def get_medical_advice_by_user(self, user_id: int) -> List[tuple]:
        """Return all medical advice records for a given user.

        Args:
            user_id: The row ID of the user.

        Returns:
            A list of row tuples (may be empty).
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM medical_advice WHERE user_id = ?",
                (user_id,),
            ).fetchall()

    def read_medical_advice(self, advice_id: int) -> Optional[tuple]:
        """Retrieve a specific medical advice entry by its ID.

        Args:
            advice_id: The row ID of the advice.

        Returns:
            A single row tuple, or None if not found.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM medical_advice WHERE id = ?",
                (advice_id,),
            ).fetchone()

    def read_all_medical_advice(self) -> List[tuple]:
        """Return all medical advice entries from the medical_advice table."""
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM medical_advice").fetchall()

    # ----------------------------------------------------------------
    # Authentication CRUD
    # ----------------------------------------------------------------

    def insert_authentication(self, auth: Authentication) -> int:
        """Insert an authentication record and return its row ID.

        Args:
            auth: An Authentication dataclass instance.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            auth_dict = asdict(auth)
            auth_dict.pop("id", None)
            columns = ", ".join(auth_dict.keys())
            placeholders = ", ".join(["?"] * len(auth_dict))
            cursor = conn.execute(
                f"INSERT INTO authentication ({columns}) VALUES ({placeholders})",
                tuple(auth_dict.values()),
            )
            return cursor.lastrowid

    def read_authentication_by_id(self, auth_id: int) -> Optional[tuple]:
        """Retrieve an authentication record by its ID.

        Args:
            auth_id: The row ID of the authentication record.

        Returns:
            A single row tuple, or None if not found.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM authentication WHERE id = ?",
                (auth_id,),
            ).fetchone()

    def read_authentication_by_user(self, user_id: int) -> List[tuple]:
        """Return all authentication records for a given user.

        Args:
            user_id: The row ID of the user.

        Returns:
            A list of row tuples (may be empty).
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM authentication WHERE user_id = ?",
                (user_id,),
            ).fetchall()

    def read_authentication_by_login(self, login: str) -> Optional[tuple]:
        """Retrieve an authentication record by login identifier.

        Args:
            login: The unique login string (e.g. email).

        Returns:
            A single row tuple, or None if not found.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM authentication WHERE login = ?",
                (login,),
            ).fetchone()

    def update_authentication(self, auth_id: int, field: str, new_value) -> None:
        """Update one field on an authentication record.

        Args:
            auth_id:   The row ID of the authentication record.
            field:     Column name to update. Must be one of: login, password, role.
            new_value: The new value.

        Raises:
            ValueError: If field is not in the allowed set.
        """
        allowed_fields = {"login", "password", "role"}
        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Allowed: {allowed_fields}")

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE authentication SET {field} = ?, updated_at = ? WHERE id = ?",
                (new_value, datetime.now().isoformat(), auth_id),
            )

    def soft_delete_authentication(self, auth_id: int) -> None:
        """Soft-delete an authentication record by setting deleted_at.

        Args:
            auth_id: The row ID of the authentication record to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE authentication SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), datetime.now().isoformat(), auth_id),
            )

    # ----------------------------------------------------------------
    # UserProfileHistory CRUD
    # ----------------------------------------------------------------

    def insert_user_profile_history(self, history: UserProfileHistory) -> int:
        """Insert a user profile history snapshot and return its row ID.

        Args:
            history: A UserProfileHistory dataclass instance.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            hist_dict = asdict(history)
            hist_dict.pop("id", None)
            columns = ", ".join(hist_dict.keys())
            placeholders = ", ".join(["?"] * len(hist_dict))
            cursor = conn.execute(
                f"INSERT INTO user_profile_history ({columns}) VALUES ({placeholders})",
                tuple(hist_dict.values()),
            )
            return cursor.lastrowid

    def read_user_profile_history_by_user(self, user_id: int) -> List[tuple]:
        """Return all profile history records for a user, newest first.

        Args:
            user_id: The row ID of the user.

        Returns:
            A list of row tuples ordered by created_at DESC.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM user_profile_history WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

    def update_user_profile_history(self, history_id: int, field: str, new_value) -> None:
        """Update one field on a user profile history record.

        Args:
            history_id: The row ID of the history record.
            field:      Column name. Must be one of: preferences, health_condition, restrictions.
            new_value:  The new value.

        Raises:
            ValueError: If field is not in the allowed set.
        """
        allowed_fields = {"preferences", "health_condition", "restrictions"}
        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Allowed: {allowed_fields}")

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE user_profile_history SET {field} = ?, updated_at = ? WHERE id = ?",
                (new_value, datetime.now().isoformat(), history_id),
            )

    def soft_delete_user_profile_history(self, history_id: int) -> None:
        """Soft-delete a user profile history record.

        Args:
            history_id: The row ID of the history record to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE user_profile_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )

    # ----------------------------------------------------------------
    # RecipeHistory CRUD
    # ----------------------------------------------------------------

    def insert_recipe_history(self, history: RecipeHistory) -> int:
        """Insert a recipe history record and return its row ID.

        Args:
            history: A RecipeHistory dataclass instance.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            hist_dict = asdict(history)
            hist_dict.pop("id", None)
            columns = ", ".join(hist_dict.keys())
            placeholders = ", ".join(["?"] * len(hist_dict))
            cursor = conn.execute(
                f"INSERT INTO recipe_history ({columns}) VALUES ({placeholders})",
                tuple(hist_dict.values()),
            )
            return cursor.lastrowid

    def read_recipe_history_by_user(self, user_id: int) -> List[tuple]:
        """Return all recipe history records for a user, newest first.

        Args:
            user_id: The row ID of the user.

        Returns:
            A list of row tuples ordered by created_at DESC.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM recipe_history WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

    def update_recipe_history(self, history_id: int, field: str, new_value) -> None:
        """Update one field on a recipe history record.

        Args:
            history_id: The row ID of the recipe history record.
            field:      Column name. Must be one of: cook_instructions, servings, ingredients, prep_time, recipe_name.
            new_value:  The new value.

        Raises:
            ValueError: If field is not in the allowed set.
        """
        allowed_fields = {"cook_instructions", "servings", "ingredients", "prep_time", "recipe_name"}
        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Allowed: {allowed_fields}")

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE recipe_history SET {field} = ?, updated_at = ? WHERE id = ?",
                (new_value, datetime.now().isoformat(), history_id),
            )

    def soft_delete_recipe_history(self, history_id: int) -> None:
        """Soft-delete a recipe history record.

        Args:
            history_id: The row ID of the recipe history record to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE recipe_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )

    # ----------------------------------------------------------------
    # NutritionHistory CRUD
    # ----------------------------------------------------------------

    def insert_nutrition_history(self, history: NutritionHistory) -> int:
        """Insert a nutrition history record and return its row ID.

        Args:
            history: A NutritionHistory dataclass instance.

        Returns:
            The integer ID of the newly inserted row.
        """
        with self._get_connection() as conn:
            hist_dict = asdict(history)
            hist_dict.pop("id", None)
            columns = ", ".join(hist_dict.keys())
            placeholders = ", ".join(["?"] * len(hist_dict))
            cursor = conn.execute(
                f"INSERT INTO nutrition_history ({columns}) VALUES ({placeholders})",
                tuple(hist_dict.values()),
            )
            return cursor.lastrowid

    def read_nutrition_history_by_user(self, user_id: int) -> List[tuple]:
        """Return all nutrition history records for a user, newest first.

        Args:
            user_id: The row ID of the user.

        Returns:
            A list of row tuples ordered by created_at DESC.
        """
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT * FROM nutrition_history WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

    def update_nutrition_history(self, history_id: int, field: str, new_value) -> None:
        """Update one field on a nutrition history record.

        Args:
            history_id: The row ID of the nutrition history record.
            field:      Column name. Must be one of: calories, protein, fat,
                        carbohydrates, fiber, sugar, sodium.
            new_value:  The new value.

        Raises:
            ValueError: If field is not in the allowed set.
        """
        allowed_fields = {"calories", "protein", "fat", "carbohydrates", "fiber", "sugar", "sodium"}
        if field not in allowed_fields:
            raise ValueError(f"Invalid field '{field}'. Allowed: {allowed_fields}")

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE nutrition_history SET {field} = ?, updated_at = ? WHERE id = ?",
                (new_value, datetime.now().isoformat(), history_id),
            )

    def soft_delete_nutrition_history(self, history_id: int) -> None:
        """Soft-delete a nutrition history record.

        Args:
            history_id: The row ID of the nutrition history record to soft-delete.
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE nutrition_history SET deleted_at = ? WHERE id = ?",
                (datetime.now().isoformat(), history_id),
            )
