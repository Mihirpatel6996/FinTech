"""
User Management Module

This module provides functionality for user authentication, registration,
and profile management.
"""

import sqlite3
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple


class UserManager:
    """
    Class for managing user accounts.
    """

    def __init__(self, db_path: str = "stock_data.db"):
        """
        Initialize the user manager.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        print(f"UserManager initialized with database: {db_path}")
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database with user tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    created_date TEXT,
                    last_login TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Create sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_date TEXT,
                    expires_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

    def _hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        salt = "stocksense"  # In production, use a unique salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def register_user(self, email: str, password: str, first_name: str = "", last_name: str = "") -> Tuple[bool, str]:
        """
        Register a new user.

        Args:
            email: User's email address
            password: User's password
            first_name: User's first name (optional)
            last_name: User's last name (optional)

        Returns:
            Tuple of (success, message)
        """
        print(f"Attempting to register user with email: {email}")

        # Validate email
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            print("Registration failed: Invalid email address")
            return False, "Invalid email address"

        # Validate password
        if len(password) < 8:
            print("Registration failed: Password too short")
            return False, "Password must be at least 8 characters long"

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if user already exists
                cursor = conn.execute("SELECT user_id FROM users WHERE email = ?", (email,))
                if cursor.fetchone():
                    print(f"Registration failed: Email {email} already registered")
                    return False, "Email address already registered"

                # Hash password
                password_hash = self._hash_password(password)

                # Insert new user
                conn.execute(
                    "INSERT INTO users (email, password_hash, first_name, last_name, created_date) VALUES (?, ?, ?, ?, ?)",
                    (email, password_hash, first_name, last_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )

                print(f"User {email} registered successfully")
                return True, "User registered successfully"
        except Exception as e:
            print(f"Registration error: {str(e)}")
            return False, f"Error registering user: {str(e)}"

    def authenticate_user(self, email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Authenticate a user.

        Args:
            email: User's email address
            password: User's password

        Returns:
            Tuple of (success, user_data, message)
        """
        print(f"Attempting to authenticate user: {email}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get user
                cursor = conn.execute(
                    "SELECT user_id, email, password_hash, first_name, last_name, is_active FROM users WHERE email = ?",
                    (email,)
                )
                user = cursor.fetchone()

                if not user:
                    print(f"Authentication failed: User {email} not found")
                    return False, None, "Invalid email or password"

                # Check if user is active
                if not user['is_active']:
                    print(f"Authentication failed: Account {email} is inactive")
                    return False, None, "Account is inactive"

                # Verify password
                password_hash = self._hash_password(password)
                if password_hash != user['password_hash']:
                    print(f"Authentication failed: Invalid password for {email}")
                    return False, None, "Invalid email or password"

                # Update last login
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE user_id = ?",
                    (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user['user_id'])
                )

                # Create user data dictionary
                user_data = {
                    'user_id': user['user_id'],
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name']
                }

                print(f"User {email} authenticated successfully")
                return True, user_data, "Authentication successful"
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False, None, f"Error authenticating user: {str(e)}"

    def create_session(self, user_id: int, expires_days: int = 30) -> Optional[str]:
        """
        Create a new session for a user.

        Args:
            user_id: User ID
            expires_days: Number of days until session expires

        Returns:
            Session ID or None if error
        """
        try:
            # Generate session ID
            session_id = secrets.token_hex(32)

            # Calculate expiration date
            created_date = datetime.now()
            expires_date = created_date + timedelta(days=expires_days)

            with sqlite3.connect(self.db_path) as conn:
                # Insert session
                conn.execute(
                    "INSERT INTO sessions (session_id, user_id, created_date, expires_date) VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        user_id,
                        created_date.strftime('%Y-%m-%d %H:%M:%S'),
                        expires_date.strftime('%Y-%m-%d %H:%M:%S')
                    )
                )

                return session_id
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            return None

    def validate_session(self, session_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a session and get user data.

        Args:
            session_id: Session ID

        Returns:
            Tuple of (valid, user_data)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get session
                cursor = conn.execute(
                    """
                    SELECT s.user_id, s.expires_date, u.email, u.first_name, u.last_name
                    FROM sessions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.session_id = ?
                    """,
                    (session_id,)
                )
                session = cursor.fetchone()

                if not session:
                    return False, None

                # Check if session is expired
                expires_date = datetime.strptime(session['expires_date'], '%Y-%m-%d %H:%M:%S')
                if expires_date < datetime.now():
                    # Delete expired session
                    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                    return False, None

                # Create user data dictionary
                user_data = {
                    'user_id': session['user_id'],
                    'email': session['email'],
                    'first_name': session['first_name'],
                    'last_name': session['last_name']
                }

                return True, user_data
        except Exception as e:
            print(f"Error validating session: {str(e)}")
            return False, None

    def end_session(self, session_id: str) -> bool:
        """
        End a session.

        Args:
            session_id: Session ID

        Returns:
            Success
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                return True
        except Exception as e:
            print(f"Error ending session: {str(e)}")
            return False

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user data by ID.

        Args:
            user_id: User ID

        Returns:
            User data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    "SELECT user_id, email, first_name, last_name, created_date, last_login FROM users WHERE user_id = ?",
                    (user_id,)
                )
                user = cursor.fetchone()

                if not user:
                    return None

                return dict(user)
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None

    def update_user_profile(self, user_id: int, first_name: str = None, last_name: str = None) -> Tuple[bool, str]:
        """
        Update user profile.

        Args:
            user_id: User ID
            first_name: New first name (optional)
            last_name: New last name (optional)

        Returns:
            Tuple of (success, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if user exists
                cursor = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                if not cursor.fetchone():
                    return False, "User not found"

                # Update fields
                updates = []
                params = []

                if first_name is not None:
                    updates.append("first_name = ?")
                    params.append(first_name)

                if last_name is not None:
                    updates.append("last_name = ?")
                    params.append(last_name)

                if not updates:
                    return True, "No changes to update"

                # Build and execute query
                query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"
                params.append(user_id)

                conn.execute(query, params)

                return True, "Profile updated successfully"
        except Exception as e:
            return False, f"Error updating profile: {str(e)}"

    def change_password(self, user_id: int, current_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            Tuple of (success, message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current password hash
                cursor = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,))
                user = cursor.fetchone()

                if not user:
                    return False, "User not found"

                # Verify current password
                current_hash = self._hash_password(current_password)
                if current_hash != user[0]:
                    return False, "Current password is incorrect"

                # Validate new password
                if len(new_password) < 8:
                    return False, "New password must be at least 8 characters long"

                # Update password
                new_hash = self._hash_password(new_password)
                conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (new_hash, user_id))

                return True, "Password changed successfully"
        except Exception as e:
            return False, f"Error changing password: {str(e)}"
