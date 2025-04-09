"""
Authentication service for user management.

This module provides functions for user registration, authentication, and management.
"""
import logging
import secrets
import sqlite3
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from flask import current_app, g
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import get_db

# Setup logger
logger = logging.getLogger(__name__)


class UserAuth:
    """User authentication class that implements Flask-Login UserMixin interface."""
    
    def __init__(self, user_data: Dict[str, Any]):
        """
        Initialize a user authentication object.
        
        Args:
            user_data: Dictionary containing user data from the database
        """
        self.id = user_data['id']
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data.get('password_hash', user_data.get('password', ''))
        self.created_at = user_data['created_at']
        self.last_login = user_data['last_login']
        self.preferences = user_data.get('preferences')
    
    # For Flask-Login compatibility
    def is_authenticated(self):
        """Check if the user is authenticated."""
        return True
    
    def is_active(self):
        """Check if the user is active."""
        return True
    
    def is_anonymous(self):
        """Check if the user is anonymous."""
        return False
        
    def get_id(self):
        """Get the user ID as a string."""
        return str(self.id)


def register_user(username: str, email: str, password: str) -> Optional[UserAuth]:
    """
    Register a new user.
    
    Args:
        username: Username for the new user
        email: Email address for the new user
        password: Password for the new user
        
    Returns:
        UserAuth object if registration is successful, None otherwise
    """
    # Validate input
    if not username or not email or not password:
        logger.warning("Registration failed: Missing required fields")
        return None
    
    # Hash the password
    password_hash = generate_password_hash(password)
    
    db = get_db()
    try:
        # Check if username already exists
        cursor = db.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone() is not None:
            logger.warning(f"Registration failed: Username '{username}' already exists")
            return None
        
        # Check if email already exists
        cursor = db.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone() is not None:
            logger.warning(f"Registration failed: Email '{email}' already exists")
            return None
        
        # Insert the new user
        db.execute(
            'INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)',
            (username, email, password_hash, datetime.now().isoformat())
        )
        db.commit()
        
        # Get the user data for the newly registered user
        cursor = db.execute('SELECT * FROM users WHERE username = ?', (username,))
        user_data = dict(cursor.fetchone())
        
        logger.info(f"User '{username}' registered successfully")
        return UserAuth(user_data)
        
    except sqlite3.Error as e:
        logger.error(f"Database error during registration: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        return None


def authenticate_user(username: str, password: str) -> Optional[UserAuth]:
    """
    Authenticate a user.
    
    Args:
        username: Username or email
        password: Password
        
    Returns:
        UserAuth object if authentication is successful, None otherwise
    """
    db = get_db()
    try:
        # Check if username or email exists
        cursor = db.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, username)
        )
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"Authentication failed: User '{username}' not found")
            return None
        
        # Convert the row to a dictionary
        user_dict = dict(user_data)
        
        # Check password
        if not check_password_hash(user_dict['password'], password):
            logger.warning(f"Authentication failed: Invalid password for user '{username}'")
            return None
        
        # Update last login
        db.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now().isoformat(), user_dict['id'])
        )
        db.commit()
        
        logger.info(f"User '{username}' authenticated successfully")
        return UserAuth(user_dict)
        
    except sqlite3.Error as e:
        logger.error(f"Database error during authentication: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {str(e)}")
        return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a user by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing user data if found, None otherwise
    """
    db = get_db()
    try:
        cursor = db.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"User with ID {user_id} not found")
            return None
        
        return dict(user_data)
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving user {user_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving user {user_id}: {str(e)}")
        return None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Get a user by username.
    
    Args:
        username: Username
        
    Returns:
        Dictionary containing user data if found, None otherwise
    """
    db = get_db()
    try:
        cursor = db.execute('SELECT * FROM users WHERE username = ?', (username,))
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"User with username '{username}' not found")
            return None
        
        return dict(user_data)
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving user '{username}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving user '{username}': {str(e)}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Get a user by email.
    
    Args:
        email: Email address
        
    Returns:
        Dictionary containing user data if found, None otherwise
    """
    db = get_db()
    try:
        cursor = db.execute('SELECT * FROM users WHERE email = ?', (email,))
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"User with email '{email}' not found")
            return None
        
        return dict(user_data)
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving user with email '{email}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving user with email '{email}': {str(e)}")
        return None


def update_user(user_id: int, user_data: Dict[str, Any]) -> bool:
    """
    Update a user's information.
    
    Args:
        user_id: User ID
        user_data: Dictionary containing user data to update
        
    Returns:
        True if update is successful, False otherwise
    """
    # Ensure we're not updating sensitive fields like id or password
    safe_fields = ['username', 'email', 'preferences']
    update_data = {k: v for k, v in user_data.items() if k in safe_fields}
    
    if not update_data:
        logger.warning("No valid fields to update")
        return False
    
    db = get_db()
    try:
        # Build update query
        query = 'UPDATE users SET '
        query += ', '.join([f'{field} = ?' for field in update_data.keys()])
        query += ' WHERE id = ?'
        
        # Execute update
        db.execute(query, list(update_data.values()) + [user_id])
        db.commit()
        
        logger.info(f"User {user_id} updated successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database error updating user {user_id}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating user {user_id}: {str(e)}")
        return False


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    """
    Change a user's password.
    
    Args:
        user_id: User ID
        current_password: Current password
        new_password: New password
        
    Returns:
        True if password change is successful, False otherwise
    """
    db = get_db()
    try:
        # Get current user data
        cursor = db.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"Password change failed: User {user_id} not found")
            return False
        
        # Verify current password
        if not check_password_hash(user_data['password'], current_password):
            logger.warning(f"Password change failed: Invalid current password for user {user_id}")
            return False
        
        # Update password
        password_hash = generate_password_hash(new_password)
        db.execute(
            'UPDATE users SET password = ? WHERE id = ?',
            (password_hash, user_id)
        )
        db.commit()
        
        logger.info(f"Password changed successfully for user {user_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database error changing password for user {user_id}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error changing password for user {user_id}: {str(e)}")
        return False


def generate_reset_token(email: str) -> Optional[str]:
    """
    Generate a password reset token for a user.
    
    Args:
        email: User's email address
        
    Returns:
        Reset token if successful, None otherwise
    """
    db = get_db()
    try:
        # Check if email exists
        cursor = db.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_data = cursor.fetchone()
        
        if user_data is None:
            logger.warning(f"Reset token generation failed: Email '{email}' not found")
            return None
        
        # Generate token
        token = secrets.token_urlsafe(32)
        expires = (datetime.now() + timedelta(hours=24)).timestamp()
        
        # Store token in preferences field as JSON (in a real app, this would be a separate table)
        db.execute(
            'UPDATE users SET preferences = json_set(COALESCE(preferences, "{}"), "$.reset_token", ?) WHERE email = ?',
            (f"{token}:{expires}", email)
        )
        db.commit()
        
        logger.info(f"Reset token generated for user with email '{email}'")
        return token
        
    except sqlite3.Error as e:
        logger.error(f"Database error generating reset token for '{email}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating reset token for '{email}': {str(e)}")
        return None


def is_token_valid(token: str) -> bool:
    """
    Check if a password reset token is valid.
    
    Args:
        token: Password reset token
        
    Returns:
        True if token is valid, False otherwise
    """
    db = get_db()
    try:
        # Find user with this token
        cursor = db.execute(
            'SELECT preferences FROM users WHERE json_extract(preferences, "$.reset_token") LIKE ?',
            (f"{token}:%",)
        )
        user_data = cursor.fetchone()
        
        if user_data is None:
            return False
        
        # Parse token and expiration
        token_data = user_data['preferences'].get('reset_token', '').split(':')
        if len(token_data) != 2:
            return False
        
        stored_token, expires = token_data[0], float(token_data[1])
        
        # Check if token matches and is not expired
        return stored_token == token and time.time() < expires
        
    except (sqlite3.Error, ValueError, AttributeError) as e:
        logger.error(f"Error validating token: {str(e)}")
        return False


def update_last_login(user_id: int) -> bool:
    """
    Update a user's last login timestamp.
    
    Args:
        user_id: User ID
        
    Returns:
        True if update is successful, False otherwise
    """
    db = get_db()
    try:
        db.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now().isoformat(), user_id)
        )
        db.commit()
        
        logger.info(f"Last login updated for user {user_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database error updating last login for user {user_id}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating last login for user {user_id}: {str(e)}")
        return False 