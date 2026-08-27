"""Authentication and user-account service functions."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from database.db import db
from database.models import User

logger = logging.getLogger(__name__)

# Backwards-compatible name used by older imports. Flask-Login can use User directly.
UserAuth = User


def register_user(username: str, email: str, password: str) -> Optional[User]:
    """Create a new application user and return it on success."""
    username = (username or "").strip()
    email = (email or "").strip().lower()
    if not username or not email or not password:
        return None

    if get_user_by_username(username) or get_user_by_email(email):
        return None

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)

    try:
        db.session.commit()
        return user
    except IntegrityError:
        db.session.rollback()
        logger.info("Registration rejected because username or email already exists")
        return None
    except Exception:
        db.session.rollback()
        logger.exception("Unexpected error while registering user")
        return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate by username or email and update the last-login timestamp."""
    identifier = (username or "").strip()
    if not identifier or not password:
        return None

    statement = db.select(User).where(
        or_(User.username == identifier, User.email == identifier.lower())
    )
    user = db.session.execute(statement).scalar_one_or_none()

    if user is None or not user.check_password(password):
        return None

    user.last_login = datetime.now(timezone.utc)
    db.session.commit()
    return user


def get_user_by_id(user_id: int) -> Optional[User]:
    """Return an application user by primary key."""
    return db.session.get(User, user_id)


def get_user_by_username(username: str) -> Optional[User]:
    """Return an application user by username."""
    if not username:
        return None
    statement = db.select(User).where(User.username == username.strip())
    return db.session.execute(statement).scalar_one_or_none()


def get_user_by_email(email: str) -> Optional[User]:
    """Return an application user by normalized email address."""
    if not email:
        return None
    statement = db.select(User).where(User.email == email.strip().lower())
    return db.session.execute(statement).scalar_one_or_none()


def update_user(user_id: int, user_data: Dict[str, Any]) -> bool:
    """Update the mutable profile fields of an application user."""
    user = get_user_by_id(user_id)
    if user is None:
        return False

    if "username" in user_data and user_data["username"]:
        user.username = str(user_data["username"]).strip()
    if "email" in user_data and user_data["email"]:
        user.email = str(user_data["email"]).strip().lower()

    try:
        db.session.commit()
        return True
    except IntegrityError:
        db.session.rollback()
        return False
    except Exception:
        db.session.rollback()
        logger.exception("Unexpected error while updating user %s", user_id)
        return False


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    """Replace a user's password after verifying the current password."""
    user = get_user_by_id(user_id)
    if user is None or not user.check_password(current_password) or not new_password:
        return False

    user.set_password(new_password)
    db.session.commit()
    return True


def generate_reset_token(email: str) -> Optional[str]:
    """Password-reset persistence is deferred until a migration system exists."""
    logger.warning("Password reset tokens are not enabled yet")
    return None


def is_token_valid(token: str) -> bool:
    """Password-reset persistence is deferred until a migration system exists."""
    return False


def update_last_login(user_id: int) -> bool:
    """Update a user's last-login timestamp."""
    user = get_user_by_id(user_id)
    if user is None:
        return False
    user.last_login = datetime.now(timezone.utc)
    db.session.commit()
    return True
