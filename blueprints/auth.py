"""
Authentication blueprint for user login and registration.
"""
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from services.auth_service import register_user, authenticate_user, get_user_by_username
from database.models import User
from forms.auth_forms import LoginForm, RegistrationForm

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    # Redirect if user is already authenticated
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = LoginForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        remember = form.remember.data
        
        # Authenticate user
        user = authenticate_user(username, password)
        
        if user:
            # Log in user
            login_user(user, remember=remember)
            
            # Redirect to the page the user was trying to access
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            else:
                return redirect(url_for('user.profile'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('auth/login.html', form=form)


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    # Redirect if user is already authenticated
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        
        # Check if username already exists
        if get_user_by_username(username):
            flash('Username already exists', 'danger')
            return render_template('auth/register.html', form=form)
        
        # Register user
        user = register_user(username, email, password)
        
        if user:
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Registration failed', 'danger')
    
    return render_template('auth/register.html', form=form)


@auth.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('main.index')) 