"""Authentication routes."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from forms.auth_forms import LoginForm, RegistrationForm
from services.auth_service import (
    authenticate_user,
    get_user_by_email,
    get_user_by_username,
    register_user,
)

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = authenticate_user(form.username.data, form.password.data)
        if user is not None:
            login_user(user, remember=form.remember.data)
            next_page = request.args.get("next", "")
            if next_page.startswith("/") and not next_page.startswith("//"):
                return redirect(next_page)
            return redirect(url_for("main.index"))
        flash("Invalid username/email or password.", "danger")

    return render_template("auth/login.html", form=form)


@auth.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.index"))

    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        email = form.email.data.strip().lower()

        if get_user_by_username(username):
            form.username.errors.append("That username is already in use.")
        elif get_user_by_email(email):
            form.email.errors.append("That email address is already registered.")
        else:
            user = register_user(username, email, form.password.data)
            if user is not None:
                flash("Registration successful. You can now sign in.", "success")
                return redirect(url_for("auth.login"))
            flash("Registration could not be completed.", "danger")

    return render_template("auth/register.html", form=form)


@auth.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.index"))
