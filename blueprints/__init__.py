"""Blueprint registration for the web application."""


def register_blueprints(app):
    """Register the canonical application blueprints."""
    # Imports remain local so extension/model setup happens before routes are loaded.
    from blueprints.auth import auth
    from blueprints.main import main
    from blueprints.movies import movies
    from blueprints.recommendations import recommendations
    from blueprints.user import user

    app.register_blueprint(main)
    app.register_blueprint(movies)
    app.register_blueprint(auth)
    app.register_blueprint(user)
    app.register_blueprint(recommendations)
