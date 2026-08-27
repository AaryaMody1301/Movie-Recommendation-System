"""Blueprint registration for the web application."""


def register_blueprints(app):
    """Register the stable Phase 2 web surface on the Flask application."""
    # Keep imports local so unfinished Phase 3 user/recommendation modules do not
    # break application startup merely by importing the blueprints package.
    from blueprints.auth import auth
    from blueprints.main import main
    from blueprints.movies import movies

    app.register_blueprint(main)
    app.register_blueprint(movies)
    app.register_blueprint(auth)
