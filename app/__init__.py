from flask import Flask

from app.config import UPLOAD_DIR
from app.database import init_db


def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config["SECRET_KEY"] = "practice-demo-2026"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    import os

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_db()

    from app.models import get_model_name
    from app.routes import bp

    @app.template_filter("model_name")
    def model_name_filter(model_id):
        return get_model_name(model_id)

    app.register_blueprint(bp)
    return app
