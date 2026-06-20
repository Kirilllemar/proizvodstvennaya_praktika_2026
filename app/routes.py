import os
import uuid

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from app.config import ALLOWED_EXTENSIONS, UPLOAD_DIR
from app.database import get_history, get_statistics, save_prediction
from app.models import compare_all_models, load_confusion_matrix, load_experiments, predict
from app.reports import generate_excel, generate_pdf, save_history_json

bp = Blueprint("main", __name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/")
def index():
    experiments = load_experiments()
    stats = get_statistics()
    best = next(a for a in experiments["architectures"] if a["id"] == experiments["best_model"])
    return render_template(
        "index.html",
        experiments=experiments,
        stats=stats,
        best_model=best,
    )


@bp.route("/recognize", methods=["GET", "POST"])
def recognize():
    experiments = load_experiments()
    result = None
    compare_result = None
    uploaded_name = None

    if request.method == "POST":
        model_id = request.form.get("model_id", experiments["best_model"])
        compare_all = request.form.get("compare_all") == "on"
        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Выберите изображение", "error")
            return redirect(url_for("main.recognize"))

        if not allowed_file(file.filename):
            flash("Формат: PNG, JPG, JPEG, WEBP, GIF", "error")
            return redirect(url_for("main.recognize"))

        ext = file.filename.rsplit(".", 1)[1].lower()
        uploaded_name = f"{uuid.uuid4().hex[:12]}.{ext}"
        path = os.path.join(UPLOAD_DIR, uploaded_name)
        file.save(path)

        if compare_all:
            compare_result = compare_all_models(path)
            top3 = compare_result[0]["top3"]
            ms = compare_result[0]["inference_ms"]
            save_prediction(uploaded_name, "all_models", top3, ms)
        else:
            top3, ms = predict(path, model_id)
            save_prediction(uploaded_name, model_id, top3, ms)
            result = {"top3": top3, "inference_ms": ms, "model_id": model_id}

    return render_template(
        "recognize.html",
        experiments=experiments,
        result=result,
        compare_result=compare_result,
        uploaded_name=uploaded_name,
    )


@bp.route("/compare")
def compare():
    experiments = load_experiments()
    return render_template("compare.html", experiments=experiments)


@bp.route("/confusion")
def confusion():
    data = load_confusion_matrix()
    return render_template("confusion.html", data=data)


@bp.route("/history")
def history():
    items = get_history()
    stats = get_statistics()
    return render_template("history.html", items=items, stats=stats)


@bp.route("/about")
def about():
    experiments = load_experiments()
    return render_template("about.html", experiments=experiments)


@bp.route("/export/excel")
def export_excel():
    buffer = generate_excel()
    return send_file(
        buffer,
        as_attachment=True,
        download_name="food_recognition_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@bp.route("/export/pdf")
def export_pdf():
    buffer = generate_pdf()
    return send_file(
        buffer,
        as_attachment=True,
        download_name="food_recognition_report.pdf",
        mimetype="application/pdf",
    )


@bp.route("/export/json")
def export_json():
    path = save_history_json()
    return send_file(path, as_attachment=True, download_name="history.json")
