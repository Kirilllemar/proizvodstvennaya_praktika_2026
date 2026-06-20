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

from app.config import ALLOWED_EXTENSIONS, FOOD_CLASSES_CATALOG, UPLOAD_DIR
from app.database import clear_history, delete_prediction, get_history, get_statistics, save_prediction
from app.models import (
    compare_all_models,
    get_model_name,
    load_confusion_matrix,
    load_experiments,
    predict,
    predict_multi_dishes,
)
from app.reports import generate_excel, generate_pdf, save_history_json

bp = Blueprint("main", __name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _remove_upload(filename):
    if not filename:
        return
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)


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
    multi_result = None
    uploaded_name = None

    if request.method == "POST":
        model_id = request.form.get("model_id", experiments["best_model"])
        compare_all = request.form.get("compare_all") == "on"
        multi_dishes = request.form.get("multi_dishes") == "on"
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
            result = {
                "top3": top3,
                "inference_ms": ms,
                "model_id": model_id,
                "model_name": get_model_name(model_id),
            }
            if multi_dishes:
                dishes, multi_ms = predict_multi_dishes(path, model_id)
                multi_result = {
                    "dishes": dishes,
                    "inference_ms": multi_ms,
                    "model_name": get_model_name(model_id),
                }

    return render_template(
        "recognize.html",
        experiments=experiments,
        result=result,
        compare_result=compare_result,
        multi_result=multi_result,
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


@bp.route("/history/delete/<int:record_id>", methods=["POST"])
def history_delete(record_id):
    deleted, filename = delete_prediction(record_id)
    if not deleted:
        flash("Запись не найдена", "error")
    else:
        _remove_upload(filename)
        flash("Запись удалена", "success")
    return redirect(url_for("main.history"))


@bp.route("/history/clear", methods=["POST"])
def history_clear():
    filenames = clear_history()
    for filename in filenames:
        _remove_upload(filename)
    flash("История очищена", "success")
    return redirect(url_for("main.history"))


@bp.route("/about")
def about():
    experiments = load_experiments()
    return render_template(
        "about.html",
        experiments=experiments,
        food_classes=FOOD_CLASSES_CATALOG,
    )


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
