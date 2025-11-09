from flask import Blueprint
from .handlers import handle_query, handle_search

router = Blueprint("api", __name__, url_prefix="/api/v1")

router.route("/query", methods=["GET"])(handle_query)
router.route("/search", methods=["GET"])(handle_search)
