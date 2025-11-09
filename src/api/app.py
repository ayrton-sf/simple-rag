from flask import Flask

from .session import SessionManager
from .routes import router
from ..rag.graph import RAGGraph

class RAGAPI:
    def __init__(self, rag_graph: RAGGraph, session_manager: SessionManager,):
        self.app = Flask(__name__)
        self.app.config.update({
            "rag_graph": rag_graph,
            "session_manager": session_manager
        })
        self.app.register_blueprint(router)

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False) -> None:
        self.app.run(host=host, port=port, debug=debug)
