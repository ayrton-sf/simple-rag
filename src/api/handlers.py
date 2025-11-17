from flask import request, jsonify, current_app
from typing import Tuple, Dict, Any

from .session import SessionManager
from ..rag.graph import RAGGraph

"""HTTP handlers for the RAG API endpoints."""

JsonResponse = Tuple[Dict[str, Any], int]


def handle_query() -> JsonResponse:
    """Handle a user query: run RAG, manage session cookies, and return the reply."""
    rag_graph: RAGGraph = current_app.config["rag_graph"]
    session_manager: SessionManager = current_app.config["session_manager"]
    
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing required parameter 'q'"}), 400
        

    cookie_session_id = request.cookies.get("session-id")
    session_id = session_manager.resolve(cookie_session_id)
    
    response = rag_graph.run(
        convo_id=session_id,
        query=query,
    )

    resp = jsonify({
        "response": response,
    })

    if cookie_session_id is None:
        resp.set_cookie("session-id", session_id, httponly=True, samesite="Lax")
    return resp, 200
        

def handle_search() -> JsonResponse:
    """Search documents using the RAG retriever and return found results."""
    rag_graph: RAGGraph = current_app.config["rag_graph"]

    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing required parameter 'q'"}), 400
        
    top_k = request.args.get("n_results", type=int)
    category = request.args.get("category")
    
    try:
        results = rag_graph.retrieve(query, top_k, category)
        
        return jsonify({
            "results": results,
        }), 200
        
    except Exception as e:
        print(f"Error in handle_search: {str(e)}")
        return jsonify({
            "error": "Failed to search documents",
            "details": str(e)
        }), 500
