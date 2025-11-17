from typing import Any
from ..ai.embeddings.embedding_service import EmbeddingService
from ..ai.llm.llm_service import LLMService
from .state import RAGState
from ..vdb.chromadb_service import ChromaDBService


def build_embed_query_func(embedding_service: EmbeddingService):
    """Builds the query processing function"""

    def embed_query(state: RAGState):
        return {"query_embed": embedding_service.embed(state["messages"][-1].content)}

    return embed_query

def build_retrieve_func(db_service: ChromaDBService):
    """Builds the retrieval function"""

    def retrieve(state: RAGState):
        retrieved = db_service.query(state["query_embed"], state["top_k"], state["category"])
        return {"retrieved": retrieved}

    return retrieve 

def build_generate_res_func(llm_service: LLMService) -> dict[str, Any]:
    """Builds the response generator function"""

    def generate_response(state: RAGState):
        response = llm_service.rag_response(
            messages=state["messages"],
            retrieved=[],    
        )
        return {"messages": [response]}

    return generate_response