
from dotenv import load_dotenv
from src.api.session import SessionManager
from src.rag.graph import RAGGraph
from src.config import Config
from src.ai.embeddings.embedding_service import EmbeddingService
from src.ai.llm.llm_service import LLMService
from src.vdb.chromadb_service import ChromaDBService
from src.api.app import RAGAPI
from src.cli import parse_args, handle_cli

load_dotenv()


def init_services(config: Config):
    embedding_service = EmbeddingService(config)
    llm_service = LLMService(config)
    db_service = ChromaDBService(config)
    session_manager = SessionManager()
    rag_service = RAGGraph(llm_service, embedding_service, db_service, config)
    
    return embedding_service, rag_service, db_service, session_manager


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    embedding_service, rag_graph, db_service, session_manager = init_services(config)
    if not handle_cli(args, embedding_service, db_service):
        api = RAGAPI(rag_graph, session_manager)
        api.run(host=args.host, port=args.port, debug=args.debug)
