from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from .nodes import build_embed_query_func, build_generate_res_func, build_retrieve_func
from .state import RAGState
from ..ai.embeddings.embedding_service import EmbeddingService
from ..vdb.chromadb_service import ChromaDBService
from ..ai.llm.llm_service import LLMService
from ..config import Config


class RAGGraph():
    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        db_service: ChromaDBService,
        config: Config
    ):
        self.config: Config = config
        self.embedding_service: EmbeddingService = embedding_service
        self.llm_service: LLMService = llm_service
        self.db_service: ChromaDBService = db_service
        self.checkpointer = MemorySaver()

    def _build_rag_graph(self):
        workflow = StateGraph(state_schema=RAGState)
        workflow.add_edge(START, "embed")
        workflow.add_node("embed", build_embed_query_func(self.embedding_service))
        workflow.add_edge("embed", "retrieve")
        workflow.add_node("retrieve", build_retrieve_func(self.db_service))
        workflow.add_edge("retrieve", "generate")
        workflow.add_node("generate", build_generate_res_func(self.llm_service))

        return workflow.compile(checkpointer=self.checkpointer)
    
    def _build_retriever_graph(self):
        workflow = StateGraph(state_schema=RAGState)
        workflow.add_edge(START, "embed")
        workflow.add_node("embed", build_embed_query_func(self.embedding_service))
        workflow.add_edge("embed", "retrieve")
        workflow.add_node("retrieve", build_retrieve_func(self.db_service))

        return workflow.compile()

    def retrieve(self, query: str, top_k=None, category=None) -> str:
        retrieval_workflow = self._build_retriever_graph()
        query_as_msg = HumanMessage(query)
        if top_k is None:
            top_k = self.config.top_k
        result = retrieval_workflow.invoke({"messages": [query_as_msg], "top_k": top_k, "category": category})

        return result["retrieved"]


    def run(self, convo_id: str, query: str) -> str:
        query_as_msg = HumanMessage(query)
        config = {"configurable": {"thread_id": convo_id}}
        rag_workflow = self._build_rag_graph()
        
        initial_state = {
            "messages": [query_as_msg],
            "top_k": self.config.top_k,
            "category": None
        }
        
        result = rag_workflow.invoke(initial_state, config)
        response = result["messages"][-1]

        return response.content