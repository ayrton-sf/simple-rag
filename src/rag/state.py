from typing import List, NotRequired, Annotated, List, Optional
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from operator import add
class RAGState(TypedDict):
    """In-memory State for both RAG and Retriever graphs"""
    query_embed: NotRequired[List[float]]
    retrieved: Annotated[List[str], add]
    messages:Annotated[List[AnyMessage],add_messages]
    top_k: Optional[int]
    category: Optional[int]