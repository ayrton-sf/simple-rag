from typing import List, NotRequired, Annotated, List, Optional
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict

class RAGState(TypedDict):
    query_embed: NotRequired[List[float]]
    retrieved: NotRequired[List[str]]
    messages:Annotated[List[AnyMessage],add_messages]
    top_k: Optional[int]
    category: Optional[int]