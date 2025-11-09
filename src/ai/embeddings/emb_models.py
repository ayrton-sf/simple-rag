from enum import Enum
from ..providers import BaseModelEnum
from ..providers import ModelProvider

class EmbeddingModel(BaseModelEnum):
    VOYAGE_CONTEXT_3 = ("voyage-context-3", ModelProvider.VOYAGE_AI)
    VOYAGE_3_LARGE = ("voyage-3-large", ModelProvider.VOYAGE_AI)
    VOYAGE_3_5 = ("voyage-3.5", ModelProvider.VOYAGE_AI)
    VOYAGE_3_5_LITE = ("voyage-3.5-lite", ModelProvider.VOYAGE_AI)
    VOYAGE_CODE_3 = ("voyage-code-3", ModelProvider.VOYAGE_AI)
    TEXT_EMBEDDING_3_SMALL = ("text-embedding-3-small", ModelProvider.OPEN_AI)
    TEXT_EMBEDDING_3_LARGE = ("text-embedding-3-large", ModelProvider.OPEN_AI)
    TEXT_EMBEDDING_ADA_2 = ("text-embedding-ada-002", ModelProvider.OPEN_AI)