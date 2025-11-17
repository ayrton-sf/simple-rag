import os
from .ai.embeddings.emb_models import EmbeddingModel
from .ai.llm.llm_models import LLModel


class Config:
    def __init__(self):
        self.llm_provider = None
        self.llm_api_key = None
        self.embed_provider = None
        self.embed_api_key = None
        self.llm_model = self._determine_env_var("LLM_MODEL")
        self.embeddings_model = self._determine_env_var("EMBEDDINGS_MODEL")
        self.chroma_db_dir = self._determine_env_var("CHROMA_DB_DIR")
        self.system_prompt = os.getenv("SYSTEM_PROMPT", None)
        self._set_llm_provider()
        self._set_embeddings_provider()
        self.top_k = 5

    def _set_llm_provider(self):
        llm_model = self.llm_model
        self.llm_provider = LLModel.provider_from_str(llm_model)
        self.llm_api_key = self._determine_env_var(self.llm_provider.value)

    def _set_embeddings_provider(self):
        self.embeddings_provider = EmbeddingModel.provider_from_str(self.embeddings_model)
        self.embeddings_api_key = self._determine_env_var(self.embeddings_provider.value)

    def _determine_env_var(self, var_key: str) -> str:
        var_value = os.getenv(var_key)
        if var_value is None:
            raise ValueError(f"Missing {var_key} in .env")
        if var_value == "":
            raise ValueError(f"{var_key} .env var is empty")
        
        return var_value