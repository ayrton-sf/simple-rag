from enum import Enum

class ModelProvider(Enum):
    OPEN_AI = "OPENAI_API_KEY"
    ANTHROPIC = "ANTHROPIC_API_KEY"
    VOYAGE_AI = "VOYAGE_AI_API_KEY"

class BaseModelEnum(Enum):
    def __new__(cls, value, provider: ModelProvider):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.provider = provider
        return obj
    
    @classmethod
    def provider_from_str(cls, model_name: str) -> ModelProvider:
        for m in cls:
            if m.value == model_name:
                return m.provider
        raise ValueError(f"Unknown LLM model: {model_name}")