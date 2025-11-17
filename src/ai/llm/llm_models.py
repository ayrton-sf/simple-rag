from ..providers import BaseModelEnum, ModelProvider


class LLModel(BaseModelEnum):
    GPT_4_O = ("gpt-4o", ModelProvider.OPEN_AI)
    GPT_4_1 = ("gpt-4.1", ModelProvider.OPEN_AI)
    O3 = ("o3", ModelProvider.OPEN_AI)
    O4_MINI = ("o4-mini", ModelProvider.OPEN_AI)
    CLAUDE_HAIKU_4_5 = ("claude-haiku-4-5-20251001", ModelProvider.ANTHROPIC)
    CLAUDE_SONNET_3_5 = ("claude-3-5-sonnet-latest", ModelProvider.ANTHROPIC)
    CLAUDE_SONNET_3_7 = ("claude-3-7-sonnet-latest", ModelProvider.ANTHROPIC)
    CLAUDE_SONNET_4 = ("claude-sonnet-4-20250514", ModelProvider.ANTHROPIC)
    CLAUDE_SONNET_4_5 = ("claude-sonnet-4-5-20250929", ModelProvider.ANTHROPIC)
    #Older model for testing purposes
    CLAUDE_HAIKU_3 = ("claude-3-haiku-20240307", ModelProvider.ANTHROPIC)