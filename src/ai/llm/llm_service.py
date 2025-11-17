from typing import List
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ...config import Config
from ..providers import ModelProvider

class PromptConfig:
    """Container for prompt-related defaults."""

    DEFAULT_ASSISTANT = "./prompts/rag_response.txt"


class LLMService:
    def __init__(self, config: Config):
        """Initialize the LLM service with configuration and client selection."""
        self.config: Config = config
        self.model = self._get_llm_client(self.config.llm_provider)

    def _load_prompt(self, prompt_path: str) -> str:
        """Load and return a prompt template from disk."""

        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
        
    def _get_llm_client(self, provider: ModelProvider) -> BaseChatModel:
        """Create and return a chat model client based on the configured provider."""

        model_factory = {
            ModelProvider.ANTHROPIC: lambda: ChatAnthropic(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                max_tokens=4096,
                max_retries=3
            ),
            ModelProvider.OPEN_AI: lambda: ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                max_retries=3,
                timeout=None,
                stop=None,
            )
        }.get(provider)

        return model_factory()
    
    def rag_response(self, messages: List[BaseMessage], retrieved: List[str]) -> AIMessage:
        """Generate an AI response using RAG-style prompt templates and retrieved context."""

        formatted_messages = self.build_rag_templates(messages, retrieved)
        response = self.model.invoke(formatted_messages)
        return response

    def build_rag_templates(self, messages: List[BaseMessage], retrieved: List[str]):
        """Build chat messages by combining the conversation with retrieved context templates."""

        prompt_path = PromptConfig.DEFAULT_ASSISTANT

        if self.config.system_prompt:
            prompt_path = self.config.system_prompt

        rag_prompt = self._load_prompt(prompt_path)
        prompt_w_templates = rag_prompt + "\n" "{messages}" + "\n" + "{retrieved}"
        prompt_template = ChatPromptTemplate.from_messages([
            ("human",prompt_w_templates),
        ])
        
        retrieved_formatted = prompt_template.format_messages(
            messages=messages,
            retrieved=retrieved
        )    

        formatted = messages + retrieved_formatted

        return formatted
