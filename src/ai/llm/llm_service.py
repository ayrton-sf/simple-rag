import os
from typing import Any, List
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, MessagesState, StateGraph
from ...config import Config
from ..providers import ModelProvider
from langgraph.checkpoint.memory import MemorySaver

class PromptConfig:
    RAG_RESPONSE = "prompts/rag_response.txt"


class LLMService:
    def __init__(self, config: Config):
        self.config: Config = config
        self.model = self._get_llm_client(self.config.llm_provider)

    def _load_prompt(self, prompt_path: str) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        full_path = os.path.join(project_root, prompt_path)
        with open(full_path, "r", encoding="utf-8") as file:
            return file.read().strip()
        
    def _get_llm_client(self, provider: ModelProvider) -> BaseChatModel:
        model_factory = {
            ModelProvider.ANTHROPIC: lambda: ChatAnthropic(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                max_tokens=32000,
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
        formatted_messages = self.build_rag_templates(messages, retrieved)
        response = self.model.invoke(formatted_messages)
        return response

    def build_rag_templates(self, messages: List[BaseMessage], retrieved: List[str]):
        rag_prompt = self._load_prompt(PromptConfig.RAG_RESPONSE)

        prompt_template = ChatPromptTemplate.from_messages([
            ("human", rag_prompt),
        ])
        
        retrieved_formatted = prompt_template.format_messages(
            messages=messages,
            retrieved=retrieved
        )    

        formatted = messages + retrieved_formatted

        return formatted
