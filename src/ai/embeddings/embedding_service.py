from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from ..providers import ModelProvider
from ...config import Config


class EmbeddingService:
    """Simple wrapper around embedding providers.

    Chooses an embeddings client based on configuration and exposes a
    convenience method to embed text into a vector.
    """
    def __init__(self, config: Config):
        """Initialize the service with configuration and prepare the client."""
        self.config: Config = config
        self.client = self._get_embeddings_client()

    def embed(self, content: str) -> list[float]:
        """Return the embedding vector for the provided text content."""

        return self.client.embed_query(content)
    
    def _get_embeddings_client(self) -> Embeddings:
        """Create and return a configured embeddings client instance."""

        client_factory = {
            ModelProvider.VOYAGE_AI: lambda: VoyageAIEmbeddings(
                voyage_api_key=self.config.embeddings_api_key, model=self.config.embeddings_model
            ),
            ModelProvider.OPEN_AI: lambda: OpenAIEmbeddings(
                model=self.config.embeddings_model
            )
        }.get(self.config.embeddings_provider)

        return client_factory()