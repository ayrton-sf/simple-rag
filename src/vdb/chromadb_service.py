import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..config import Config

class ChromaDBService:
    def __init__(self, config: Config):
        """Initialize persistent ChromaDB client and main collection."""
        self.config: Config = config
        self.persist_directory = Path(self.config.chroma_db_dir)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        self.collection: Collection = self.client.get_or_create_collection(
            name="main",
            metadata={"description": "Main collection"}
        )

    def upsert(self, id: str, embed: List[float], content: str, category: str) -> None:
        """Insert or update a document with embedding and category."""
        self.collection.upsert(
            embeddings=[embed],
            ids=[id],
            documents=[content],
            metadatas=[{"category": category}]
        )
    
    def query(
        self,
        query_embed: List[float],
        top_k: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query documents by embedding, optionally filtering by category."""
        if top_k is None:
            top_k = self.config.top_k

        where = {"category": category} if category else None

        results = self.collection.query(
            query_embeddings=[query_embed],
            n_results=top_k,
            where=where,
        )
        
        parsed_results = []
        for i in range(len(results["ids"][0])):
            parsed_result = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "category": results["metadatas"][0][i]["category"]
            }
            parsed_results.append(parsed_result)
        

        return parsed_results

    def delete_documents(self, ids: List[str] = None, category: Optional[str] = None) -> None:
        """Delete by IDs, category, or all documents."""
        if ids is not None:
            self.collection.delete(ids=ids)
        elif category is not None:
            self.collection.delete(where={"category": category})
        else:
            self.collection.delete()
            
    def get_documents(self) -> Dict[str, Any]:
        """Return raw collection data."""
        return self.collection.get()