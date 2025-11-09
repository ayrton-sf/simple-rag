import pandas as pd
from pathlib import Path
from typing import Optional
import hashlib
from ..vdb.chromadb_service import ChromaDBService
from ..ai.embeddings.embedding_service import EmbeddingService


class DocumentLoader: 
    def __init__(self, embedding_service: EmbeddingService, db_service: ChromaDBService):
        self.embedding_service = embedding_service
        self.db_service = db_service

    def load_documents(self, file_path: str, category: str) -> None:
        file_path = str(Path(file_path).resolve())
        
        if not Path(file_path).exists():
            raise ValueError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            self._load_csv(file_path, category)
        elif file_extension == '.txt':
            self._load_txt(file_path, category)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _load_csv(self, file_path: str, category: str) -> None:
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        print(f"Found {len(df)} documents to process")

        for idx, row in df.iterrows():

            row_dict = row.to_dict()
            if 'id' in row_dict:
                doc_id = str(row_dict.pop('id'))
            else:
                doc_id = None

            content_parts = [f"{{{key}: {value}}}" for key, value in row_dict.items()]
            content = ", ".join(content_parts)
            
            self._upsert_doc(content, category, doc_id)
            print(f"Successfully processed document {idx + 1}/{len(df)}")
        
        print(f"Finished processing documents for category '{category}'")

    def _load_txt(self, file_path: str, category: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("File is empty")
        
        self._process_document(content, category)
        print(f"Successfully processed file: {file_path}")

    def _validate_csv(self, df: pd.DataFrame) -> None:
        if 'id' not in df.columns:
            raise ValueError("CSV must have an 'id' column")
        
        if df['id'].isna().all():
            raise ValueError("No valid IDs found in CSV")

    def _upsert_doc(self, content: str, category: str, doc_id: Optional[str] = None) -> None:
        if not doc_id:
            doc_id = self._generate_document_id(content, category)
        
        embedding = self.embedding_service.embed(content)
        
        self.db_service.upsert(
            id=doc_id,
            embed=embedding,
            content=content,
            category=category
        )

    def _generate_document_id(self, content: str, category: str) -> str:
        hash_input = f"{content}{category}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def clear_documents(self, category: Optional[str] = None) -> None:
        if category:
            print(f"Clearing documents in category: {category}")
            self.db_service.delete_documents(category=category)
        else:
            print("Clearing all documents from storage")
            self.db_service.delete_documents()
            
    def list_categories(self) -> None:
        categories = {}
        collection_data = self.db_service.get_documents()
        
        if not collection_data or 'metadatas' not in collection_data or not collection_data['metadatas']:
            print("No documents found in storage")
            return
            
        for metadata in collection_data['metadatas']:
            if isinstance(metadata, dict) and 'category' in metadata:
                category = metadata['category']
                categories[category] = categories.get(category, 0) + 1
                
        if not categories:
            print("No categories found")
            return
            
        print("\nDocuments info:")
        print("-" * 50)
        print(f"{'Category':<30} | {'Document Count':>15}")
        print("-" * 50)
        
        for category, count in sorted(categories.items()):
            print(f"{category:<30} | {count:>15}")
        print("-" * 50)