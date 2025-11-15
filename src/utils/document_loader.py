import pandas as pd
from pathlib import Path
from typing import Optional
import hashlib
import json
from ..vdb.chromadb_service import ChromaDBService
from ..ai.embeddings.embedding_service import EmbeddingService


class DocumentLoader:
    """
    Loads structured documents from CSV or JSONL files, embeds them,
    and upserts them into a ChromaDB collection. Supports cleanup and
    category listing.
    """
    def __init__(self, embedding_service: EmbeddingService, db_service: ChromaDBService):
        self.embedding_service = embedding_service
        self.db_service = db_service

    def load_documents(self, file_path: str, category: str) -> None:
        """
        Detect the file type and load its contents into the vector database.
        Supports CSV and JSONL/NDJSON formats.
        """
        file_path = str(Path(file_path).resolve())
        
        if not Path(file_path).exists():
            raise ValueError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.csv':
            self._load_csv(file_path, category)
        elif file_extension in ('.jsonl', '.ndjson'):
            self._load_jsonl(file_path, category)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported: .csv, .jsonl, .ndjson")

    def _load_csv(self, file_path: str, category: str) -> None:
        """
        Load records from a CSV file, generate content strings, embed them,
        and upsert into the vector DB. Each row becomes one document.
        """
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        print(f"Found {len(df)} documents to process")

        for _, row in df.iterrows():

            row_dict = row.to_dict()
            if 'id' in row_dict:
                doc_id = str(row_dict.pop('id'))
            else:
                doc_id = None

            content_parts = [f"{{{key}: {value}}}" for key, value in row_dict.items()]
            content = ", ".join(content_parts)
            
            self._upsert_doc(content, category, doc_id)
        
        print(f"Finished processing documents for category '{category}'")

    def _validate_csv(self, df: pd.DataFrame) -> None:
        """Ensure CSV includes a usable 'id' column."""
        if 'id' not in df.columns:
            raise ValueError("CSV must have an 'id' column")
        
        if df['id'].isna().all():
            raise ValueError("No valid IDs found in CSV")

    def _upsert_doc(self, content: str, category: str, doc_id: Optional[str] = None) -> None:
        """Embed document content and upsert it into ChromaDB."""
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
        """Create a deterministic MD5 hash ID based on content + category."""
        hash_input = f"{content}{category}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def _load_jsonl(self, file_path: str, category: str) -> None:
        """Load newline-delimited JSON where each record must contain an 'id' field.

        Each record may provide a `content` or `text` field; otherwise the loader will
        serialize remaining fields into a string to be embedded.
        """

        processed = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}")

                if not isinstance(record, dict):
                    raise ValueError(f"Each JSONL line must be an object (line {line_no})")

                if 'id' not in record:
                    raise ValueError(f"Record on line {line_no} missing required 'id' field")

                doc_id = str(record.get('id'))

                try:
                    content = json.dumps(record, ensure_ascii=False)
                except (TypeError, ValueError):
                    content = str(record)

                self._upsert_doc(content, category, doc_id)
                processed += 1

        print(f"Finished processing {processed} records for category '{category}'")

    def clear_documents(self, category: Optional[str] = None) -> None:
        """Delete documents from a specific category or all categories."""
        if category:
            print(f"Clearing documents in category: {category}")
        else:
            print("Deleting all documents")

        self.db_service.delete_documents(category=category)

            
    def list_categories(self) -> None:
        """
        Print category-level document counts from the vector DB.
        Helps inspect what's stored.
        """
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