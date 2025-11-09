import argparse
from src.utils.document_loader import DocumentLoader

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--load", nargs=2, 
                       metavar=('CSV_FILE', 'CATEGORY'),
                       help="Load documents from CSV_FILE into CATEGORY")
    parser.add_argument("--delete", 
                       help="Delete all documents in the specified category")
    parser.add_argument("--reset", action="store_true",
                       help="Reset the entire document storage")
    parser.add_argument("--list", action="store_true",
                       help="List all categories and their document counts")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="API host")
    parser.add_argument("--port", type=int, default=8000, 
                       help="API port")
    parser.add_argument("--debug", action="store_true", 
                       help="Run in debug mode")
    return parser.parse_args()


def handle_cli(args, embedding_service, db_service):
    """Handle CLI document management commands."""
    if args.load or args.delete or args.reset or args.list:
        try:
            document_loader = DocumentLoader(embedding_service, db_service)
            if args.reset:
                document_loader.clear_documents()
            elif args.load:
                csv_file, category = args.load
                print(f"Loading documents from {csv_file} into category '{category}'...")
                document_loader.load_documents(csv_file, category)
            elif args.delete:
                print(f"Deleting all documents in category '{args.delete}'...")
                document_loader.clear_documents(args.delete)
            elif args.list:
                document_loader.list_categories()
        except Exception as e:
            print(f"Error: {str(e)}")
            exit(1)
        return True
    return False
