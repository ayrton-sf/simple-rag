# RAG Conversational Chatbot

A RAG-based conversational chatbot using LangGraph for LLM integration and ChromaDB for local vector storage.

## Features

- **LLM Support**: Anthropic (Claude) and OpenAI
- **Embeddings**: VoyageAI and OpenAI
- **Vector Storage**: ChromaDB (local)
- **Memory**: Full conversation history with retrieved documents
- **CLI-based**: Simple command-line interface
- **API Documentation**: Swagger docs available at `/docs`


## How It Works

Simple RAG pipeline:
1. Query is vectorized and used to search documents on every turn
2. All messages (human and AI) are saved to context
3. Retrieved documents are saved to context

## Setup

### 1. Create and activate a virtual environment

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `example.env` to `.env` and fill in the required values:
```env
# Required - choose one LLM model
LLM_MODEL=claude-sonnet-4-20250514

# Required - choose one embedding model
EMBEDDINGS_MODEL=voyage-3.5

# API Keys (fill based on your model choices)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
VOYAGEAI_API_KEY=your_key_here

# Optional
SYSTEM_PROMPT_PATH=prompts/system.txt
```

### Available Models

**LLM Models**:
- OpenAI: `gpt-4o`, `gpt-4.1`, `o3`, `o4-mini`
- Anthropic: `claude-haiku-4-5-20251001`, `claude-3-5-sonnet-latest`, `claude-3-7-sonnet-latest`, `claude-sonnet-4-20250514`

**Embedding Models**:
- VoyageAI: `voyage-context-3`, `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3`
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

## Usage

### Start the chatbot
```bash
python main.py --host 127.0.0.1 --port 8000
```

Access Swagger documentation at: `http://127.0.0.1:8000/docs`

### Load data
```bash
python main.py --load current-stock.csv cars
```

**Data requirements**:
- CSV files must have an `id` column
- JSONL files must have an `id` attribute
- Using an existing ID will update that record

### Delete a category
```bash
python main.py --delete cars
```

### List database contents
```bash
python main.py --list
```

### Reset database
```bash
python main.py --reset
```

## License

MIT