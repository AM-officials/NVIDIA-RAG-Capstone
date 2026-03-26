# NVIDIA RAG Agent (LangChain + Gradio + FastAPI)

A production-ready Retrieval-Augmented Generation (RAG) pipeline using NVIDIA endpoints, LangChain, and FAISS for intelligent document Q&A.

## Features

- Document retrieval with FAISS semantic search
- Grounded response generation with NVIDIA-hosted LLMs
- Gradio chat interface for interactive Q&A
- FastAPI wrapper with `GET /health` and `POST /ask`
- Conversational memory for multi-turn follow-up questions
- Basic evaluation metrics (latency and retrieved-doc count)
- Source citations derived from retrieved document metadata
- Optimized `/ask` flow that reuses retrieved docs for source titles (no duplicate retrieval call)
- Reusable pre-built vector store for faster startup

## Tech Stack

- LLM: NVIDIA AI Endpoints (`openai/gpt-oss-120b` in notebook)
- Embeddings: NVIDIA `nvidia/nv-embed-v1`
- Framework: LangChain (LCEL)
- Vector Store: FAISS
- Interfaces: Gradio + FastAPI

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA API key ([Get one free](https://build.nvidia.com))

### Installation

```bash
git clone https://github.com/AM-officials/NVIDIA-RAG-Capstone.git
cd NVIDIA-RAG-Capstone
pip install -r requirements.txt
```

Set your NVIDIA API key before running the notebook.

### Usage

1. Open `RagAgent.ipynb` in Jupyter or VS Code notebooks.
2. Set your NVIDIA API key in the environment setup cell.
3. Load the pre-built vector store (fast path) or rebuild it from arXiv papers.
4. Run the retrieval and generation chain cells.
5. Test via the Gradio interface.
6. Optionally run the FastAPI section to expose REST endpoints.

## API Endpoints

After running the FastAPI cell in `RagAgent.ipynb`:

- `GET /health`: Returns service status, model metadata, and vector count
- `POST /ask`: Accepts `{ "question": "..." }` and returns:
  - `question`
  - `answer`
  - `sources` (titles from retrieved docs)
  - `response_time_ms`

## How It Works

1. Load papers and split into chunks.
2. Embed chunks with `nvidia/nv-embed-v1`.
3. Index embeddings in FAISS.
4. Retrieval chain gets top-k docs, reorders context, and returns both formatted context and document objects.
5. Generation chain streams grounded answers using retrieved context.
6. `/ask` reuses already-retrieved documents to build source titles in one retrieval pass.

## Project Structure

```text
NVIDIA-RAG-Capstone/
├── RagAgent.ipynb
├── README.md
├── requirements.txt
├── docstore_index.zip
├── docstore_index/
│   ├── index.faiss
│   └── index.pkl
└── anaconda_projects/
```

## Example Questions

- What are the latest developments in large language models?
- Explain retrieval-augmented generation.
- What are the key innovations in Llama 2?

## Future Enhancements

- Hybrid retrieval (dense + sparse)
- API authentication and rate limiting
- Persistent memory backend (Redis/Postgres)
- Managed vector database for larger corpora
- Multimodal document support

## Acknowledgments

- NVIDIA for model and embedding endpoints
- LangChain for orchestration primitives
- FAISS for efficient vector search
