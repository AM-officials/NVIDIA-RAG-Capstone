# 🤖 NVIDIA RAG Agent (LangChain + Gradio)

A minimal **Retrieval-Augmented Generation (RAG)** pipeline using NVIDIA LLM endpoints, LangChain, and FAISS for intelligent document Q&A.


## ✨ Features

- **Document Retrieval**: Semantic search with FAISS vector store
- **Contextual Generation**: NVIDIA Llama 3.1 70B for grounded responses
- **Interactive UI**: Simple Gradio chat interface
- **Citation Support**: Answers include source references
- **Reusable Vector Store**: Pre-built vector store included for 4 research papers (saves ~2 min setup time)

## 🛠️ Tech Stack

- **LLM**: NVIDIA AI Endpoints (Llama 3.1 8B Instruct)
- **Embeddings**: NVIDIA nv-embed-v1
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Interface**: Gradio

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA API Key ([Get one free](https://build.nvidia.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-agent.git
cd rag-agent

# Install dependencies
pip install -r requirements.txt

# Set your NVIDIA API key
export NVIDIA_API_KEY="nvapi-your-key-here"

# Run the notebook
jupyter notebook RagAgent.ipynb
```

### Usage

1. Open `RagAgent.ipynb` in Jupyter or Google Colab
2. **Replace `YOUR_NVIDIA_API_KEY_HERE`** with your actual NVIDIA API key in cell 3
3. **Choose one of two options:**
   - **Option A (Fast)**: Run cell 7 to load the pre-built vector store (included in repo)
   - **Option B (Fresh)**: Run cell 6 to create a new vector store from arXiv papers
4. Continue running remaining cells to build the RAG pipeline
5. Use the Gradio interface to ask questions!

**Example Queries:**
- "What are the latest developments in large language models?"
- "Explain retrieval-augmented generation"
- "How do transformer architectures work?"

## ⚡ Performance Note

**Vector Store Creation Time**: Creating embeddings for documents takes time. As shown below, processing just 4 research papers takes approximately **2 minutes**:

![Vector Store Creation Time](image.png)

**💡 Tip**: Use the pre-built `docstore_index.zip` included in this repo to skip the vectorization step and start immediately! As the number of papers increases, vectorization time grows linearly - making the reusable vector store essential for production use.

## 📁 Project Structure

```
NVIDIA-RAG-Capstone/
├── 📓 RagAgent.ipynb           # Main notebook with complete RAG pipeline
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore              # Git ignore rules
├── 📦 docstore_index.zip      # Pre-built vector store (4 papers)
├── � docstore_index/         # Extracted vector store files
│   ├── index.faiss            # FAISS index file
│   └── index.pkl              # Document metadata
└── 🖼️ image.png               # Performance benchmark screenshot
```

## 🎯 How It Works

1. **Document Loading**: Load research papers from arXiv (or use pre-built vector store)
2. **Embedding**: Convert documents to vectors using NVIDIA embeddings
3. **Indexing**: Store embeddings in FAISS for fast retrieval
4. **Query Processing**: User query → Retrieve relevant docs → Generate answer
5. **Response**: LLM generates contextual answer with citations

**Included Papers** (in pre-built vector store):
- RAG: Retrieval-Augmented Generation (2005.11401)
- Attention Is All You Need - Transformers (1706.03762)
- GPT-4 Technical Report (2304.08485)
- Llama 2 Paper (2307.09288)

## 🔧 Configuration

Customize models and papers in the notebook:

```python
# Embedding model
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# LLM model (choose one)
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")  # Powerful
# llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")  # Faster

# Add more arXiv papers (in cell 6)
arxiv_ids = [
    "2005.11401",  # Your paper ID here
    "1706.03762",  # Add as many as needed
]

# Retrieval parameters
retriever = docstore.as_retriever(search_kwargs={'k': 5})
```
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# LLM model
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

# Retrieval parameters
retriever = docstore.as_retriever(search_kwargs={'k': 5})
```

## 📊 Sample Output

**Query**: "What is retrieval-augmented generation?"

**Response**: "According to [Paper Title], Retrieval-Augmented Generation (RAG) is a technique that enhances language model outputs by incorporating relevant documents from a knowledge base..."

## � Future Enhancements

- [ ] Add conversational memory for multi-turn dialogue
- [ ] Implement hybrid search (dense + sparse)
- [ ] Support multi-modal documents (images, tables)
- [ ] Add evaluation metrics (precision, recall)
- [ ] Deploy as web service


## 🙏 Acknowledgments

- **NVIDIA** for AI endpoints
- **LangChain** for the orchestration framework
- **FAISS** for vector search

---

**Built with ❤️ for the AI community**
