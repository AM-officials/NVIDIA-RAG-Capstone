# 🤖 NVIDIA RAG Agent (LangChain + Gradio)

A minimal **Retrieval-Augmented Generation (RAG)** pipeline using NVIDIA LLM endpoints, LangChain, and FAISS for intelligent document Q&A.


## ✨ Features

- **Document Retrieval**: Semantic search with FAISS vector store
- **Contextual Generation**: NVIDIA Llama 3.1 8B for grounded responses
- **Interactive UI**: Simple Gradio chat interface
- **Citation Support**: Answers include source references

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
2. Run all cells to initialize the RAG pipeline
3. Use the Gradio interface to ask questions!

**Example Queries:**
- "What are the latest developments in large language models?"
- "Explain retrieval-augmented generation"
- "How do transformer architectures work?"

## � Project Structure

```
RAG-Agent-NVIDIA/
├── 📓 RagAgent.ipynb        # Main notebook with complete RAG pipeline
├── 📄 README.md             # This file
├── 📄 requirements.txt      # Python dependencies
├── 📄 LICENSE              # MIT license
└── 📄 .gitignore           # Git ignore rules
```

## 🎯 How It Works

1. **Document Loading**: Load research papers from arXiv
2. **Embedding**: Convert documents to vectors using NVIDIA embeddings
3. **Indexing**: Store embeddings in FAISS for fast retrieval
4. **Query Processing**: User query → Retrieve relevant docs → Generate answer
5. **Response**: LLM generates contextual answer with citations

## 🔧 Configuration

Customize models in the notebook:

```python
# Embedding model
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

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** for AI endpoints
- **LangChain** for the orchestration framework
- **FAISS** for vector search

---

**Built with ❤️ for the AI community**
