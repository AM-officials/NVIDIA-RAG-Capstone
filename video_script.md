# 5–7 Minute Screen Recording Script
## RAG Agent: Jupyter Notebook Demo

---

> **Setup before recording:**  
> - Open `RagAgent.ipynb` in VS Code or Jupyter Lab  
> - Have all cells already run (outputs visible) except Sections 7, 8, 9  
> - Have a terminal open alongside for the API demo  
> - Font size: 16+, dark theme preferred  

---

## [0:00 – 0:45] Introduction & Project Overview

*(Show the notebook title cell — Section 1 markdown)*

> "Hi! I'm Anmol. I built a production-ready Retrieval-Augmented Generation system using NVIDIA AI Endpoints, LangChain, and FAISS. Let me walk you through what it does and the improvements I added."

*(Scroll slowly through the overview markdown)*

> "The system loads research papers from arXiv — papers on RAG itself, the Transformer architecture, GPT-4, and Llama 2. It creates semantic embeddings using NVIDIA's nv-embed-v1 model, stores them in a FAISS vector store, and can answer questions about those papers with grounded, citation-backed responses."

---

## [0:45 – 2:00] Core Pipeline Walkthrough

*(Jump to Section 2 — AI Models cell, show the code)*

> "The embedding model converts text into 1024-dimensional vectors. The LLM — here I'm using OpenAI's gpt-oss-120b via NVIDIA's endpoint — generates the actual responses."

*(Jump to Section 4 — RAG Pipeline cells)*

> "The pipeline is built with LangChain Expression Language — these pipe operators chain Runnables together. The retrieval chain looks up the top-5 most relevant document chunks from FAISS, applies LongContextReorder — which places the most relevant chunks at the edges because LLMs pay less attention to information in the middle of long contexts — and formats the context with source citations."

*(Show the `rag_chain = retrieval_chain | generator_chain` line)*

> "The full pipeline is just this one line. The retrieval chain feeds into the generator chain. Clean, composable, streaming-capable."

---

## [2:00 – 2:45] Existing Query Demo

*(Jump to Section 5 — show the existing output for the LLM query)*

> "Here you can see the pipeline already working. The question 'What are the latest developments in large language models?' gets answered with citations to the specific papers — Llama 2, BLOOM, OPT — retrieved directly from our FAISS index. Every fact traces back to a retrieved document."

---

## [2:45 – 3:45] Improvement 1: Evaluation Metrics *(NEW)*

*(Scroll to Section 7 markdown, read it briefly, then run the code cell)*

> "The first improvement I added is automated evaluation. I run 5 representative test questions through the pipeline and measure response latency and how many source documents are retrieved per query."

*(Wait for the table to print)*

> "You can see the output here — a clean summary table. The average response time, number of documents retrieved. This is what you'd use to benchmark before a production deployment — to catch regressions if you swap out models or change chunk sizes."

---

## [3:45 – 5:00] Improvement 2: Conversational Memory *(NEW)*

*(Scroll to Section 8 markdown)*

> "The second improvement is multi-turn conversational memory. The original pipeline was stateless — each question was independent. I added `ConversationBufferWindowMemory` with a window of 5 exchanges. It stores the last 5 question-answer pairs and injects them as context on each new turn."

*(Run the Section 8a setup cell)*

> "The setup is clean. The memory object, a new prompt template that includes a `{chat_history}` slot, and a wrapper function `ask_with_memory` that loads history → retrieves context → generates response → saves the exchange back to memory."

*(Run the Section 8b demo cell — this is the key moment)*

> "Now watch the demo. I ask three questions. First: 'What is RAG?' — standard retrieval. Then: 'What are its main limitations?' — notice the pronoun 'its'. And finally: 'How does it compare to traditional IR?' — again a pronoun reference."

*(As output streams)*

> "The agent resolves 'its' and 'it' correctly back to RAG — because the previous exchanges are in the prompt. This is what makes a chatbot feel natural versus a stateless Q&A bot."

---

## [5:00 – 6:15] Improvement 3: FastAPI REST Wrapper *(NEW)*

*(Scroll to Section 9 markdown)*

> "The third improvement wraps the entire pipeline as a REST API using FastAPI. Two endpoints — GET /health and POST /ask."

*(Run the Section 9 code cell)*

> "The implementation uses `nest_asyncio` to allow Uvicorn to run inside Jupyter's event loop. The server starts in a background daemon thread so the notebook stays interactive."

*(Switch to terminal)*

> "Let me hit the health endpoint to confirm it's running."

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

> "You can see it returns the status, the embedding model name, the LLM name, and how many vectors are in the store."

*(Now test /ask)*

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is RAG?\"}" | python -m json.tool
```

> "And the /ask endpoint returns the full answer, the source document titles, and the response time in milliseconds. This is now a deployable API — you could put a reverse proxy in front of it and ship it."

---

## [6:15 – 7:00] Summary & What's Next

*(Scroll back to top of notebook for a final overview shot)*

> "To summarize: I built a complete RAG pipeline on NVIDIA AI Endpoints and added three production-oriented improvements — automated evaluation for benchmarking, conversational memory for natural multi-turn dialogue, and a FastAPI wrapper for deployment."

> "Natural next steps would be: replacing FAISS with a persistent vector database like Pinecone for multi-user scale, adding RAGAS-based quality metrics to the evaluation section, and adding authentication to the API. The code is clean, modular, and each section is independent — easy to extend."

> "Thanks for watching!"

---

## Timing Reference

| Segment | Time | Key Action |
|---------|------|-----------|
| Intro & overview | 0:00–0:45 | Scroll title + overview markdown |
| Core pipeline walkthrough | 0:45–2:00 | Show Section 2 + Section 4 code |
| Existing query demo | 2:00–2:45 | Show existing Section 5 output |
| **Eval Metrics (NEW)** | 2:45–3:45 | Run Section 7 cell live |
| **Memory (NEW)** | 3:45–5:00 | Run Section 8a + 8b cells live |
| **FastAPI (NEW)** | 5:00–6:15 | Run Section 9 + curl in terminal |
| Summary | 6:15–7:00 | Scroll to top, close |
