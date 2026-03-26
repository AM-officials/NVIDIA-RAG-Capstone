# RAG Agent: Deep Interview Preparation Guide

## What Was Built

**A production-ready Retrieval-Augmented Generation (RAG) system** powered by:
- **NVIDIA AI Endpoints** (NVIDIAEmbeddings + ChatNVIDIA)
- **LangChain LCEL** (LangChain Expression Language) for composable chains
- **FAISS** vector store for fast approximate nearest-neighbour search
- **arXiv papers** as the knowledge base (4 papers: RAG, Transformers, Llama 2, GPT-4)

Three production improvements were added: **Evaluation Metrics**, **Conversational Memory**, and a **FastAPI REST wrapper**.

---

## Core Concepts Deep Dive

### 1. What is RAG and Why?

**Definition:** RAG augments a language model by retrieving relevant documents at inference time and injecting them as context. The LLM generates answers *grounded* in those documents rather than relying on parametric (baked-in) knowledge.

**Why it matters:**
- Reduces hallucinations — the model is constrained to retrieved facts
- Knowledge doesn't become stale — swap the vector store to update knowledge
- Provides citations — every claim traces to a source document
- More efficient than fine-tuning — no re-training needed

**Formula for an interviewer:** `LLM + Retriever = RAG`. The retriever fetches context; the LLM synthesises an answer.

---

### 2. Pipeline Architecture (Step-by-Step)

```
User Query
    │
    ▼
[Embedder] nvidia/nv-embed-v1
    │  Converts query to 1024-dim vector
    ▼
[FAISS Vector Store]
    │  cosine similarity search → top-5 chunks
    ▼
[LongContextReorder]
    │  Most relevant chunks placed at edges (middle is lost)
    ▼
[Context Formatter] docs2str()
    │  Wraps chunks with paper titles for citations
    ▼
[ChatPromptTemplate]
    │  Slots: {input}, {context} (+ {chat_history} for memory)
    ▼
[ChatNVIDIA] openai/gpt-oss-120b
    │  Generates grounded response (streaming)
    ▼
[StrOutputParser]  →  Final answer string
```

**Key code object:** `rag_chain = retrieval_chain | generator_chain`  
The `|` operator is LCEL's pipe — same concept as Unix pipes.

---

### 3. FAISS — What It Is and Why

**FAISS** (Facebook AI Similarity Search) is an in-memory vector index library.

- **index.faiss** stores the raw float32 vectors
- **index.pkl** stores the document metadata and text
- Search is `O(n)` for flat index, `O(log n)` for IVF variants
- Used `k=5` — retrieves top 5 most semantically similar chunks

**Interview question:** *"Why FAISS over a cloud vector DB like Pinecone?"*  
→ FAISS is free, runs locally, zero latency, perfect for prototypes and offline demos. Pinecone is better for production scale (millions of vectors, multi-user, persistent).

---

### 4. NVIDIAEmbeddings — How Embeddings Work

**Model:** `nvidia/nv-embed-v1` → outputs 1024-dimensional vectors

- Each document chunk is converted to a vector that encodes *semantic meaning*
- Similar meaning = similar vectors (close in cosine distance)
- At query time, the question gets embedded the same way → FAISS finds nearest vectors

**Interview question:** *"What happens if you change the embedding model?"*  
→ You must re-embed ALL documents. The embedding space is model-specific — mixing embeddings from different models is meaningless.

---

### 5. LangChain LCEL Chain Composition

**LCEL** (LangChain Expression Language) uses the `|` operator to compose Runnables:

```python
# Each component is a Runnable (supports invoke/stream/batch)
retrieval_chain = {'input': identity} | RunnableAssign({'context': context_getter})
generator_chain = {'output': response_chain} | RunnableLambda(output_puller)
rag_chain = retrieval_chain | generator_chain
```

**RunnableAssign** — adds a new key to a dict without losing existing keys.  
**RunnableLambda** — wraps any Python function as a Runnable.  
**itemgetter** — extracts a key from a dict, used to route dict fields between chain steps.

---

### 6. LongContextReorder — What and Why

**Problem:** LLMs perform worse on information in the *middle* of long contexts (the "lost in the middle" phenomenon, Liu et al. 2023).

**Solution:** After retrieving 5 chunks, `LongContextReorder` reorders them so the most relevant chunks are at the *beginning and end*, where the LLM pays most attention.

---

### 7. Change 1 — Conversational Memory

**Class:** `ConversationBufferWindowMemory(k=5)`

**How it works:**
- `memory.save_context({"input": q}, {"output": a})` — stores each exchange
- `memory.load_memory_variables({})["chat_history"]` — retrieves formatted history
- Window of `k=5` means only the last 5 exchanges are kept (prevents token bloat)

**Why it handles pronouns:** The new `memory_prompt` injects `{chat_history}` before the current question. When the LLM sees "What are its limitations?", it can look up the previous exchange where "RAG" was discussed.

**Interview question:** *"What's the difference between ConversationBufferMemory and ConversationBufferWindowMemory?"*  
→ Buffer keeps ALL history (grows unboundedly, risks exceeding context window). Window keeps only the last `k` exchanges — bounded, production-safe.

---

### 8. Change 2 — FastAPI Wrapper

**Why FastAPI over Flask?**
- Auto-generates OpenAPI/Swagger docs at `/docs`
- Pydantic validation (the `Question` BaseModel validates incoming JSON automatically)
- Async-native (better concurrency)
- Industry standard for ML model serving

**Key implementation detail:** `nest_asyncio.apply()` is required because Jupyter already runs an event loop, and Uvicorn needs to start its own. `nest_asyncio` allows nested event loops.

**The `threading.Thread(daemon=True)` pattern** — runs the server in background so the notebook cell completes and you can keep running other cells.

**Endpoints:**
```
GET  /health  →  {status, embedding_model, llm, vector_count}
POST /ask     →  {question, answer, sources, response_time_ms}
```

---

### 9. Change 3 — Evaluation Metrics

**Metrics measured:**
- **Response time (ms):** End-to-end latency from query to full response
- **Docs retrieved:** Always 5 (because `k=5` in the retriever) — confirms retriever is working

**What a real evaluation would add (for follow-up questions):**
- **Faithfulness:** Does the answer only use retrieved context? (RAGAS metric)
- **Answer Relevance:** Is the answer relevant to the question?
- **Context Recall:** Were the right documents retrieved?

---

## Likely Interview Questions & Strong Answers

### Q: "Walk me through how a user question becomes an answer."
**A:** The question is embedded into a 1024-dim vector using NVIDIA's nv-embed-v1 model. FAISS searches the index for the 5 closest document chunk vectors. Those chunks are reordered (LongContextReorder) and formatted into a context string. A ChatPromptTemplate inserts the question and context into a structured prompt. ChatNVIDIA (gpt-oss-120b) generates a streaming response that's constrained to the retrieved facts.

### Q: "Why do you need a vector database? Why not just send all documents to the LLM?"
**A:** LLM context windows are limited (128K tokens for most models). Our corpus has 577 chunks × ~800 tokens = ~460K tokens — far exceeds any context window. Retrieval selects only the relevant ~5K tokens. It's also faster and cheaper to retrieve semantically than to brute-force the whole corpus.

### Q: "How would you productionize this beyond the FastAPI wrapper?"
**A:** (1) Replace FAISS with a persistent vector DB (Pinecone, Weaviate, or pgvector). (2) Add authentication to the API. (3) Add a caching layer for repeated queries. (4) Add RAGAS-based evaluation in CI/CD. (5) Deploy on NVIDIA NIM or AWS SageMaker. (6) Add structured logging and async request handling.

### Q: "What are RAG's limitations?"
**A:** (1) Retrieval quality bottleneck — if the wrong chunks are retrieved, the answer will be wrong. (2) Multi-hop reasoning is hard — needing to connect facts across multiple documents. (3) Up-to-date knowledge requires re-indexing. (4) Latency — two calls (embedding + generation) vs. one. (5) Hallucinations aren't fully eliminated — the model can still hallucinate within the retrieved context.

### Q: "How does the memory know that 'its' refers to RAG?"
**A:** The previous conversation is serialized as a formatted string and injected into the prompt under the "Conversation History" section. The LLM performs coreference resolution — it reads the prior exchange ("What is RAG? / RAG is...") and resolves "its" accordingly. This is standard in-context reasoning, not a special mechanism.

### Q: "What is LongContextReorder and why does it help?"
**A:** It's based on the "Lost in the Middle" paper (Liu et al., 2023) which showed LLMs pay most attention to content at the beginning and end of long contexts. LongContextReorder places the most relevant retrieved chunks at the edges, so the model focuses on the best information.

### Q: "Why use NVIDIA AI Endpoints instead of OpenAI?"
**A:** NVIDIA provides optimized inference for their own GPU hardware, and hosts models like Llama 2, Mistral, and their own Nemotron fine-tunes. For this project, it demonstrates integration with enterprise-grade, GPU-accelerated inference services — relevant in production ML engineering. NVIDIA also offers specialized embedding models (nv-embed-v1) trained for retrieval tasks.

### Q: "What's LCEL and why is it better than the old LangChain chains?"
**A:** LangChain Expression Language uses the `|` (pipe) operator to compose Runnables. It supports streaming, async, parallel execution, and batching natively. The old `ConversationalRetrievalChain` was a black box — LCEL makes every step explicit and customizable. It's also easier to debug because you can `invoke()` any intermediate step independently.

---

## Technical Numbers to Know

| Metric | Value |
|--------|-------|
| Papers loaded | 4 (RAG, Transformers, GPT-4, Llama 2) |
| Document chunks | ~577 |
| Chunk size | 1,000 tokens (200 overlap) |
| Embedding dimensions | 1,024 (nv-embed-v1) |
| Retriever k (top-K) | 5 chunks per query |
| Memory window | 5 exchanges |
| API port | 8000 |
| Avg. response time | ~3-8 seconds (NVIDIA API latency) |

---

## Potential Gotchas (Show You've Thought Deeply)

1. **API key exposure** — the key is hardcoded in the notebook. In production, use environment variables or AWS Secrets Manager.
2. **FAISS is not persistent across sessions** — must reload from `docstore_index.pkl` each time.
3. **Thread safety of FastAPI** — the `retrieval_chain` and `generator_chain` are shared across requests; they are stateless Runnables so this is safe, but `memory` is session-scoped (single notebook session = single user).
4. **`allow_dangerous_deserialization=True`** in FAISS.load_local — required because pickle is used; safe when you control the file source.
5. **Gradio deprecation warning** — `tuples` format for chatbot is deprecated; should use `type='messages'` with OpenAI-style dicts.
