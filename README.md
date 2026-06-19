# Building Local Private RAG Pipeline

A hands-on notebook for building a **local, private RAG (Retrieval-Augmented Generation)** pipeline using open-source tools — no API keys, no data sent to the cloud.

---

## What You'll Build

A conversational agent that can read any PDF and answer questions about it — using only tools running on your machine.

---

## What is RAG?

Imagine you ask a knowledgeable friend a question. They know a lot in general, but if you ask something very specific — like what's in a particular research paper or a private company document — they'd need to look it up first. RAG gives an AI model the same ability: **look something up before answering**.

### The Problem with Plain LLMs

A large language model (LLM) like Gemma is trained on a massive snapshot of text from the internet. That training gives it broad general knowledge, but it has a hard cutoff — it knows nothing about documents you wrote yesterday, your company's internal data, or anything outside its training window.

More critically, when an LLM doesn't know something, it doesn't say "I don't know." It generates a plausible-sounding answer anyway. These confident but wrong answers are called **hallucinations**.

```
User question ──► LLM ──► Response
                          (may be wrong, outdated, or fabricated — no source to check)
```

### The RAG Solution: Retrieve Before You Generate

RAG (Retrieval-Augmented Generation) fixes this by giving the LLM access to a document store at query time. Instead of relying solely on memorized training data, the model is shown the most relevant excerpts from your actual documents before it answers.

```
Your documents ──► Index (chunk → embed → store in vector DB)
                                            │
User question ──► Embed ──► Retrieve ───────┘
                             top-k chunks
                                 │
                    Augmented Prompt = question + retrieved chunks
                                 │
                                LLM ──► Grounded response
```

The response is now grounded in real text you provided — text you can trace back to a source. The LLM doesn't hallucinate facts that aren't in your documents; it works with what it's shown.

### The RAG Pipeline, Step by Step

```
─────────────────── INDEXING (done once at setup) ───────────────────

   ┌──────────┐     ┌──────────────┐     ┌──────────────┐
   │          │     │    Embed     │     │    Vector    │
   │ Sources  ├────►│   Sources   ├────►│    Store    │
   │    ①    │     │      ②      │     │      ③      │
   └──────────┘     └──────────────┘     └──────┬───────┘
                                                 │
─────────────────── QUERYING (every question) ───┼────────────────────
                                                 │
   ┌──────────┐     ┌──────────────┐     ┌──────▼───────┐     ┌──────────────┐     ┌──────────┐
   │          │     │    Embed     │     │              │     │  Retrieved   │     │          │
   │  Prompt  ├────►│   Prompt    ├────►│  Retriever  ├────►│    Text     ├────►│   LLM    ├────► Response
   │    ④    │     │      ⑤      │     │      ⑥      │     │    ⑦  +     │     │    ⑧    │
   └────┬─────┘     └──────────────┘     └─────────────┘     └──────────────┘     └──────────┘
        │                                                                                ▲
        └────────────────────────── original prompt (passed directly) ──────────────────┘
```

**Steps:**

| # | Step | What happens |
|---|------|-------------|
| ① | **Gather Sources** | Collect documents — PDFs, policies, reports — that will serve as the knowledge base |
| ② | **Embed Sources** | Pass each text chunk through an embedding model, converting it to a fixed-length numeric vector that captures its meaning |
| ③ | **Store Vectors** | Save the vectors in a vector database optimized for similarity search |
| ④ | **Obtain User Prompt** | Receive the user's question |
| ⑤ | **Embed User Prompt** | Embed the question using the same model as step ②, so it lives in the same vector space |
| ⑥ | **Retrieve Relevant Data** | Find the top-k vectors in the store that are closest to the question vector; return those text chunks |
| ⑦ | **Create Augmented Prompt** | Combine the retrieved chunks with the original prompt into a single enriched input |
| ⑧ | **Obtain Response** | Feed the augmented prompt to the LLM; it generates an answer grounded in the retrieved context |

RAG has two phases: **indexing** (done once) and **retrieval + generation** (done on every query).

#### Phase 1 — Indexing

**1. Load your document**
The source document — a PDF, a web page, a text file — is loaded and converted to raw text. In this notebook, `PyPDFLoader` reads each page of a PDF and passes the text downstream.

**2. Split into chunks**
Long documents are broken into smaller overlapping chunks (e.g., 500-word chunks with 50-word overlap). This is essential because retrieval compares the user's question against individual chunks, not the entire document. Smaller chunks = more precise matching.

**3. Embed each chunk**
Every chunk is passed through an **embedding model** — a neural network that converts text into a fixed-length list of numbers called a **vector**. The key property: semantically similar text produces numerically similar vectors. This notebook uses `all-MiniLM-L6-v2`, which produces 384-dimensional vectors and runs fully locally.

**4. Store in a vector database**
The vectors are saved in a **vector store** — a database optimized for similarity search over thousands of vectors in milliseconds. This notebook uses ChromaDB (in-memory). Alternatives include FAISS, Milvus, and Pinecone.

#### Phase 2 — Retrieval + Generation (every query)

**5. Embed the user's question**
The incoming question is embedded using the same model (`all-MiniLM-L6-v2`). This puts the question and the stored chunks in the same vector space so they can be meaningfully compared.

**6. Retrieve the top-k most relevant chunks**
The vector store finds chunks whose vectors are closest to the question vector (via cosine similarity). The top 4 chunks are returned — not the whole document, just the most relevant passages.

**7. Build the augmented prompt**
The retrieved chunks are combined with the user's original question to form a single input — the augmented prompt. The simplest approach is concatenation: paste the chunks and the question together. More sophisticated systems use a structured template that tells the LLM how to use the context, what to do if the answer isn't there, and what tone or format to respond in. Roughly:

```
Use the following context to answer the question.
If the answer isn't in the context, say you don't know — don't guess.

Context:
  [chunk 1 text]
  [chunk 2 text]
  [chunk 3 text]
  [chunk 4 text]

Question: What is the main finding of this paper?
```

In this notebook, LangChain handles this automatically via `RetrievalQA` and `ConversationalRetrievalChain` — it retrieves the chunks, slots them into a `PromptTemplate`, and passes the final string to the LLM, so you never have to assemble the prompt manually.

**8. Send to the LLM and get a response**
The augmented prompt is sent to Gemma 2 2B running locally via Ollama. The model reads the retrieved context and generates a response grounded in your document rather than its training memory.

### How This Maps to the Notebook

| RAG concept | This notebook |
|---|---|
| Document loader | `PyPDFLoader` |
| Text splitter | `RecursiveCharacterTextSplitter` |
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers, runs locally) |
| Vector store | ChromaDB (in-memory) |
| Retriever | ChromaDB cosine similarity search, top-4 chunks |
| LLM | Gemma 2 2B via Ollama (runs locally, no API key) |
| Orchestration | LangChain `RetrievalQA` + `ConversationalRetrievalChain` |
| Conversation memory | `ConversationBufferMemory` — retains chat history across turns |

The `ConversationalRetrievalChain` adds one layer on top of plain RAG: it rewrites follow-up questions to be self-contained before embedding them. So a follow-up like "what did they find?" after an earlier exchange about a paper still retrieves the right chunks instead of matching against vague pronouns.

---

## Stack

| Component | Tool |
|-----------|------|
| LLM | Gemma 2 2B via [Ollama](https://ollama.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Document Loader | PyPDFLoader |

---

## How Ollama Works Locally

**Why Ollama?**

You can download Gemma's weights directly from HuggingFace, but raw model weights are just binary files — they don't run on their own. To actually perform inference you need a runtime that handles tokenization, memory layout, attention computation, sampling, and context management. The standard Python path is `transformers` + PyTorch — where `transformers` provides the model architecture and tokenizer in Python, and PyTorch is the tensor computation engine that runs the actual matrix math — but on a CPU that stack is slow, heavyweight, and requires several gigabytes of additional dependencies. Ollama takes a different approach: it packages models in GGUF format and runs them through `llama.cpp`, a highly optimized C++ inference engine built specifically for quantized CPU inference. On an Intel CPU with AVX2, `llama.cpp` runs inference several times faster than an equivalent PyTorch setup. Ollama adds model management (pull, list, remove), automatic CPU backend selection, a keep-alive cache, and a clean HTTP API on top — reducing what would be a complex multi-step setup to a single `ollama pull` command. From the notebook's perspective, all of that complexity is hidden behind one HTTP call.

---

Ollama is a local model runtime that lets you download and run open-source LLMs entirely on your own machine — no cloud account, no API key, no data leaving your device. Think of it like Docker, but for AI models: it pulls a model once, manages it on disk, and serves it through a lightweight HTTP API bound exclusively to `localhost`.

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Your Machine                       │
│                                                     │
│   Jupyter Notebook                                  │
│        │                                            │
│        │  LangChain builds prompt                   │
│        │  (retrieved chunks + user question)        │
│        ▼                                            │
│   OllamaLLM.invoke()                               │
│        │                                            │
│        │  HTTP POST → localhost (Ollama port)       │
│        ▼                                            │
│   Ollama Server  ──► llama-server (llama.cpp)      │
│        │                                            │
│        │  loads model weights from disk into RAM    │
│        │  runs token-by-token CPU inference         │
│        ▼                                            │
│   JSON response ──► LangChain ──► Notebook output  │
│                                                     │
│   ~/.ollama/models/  (model weights stored here)   │
└─────────────────────────────────────────────────────┘
```

All traffic stays on `localhost`. Ollama does not open any external ports or connect to the internet during inference.

---

### Downloading and Managing Models

When you run `ollama pull gemma2:2b`, Ollama downloads the model weights in **GGUF format** — a compact, quantized binary format optimized for CPU inference developed by the `llama.cpp` project. The weights are stored permanently in `~/.ollama/models/blobs/`. Subsequent runs load from disk with no re-download.

```bash
ollama pull gemma2:2b       # ~1.6 GB — used in this notebook
ollama pull llama3.2        # ~2.0 GB — good general-purpose alternative
ollama pull phi3            # ~2.2 GB — strong reasoning for its size
ollama pull mistral         # ~4.1 GB — requires ~5 GB free RAM
ollama list                 # show all downloaded models
ollama rm mistral           # remove a model to free disk space
```

Ollama supports dozens of open-source models — Llama, Mistral, Gemma, Phi, Qwen, DeepSeek, CodeLlama and more. Each model can have multiple variants tagged by size and quantization level (e.g. `mistral:7b-instruct-q4_0` = 7 billion parameters, instruction-tuned, 4-bit quantized).

**What is quantization?**
Full-precision models store each weight as a 32-bit float. Quantization reduces this to 4-bit integers, cutting memory usage by ~8× with minimal quality loss. A 7B-parameter model that would need ~28 GB at full precision fits in ~4 GB at Q4.

---

### How the Inference Server Works

Ollama runs a background server process that:
1. Listens for requests on a local-only port (configurable via `OLLAMA_HOST`, defaults to `127.0.0.1`)
2. Loads the requested model's GGUF weights into RAM on first use
3. Passes prompts to `llama-server` — a C++ inference engine built on [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
4. Runs token-by-token autoregressive generation on your CPU (or GPU if available)
5. Streams or returns the generated tokens as a JSON response

The server keeps the model loaded in RAM between requests for `OLLAMA_KEEP_ALIVE` seconds (default: 5 minutes). This means the first query after loading pays a warm-up cost; subsequent queries within that window are faster.

**CPU optimization:**
Ollama automatically selects the best GGML backend for your CPU architecture. On modern Intel CPUs with AVX2 support it uses optimized vector math backends, significantly speeding up inference compared to a generic build.

---

### How the Jupyter Notebook Talks to Ollama

`OllamaLLM` from `langchain-ollama` is a thin HTTP client. When you instantiate it:

```python
llm = OllamaLLM(
    model="gemma2:2b",
    temperature=0.5,
    num_predict=256,   # max tokens to generate
    num_ctx=2048,      # context window size (affects RAM usage)
)
```

Every `.invoke()` call — whether from a `RetrievalQA` chain or `ConversationalRetrievalChain` — translates into an HTTP POST to the Ollama API:

```
POST http://127.0.0.1:<ollama-port>/api/generate
Content-Type: application/json

{
  "model": "gemma2:2b",
  "prompt": "<system prompt + retrieved chunks + user question>",
  "options": {
    "temperature": 0.5,
    "num_predict": 256,
    "num_ctx": 2048
  },
  "stream": false
}
```

Ollama processes the request and returns:

```json
{
  "model": "gemma2:2b",
  "response": "The paper introduces Sentinel Kiln DB...",
  "done": true,
  "total_duration": 24460000000,
  "eval_count": 128
}
```

LangChain extracts the `response` field and passes it back up the chain as the LLM's answer.

**What goes into the prompt?**
In a RAG pipeline, LangChain constructs the prompt automatically. For `RetrievalQA`, it:
1. Embeds the user's question using `all-MiniLM-L6-v2`
2. Queries ChromaDB for the top-k most similar document chunks
3. Concatenates them with the question into a single prompt string
4. Sends that string to Ollama

The model never sees the full document — only the most relevant chunks retrieved for each query. This keeps the prompt within the context window and makes responses accurate and grounded.

---

### Key Configuration Parameters

| Parameter | What it controls | Default |
|-----------|-----------------|---------|
| `num_ctx` | Context window size in tokens. Higher = more RAM. | 2048 |
| `num_predict` | Max tokens to generate per response | 128 |
| `temperature` | Randomness: 0 = deterministic, 1 = creative | 0.8 |
| `OLLAMA_KEEP_ALIVE` | How long to keep model loaded in RAM | 5 minutes |
| `OLLAMA_HOST` | Network interface the server binds to | `127.0.0.1` |
| `OLLAMA_CONTEXT_LENGTH` | Server-level context cap (overrides per-request) | model default |

---

## Prerequisites

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull Gemma 2 2B (~1.6 GB)

```bash
ollama pull gemma2:2b
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Open the notebook in Jupyter or PyCharm:

```bash
jupyter lab
```

Then open `Summarizing_Private_Documents_Using_RAG_LangChain_and_Ollama.ipynb` and run cells top to bottom.

The notebook will:
1. Auto-start the Ollama server if it's not running
2. Download the sample PDF
3. Split, embed, and index it in ChromaDB
4. Let you query it with natural language — with memory across turns

To use your own PDF, replace the `filename` and `url` in the load cell with your file path.

---

## Hardware Requirements

| Resource | Minimum |
|----------|---------|
| RAM | 4 GB free (Gemma 2 2B needs ~1.6 GB) |
| Disk | 3 GB free (model + dependencies) |
| GPU | Not required — runs on CPU |

Tested on Intel Core i5-1235U, 14 GB RAM, no GPU.

---

## Notebook Structure

1. **Setup** — install dependencies, start Ollama
2. **Preprocessing** — load PDF, split into chunks, embed into ChromaDB
3. **LLM Construction** — connect to local Gemma 2 via Ollama
4. **RetrievalQA** — simple Q&A over the document
5. **Prompt Template** — prevent hallucination with a grounded prompt
6. **Conversation Memory** — follow-up questions with context
7. **Interactive Agent** — chat loop over your document
8. **Exercises** — try your own PDF, return source chunks, swap models

---

## Sample Document

The notebook uses the [Sentinel Kiln DB paper](https://openreview.net/pdf?id=efGzsxVSEC) as a demo document — a research paper on satellite-based brick kiln detection in South Asia.

---

## Why Local?

| | Cloud LLM API | This Notebook |
|---|---|---|
| LLM | Remote model via API | Gemma 2 2B via Ollama (local) |
| Auth | API key required | None |
| Data leaves machine? | Yes | No |
| Cost | Per-token billing | Free |
