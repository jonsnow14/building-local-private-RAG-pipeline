# RAG Learning Resources

A hands-on notebook for building a **local, private RAG (Retrieval-Augmented Generation)** pipeline using open-source tools — no API keys, no data sent to the cloud.

---

## What You'll Build

A conversational agent that can read any PDF and answer questions about it — using only tools running on your machine.

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

You can download Gemma's weights directly from HuggingFace, but raw model weights are just binary files — they don't run on their own. To actually perform inference you need a runtime that handles tokenization, memory layout, attention computation, sampling, and context management. The standard Python path is `transformers` + PyTorch, but on a CPU that stack is slow, heavyweight, and requires several gigabytes of additional dependencies. Ollama takes a different approach: it packages models in GGUF format and runs them through `llama.cpp`, a highly optimized C++ inference engine built specifically for quantized CPU inference. On an Intel CPU with AVX2, `llama.cpp` runs inference several times faster than an equivalent PyTorch setup. Ollama adds model management (pull, list, remove), automatic CPU backend selection, a keep-alive cache, and a clean HTTP API on top — reducing what would be a complex multi-step setup to a single `ollama pull` command. From the notebook's perspective, all of that complexity is hidden behind one HTTP call.

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
