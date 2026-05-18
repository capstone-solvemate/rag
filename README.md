# Enterprise RAG Chatbot

A retrieval-augmented generation (RAG) API for enterprise documents.
Retrieves relevant document chunks from a vector store and generates
grounded answers using OpenAI. Designed to be consumed by any HTTP
client — including an Express.js backend.

---

## Current Status

| Metric                | Value                                |
| --------------------- | ------------------------------------ |
| Retrieval Precision@5 | 0.96 (target ≥ 0.60) ✅              |
| API layer             | FastAPI — 5 endpoints across 3 routers |
| Generation model      | gpt-4o-mini                          |
| Embedding model       | text-embedding-3-small               |
| Vector store          | Chroma (persisted to disk)           |
| Python version        | 3.11.x (recommended) / 3.14.3 (dev) |

---

## Architecture

```
[HTTP Client / Express.js]
          ↓
  [FastAPI Service :8000]
    GET  /health
    POST /chat
    POST /chat/image
    POST /knowledge-base
    DELETE /knowledge-base/{doc_id}
          ↓
  [Retrieval Layer]
    Chroma vector store
    OpenAI embeddings
          ↓
  [Generation Layer]
    gpt-4o-mini
    Context-grounded answer
```

### Request flow — `POST /chat`

```
POST /chat  { history, k }
    → rewrite_query()           standalone English query from conversation history
    → similarity_search()       top-k chunks from Chroma
    → build_context()           numbered context block + sources
    → generate_answer()         grounded answer from gpt-4o-mini
    → { query, answer, sources }
```

### Request flow — `POST /chat/image`

```
POST /chat/image  { query, image_base64, media_type, k }
    → similarity_search()           top-k chunks from Chroma (text query only)
    → build_context()               numbered context block + sources
    → generate_answer_with_image()  grounded answer from gpt-4o-mini (vision)
    → { query, answer, sources }
```

### Request flow — `POST /knowledge-base`

```
POST /knowledge-base  { doc_id, file_path, file_name }
    → load_single_document()    load file from shared storage path
    → chunk_documents()         split into overlapping chunks
    → index_documents()         embed + persist to Chroma with doc_id stamp
    → { doc_id, file_name, chunks_indexed, status: "indexed" }
```

### Request flow — `DELETE /knowledge-base/{doc_id}`

```
DELETE /knowledge-base/{doc_id}
    → get_ids_by_doc_id()   fetch Chroma vector IDs for doc_id
    → delete_by_doc_id()    remove all matching vectors
    → { doc_id, chunks_deleted, status: "deleted" }
```

---

## Project Structure

```
enterprise-rag-chatbot/
├── data/
│   ├── raw/                    input documents (.pdf, .docx, .txt)
│   ├── processed/
│   │   └── chunks.json         chunk inspection artifact
│   └── chroma/                 persisted vector store
├── src/
│   ├── config.py               central configuration
│   ├── main.py                 FastAPI app entrypoint
│   ├── core/
│   │   └── exceptions.py       shared exception types
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py       GET /health
│   │   │   ├── chat.py         POST /chat
│   │   │   ├── image_chat.py   POST /chat/image
│   │   │   └── knowledge_base.py  POST + DELETE /knowledge-base
│   │   └── schemas/
│   │       ├── chat.py         ChatRequest, ChatResponse, SourceDocument
│   │       ├── image_chat.py   ImageChatRequest
│   │       ├── knowledge_base.py  AddKnowledgeBaseRequest/Response, DeleteKnowledgeBaseResponse
│   │       ├── health.py       HealthResponse
│   │       └── common.py       ErrorResponse
│   ├── data/
│   │   ├── loader.py           document loading (.pdf, .docx, .txt)
│   │   └── chunker.py          text splitting + chunks.json artifact
│   ├── embedding/
│   │   ├── embedder.py         OpenAI embedding client
│   │   └── indexer.py          Chroma vector store init, indexing, and doc_id adapter
│   ├── llm/
│   │   ├── context_builder.py  retrieved docs → formatted context string
│   │   ├── prompt_templates.py system prompt, user prompt, vision messages builder
│   │   └── generator.py        answer generation (text and vision)
│   ├── retrieval/
│   │   ├── retriever.py        similarity and MMR search
│   │   └── evaluator.py        Precision@K evaluation
│   └── utils/
│       └── logger.py           logger factory
├── tests/
│   ├── conftest.py
│   ├── api/
│   │   ├── test_api_health.py
│   │   ├── test_api_chat.py
│   │   └── test_image_chat.py
│   ├── test_context_builder.py
│   ├── test_prompt_templates.py
│   ├── test_generator.py
│   ├── test_e2e_rag_pipeline.py
│   └── test_openai_connection.py
├── pyproject.toml
└── .python-version
```

---

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — used for environment creation and dependency management
- An OpenAI API key

> **Note:** This project was developed on Python 3.14.3 (pre-release).
> Python 3.11.x or 3.12.x is recommended for production deployments
> due to broader package wheel availability.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd enterprise-rag-chatbot

# Create virtual environment and install all dependencies (deterministic)
uv sync

# Install the package in editable mode
uv pip install -e .
```

### Environment variables

Create a `.env` file in the repo root:

```env
OPENAI_API_KEY=sk-...
APP_ENV=development
LOG_LEVEL=DEBUG
```

---

## Indexing Documents

Place your documents in `data/raw/`. Supported formats: `.pdf`, `.docx`, `.txt`.

```bash
# Step 1 — chunk documents and write inspection artifact
uv run python src/data/chunker.py

# Step 2 — embed chunks and persist to Chroma
uv run python src/embedding/indexer.py
```

After indexing, `data/processed/chunks.json` contains the chunked text
for inspection and `data/chroma/` contains the persisted vector store.

---

## Running the API

```bash
uv run uvicorn src.main:app --reload
```

The service starts on `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

---

## API Reference

### `GET /health`

Returns service status, Chroma document count, and OpenAI reachability.

**Response**

```json
{
  "status": "ok",
  "chroma_doc_count": 263,
  "openai_reachable": true,
  "python_version": "3.11.9",
  "app_env": "development"
}
```

Status is `"degraded"` if OpenAI is unreachable or no documents are indexed.

---

### `POST /chat`

Accepts a conversation history and returns a grounded answer with source citations.
The last message in `history` must have `role: "user"`. Multi-turn conversations
are supported — prior turns are used to rewrite the query before retrieval.

**Request body**

```json
{
  "history": [
    { "role": "user", "content": "How do I replace the ink cartridge?" }
  ],
  "k": 5
}
```

| Field     | Type    | Required | Default | Constraints                              |
| --------- | ------- | -------- | ------- | ---------------------------------------- |
| `history` | array   | Yes      | —       | Min 1 message. Last message must be user |
| `k`       | integer | No       | 5       | 1–20                                     |

**Response**

```json
{
  "query": "How do I replace the ink cartridge?",
  "answer": "To replace the ink cartridge, open the printer cover... [1][2]",
  "sources": [
    {
      "file_name": "manual L3210.pdf",
      "file_path": "/data/raw/manual L3210.pdf",
      "chunk_index": 42
    }
  ]
}
```

**Error responses**

| Status | Condition                                              |
| ------ | ------------------------------------------------------ |
| 422    | Invalid request body (last message not user, k out of range) |
| 503    | Query rewrite or answer generation failed              |
| 500    | Unexpected internal error                              |

---

### `POST /chat/image`

Accepts a text query and a base64-encoded image. Retrieval runs on the text query;
the image is sent to gpt-4o-mini as additional visual context. Single-turn only —
no conversation history.

**Request body**

```json
{
  "query": "What does the chart in this image show?",
  "image_base64": "<base64-encoded image, no data URI prefix>",
  "media_type": "image/jpeg",
  "k": 5
}
```

| Field          | Type    | Required | Default | Constraints                                          |
| -------------- | ------- | -------- | ------- | ---------------------------------------------------- |
| `query`        | string  | Yes      | —       | Non-blank                                            |
| `image_base64` | string  | Yes      | —       | Non-blank, no `data:...;base64,` prefix              |
| `media_type`   | string  | Yes      | —       | One of: `image/jpeg`, `image/png`, `image/webp`, `image/gif` |
| `k`            | integer | No       | 5       | 1–20                                                 |

**Response** — same shape as `POST /chat`.

**Error responses**

| Status | Condition                                        |
| ------ | ------------------------------------------------ |
| 422    | Blank query, blank image, unsupported media type |
| 503    | Answer generation failed                         |
| 500    | Retrieval or context building failed             |

---

### `POST /knowledge-base`

Indexes a document into the vector store. The file must already exist at `file_path`
on storage accessible to the RAG service.

**Request body**

```json
{
  "doc_id": "doc_abc123",
  "file_path": "/shared/uploads/manual.pdf",
  "file_name": "manual.pdf"
}
```

**Response**

```json
{
  "doc_id": "doc_abc123",
  "file_name": "manual.pdf",
  "chunks_indexed": 38,
  "status": "indexed"
}
```

**Error responses**

| Status | Condition                              |
| ------ | -------------------------------------- |
| 409    | `doc_id` already exists in the store   |
| 422    | Invalid request body                   |
| 500    | File loading or indexing failed        |

---

### `DELETE /knowledge-base/{doc_id}`

Removes all vectors associated with a document from the vector store.

**Response**

```json
{
  "doc_id": "doc_abc123",
  "chunks_deleted": 38,
  "status": "deleted"
}
```

**Error responses**

| Status | Condition                            |
| ------ | ------------------------------------ |
| 404    | `doc_id` not found in the store      |
| 500    | Unexpected internal error            |

---

## Express.js Integration

```javascript
// POST /chat — conversational query
const chatResponse = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    history: [{ role: "user", content: query }],
    k: 5,
  }),
});

// POST /chat/image — query with image context
const imageResponse = await fetch("http://localhost:8000/chat/image", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query,
    image_base64: base64String,   // strip the "data:...;base64," prefix first
    media_type: "image/jpeg",
    k: 5,
  }),
});

// POST /knowledge-base — index a document
await fetch("http://localhost:8000/knowledge-base", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ doc_id, file_path, file_name }),
});

// DELETE /knowledge-base/:docId — remove a document
await fetch(`http://localhost:8000/knowledge-base/${docId}`, {
  method: "DELETE",
});
```

In production, replace `http://localhost:8000` with the internal
service address of the FastAPI container.

---

## Running Tests

```bash
# Full test suite
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# Single test file
uv run pytest tests/api/test_image_chat.py -v
```

No `OPENAI_API_KEY` is required to run the test suite — all external calls are mocked.

---

## Retrieval Evaluation

```bash
uv run python src/retrieval/evaluator.py
```

Runs Precision@K against the hardcoded test query set and reports
pass/fail against the ≥ 0.60 threshold.

---

## Known Issues and Deferred Items

| Item                                                          | Severity | Target  |
| ------------------------------------------------------------- | -------- | ------- |
| No API authentication                                         | Critical | Phase 5 |
| No containerization (Dockerfile)                              | Major    | Phase 3 |
| No structured logging with correlation IDs                    | Medium   | Backlog |
| Python 3.14.3 is pre-release — not recommended for production | Major    | Phase 3 |

---

## Roadmap

| Phase   | Focus                                                                    | Status      |
| ------- | ------------------------------------------------------------------------ | ----------- |
| Phase 0 | Import cleanup, path normalization                                       | ✅ Done     |
| Phase 1 | API layer, LLM generation, end-to-end pipeline                           | ✅ Done     |
| Phase 2 | Knowledge base CRUD, multi-turn chat, query rewrite                      | ✅ Done     |
| Phase 3 | Unit tests, Chroma adapter refactor, Dockerfile, README                  | ✅ Done     |
| Phase 4 | Image input (vision) via `POST /chat/image`                              | ✅ Done     |
| Phase 5 | Query intelligence: gap logging, category retrieval, SSE streaming       | 🔲 Planned  |
