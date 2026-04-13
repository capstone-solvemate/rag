The user just wants a translation of the markdown document — no file creation needed for a translation task. I'll translate it directly in chat.

---

# RAG Pipeline

[en](#) · [id](../../../id/features/01-chat-response/rag-pipeline.md)

## Project Summary

Enterprise RAG chatbot receives a query from the IPC handler, retrieves relevant context from a vector DB, then generates an answer via LLM.

---

## Goal of This Step

Implement `pipeline.py` as the final orchestrator: connecting the retriever, context builder, and LLM into one complete flow.

---

## Current Implementation Status

| Component | File | Status |
|---|---|---|
| `similarity_search()` | `retrieval/retriever.py` | ✅ Done |
| `mmr_search()` | `retrieval/retriever.py` | ✅ Done |
| `evaluate_retriever()` | `retrieval/evaluator.py` | ✅ Done |
| `build_context()` | `llm/context_builder.py` | ✅ Done |
| `rewrite_query()` | `llm/query_rewriter.py` | ❌ Not yet |
| `_build_rewrite_prompt()` | `llm/query_rewriter.py` | ❌ Not yet |
| `_build_prompt()` | `pipeline.py` | ❌ Not yet |
| `_call_llm()` | `pipeline.py` | ❌ Not yet |
| `run_rag_pipeline()` | `pipeline.py` | ❌ Not yet |
| LLM constants | `config.py` | ❌ Not yet |

---

## Required Changes

### `config.py` — Add LLM constants

The following constants do not yet exist in `config.py` and must be added before building the pipeline:

```python
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024
LLM_TIMEOUT: int = 30  # seconds
RETRIEVAL_K: int = 5
```

> **Note:** `LLM_TEMPERATURE = 0.0` is recommended for RAG — deterministic answers, not "creative". Adjust if needed.

---

## Module Structure

```
src/
├── retrieval/
│   ├── retriever.py       ✅ similarity_search, mmr_search
│   └── evaluator.py       ✅ evaluate_retriever
├── llm/
│   ├── context_builder.py ✅ build_context, _get_meta
│   └── query_rewriter.py  ❌ (new) — cleanup + translate query to English
├── pipeline.py            ❌ (new) — full orchestration
└── config.py              ⚠️  needs LLM constants added
```

---

## Function Plan

### `llm/query_rewriter.py`

| Function | Purpose | Parameters | Output |
|---|---|---|---|
| `_build_rewrite_prompt()` | Compose rewrite instruction prompt | `query: str` | `str` |
| `rewrite_query()` | Cleanup + translate query to English via LLM | `query: str` | `str` |

### `pipeline.py`

| Function | Purpose | Parameters | Output |
|---|---|---|---|
| `_build_prompt()` | Compose prompt from system instruction + context + query | `query: str`, `context: str` | `str` |
| `_call_llm()` | Send prompt to OpenAI, return answer | `prompt: str` | `str` |
| `run_rag_pipeline()` | Full orchestration: retrieve → context → generate | `query: str`, `k: int` | `ChatResponse` |

---

## System Flow

```
Query received (from IPC handler)
    │
    ▼
rewrite_query(query)                        ← llm/query_rewriter.py
    │  Cleanup + translate to English via LLM
    │  Example: "printer saya bunyi aneh" → "printer making unusual noise"
    ▼
similarity_search(rewritten_query, ...)     ← retrieval/retriever.py
    │  Fetch k most relevant documents from Chroma
    ▼
build_context(documents)                    ← llm/context_builder.py
    │  Format documents → numbered context string
    │  + extract List[SourceDocument]
    ▼
_build_prompt(query, context)               ← pipeline.py
    │  Combine system instruction + context + ORIGINAL query
    │  (not rewritten — LLM answer stays in user's language)
    ▼
_call_llm(prompt)                           ← pipeline.py
    │  Send to OpenAI (model, temperature, timeout from config)
    ▼
ChatResponse(answer, sources)
    │
    ▼
Returned to IPC handler
```

---

## Recommended Implementation Order

### 1. Add LLM constants to `config.py`

- [ ] Add `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `RETRIEVAL_K`
- [ ] Ensure no values are hardcoded in `pipeline.py`

---

### 2. `query_rewriter.py`

- [ ] `_build_rewrite_prompt()` — compose rewrite instruction prompt (see example below)
- [ ] `rewrite_query()` — call LLM with the rewrite prompt, return the rewritten string

Example rewrite prompt structure:

```
You are a query rewriting assistant.
Your job is to rewrite the user's question so it is:
- In English
- Free of typos and grammatical errors
- Concise and technically clear
- Suitable for semantic search against a product manual knowledge base

Return ONLY the rewritten query. No explanation. No punctuation changes beyond corrections.

Original query: {query}
```

Example input → output:

| Input (user) | Output (rewritten) |
|---|---|
| `printer saya bunyi aneh` | `printer making unusual noise` |
| `gimana cara ganti tinta` | `how to replace ink cartridge` |
| `paper nyangkut terus` | `paper jam issue` |
| `how do i scan a dokumen` | `how do I scan a document` |

---

### 3. `_build_prompt()`

- [ ] Create a system instruction that tells the LLM to answer **only** based on the context
- [ ] Include the numbered context from `build_context()`
- [ ] Append the user's query at the end of the prompt

Example prompt structure:

```
You are a helpful assistant. Answer the user's question based ONLY
on the provided context. If the answer is not in the context, say
"I don't know based on the available documents."

Context:
[1] Source: manual.pdf | Type: .pdf | Chunk: 2 | Page: 4
...chunk text...

[2] Source: policy.docx | Type: .docx | Chunk: 7
...chunk text...

Question: {query}
```

---

### 4. `_call_llm()`

- [ ] Initialize `OpenAI` client using `config.OPENAI_API_KEY`
- [ ] Send with `model`, `temperature`, `max_tokens`, `timeout` from `config`
- [ ] Return `response.choices[0].message.content`
- [ ] Handle `openai.APITimeoutError` and `openai.APIError` — re-raise as a clear exception

---

### 5. `run_rag_pipeline()`

- [ ] Load vector store via `get_vector_store()`
- [ ] Call `rewrite_query(query)` — use the rewritten result for similarity search
- [ ] Call `similarity_search(rewritten_query, vector_store, k)`
- [ ] Call `build_context(documents)` — handle `ValueError` if docs are empty
- [ ] Call `_build_prompt(query, context)` — use the **original** query, not rewritten
- [ ] Call `_call_llm(prompt)`
- [ ] Return `ChatResponse(answer=answer, sources=sources)`

---

### 6. Manual Verification

- [ ] Run `run_rag_pipeline()` directly from the terminal with an example query **in Indonesian**
- [ ] Confirm `rewrite_query()` produces an accurate English query
- [ ] Confirm the `answer` is relevant to the query
- [ ] Confirm `sources` contains the correct file names and page numbers
- [ ] Try a query that is **not in the documents** — confirm the LLM responds with "I don't know"

---

## Implementation Notes

- **Never hardcode** model name or temperature — always read from `config`
- **`rewrite_query()` is for retrieval only** — the original query is still used in `_build_prompt()` so the LLM answer follows the user's language
- **`build_context()` raises `ValueError`** if `documents` is empty — handle it in `run_rag_pipeline()`, don't let it crash into the IPC handler
- **Timeout is required** — without a timeout, the OpenAI connection can hang and block the IPC server
- **`similarity_search()` vs `mmr_search()`** — use `similarity_search()` first; switch to `mmr_search()` if retrieval results feel redundant
- **Vector store** — load once at startup if possible, don't reload on every request

---

## Additions to `src/config.py`

```python
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024
LLM_TIMEOUT: int = 30
RETRIEVAL_K: int = 5
```