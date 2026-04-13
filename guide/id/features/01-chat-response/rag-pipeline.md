# RAG Pipeline

[en](../../../en/features/01-chat-response/rag-pipeline.md) · [id](#)

## Ringkasan Proyek

RAG chatbot enterprise menerima query dari IPC handler, mengambil konteks relevan dari vector DB, lalu menghasilkan jawaban via LLM.

---

## Tujuan Langkah Ini

Mengimplementasikan `pipeline.py` sebagai orkestrator akhir: menghubungkan retriever, context builder, dan LLM menjadi satu alur penuh.

---

## Status Implementasi Saat Ini

| Komponen | File | Status |
|---|---|---|
| `similarity_search()` | `retrieval/retriever.py` | ✅ Selesai |
| `mmr_search()` | `retrieval/retriever.py` | ✅ Selesai |
| `evaluate_retriever()` | `retrieval/evaluator.py` | ✅ Selesai |
| `build_context()` | `llm/context_builder.py` | ✅ Selesai |
| `rewrite_query()` | `llm/query_rewriter.py` | ❌ Belum ada |
| `_build_rewrite_prompt()` | `llm/query_rewriter.py` | ❌ Belum ada |
| `_build_prompt()` | `pipeline.py` | ❌ Belum ada |
| `_call_llm()` | `pipeline.py` | ❌ Belum ada |
| `run_rag_pipeline()` | `pipeline.py` | ❌ Belum ada |
| Konstanta LLM | `config.py` | ❌ Belum ada |

---

## Perubahan yang Diperlukan

### `config.py` — Tambahkan konstanta LLM

Konstanta berikut belum ada di `config.py` dan harus ditambahkan sebelum pipeline dibuat:

```python
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024
LLM_TIMEOUT: int = 30  # detik
RETRIEVAL_K: int = 5
```

> **Catatan:** `LLM_TEMPERATURE = 0.0` direkomendasikan untuk RAG — jawaban deterministik,
> tidak "kreatif". Sesuaikan jika dibutuhkan.

---

## Struktur Modul

```
src/
├── retrieval/
│   ├── retriever.py       ✅ similarity_search, mmr_search
│   └── evaluator.py       ✅ evaluate_retriever
├── llm/
│   ├── context_builder.py ✅ build_context, _get_meta
│   └── query_rewriter.py  ❌ (baru) — cleanup + translate query ke English
├── pipeline.py            ❌ (baru) — orkestrasi penuh
└── config.py              ⚠️  perlu tambahan konstanta LLM
```

---

## Rencana Fungsi

### `llm/query_rewriter.py`

| Fungsi | Tujuan | Parameter | Output |
|---|---|---|---|
| `_build_rewrite_prompt()` | Susun prompt instruksi rewrite query | `query: str` | `str` |
| `rewrite_query()` | Cleanup + translate query ke English via LLM | `query: str` | `str` |

### `pipeline.py`

| Fungsi | Tujuan | Parameter | Output |
|---|---|---|---|
| `_build_prompt()` | Susun prompt dari system instruction + konteks + query | `query: str`, `context: str` | `str` |
| `_call_llm()` | Kirim prompt ke OpenAI, return jawaban | `prompt: str` | `str` |
| `run_rag_pipeline()` | Orkestrasi penuh: retrieve → context → generate | `query: str`, `k: int` | `ChatResponse` |

---

## Alur Sistem

```
Query masuk (dari IPC handler)
    │
    ▼
rewrite_query(query)                        ← llm/query_rewriter.py
    │  Cleanup + translate ke English via LLM
    │  Contoh: "printer saya bunyi aneh" → "printer making unusual noise"
    ▼
similarity_search(rewritten_query, ...)     ← retrieval/retriever.py
    │  Ambil k dokumen paling relevan dari Chroma
    ▼
build_context(documents)                    ← llm/context_builder.py
    │  Format dokumen → string konteks bernomor
    │  + ekstrak List[SourceDocument]
    ▼
_build_prompt(query, context)               ← pipeline.py
    │  Gabungkan system instruction + konteks + query ASLI
    │  (bukan rewritten — jawaban tetap dalam bahasa user)
    ▼
_call_llm(prompt)                           ← pipeline.py
    │  Kirim ke OpenAI (model, temperature, timeout dari config)
    ▼
ChatResponse(answer, sources)
    │
    ▼
Dikembalikan ke IPC handler
```

---

## Rekomendasi Urutan Implementasi

### 1. Tambahkan konstanta LLM ke `config.py`

- [ ] Tambahkan `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `RETRIEVAL_K`
- [ ] Pastikan tidak ada nilai yang di-hardcode di `pipeline.py`

---

### 1. Tambahkan konstanta LLM ke `config.py`

- [ ] Tambahkan `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`, `RETRIEVAL_K`
- [ ] Pastikan tidak ada nilai yang di-hardcode di `pipeline.py`

---

### 2. `query_rewriter.py`

- [ ] `_build_rewrite_prompt()` — susun prompt instruksi rewrite (lihat contoh di bawah)
- [ ] `rewrite_query()` — panggil LLM dengan prompt rewrite, return string hasil rewrite

Contoh struktur prompt rewrite:

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

Contoh input → output:

| Input (user) | Output (rewritten) |
|---|---|
| `printer saya bunyi aneh` | `printer making unusual noise` |
| `gimana cara ganti tinta` | `how to replace ink cartridge` |
| `paper nyangkut terus` | `paper jam issue` |
| `how do i scan a dokumen` | `how do I scan a document` |

---

### 3. `_build_prompt()`

- [ ] Buat system instruction yang memerintahkan LLM untuk menjawab **hanya** berdasarkan konteks
- [ ] Sertakan konteks bernomor dari `build_context()`
- [ ] Tambahkan query user di bagian akhir prompt

Contoh struktur prompt:

```
You are a helpful assistant. Answer the user's question based ONLY
on the provided context. If the answer is not in the context, say
"I don't know based on the available documents."

Context:
[1] Source: manual.pdf | Type: .pdf | Chunk: 2 | Page: 4
...teks chunk...

[2] Source: policy.docx | Type: .docx | Chunk: 7
...teks chunk...

Question: {query}
```

---

### 4. `_call_llm()`

- [ ] Inisialisasi `OpenAI` client menggunakan `config.OPENAI_API_KEY`
- [ ] Kirim dengan `model`, `temperature`, `max_tokens`, `timeout` dari `config`
- [ ] Return `response.choices[0].message.content`
- [ ] Tangani `openai.APITimeoutError` dan `openai.APIError` — raise ulang sebagai exception yang jelas

---

### 5. `run_rag_pipeline()`

- [ ] Load vector store via `get_vector_store()`
- [ ] Panggil `rewrite_query(query)` — gunakan hasil rewrite untuk similarity search
- [ ] Panggil `similarity_search(rewritten_query, vector_store, k)`
- [ ] Panggil `build_context(documents)` — tangani `ValueError` jika docs kosong
- [ ] Panggil `_build_prompt(query, context)` — gunakan query **asli**, bukan rewritten
- [ ] Panggil `_call_llm(prompt)`
- [ ] Return `ChatResponse(answer=answer, sources=sources)`

---

### 6. Verifikasi Manual

- [ ] Jalankan `run_rag_pipeline()` langsung dari terminal dengan query contoh **berbahasa Indonesia**
- [ ] Pastikan `rewrite_query()` menghasilkan query English yang tepat
- [ ] Pastikan `answer` relevan dengan query
- [ ] Pastikan `sources` berisi nama file dan nomor halaman yang benar
- [ ] Coba query yang **tidak ada** di dokumen — pastikan LLM menjawab "I don't know"

---

## Catatan Implementasi

- **Jangan hardcode** model name atau temperature — selalu ambil dari `config`
- **`rewrite_query()` untuk retrieval saja** — query asli tetap dipakai di `_build_prompt()` agar jawaban LLM mengikuti bahasa user
- **`build_context()` raise `ValueError`** jika `documents` kosong — tangani di `run_rag_pipeline()`, jangan biarkan crash ke IPC handler
- **Timeout wajib ada** — tanpa timeout, koneksi ke OpenAI bisa hang dan memblokir IPC server
- **`similarity_search()` vs `mmr_search()`** — gunakan `similarity_search()` dulu; ganti ke `mmr_search()` jika hasil retrieval terasa redundan
- **Vector store** — load sekali saat startup jika memungkinkan, jangan load ulang tiap request

---

## Tambahan di `src/config.py`

```python
LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024
LLM_TIMEOUT: int = 30
RETRIEVAL_K: int = 5
```