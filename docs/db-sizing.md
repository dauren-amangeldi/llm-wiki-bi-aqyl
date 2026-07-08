# PostgreSQL sizing — BI AQYL / LLM-Wiki

Sizing for the app database **after the storage consolidation** (all state in
Postgres + S3). Use this to fill the DB provisioning request: **initial size**,
**monthly growth**, and **scaling**.

> TL;DR — request **20 GB** to start and **~0.5 GB/month** growth for a 20-user
> MVP. Give the instance **4–8 GB RAM** so the pgvector (HNSW) indexes stay in
> memory. Storage grows with *unique* content only (SHA-256 dedup).

---

## 1. What lives in the database

| Table | Holds | Size driver |
|---|---|---|
| `heading_embeddings` | 1 vector(1536) per wiki page (title) | **vectors + HNSW index** |
| `chunk_embeddings` | 1 vector(1536) + text per ~500-token chunk | **vectors + HNSW index** (largest) |
| `wiki_fts` | full page markdown (`body`) + generated `tsvector` | wiki text |
| `wiki_index` | slug/title/section/level per page | tiny |
| `files` | upload metadata (name, sha, status, cost, pages) | tiny |
| `cases`, `chat_messages` | cases + chat history | grows with usage |
| `issues_report`, `embedding_meta`, `skills`, `users` | reports/config/seed | negligible |

Raw uploaded files (PDF/MD) and generated wiki markdown do **not** grow the DB —
raw files live in **S3** (date-partitioned), wiki text lives in `wiki_fts`.

---

## 2. Per-document estimate

A `vector(1536)` column = `1536 × 4 B = 6 KB`. An **HNSW** index adds roughly
another ~0.5× on top, so budget **~9 KB per vector** (data + index).

Assumptions (typical document):

| Item | Count | Size | Subtotal |
|---|---|---|---|
| wiki pages / doc | ~4 | — | — |
| chunks / doc (~7 per page) | ~28 | 9 KB (vec+idx) + 2 KB (text) | ~310 KB |
| heading vectors / doc | ~4 | 9 KB (vec+idx) | ~36 KB |
| wiki_fts body + tsvector | 4 pages | ~8 KB / page | ~32 KB |
| metadata rows (files/index/…) | — | — | ~15 KB |
| **Total per document** | | | **≈ 0.4–0.5 MB** |

**Vectors are ~80% of the footprint.** Chat history adds ~1 KB per message,
independent of documents.

---

## 3. Growth table

| Scale | Documents | Approx DB size |
|---|---|---|
| Empty (schema + indexes + seed) | 0 | ~50 MB |
| Light | 1,000 | ~0.5 GB |
| Growing | 10,000 | ~5 GB |
| Fills the initial request | ~40,000 | ~20 GB |

**Request:**
- **Initial size: 20 GB** — comfortable to ~40k documents.
- **Monthly growth: ~0.5 GB/month** for a 20-user MVP.
  (≈ 200–500 docs/month × ~0.5 MB + chat ≈ a few MB, rounded up for headroom.)
- Also enable the **`pgvector` extension** (`CREATE EXTENSION vector`) — DBA-level.

---

## 4. RAM — the pgvector caveat

HNSW vector search is fast only when the index is served from memory
(`shared_buffers` / OS page cache). So the instance **RAM should cover the vector
index**, not just the working set:

| Documents | ~Chunk vectors | ~Vector data | Suggested DB RAM |
|---|---|---|---|
| ≤ 5,000 | ≤ ~150k | ~1 GB | **4–8 GB** |
| ~20,000 | ~600k | ~4 GB | 8–16 GB |
| ~40,000 | ~1.3M | ~8 GB | 16–32 GB |

For the MVP (a few thousand docs) **2 vCPU / 4–8 GB RAM / 20 GB storage** is a
solid starting instance.

---

## 5. Horizontal scaling

"Horizontal scaling" of the DB here means **read scaling**, not sharding
(single-primary Postgres):

- **Read replicas** — search / ask / wiki browsing are read-heavy (vector + FTS
  queries); route reads to replicas as load grows.
- **PgBouncer** connection pooling — many stateless app pods must not each open a
  pool of direct DB connections; front the DB with PgBouncer (transaction mode).
- Storage itself scales **vertically** (bigger volume) and grows with *unique*
  content — see dedup below.

The app pods stay stateless, so they scale horizontally freely; the DB is the
shared backing service and scales via replicas + pooling + a bigger instance.

---

## 6. Deduplication effect

Uploads are deduplicated by **SHA-256 of file content** before any work:

- An identical re-upload creates **no** new `files` row, **no** wiki pages, **no**
  vectors, and **no** S3 object.
- So real growth tracks **unique content**, not raw upload count.

For organisations where people re-share the same documents, this materially
slows both DB and S3 growth versus the naive "N uploads × size" estimate.

---

## 7. Summary for the request form

| Field | Value |
|---|---|
| Initial DB size | **20 GB** |
| Monthly growth (MVP, 20 users) | **~0.5 GB/month** |
| DB instance (start) | **2 vCPU / 4–8 GB RAM** |
| Required extension | **pgvector** (`CREATE EXTENSION vector`) |
| Scaling path | read replicas + PgBouncer; grow RAM with vector count |
| S3 bucket (raw files) | **10 GB** start, **~0.5–1 GB/month** |
