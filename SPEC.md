# mcp-memory-service-rs — Rust Port Specification

**Status:** Draft v0.1 — 2026-04-17
**Source:** `/Users/hkr/Repositories/mcp-memory-service` (Python, all features)
**Target:** Schema-compatible Rust drop-in, reduced tool surface, same SQLite file.

---

## 1. Goal

Port the 7 most-used MCP memory tools to Rust so existing SQLite data works unchanged when the binary swaps in. Wins: ~10× cold-start, ~4× RAM, ~2–3× query latency, single-binary distribution.

## 2. Non-Goals (Explicitly Dropped)

- Consolidation (decay, clustering, associations, forgetting)
- Harvest / ingestion (PDF, CSV, JSON loaders)
- Quality scoring / AI evaluator / ONNX reranker
- Graph storage, Cloudflare backend, hybrid sync
- Web dashboard, mDNS discovery
- Backup scheduler
- HTTP transport (stdio MCP only for MVP)

If any of those is load-bearing for Henry today, flag before MVP lock.

## 3. Tool Surface (7 tools, stdio MCP)

All tools stay wire-compatible with Python version. JSON shapes identical.

### 3.1 `store_memory`
```
content: string (required)
tags: string | string[] | null         # "a,b" or ["a","b"]; strip() each
memory_type: string = "note"
metadata: object | null                # merge metadata.tags into tags
client_hostname: string | null
→ { success, content_hash, split? } | { success:false, error }
```
Dedup by `content_hash` (SHA256 of content). UNIQUE collision → no-op + return existing hash. Skip the Python 800-char chunking (that's only Cloudflare/Hybrid).

### 3.2 `retrieve_memory`
```
query: string
n_results: int = 10
quality_boost: bool | null             # MVP: ignore, always false
quality_weight: float | null           # MVP: ignore
→ [{ memory, relevance_score: 0..1, debug_info }]
```
Cosine similarity via sqlite-vec KNN. Normalize embeddings L2 before store and query.

### 3.3 `search_by_tag`
```
tags: string[]
tag_match: "any" | "all" = "any"
memory_type: string | null
→ [Memory]
```
**Preserve Python behavior exactly:** GLOB pattern `(',' || REPLACE(tags, ' ', '') || ',') GLOB '*,<tag>,*'`. Case-sensitive. Whitespace stripped in comparison, preserved in storage. Always `WHERE deleted_at IS NULL`.

### 3.4 `delete_memory`
```
content_hash: string | null
tags: string[] | null
tag_match: "any" | "all" = "any"
before: ISO8601 | null
after: ISO8601 | null
dry_run: bool = false
→ { success, deleted_count, deleted_hashes[], dry_run, error? }
```
**Soft-delete only.** Set `deleted_at = unixepoch()`. Never `DELETE FROM`.

### 3.5 `list_memories`
```
page: int = 1
page_size: int = 10
tag: string | null        # single tag, GLOB-matched same as search_by_tag
memory_type: string | null
→ { memories[], total, page, page_size, total_pages }
```
Filter `deleted_at IS NULL`. `ORDER BY created_at DESC`. Tag GLOB rule identical to §3.3.

### 3.6 `check_database_health`
```
→ { status, total_memories, storage_backend: "sqlite_vec_rs",
    database_info: { path, size_bytes, wal_size_bytes }, timestamp }
```

### 3.7 `get_cache_stats`
```
→ { embedding_model_load_ms, storage_init_ms,
    query_count, store_count, process_uptime_s }
```
Cheap process-local counters. Not the Python module-cache stats.

---

## 4. SQLite Schema (verbatim from Python, MUST match)

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,                       -- comma-separated, case-preserved
    memory_type TEXT,
    metadata TEXT,                   -- JSON blob
    created_at REAL,                 -- Unix epoch float
    updated_at REAL,
    created_at_iso TEXT,             -- ISO8601
    updated_at_iso TEXT,
    deleted_at REAL DEFAULT NULL     -- soft-delete marker
);

CREATE INDEX IF NOT EXISTS idx_deleted_at ON memories(deleted_at);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- sqlite-vec virtual table (managed by extension):
CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
    content_hash TEXT PRIMARY KEY,
    embedding FLOAT[384]
);
```

**Pragmas on open:**
```
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
```
Honor `MCP_MEMORY_SQLITE_PRAGMAS` env override (comma-separated `k=v`).

## 5. Data Model (Rust)

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct Memory {
    pub content: String,
    pub content_hash: String,          // SHA256 hex lowercase
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    pub metadata: serde_json::Value,   // object
    pub created_at: Option<f64>,       // unix epoch
    pub created_at_iso: Option<String>,
    pub updated_at: Option<f64>,
    pub updated_at_iso: Option<String>,
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,   // 384 f32
}

pub struct MemoryQueryResult {
    pub memory: Memory,
    pub relevance_score: f32,          // 0..1
    pub debug_info: serde_json::Map<String, Value>,
}
```

Write both `created_at` (f64) and `created_at_iso` (String) on every insert. Preserve `created_at` on update unless caller passes `preserve_timestamps=false`.

## 6. Embedding Layer

- **Model:** `all-MiniLM-L6-v2`, 384 dims
- **Source archive:** `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`
- **SHA256:** `913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3`
- **Cache path:** `~/.cache/mcp_memory/onnx_models/all-MiniLM-L6-v2/onnx/`
- **Runtime:** `ort` crate (ONNX Runtime bindings)
- **Tokenizer:** `tokenizers` crate, load `tokenizer.json` from archive
- **Normalization:** L2 after mean-pool over token dims
- **Fallback:** If `MCP_EMBEDDING_BACKEND=external`, POST to `MCP_EXTERNAL_EMBEDDING_API_URL` with `MCP_EXTERNAL_EMBEDDING_API_KEY`, model `MCP_EXTERNAL_EMBEDDING_MODEL` (default `nomic-embed-text`). Response shape: OpenAI-style `{data: [{embedding: [...]}]}`.

Lazy init on first embed call. Cache model handle in a `OnceCell`.

## 7. Behavior Preservation Checklist

These are the subtle traps. Each one is a test case.

| # | Behavior | Source |
|---|----------|--------|
| 1 | Tag GLOB match: whitespace-stripped, case-sensitive | sqlite_vec.py ~1650 |
| 2 | Soft-delete: `deleted_at IS NULL` filter in every read | sqlite_vec.py ~450 |
| 3 | Dedup on content_hash UNIQUE, no overwrite by default | sqlite_vec.py (store path) |
| 4 | Dual timestamp (REAL + ISO) written atomically | sqlite_vec.py |
| 5 | `preserve_timestamps` skips `updated_at` advance | sqlite_vec.py ~1200 |
| 6 | `tags` metadata merged with top-level `tags` param | mcp_server.py (store_memory) |
| 7 | Tags stored comma-joined, raw whitespace kept on disk | sqlite_vec.py |
| 8 | Retrieve re-ranking: fetch 3× candidates when quality_boost (MVP: skip) | mcp_server.py |
| 9 | Embeddings L2-normalized identically on store + query | onnx_embeddings.py |
| 10 | Delete by time range uses `created_at REAL` not ISO | sqlite_vec.py |

## 8. Config Surface (env vars)

```
MCP_MEMORY_STORAGE_BACKEND = sqlite_vec    # only value supported
MCP_MEMORY_DB_PATH          = ~/.mcp_memory_service.db
MCP_MEMORY_SQLITE_PRAGMAS   = (optional k=v,k=v overrides)
MCP_EMBEDDING_BACKEND       = onnx | external
MCP_EXTERNAL_EMBEDDING_API_URL
MCP_EXTERNAL_EMBEDDING_API_KEY
MCP_EXTERNAL_EMBEDDING_MODEL = nomic-embed-text
RUST_LOG                    = info
```

No HTTP/SSE env vars — stdio only in MVP.

## 9. Cargo Stack

```toml
[dependencies]
rmcp = "0.x"                  # official MCP SDK (check latest)
tokio = { version = "1", features = ["full"] }
rusqlite = { version = "0.31", features = ["bundled"] }
sqlite-vec = "0.1"            # or load extension at runtime
ort = "2"                     # ONNX Runtime
tokenizers = "0.20"
ndarray = "0.15"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
anyhow = "1"
thiserror = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
chrono = { version = "0.4", features = ["serde"] }
tar = "0.4"
flate2 = "1"
dirs = "5"
once_cell = "1"
```

Target binary size: <25MB stripped (sans ONNX model, which is ~90MB cached separately).

## 10. Migration Strategy

1. User stops Python server.
2. User swaps MCP client config from `uvx mcp-memory-service` → `/path/to/mcp-memory-service-rs`.
3. Same `MCP_MEMORY_DB_PATH` env var → same SQLite file → same data.
4. First query triggers ONNX model cache check (reuses Python's cache dir).
5. Rollback = swap binary back. Zero data migration.

Provide smoke-test CLI: `mcp-memory-service-rs verify --db <path>` → opens DB, counts memories, runs a dummy embed+store+retrieve+delete cycle on a test tag, reports timings. Ship in the binary.

## 11. Resolved Questions (2026-04-17)

### Q1 — rmcp API shape ✅
Use attribute-macro pattern with `#[tool_router]` + `#[tool]`. Each tool is a method on a server struct; parameters come in as `Parameters<T>` (T is a serde/schemars struct), returns wrap in `Json<T>`. Requires `macros` + `server` crate features. Stdio transport is the default example in the docs.

Sketch:
```rust
struct MemoryServer { tool_router: ToolRouter<Self>, storage: Storage, embedder: Embedder }

#[derive(Deserialize, schemars::JsonSchema)]
struct StoreParams { content: String, tags: Option<serde_json::Value>, memory_type: Option<String>, metadata: Option<serde_json::Value> }

#[tool_router]
impl MemoryServer {
    #[tool(name = "store_memory", description = "...")]
    async fn store_memory(&self, Parameters(p): Parameters<StoreParams>) -> Json<StoreResult> { ... }
}
```

### Q2 — sqlite-vec loading ✅
Use the `sqlite-vec` Rust crate. It embeds C source + compiles statically via `cc`. No runtime `.load_extension` gymnastics, no shared-library shipping.

Register once at startup BEFORE any `Connection::open`:
```rust
unsafe {
    rusqlite::ffi::sqlite3_auto_extension(Some(
        std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ())
    ));
}
```
After that, every new connection auto-loads vec0. Query vectors with `zerocopy::AsBytes` to pass `Vec<f32>` as blob.

### Q3 — ONNX cache path ✅
**Mirror Python exactly.** Python uses:
```
Path.home() / ".cache" / "mcp_memory" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx"
```
Reference: `onnx_embeddings.py:63`. Rust MUST use identical path so existing downloads are reused. Use `dirs::home_dir()` + hardcoded subpath. No override env var in MVP — keep it dumb.

### Q4 — list_memories `tag` semantics ✅
Single tag string, matched via the same GLOB rule as `search_by_tag` with `tags=[tag]`, `tag_match="any"`. Case-sensitive, whitespace-stripped. Reference: `mcp_server.py:619`. Returns `{memories, total, page, page_size, total_pages}` (note: Python returns `total` and `total_pages`, not `total_count` — I had this wrong in §3.5, fix below).

### Q5 — get_cache_stats surface ✅
Python's full shape: `{total_calls, hit_rate, storage_cache{hits,misses,size}, service_cache{hits,misses,size}, performance{avg,min,max init_ms}, backend_info}`. That shape is built around Python's module-level cache which doesn't exist in Rust (no stateless HTTP reinits).

**MVP decision:** Return a compatible-shape but semantically Rust-native response. Fake the nested structure but fill with Rust-relevant numbers:
```json
{
  "total_calls": <store_count + retrieve_count + ...>,
  "hit_rate": 1.0,
  "storage_cache": { "hits": <total_calls>, "misses": 0, "size": 1 },
  "service_cache": { "hits": <total_calls>, "misses": 0, "size": 1 },
  "performance": { "avg_init_ms": <embed_load_ms>, "min_init_ms": <db_init_ms>, "max_init_ms": <embed_load_ms> },
  "backend_info": { "backend": "sqlite_vec_rs", "db_path": "...", "model": "all-MiniLM-L6-v2" }
}
```
Clients that grep these fields keep working. Document honestly in README that Rust has no cache-miss path.

### Q6 — stdio logging discipline ✅
- stdout = JSON-RPC protocol **only**. Any print/log to stdout = protocol corruption.
- stderr = all logs. Use `tracing_subscriber::fmt().with_writer(std::io::stderr).init()`.
- Exit code: `0` on clean shutdown, `1` on init failure (DB open, model load), `2` on protocol error.
- Panics must not escape to stdout. Wrap main in `anyhow` + explicit stderr dump before exit.

### Corrections from resolved questions
- **§3.5 `list_memories` return shape:** was `{memories, page, page_size, total_count}` → correct is `{memories, total, page, page_size, total_pages}`. Updated wire-compat requirement.

## 12. Milestones

- **M0 Scaffold — ✅ done (commit `0e39a2e`, 2026-04-17):** Cargo project, rmcp 1.5 stdio server with stub `ping` tool, SQLite open with pragmas per §4, schema migration including sqlite-vec `vec0` virtual table, `verify` subcommand. Clean build, stdio handshake verified end-to-end (initialize → tools/list → EOF shutdown).
- **M1 Store + Retrieve — ✅ done (2026-04-18):** ONNX model load via `ort` 2.0.0-rc.10 + `download-binaries` (self-contained static link, no system ONNX runtime required), `all-MiniLM-L6-v2` auto-download + SHA256 verify + safe tar extract into Python-compatible cache path, mean-pool with attention mask + L2 normalize. `store_memory` and `retrieve_memory` MCP tools wired. New `embed` subcommand for pipeline smoke test. Verified end-to-end: stored 3 memories, queried "compiled programming languages" → Rust first (0.56), "tropical fruits" → Bananas first (0.59). Warm cold-start: **114 ms** model load, **~11 ms** per embed.
- **M2 Tag search + Delete + List — ✅ done (2026-04-18, commit `b2d1014`):** `search_by_tag` with GLOB semantics matching Python exactly (case-sensitive, whitespace-stripped, `any`/`all` modes), `delete_memory` soft-delete via `deleted_at = unixepoch()` with multi-selector filter (hash, tags, time range, memory_type) and `dry_run`, `list_memories` pagination with correct `total` / `total_pages` / `has_more` shape (matches upstream post-PR #731). Refuses empty-filter deletes. Verified end-to-end via stdio MCP against the M1 fixture DB.
- **M3 Health + Cache stats + external API fallback — ✅ done (2026-04-18):** `check_database_health` returns status, alive/soft-deleted counts, DB path + size + WAL size, `sqlite-vec` version, embedding model + dim, timestamp. `get_cache_stats` emits Python-compatible nested shape filled with honest Rust numbers (per SPEC §11-Q5) plus per-tool counters and init timings. External embedding backend wired via `MCP_EMBEDDING_BACKEND=external` → OpenAI-style `/v1/embeddings` POST with optional bearer auth; responses L2-normalized locally and dim-checked against schema. Embedder refactored to an enum (`Onnx` / `External`) with async dispatch. **All 8 M0–M3 tools now wired and exchange traffic end-to-end** — M4 is the remaining gate.
- **M4 Parity test suite — ✅ done (2026-04-18):** `scripts/m4_parity.py` populates a fresh temp DB through the upstream Python `MemoryService`, then runs 23 shape / count / ordering / semantic assertions across all six read-side Rust tools against the same DB. 23/23 passing. First run exposed a real schema bug the embedding-only parity script never would have caught: the Rust port was writing vec0 under `content_hash` FK with an `embedding` column, but the Python upstream uses `rowid` keyed by `memories.id` with a `content_embedding` column declared with `distance_metric=cosine`. Rust now matches verbatim, and the KNN scoring formula simplified from `1 - d²/2` (L2 assumption) to `1 - d` (cosine distance), with 3× oversampling to cover soft-deleted matches. **Ship gate passed.**
- **v0.1.0 released — 2026-04-18:** [tag `v0.1.0`](https://github.com/doobidoo/mcp-memory-service-rs/releases/tag/v0.1.0). CI wired via `.github/workflows/ci.yml` (cargo build+test on ubuntu+macOS, parity suite against freshly-cloned upstream). Repo flipped to public. Benchmark numbers captured in README: cold-start **36× faster** (4475ms → 123ms), RSS **2.4× smaller** (576MB → 242MB), store p50 1.1× faster, retrieve p50 **0.3× (slower)** — the retrieve regression is a known follow-up, suspected to be the KNN oversampling factor.

Total realistic effort with Opus pair: **2–3 weekends.**

## 13. Change Log

- **2026-04-17 — SPEC v0.1** drafted.
- **2026-04-17 — Upstream bug filed:** `list_memories` return-shape docstring drift → [doobidoo/mcp-memory-service#731](https://github.com/doobidoo/mcp-memory-service/pull/731). §3.5 and §11-Q4 updated to reflect real shape (`total`, `total_pages`, `has_more`).
- **2026-04-17 — M0 landed:** scaffold commit `0e39a2e`. Deviations from §9 dep list:
  - `schemars` pinned to `"1"` (rmcp 1.5 requires schemars 1.x; the examples-era docs referenced 0.8).
  - `ort` deferred to M1 — `2.0.0-rc.12` fails to compile on macOS (`src/ep/vitis.rs` references a non-existent `OrtApi` field). M1 will pin to `=2.0.0-rc.10` or evaluate `candle-core` + `candle-transformers` as an alternative.
  - `tokenizers`, `ndarray` deferred along with `ort` (only needed once embeddings are wired).
- **2026-04-17 — Upstream merged:** PR #731 shipped in `doobidoo/mcp-memory-service` v10.38.3.
- **2026-04-18 — M1 landed:** `ort` pinned to `=2.0.0-rc.10` with features `["std", "download-binaries", "ndarray"]` — self-contained binary, no system ONNX runtime required. Noted concurrency nuance: rmcp dispatches tool calls on independent tasks, so a client firing `store_memory` and `retrieve_memory` back-to-back without awaiting the first response can see a race. Real MCP clients await responses sequentially, so this is a test-harness artifact rather than a prod bug — documented here so M4 parity tests don't get surprised.
- **2026-04-18 — Embedding parity verified:** `scripts/parity_check.py` runs each fixture text through the upstream Python `ONNXEmbeddingModel` and this binary's `embed` subcommand, asserting first-5 coefficients agree to 4 decimals. 4/4 fixtures pass. L2 norm 1.000000 both sides. Cosine similarities between stored memories and queries match Python to 4 decimals (including the negative-cosine corner case that originally caused us to clamp in Rust — now fixed).
- **2026-04-18 — License decision:** Switched the Rust port from MIT (placeholder from `cargo init`) to **PolyForm Noncommercial 1.0.0** with a separate commercial track (see `LICENSE`, `COMMERCIAL.md`). Upstream Python project stays Apache-2.0 for now; revisiting upstream relicensing requires contributor consent and is a separate planning item. The two repos therefore have different licenses by design until that planning completes.
